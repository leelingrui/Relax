# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""The task-scoped CPU owner of every inference role's engines and state."""

import asyncio
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from threading import RLock
from typing import Any, Callable
from uuid import uuid4

import ray

from relax.core.node_group_affinity import with_control_plane_affinity
from relax.engine.inference.config import ModelConfig
from relax.engine.inference.lifecycle import (
    ActivationGroupSnapshot,
    ActivationToken,
    ErrorCode,
    HandoffResult,
    LifecycleCoordinator,
    OperationError,
    OperationResult,
    OperationStatus,
    PhaseHandle,
    PhasePlan,
    PhaseResult,
    ReleaseEvidence,
)
from relax.engine.inference.manager import (
    InferenceManager,
    OperationSnapshot,
    RequestPermit,
)
from relax.engine.inference.phase_plans import phase_plans_from_contentions
from relax.engine.inference.placement import (
    PhaseContention,
    PlacementGroupView,
    PlacementPlanner,
    PlacementRelease,
    PlacementRequest,
    PlacementSlice,
)
from relax.engine.inference.types import LifecycleState, ModelRef, Role, RoleSnapshot, RoutingSpec
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


class _FixedReleaseAdapter:
    """Carry already-collected training evidence to the Coordinator."""

    def __init__(self, evidence: ReleaseEvidence) -> None:
        self._evidence = evidence

    def prepare_training_handoff(
        self, batch_id: str, policy_version: str | None, *, operation_id: str, activation_token: Any
    ) -> HandoffResult:
        return HandoffResult(operation_id, batch_id, True, policy_version)

    def release_training_resources(self, *, operation_id: str, activation_token: Any) -> ReleaseEvidence:
        return self._evidence


class TaskInferenceManager:
    """CPU control-plane owner for every inference role of one task.

    Rollout, GenRM and Teacher engine pools are all created in this owner's
    process and bound to its one :class:`InferenceManager`, so discovery,
    admission, lifecycle, placement and shutdown have a single writer.
    """

    def __init__(self) -> None:
        # Guards only compound updates that do not cross a remote call: a drain
        # blocks inside this actor while the Gateway reports completions on
        # another thread, so nothing may hold a lock across an RPC.
        self._lock = RLock()
        self._placement_planner = PlacementPlanner()
        self._control_manager = InferenceManager()
        # One epoch for the whole task: snapshots, permits and operations all
        # carry the control manager's, so a permit always matches the snapshot
        # it was admitted against.
        self.manager_epoch = self._control_manager.manager_epoch
        self._operations: dict[str, OperationSnapshot] = {}
        self._role_pools: dict[Role, dict[str, Any]] = {}
        # The rollout engine pool, once this owner created it. It is one pool
        # object covering every rollout model, unlike the static roles where
        # each model has its own adapter.
        self._rollout_pool: Any = None
        self._coordinator: LifecycleCoordinator | None = None

    def registered_roles(self) -> tuple[str, ...]:
        return tuple(role.value for role in self._role_pools)

    def role_models(self, role: Role | str) -> tuple[str, ...]:
        """The model IDs this owner serves for ``role``, in route order."""
        return tuple(self._role_pools.get(Role(role), {}))

    def register_model(self, role: Role | str, spec: ModelConfig, *, operation_id: str) -> ModelConfig:
        role = Role(role)
        result = self._control_manager.register_model(spec, operation_id=operation_id, role=role)
        self._operations[operation_id] = OperationSnapshot(
            operation_id, self.manager_epoch, "completed", "register", result
        )
        return result

    def configure_routes(self, role: Role | str, routing: RoutingSpec, *, operation_id: str) -> RoutingSpec:
        role = Role(role)
        result = self._control_manager.configure_routes(routing, operation_id=operation_id, role=role)
        self._operations[operation_id] = OperationSnapshot(
            operation_id, self.manager_epoch, "completed", "routes", result
        )
        return result

    def get_operation(self, operation_id: str) -> OperationSnapshot:
        operation = self._operations.get(operation_id)
        if operation is not None:
            return operation
        for role in self._role_pools:
            try:
                return self._control_manager.get_operation(operation_id, role=role)
            except KeyError:
                continue
        raise KeyError(f"Unknown operation: {operation_id}")

    def snapshot(self, *, role: Role | str) -> RoleSnapshot:
        role = Role(role)
        if role not in self._role_pools:
            raise RuntimeError(f"Inference role is not registered: {role.value}")
        return replace(self._control_manager.snapshot(role=role), manager_epoch=self.manager_epoch)

    def admit_request(
        self,
        model_id: str,
        request_id: str | None = None,
        *,
        role: Role | str,
        target: str | None = None,
    ) -> RequestPermit:
        """Admit against a freshly committed observation, then register it.

        Registration lives in the one control manager so a drain waits on the
        same in-flight index the Gateway reports completions to.
        """
        return self._control_manager.admit_request(model_id, request_id, role=Role(role), target=target)

    def _is_registered(self, role: Role) -> bool:
        """Whether this owner holds the model definitions for ``role``."""
        return bool(self._control_manager.snapshot(role=role).models)

    def complete_request(self, permit: RequestPermit | str) -> None:
        self._control_manager.complete_request(permit)

    def cancel_request(self, permit: RequestPermit | str, *, dispatched: bool = True) -> Any:
        """Start an abort without claiming the request finished.

        ``dispatched=False`` means the request never reached an engine, which
        is the only case where dropping the registration is honest.
        """
        return self._control_manager.cancel_request(permit, dispatched=dispatched)

    def get_request(self, request_id: str) -> RequestPermit | None:
        return self._control_manager.get_request(request_id)

    # ------------------------------------------------------------------
    # Unified lifecycle: the owner implements the Coordinator's manager port.
    # ------------------------------------------------------------------
    def _unsupported(self, kind: str, targets: Sequence[ModelRef], operation_id: str) -> OperationResult | None:
        """Refuse a managed transition for a model this owner cannot control.

        A role with no registered models has no bound pool, so there is no way
        to close admission or confirm a release for it. Saying ``unsupported``
        is the only answer that does not fake a completion.
        """
        unmanaged = sorted(str(target) for target in targets if not self._is_registered(target.role))
        if not unmanaged:
            return None
        return OperationResult(
            operation_id=operation_id,
            owner_epoch=self.manager_epoch,
            status=OperationStatus.FAILED,
            kind=kind,
            targets=tuple(targets),
            error=OperationError(ErrorCode.UNSUPPORTED, f"Models are not registered with the task owner: {unmanaged}"),
        )

    def drain(
        self,
        targets: Sequence[ModelRef],
        *,
        operation_id: str,
        activation_token: ActivationToken | None = None,
        timeout_s: float | None = None,
    ) -> OperationResult:
        refusal = self._unsupported("drain", targets, operation_id)
        if refusal is not None:
            return refusal
        for target in targets:
            aborted = self._control_manager.cancelling_requests(target)
            if aborted:
                # Not a drain failure: the engine's release path pauses
                # admission, aborts in flight and waits for flush_cache to
                # confirm the scheduler is empty. Say so rather than hiding it.
                logger.info(
                    "Draining %s past %d aborted request(s) left to the engine release drain: %s",
                    target,
                    len(aborted),
                    list(aborted[:8]),
                )
        return self._control_manager.drain(
            targets, operation_id=operation_id, activation_token=activation_token, timeout_s=timeout_s
        )

    def deactivate(
        self,
        targets: Sequence[ModelRef],
        *,
        operation_id: str,
        activation_token: ActivationToken | None = None,
    ) -> OperationResult:
        refusal = self._unsupported("deactivate", targets, operation_id)
        if refusal is not None:
            return refusal
        return self._control_manager.deactivate(targets, operation_id=operation_id, activation_token=activation_token)

    def activate(
        self,
        targets: Sequence[ModelRef],
        *,
        operation_id: str,
        activation_token: ActivationToken | None = None,
        tags: list[str] | None = None,
    ) -> OperationResult:
        refusal = self._unsupported("activate", targets, operation_id)
        if refusal is not None:
            return refusal
        return self._control_manager.activate(
            targets,
            operation_id=operation_id,
            activation_token=activation_token,
            tags=tags,
        )

    def shutdown_models(
        self,
        targets: Sequence[ModelRef],
        *,
        operation_id: str,
        activation_token: ActivationToken | None = None,
        timeout_s: float | None = None,
    ) -> OperationResult:
        refusal = self._unsupported("shutdown_models", targets, operation_id)
        if refusal is not None:
            return refusal
        for target in targets:
            aborted = self._control_manager.cancelling_requests(target)
            if aborted:
                logger.warning(
                    "Shutting down %s while %d aborted request(s) are unconfirmed: %s",
                    target,
                    len(aborted),
                    list(aborted[:8]),
                )
        return self._control_manager.shutdown_models(
            targets, operation_id=operation_id, activation_token=activation_token, timeout_s=timeout_s
        )

    def get_lifecycle_operation(self, operation_id: str) -> OperationResult:
        if self._coordinator is not None:
            try:
                return self._coordinator.get_operation(operation_id)
            except Exception:
                pass
        return self._control_manager.get_lifecycle_operation(operation_id)

    def model_states(self) -> dict[ModelRef, LifecycleState | None]:
        return self._control_manager.model_states()

    # ------------------------------------------------------------------
    # Phase coordination.
    # ------------------------------------------------------------------
    def create_coordinator(
        self,
        session_id: str,
        phase_targets: Mapping[str, Sequence[ModelRef]] | None = None,
        plans: Sequence[PhasePlan] = (),
        deferred: Sequence[str] = (),
    ) -> str:
        """Create the one Coordinator for this task and take activation
        authority.

        The mutual exclusions come from this owner's own placement ledger:
        ``phase_targets`` says which models each planner phase owns, and the
        ledger says which of those phases actually share GPUs. ``deferred`` adds
        the phases that sleep outside their own scoring stage even when they
        share nothing. Binding the Coordinator to the control manager is what
        stops a user script from onloading a colocated role behind its back.

        Returns the coordinator epoch, or an empty string when the layout needs
        no sequencing at all -- no shared slice means no plan, and creating one
        would only serialize roles that were given their own GPUs.
        """
        with self._lock:
            contentions = self._placement_planner.contended_phases()
            derived = tuple(plans)
            if phase_targets:
                derived += phase_plans_from_contentions(phase_targets, contentions, deferred=deferred)
            if self._coordinator is None:
                if not derived:
                    logger.info("No contended inference phases; the task runs without a lifecycle coordinator")
                    return ""
                self._coordinator = LifecycleCoordinator(
                    self,
                    session_id=session_id,
                    plans=derived,
                    readiness=lambda target: self.model_states().get(target) or LifecycleState.STARTING,
                )
                self._control_manager.bind_coordinator(self._coordinator.coordinator_epoch)
            else:
                for plan in derived:
                    self._coordinator.register_plan(plan)
            self._coordinator.validate_placement(contentions)
            return self._coordinator.coordinator_epoch

    def locate_phase(self, phase_id: str) -> tuple[str, str] | None:
        """Resolve a planner phase label to ``(plan_id, activation_group)``.

        Callers name the phase they need -- ``genrm``, ``teacher`` -- because
        the activation group is named after the shared slice and is therefore
        only known once placement resolved.
        """
        if self._coordinator is None:
            return None
        for plan in self._coordinator.plans():
            if any(phase.phase_id == phase_id for phase in plan.phases):
                return plan.plan_id, plan.activation_group
        return None

    def _coordinator_or_raise(self) -> LifecycleCoordinator:
        if self._coordinator is None:
            raise RuntimeError("No lifecycle coordinator has been created for this task")
        return self._coordinator

    def has_coordinator(self) -> bool:
        return self._coordinator is not None

    def register_phase_plan(self, plan: PhasePlan) -> PhasePlan:
        return self._coordinator_or_raise().register_plan(plan)

    def switch_model(
        self,
        activation_group: str,
        target: ModelRef | Sequence[ModelRef],
        *,
        operation_id: str,
        timeout_s: float | None = None,
        tags: list[str] | None = None,
    ) -> PhaseResult:
        return self._coordinator_or_raise().switch_model(
            activation_group, target, operation_id=operation_id, timeout_s=timeout_s, tags=tags
        )

    def transition(
        self,
        activation_group: str,
        target_phase: str,
        *,
        operation_id: str,
        timeout_s: float | None = None,
        tags: list[str] | None = None,
    ) -> PhaseResult:
        return self._coordinator_or_raise().transition(
            activation_group, target_phase, operation_id=operation_id, timeout_s=timeout_s, tags=tags
        )

    def enter_phase(
        self,
        plan_id: str,
        phase_id: str,
        *,
        operation_id: str,
        timeout_s: float | None = None,
        tags: list[str] | None = None,
    ) -> PhaseHandle:
        return self._coordinator_or_raise().enter_phase(
            plan_id, phase_id, operation_id=operation_id, timeout_s=timeout_s, tags=tags
        )

    def finish_phase(
        self,
        handle: PhaseHandle,
        *,
        operation_id: str,
        outcome: str = "completed",
        timeout_s: float | None = None,
    ) -> PhaseResult:
        return self._coordinator_or_raise().finish_phase(
            handle, operation_id=operation_id, outcome=outcome, timeout_s=timeout_s
        )

    def get_activation_group(self, activation_group: str) -> ActivationGroupSnapshot:
        return self._coordinator_or_raise().get_activation_group(activation_group)

    def confirm_training_release(
        self, activation_group: str, evidence: ReleaseEvidence, *, operation_id: str
    ) -> ReleaseEvidence:
        """Record training's own release evidence for a shared activation
        group.

        The evidence is produced by the training backend, which is the only
        side that can speak for all of its ranks; this owner only records it,
        and an unconfirmed release keeps the group closed to the next role.
        """
        coordinator = self._coordinator_or_raise()
        return coordinator.release_training_resources(
            activation_group, _FixedReleaseAdapter(evidence), operation_id=operation_id
        )

    def adopt_phase(self, activation_group: str, phase_id: str, *, operation_id: str) -> PhaseResult:
        return self._coordinator_or_raise().adopt_phase(activation_group, phase_id, operation_id=operation_id)

    def activation_groups(self) -> tuple[str, ...]:
        return () if self._coordinator is None else self._coordinator.activation_groups()

    def coordinated_phases(self) -> tuple[str, ...]:
        """The planner phase labels this task actually sequences."""
        return () if self._coordinator is None else self._coordinator.phase_ids()

    def plan_placement(
        self,
        requests: Sequence[PlacementRequest],
        placement_group: PlacementGroupView,
        *,
        dry_run: bool = False,
    ) -> tuple[PlacementSlice, ...]:
        """Resolve a layout against the one ledger this task owns."""
        return self._placement_planner.plan(requests, placement_group, dry_run=dry_run)

    def release_placement(
        self, placement: PlacementSlice | PlacementGroupView, group_id: str | None = None
    ) -> PlacementRelease:
        """Release ledger entries and report whether the group may be
        removed."""
        return self._placement_planner.release(placement, group_id=group_id)

    def allocations(self, placement_group: PlacementGroupView | None = None) -> tuple[PlacementSlice, ...]:
        return self._placement_planner.allocations(placement_group)

    def contended_phases(self, placement_group: PlacementGroupView | None = None) -> tuple[PhaseContention, ...]:
        """Report the phase exclusions a lifecycle coordinator must honor."""
        return self._placement_planner.contended_phases(placement_group)

    def ready(self, *, role: Role | str) -> bool:
        role = Role(role)
        return role in self._role_pools and self._control_manager.ready(role=role)

    def call(self, role: Role | str, model_id: str, method: str, /, *args: Any, **kwargs: Any) -> Any:
        """Run one public pool method on a model, serialized with its
        lifecycle."""
        if method not in _POOL_METHODS:
            raise ValueError(f"Unsupported pool method: {method}")
        role = Role(role)
        if role not in self._role_pools:
            raise RuntimeError(f"Inference role is not registered: {role.value}")
        return self._control_manager.dispatch(model_id, method, *args, wait=True, role=role, **kwargs)

    def lifecycle(self, role: Role | str, model_id: str, method: str, /, *args: Any, **kwargs: Any) -> Any:
        """Run a serialized pool lifecycle operation through the task owner."""
        if method not in _LIFECYCLE_METHODS:
            raise ValueError(f"Unsupported lifecycle method: {method}")
        return self.call(role, model_id, method, *args, **kwargs)

    @ray.method(concurrency_group="rollout")
    def create_rollout_role(self, args: Any, placement_group: Any) -> dict[str, Any]:
        """Create the rollout engines here, in the owner's own process.

        The pool is built against this owner's role-scoped ``InferenceManager``
        and placement ledger, so the rollout engines are registered, routed and
        accounted for exactly like the static roles. Repeat calls reuse the
        pool instead of starting a second one.

        Returns the primary router endpoint: the pool starts the router in
        this process, so the rollout process cannot observe that endpoint by
        itself and has to write it back into its own ``args``.
        """
        from relax.distributed.ray.rollout import RolloutEnginePool

        if self._rollout_pool is None:
            manager = self._control_manager.for_role(Role.ROLLOUT)
            pool = RolloutEnginePool(
                args,
                placement_group,
                inference_manager=manager,
                placement_ledger=self._placement_planner,
            )
            self._rollout_pool = pool
            self._role_pools[Role.ROLLOUT] = pool.model_pools
            self._operations["routes:rollout:startup"] = OperationSnapshot(
                "routes:rollout:startup",
                self.manager_epoch,
                "completed",
                "routes",
                manager.snapshot().routing,
            )
        return self._rollout_pool.get_primary_router_address()

    @ray.method(concurrency_group="rollout")
    def rollout_operation(self, method: str, /, *args: Any, **kwargs: Any) -> Any:
        """Run one engine-pool operation on the rollout pool this owner holds.

        Only the rollout role's own Ray entry point calls this; the method
        names are the engine-pool surface listed in
        :data:`_ROLLOUT_POOL_METHODS`. Coroutine results are driven to
        completion here so the owner stays a threaded actor: an async method
        would put every blocking control-plane call (notably ``drain``) on a
        single event loop.
        """
        if method not in _ROLLOUT_POOL_METHODS:
            raise ValueError(f"Unsupported rollout pool method: {method}")
        if self._rollout_pool is None:
            raise RuntimeError("The rollout engine pool has not been created on this owner")
        if method not in _ROLLOUT_TRACKED_OPERATIONS:
            return self._run_rollout_method(method, *args, **kwargs)

        # An elastic operation is recorded like every other role's, so one
        # ``get_operation`` answers for the whole task. Only the externally
        # triggered ones are recorded: onload/offload and recovery run on every
        # training step and would grow this ledger without bound.
        request_id = args[0] if args and isinstance(args[0], str) else uuid4().hex
        operation_id = f"{method}:{Role.ROLLOUT.value}:{request_id}"
        self._operations[operation_id] = OperationSnapshot(operation_id, self.manager_epoch, "running", method)
        try:
            result = self._run_rollout_method(method, *args, **kwargs)
        except Exception as exc:
            self._operations[operation_id] = replace(self._operations[operation_id], status="failed", error=str(exc))
            raise
        self._operations[operation_id] = replace(self._operations[operation_id], status="completed", result=result)
        return result

    def _run_rollout_method(self, method: str, /, *args: Any, **kwargs: Any) -> Any:
        result = getattr(self._rollout_pool, method)(*args, **kwargs)
        if asyncio.iscoroutine(result):
            return asyncio.run(result)
        return result

    def create_role(
        self, args: Any, role: Role | str, pool_configs: Mapping[str, Mapping[str, Any]]
    ) -> tuple[str, ...]:
        """Create a static role's model pools here and return their model IDs.

        Every model is registered and the routes are committed before any pool
        starts a GPU process; a failure closes whatever did start.
        """
        role = Role(role)
        if role in self._role_pools:
            return self.role_models(role)
        _validate_configs(role, pool_configs)
        manager = self._control_manager.for_role(role)
        pools: dict[str, Any] = {}
        try:
            for model_id, config in pool_configs.items():
                config = deepcopy(config)
                pools[model_id] = _create_model_pool(
                    role,
                    *config.get("args", ()),
                    **config.get("kwargs", {}),
                    inference_manager=manager,
                    model_id=model_id,
                    defer_init=True,
                    placement_manager_handle=self._placement_planner,
                )
            manager.configure_routes(
                RoutingSpec(
                    default_model=next(iter(pools)) if len(pools) == 1 else None,
                    route_key_to_model=tuple((model_id, model_id) for model_id in pools),
                ),
                operation_id=f"routes:{role.value}:startup",
            )
            for pool in pools.values():
                pool.initialize()
        except Exception:
            try:
                UnifiedServiceManager(
                    role, inference_manager=manager, pools=pools, stop_routers=_stop_role_routers
                ).shutdown()
            except Exception:
                logger.exception("Failed to clean up inference role %s", role.value)
            raise
        UnifiedServiceManager(role, inference_manager=manager, pools=pools, stop_routers=_stop_role_routers)
        self._role_pools[role] = pools
        self._operations[f"routes:{role.value}:startup"] = OperationSnapshot(
            f"routes:{role.value}:startup", self.manager_epoch, "completed", "routes", manager.snapshot().routing
        )
        return self.role_models(role)

    def shutdown_role(self, role: Role | str) -> None:
        role = Role(role)
        if role is Role.ROLLOUT and self._rollout_pool is not None:
            # dispose stops the monitor threads and closes the models through
            # the same manager the pools are bound to.
            self._rollout_pool.dispose()
            self._rollout_pool = None
            self._role_pools.pop(role, None)
            return
        if role in self._role_pools:
            self._control_manager.shutdown(role=role)
            self._role_pools.pop(role, None)

    def shutdown_all(self) -> None:
        """Close every registered role through this task-scoped owner."""
        errors = []
        for role in tuple(self._role_pools):
            try:
                self.shutdown_role(role)
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError("Failed to shut down one or more inference roles") from errors[0]


# Rollout engine operations get their own execution slots: a scale-out runs
# for minutes and must not consume the slots the control plane needs for
# admission, completion reports and drains.
TaskInferenceManagerActor = ray.remote(num_cpus=1, num_gpus=0, concurrency_groups={"rollout": 8})(TaskInferenceManager)

# A drain blocks inside this actor until the registered requests report
# completion, and those reports arrive as calls on this same actor. One
# execution slot would therefore make every drain time out by construction.
_TASK_MANAGER_CONCURRENCY = 8


def create_task_inference_manager(args: Any, runtime_env: dict[str, Any] | None = None) -> Any:
    """Create the single task-level CPU inference control-plane actor.

    It is pinned to the head node because it starts the SGLang routers and the
    rest of the job resolves a router by the head node's address.
    """
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from relax.core.node_group_affinity import require_control_plane_resource_on_node
    from relax.distributed.ray.placement_group import _get_head_node_id

    head_node_id = _get_head_node_id()
    require_control_plane_resource_on_node(args, head_node_id)
    return TaskInferenceManagerActor.options(
        **with_control_plane_affinity(
            args,
            {
                "num_cpus": 1,
                "num_gpus": 0,
                "runtime_env": runtime_env,
                "max_concurrency": _TASK_MANAGER_CONCURRENCY,
                "scheduling_strategy": NodeAffinitySchedulingStrategy(node_id=head_node_id, soft=False),
            },
        )
    ).remote()


_POOL_METHODS = frozenset(
    {
        "health_check",
        "recover",
        "onload",
        "offload",
        "is_onloaded",
        "shutdown",
        "get_urls",
        "get_engine_hosts_ports",
        "fanout",
        "retire",
        "set_onloaded",
    }
)
_LIFECYCLE_METHODS = frozenset({"health_check", "recover", "onload", "offload", "shutdown"})
_OWNED_KWARGS = frozenset({"inference_manager", "model_id", "defer_init", "placement_manager_handle"})

# The elastic operations an outside caller triggers, and which therefore get a
# recorded operation. Per-step traffic (onload/offload, recovery) is left out
# on purpose: recording it would grow the owner's ledger for the whole run.
_ROLLOUT_TRACKED_OPERATIONS = frozenset(
    {
        "cancel_all_scale_out_requests",
        "cancel_scale_out",
        "execute_scale_in",
        "execute_scale_out",
        "sync_weights_for_scaled_out_engines",
    }
)

# The rollout engine-pool surface the rollout Ray entry point may drive. It is
# the pool's public API minus the pieces the owner drives itself (creation and
# shutdown), kept explicit so no caller can reach an arbitrary attribute.
_ROLLOUT_POOL_METHODS = frozenset(
    {
        "call",
        "cancel_all_scale_out_requests",
        "cancel_scale_out",
        "check_weights",
        "clear_num_new_engines",
        "complete_inference_weight_update",
        "create_scale_in_request",
        "create_scale_out_request",
        "execute_scale_in",
        "execute_scale_out",
        "get_discovery_snapshot",
        "get_engines_info",
        "get_primary_router_address",
        "get_rollout_engines_and_lock",
        "get_router_address",
        "get_scale_in_status",
        "get_scale_out_status",
        "get_status",
        "get_weight_sync_lock",
        "health_monitoring_pause",
        "health_monitoring_resume",
        "inject_ci_fault",
        "invalidate_inference_state",
        "list_all_scale_in_requests",
        "list_all_scale_out_requests",
        "offload",
        "onload",
        "onload_kv",
        "onload_weights",
        "ready",
        "recover_rollout_engines",
        "refresh_inference_state",
        "set_force_unhealthy",
        "set_weight_updating",
        "sync_weights_for_scaled_out_engines",
    }
)


def _create_model_pool(role: Role, *args: Any, **kwargs: Any) -> Any:
    from relax.distributed.ray.model_pool import create_model_pool

    return create_model_pool(role, *args, **kwargs)


def _stop_role_routers() -> None:
    from relax.distributed.ray.rollout import stop_launched_routers

    stop_launched_routers()


def _validate_configs(role: Role, pool_configs: Mapping[str, Mapping[str, Any]]) -> None:
    if role not in (Role.GENRM, Role.TEACHER):
        raise ValueError(f"Unsupported inference pool role: {role.value}")
    if not pool_configs:
        raise ValueError("At least one model pool is required")
    for model_id, config in pool_configs.items():
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("Model IDs must be non-empty strings")
        if set(config) - {"args", "kwargs"}:
            raise ValueError(f"Unknown pool config fields for {model_id}")
        if not isinstance(config.get("args", ()), (tuple, list)):
            raise ValueError(f"Pool args for {model_id} must be a positional argument sequence")
        kwargs = config.get("kwargs", {})
        if not isinstance(kwargs, Mapping) or _OWNED_KWARGS.intersection(kwargs):
            raise ValueError(f"Pool kwargs for {model_id} override role-owned constructor arguments")


class UnifiedServiceManager:
    """Plain CPU host that binds initialized pool adapters to a role manager.

    Every role -- Rollout, GenRM and Teacher -- attaches its model pools
    through this one host inside the task owner's process. Register models,
    bind backend runtimes and configure routes on the supplied manager first.
    Each adapter must expose shutdown and the public methods callers use
    (onload/offload/etc.); no methods or initialization are inferred. A manager
    accepts one host only. Construction never starts actors.
    """

    def __init__(
        self,
        role: Role | str,
        *,
        inference_manager: InferenceManager,
        pools: Mapping[str, Any],
        stop_routers: Callable[[], None] | None = None,
    ) -> None:
        self.role = Role(role)
        self.inference_manager = inference_manager
        self.pools = dict(pools)
        self.inference_manager.attach_host(self.pools, stop_routers=stop_routers)

    def ready(self) -> bool:
        return self.inference_manager.ready()

    def snapshot(self, model_names: Sequence[str] | None = None) -> RoleSnapshot:
        return self.inference_manager.snapshot(tuple(model_names) if model_names is not None else None)

    def call_wait(self, model_id: str, method: str, /, *args: Any, **kwargs: Any) -> Any:
        """Run a local host call while waiting for the model operation lock."""
        if method not in _POOL_METHODS:
            raise ValueError(f"Unsupported pool method: {method}")
        return self.inference_manager.dispatch(model_id, method, *args, wait=True, **kwargs)

    def shutdown(self) -> None:
        self.inference_manager.close()
