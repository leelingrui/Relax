# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""The task-scoped CPU control plane for every inference role.

One ``InferenceManager`` per training task owns the engine pools of Rollout,
GenRM and Teacher, their discovery state, request admission, lifecycle and
placement ledger. It runs as one CPU Ray actor on the head node; the engines it
creates live in that actor's process. Role Gateways read discovery and admit
requests through it; training and scoring code switch the GPUs shared by
deferred roles through ``switch``.

An engine pool is a :class:`~relax.distributed.ray.rollout.RolloutServer`: it
exposes ``onload``/``offload``/``recover``/``health_check``/``shutdown`` and
reports its state with ``observe``. The Manager serializes lifecycle calls per
model and is the only writer of the published snapshots.
"""

import asyncio
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field, replace
from threading import Condition, RLock, get_ident
from typing import Any
from uuid import uuid4

import ray

from relax.core.node_group_affinity import with_control_plane_affinity
from relax.engine.inference.config import ModelConfig, validate_routes
from relax.engine.inference.discovery import new_manager_epoch
from relax.engine.inference.phase_plans import PHASE_GENERATE, reject_shared_co_resident
from relax.engine.inference.placement import PlacementGroupView, PlacementPlanner, PlacementRequest, PlacementSlice
from relax.engine.inference.types import LifecycleState, ModelSnapshot, Role, RoleSnapshot, RoutingSpec
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

# Generation has finished by the time a role is switched out, so nothing should
# still be in flight; the bound turns a request that never completes into a
# failed switch instead of a hung training step.
SWITCH_DRAIN_TIMEOUT_S = 600.0

_LIFECYCLE_METHODS = frozenset({"onload", "offload", "recover", "health_check", "shutdown"})
_QUERY_METHODS = frozenset({"get_urls", "get_engine_hosts_ports"})


@dataclass(frozen=True)
class ModelHandle:
    """A picklable handle to one model on the manager.

    Shaped like a per-model actor handle, so training and service code call
    ``handle.onload.remote()`` / ``handle.offload.remote()``; every call runs
    serialized on the manager through :meth:`InferenceManager.call`.
    """

    manager: Any
    role: Role
    model_id: str

    def __getattr__(self, method: str) -> Any:
        if method.startswith("_"):
            raise AttributeError(method)
        return _RemoteCall(self, method)


@dataclass(frozen=True)
class _RemoteCall:
    handle: ModelHandle
    method: str

    def remote(self, *args: Any, **kwargs: Any) -> Any:
        return self.handle.manager.call.remote(self.handle.role, self.handle.model_id, self.method, *args, **kwargs)


@dataclass
class _Model:
    config: ModelConfig
    pool: Any
    snapshot: ModelSnapshot
    lock: RLock = field(default_factory=RLock)
    # Whether the model may hold GPU memory. Lifecycle state cannot tell:
    # DEAD is published before a failed offload or a shutdown has released
    # anything, and a weights-only onload reports SLEEPING. Set before every
    # load and cleared only once an offload or shutdown has succeeded.
    resident: bool = True


class InferenceManager:
    """Owner of every inference model of one task; see the module docstring."""

    def __init__(self) -> None:
        self.manager_epoch = new_manager_epoch()
        self.placement = PlacementPlanner()
        self._lock = RLock()
        self._requests_done = Condition(self._lock)
        self._models: dict[Role, dict[str, _Model]] = {}
        self._routing: dict[Role, RoutingSpec] = {}
        self._revision: dict[Role, int] = {}
        self._inflight: dict[str, tuple[Role, str]] = {}
        self._rollout_pool: Any = None
        # Loads (switch, onload, recover) run one at a time, owned by the
        # thread running them, and only once no conflicting role is resident.
        # A separate lock, because a waiting load must not block admission or
        # request completion.
        self._gpus_changed = Condition()
        self._gpu_owner: int | None = None

    # ------------------------------------------------------------------
    # Registration and discovery.
    # ------------------------------------------------------------------

    def register(self, role: Role | str, pools: dict[str, Any]) -> tuple[str, ...]:
        """Register a role's started pools and route each model by its own
        ID."""
        role = Role(role)
        with self._lock:
            if role in self._models:
                raise ValueError(f"Inference role is already registered: {role.value}")
            self._models[role] = {
                model_id: _Model(deepcopy(pool.model_spec), pool, ModelSnapshot(model_id))
                for model_id, pool in pools.items()
            }
            names = tuple(pools)
            self._routing[role] = RoutingSpec(
                default_model=names[0] if len(names) == 1 else None,
                route_key_to_model=tuple((name, name) for name in names),
            )
            validate_routes(self._routing[role], set(names))
            self._revision[role] = 1
        for model_id in names:
            self._observe(role, model_id)
        return names

    def roles(self) -> tuple[str, ...]:
        return tuple(role.value for role in self._models)

    def model_ids(self, role: Role | str) -> tuple[str, ...]:
        return tuple(self._models.get(Role(role), {}))

    def model_config(self, role: Role | str, model_id: str) -> ModelConfig:
        return deepcopy(self._model(role, model_id).config)

    def snapshot(self, role: Role | str) -> RoleSnapshot:
        role = Role(role)
        with self._lock:
            if role not in self._models:
                raise RuntimeError(f"Inference role is not registered: {role.value}")
            return RoleSnapshot(
                role=role,
                manager_epoch=self.manager_epoch,
                topology_revision=self._revision[role],
                phase=self._current_phase(),
                models=tuple(model.snapshot for model in self._models[role].values()),
                routing=self._routing[role],
            )

    def publish(self, role: Role | str, model: ModelSnapshot) -> None:
        """Commit one model's observation after checking its invariants."""
        role = Role(role)
        with self._lock:
            entry = self._model(role, model.model_id)
            spec = entry.config
            # Direct eligibility is derived, never reported: only a READY
            # replica of a direct-routed model may take requests without the
            # Router, so sleeping, draining and restarting replicas drop out.
            model = replace(
                model,
                replicas=tuple(
                    replace(
                        replica,
                        direct_eligible=not spec.needs_router
                        and replica.state == LifecycleState.READY
                        and bool(replica.base_url),
                    )
                    for replica in model.replicas
                ),
                pd_workers=tuple((kind, replace(worker, direct_eligible=False)) for kind, worker in model.pd_workers),
            )
            if model.admission and model.state != LifecycleState.READY:
                raise ValueError("Only READY models may admit requests")
            if model.state == LifecycleState.READY:
                ready = [r for r in model.replicas if r.state == LifecycleState.READY and r.base_url]
                if spec.needs_router and not model.router_url:
                    raise ValueError("READY requires the model's Router")
                if not spec.needs_router and model.router_url:
                    raise ValueError("A direct-routed model cannot publish a Router")
                if not ready:
                    raise ValueError("READY requires a healthy logical replica")
                if spec.needs_weight_update and (
                    model.required_weight_version is None
                    or any(r.weight_version != model.required_weight_version for r in ready)
                ):
                    raise ValueError("READY policy replicas must have the required weight version")
            if _topology(model) != _topology(entry.snapshot) or _restarted(entry.snapshot, model):
                self._revision[role] += 1
            entry.snapshot = model

    def set_state(self, role: Role | str, model_id: str, state: LifecycleState) -> None:
        """Close admission and move the model and its replicas to ``state``."""
        if state == LifecycleState.READY:
            raise ValueError("READY is only published from an observation")
        with self._lock:
            current = self._model(role, model_id).snapshot
            self.publish(
                role,
                replace(
                    current,
                    state=state,
                    admission=False,
                    replicas=tuple(replace(replica, state=state) for replica in current.replicas),
                    pd_workers=tuple((kind, replace(worker, state=state)) for kind, worker in current.pd_workers),
                ),
            )

    def _model(self, role: Role | str, model_id: str) -> _Model:
        models = self._models.get(Role(role))
        if models is None or model_id not in models:
            raise KeyError(f"Unknown inference model: {Role(role).value}/{model_id}")
        return models[model_id]

    def _observe(self, role: Role, model_id: str) -> None:
        self.publish(role, self._model(role, model_id).pool.observe())

    # ------------------------------------------------------------------
    # Request admission.
    # ------------------------------------------------------------------

    def admit_request(self, role: Role | str, model_id: str, request_id: str | None = None) -> str:
        """Record an in-flight request on a READY model, or refuse it."""
        role = Role(role)
        with self._lock:
            model = self._model(role, model_id).snapshot
            if not model.admission or model.state != LifecycleState.READY:
                raise RuntimeError(f"Model {role.value}/{model_id} is not ready for inference")
            request_id = request_id or uuid4().hex
            self._inflight[request_id] = (role, model_id)
            return request_id

    def complete_request(self, request_id: str) -> None:
        """Forget a finished, failed or aborted request."""
        with self._requests_done:
            if self._inflight.pop(request_id, None) is not None:
                self._requests_done.notify_all()

    def drain(self, roles: Sequence[Role | str], timeout: float = SWITCH_DRAIN_TIMEOUT_S) -> None:
        """Close admission for ``roles`` and wait for their requests to end."""
        roles = {Role(role) for role in roles}
        for role in roles:
            for model_id in self.model_ids(role):
                self.set_state(role, model_id, LifecycleState.DRAINING)
        deadline = time.monotonic() + timeout
        with self._requests_done:
            while any(role in roles for role, _ in self._inflight.values()):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Inference requests still in flight after {timeout}s: {sorted(self._inflight)}"
                    )
                self._requests_done.wait(remaining)

    # ------------------------------------------------------------------
    # Lifecycle.
    # ------------------------------------------------------------------

    def call(self, role: Role | str, model_id: str, method: str, /, *args: Any, **kwargs: Any) -> Any:
        """Run a lifecycle operation or a pool query on one model."""
        role = Role(role)
        if method in _LIFECYCLE_METHODS:
            return getattr(self, method)(role, model_id, *args, **kwargs)
        if method not in _QUERY_METHODS:
            raise ValueError(f"Unsupported pool method: {method}")
        return getattr(self._model(role, model_id).pool, method)(*args, **kwargs)

    def onload(self, role: Role | str, model_id: str, tags: list[str] | None = None) -> None:
        """Load a model's memory once no conflicting role holds its GPUs."""
        role = Role(role)
        with self._loading([role]):
            entry = self._model(role, model_id)
            with entry.lock:
                entry.resident = True
                self.set_state(role, model_id, LifecycleState.ONLOADING)
                try:
                    entry.pool.onload(tags)
                finally:
                    self._observe(role, model_id)

    def offload(self, role: Role | str, model_id: str) -> None:
        entry = self._model(role, model_id)
        try:
            with entry.lock:
                self.set_state(role, model_id, LifecycleState.DRAINING)
                try:
                    entry.pool.offload()
                except Exception:
                    self.set_state(role, model_id, LifecycleState.DEAD)
                    raise
                entry.resident = False
                self.set_state(role, model_id, LifecycleState.SLEEPING)
        finally:
            # A load waiting for these GPUs re-checks now.
            self._notify_gpus_changed()

    def recover(self, role: Role | str, model_id: str) -> None:
        """Restart dead engines; they allocate memory, so this waits like a
        load."""
        role = Role(role)
        with self._loading([role]):
            entry = self._model(role, model_id)
            with entry.lock:
                entry.resident = True
                self.set_state(role, model_id, LifecycleState.STARTING)
                try:
                    entry.pool.recover()
                finally:
                    self._observe(role, model_id)

    def health_check(self, role: Role | str, model_id: str) -> bool:
        entry = self._model(role, model_id)
        with entry.lock:
            healthy = entry.pool.health_check()
            self._observe(Role(role), model_id)
            return healthy

    def shutdown(self, role: Role | str, model_id: str) -> None:
        entry = self._model(role, model_id)
        with entry.lock:
            self.set_state(role, model_id, LifecycleState.DEAD)
            try:
                entry.pool.shutdown(self.placement)
                entry.resident = False
            finally:
                # Stopping engines reports a topology change; the model stays DEAD.
                self.set_state(role, model_id, LifecycleState.DEAD)
        self._notify_gpus_changed()

    def switch(
        self,
        deactivate: Sequence[Role | str],
        activate: Sequence[Role | str],
        timeout: float = SWITCH_DRAIN_TIMEOUT_S,
    ) -> None:
        """Hand shared GPUs from ``deactivate`` roles to ``activate`` roles.

        The outgoing roles stop admitting, finish their requests and release
        their memory before any incoming role is loaded, so two roles never
        hold the same GPUs at once. Like every load it runs alone, and waits
        while a role outside ``deactivate`` that conflicts with an incoming one
        is resident. Every step is idempotent.
        """
        deactivate = [Role(role) for role in deactivate]
        activate = [Role(role) for role in activate]
        clashing = [
            (a.value, b.value) for i, a in enumerate(activate) for b in activate[i + 1 :] if self._conflict(a, b)
        ]
        if clashing:
            raise ValueError(f"Roles that share GPUs cannot be active together: {clashing}")
        with self._loading(activate, released=deactivate, timeout=timeout):
            self.drain(deactivate, timeout)
            for role in deactivate:
                self._deactivate(role)
            try:
                for role in activate:
                    self._activate(role)
            except Exception:
                # An incoming role that failed to load gives its GPUs back.
                for role in activate:
                    try:
                        self._deactivate(role)
                    except Exception as exc:
                        logger.warning(f"Failed to release {role.value} after a failed switch: {exc}")
                raise

    @contextmanager
    def _loading(
        self, roles: Sequence[Role], *, released: Sequence[Role] = (), timeout: float = SWITCH_DRAIN_TIMEOUT_S
    ) -> Iterator[None]:
        """Own the GPUs while ``roles`` load.

        Waits until no other load runs and no role that conflicts with
        ``roles`` is resident, apart from ``released``, which the caller frees
        first. A load nested in the owning thread runs directly.
        """
        if self._gpu_owner == get_ident():
            yield
            return
        deadline = time.monotonic() + timeout
        with self._gpus_changed:
            while True:
                blockers = self._blockers(roles, released)
                if self._gpu_owner is None and not blockers:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    holders = [role.value for role in blockers] or ["another load"]
                    raise TimeoutError(
                        f"GPUs for {[role.value for role in roles]} still held by {holders} after {timeout}s"
                    )
                self._gpus_changed.wait(remaining)
            self._gpu_owner = get_ident()
        try:
            yield
        finally:
            with self._gpus_changed:
                self._gpu_owner = None
                self._gpus_changed.notify_all()

    def _notify_gpus_changed(self) -> None:
        with self._gpus_changed:
            self._gpus_changed.notify_all()

    def _slices(self, role: Role) -> tuple[PlacementSlice, ...]:
        prefix = f"{role.value}/"
        return tuple(item for item in self.placement.allocations() if item.group_id.startswith(prefix))

    def _conflict(self, role: Role, other: Role) -> bool:
        """Whether two roles are placed on the same GPUs in different phases,
        so they may only take turns."""
        return role is not other and any(
            mine.placement_group_key == theirs.placement_group_key
            and mine.phase != theirs.phase
            and mine.reserved_offset < theirs.reserved_offset + theirs.reserved_size
            and theirs.reserved_offset < mine.reserved_offset + mine.reserved_size
            for mine in self._slices(role)
            for theirs in self._slices(other)
        )

    def _resident(self, role: Role) -> bool:
        with self._lock:
            models = tuple(self._models.get(role, {}).values())
        return any(model.resident for model in models)

    def _blockers(self, roles: Sequence[Role], released: Sequence[Role]) -> list[Role]:
        with self._lock:
            registered = tuple(self._models)
        return [
            other
            for other in registered
            if other not in released
            and other not in roles
            and any(self._conflict(role, other) for role in roles)
            and self._resident(other)
        ]

    def _current_phase(self) -> str:
        """The phase of the resident role placed outside generation, if any."""
        with self._lock:
            registered = tuple(self._models)
        for role in registered:
            phases = {item.phase for item in self._slices(role)} - {PHASE_GENERATE}
            if phases and self._resident(role):
                return phases.pop()
        return PHASE_GENERATE

    def _deactivate(self, role: Role) -> None:
        if role is Role.ROLLOUT and self._rollout_pool is not None:
            self._rollout_pool.offload_local()
        else:
            for model_id in self.model_ids(role):
                self.offload(role, model_id)

    def _activate(self, role: Role) -> None:
        if role is Role.ROLLOUT and self._rollout_pool is not None:
            self._rollout_pool.onload_local()
        else:
            for model_id in self.model_ids(role):
                self.onload(role, model_id)

    # ------------------------------------------------------------------
    # Placement ledger.
    # ------------------------------------------------------------------

    def plan_placement(
        self, requests: Sequence[PlacementRequest], placement_group: PlacementGroupView, *, dry_run: bool = False
    ) -> tuple[PlacementSlice, ...]:
        return self.placement.plan(requests, placement_group, dry_run=dry_run)

    def allocations(self) -> tuple[PlacementSlice, ...]:
        return self.placement.allocations()

    # ------------------------------------------------------------------
    # Role creation and teardown; the engines live in this process.
    # ------------------------------------------------------------------

    @ray.method(concurrency_group="rollout")
    def create_rollout_role(self, args: Any, placement_group: Any) -> dict[str, Any]:
        """Create the rollout engines and return the primary Router address."""
        from relax.distributed.ray.rollout import RolloutEnginePool

        if self._rollout_pool is None:
            # The pool registers its servers once they are started.
            self._rollout_pool = RolloutEnginePool(args, placement_group, inference_manager=self)
        return self._rollout_pool.get_primary_router_address()

    @ray.method(concurrency_group="rollout")
    def rollout_operation(self, method: str, /, *args: Any, **kwargs: Any) -> Any:
        """Run one public rollout engine-pool operation."""
        if self._rollout_pool is None:
            raise RuntimeError("The rollout engine pool has not been created on this manager")
        if method.startswith("_") or not callable(getattr(self._rollout_pool, method, None)):
            raise ValueError(f"Unsupported rollout pool method: {method}")
        result = getattr(self._rollout_pool, method)(*args, **kwargs)
        return asyncio.run(result) if asyncio.iscoroutine(result) else result

    def create_role(
        self, role: Role | str, models: Sequence[tuple[ModelConfig, Any, dict[str, Any]]]
    ) -> tuple[str, ...]:
        """Start a static role's models here and register them.

        Each entry is ``(model_config, engine_args, placement_kwargs)``; see
        :func:`relax.distributed.ray.rollout.start_servers`. The whole role's
        layout is checked against the task ledger before its first engine
        starts; a later failure closes whatever already started.
        """
        from relax.distributed.ray.rollout import placement_preview, start_servers

        role = Role(role)
        if role in self._models:
            return self.model_ids(role)
        previews: dict[str, tuple[PlacementGroupView, list[PlacementRequest]]] = {}
        for config, engine_args, placement in models:
            view, requests = placement_preview(engine_args, [config], role=role, **placement)
            previews.setdefault(view.key, (view, []))[1].extend(requests)
        for view, requests in previews.values():
            self.placement.plan(requests, view, dry_run=True)
        pools: dict[str, Any] = {}
        try:
            for config, engine_args, placement in models:
                pools.update(
                    start_servers(engine_args, [config], planner=self.placement, role=role, **deepcopy(placement))
                )
        except Exception:
            for pool in pools.values():
                pool.shutdown(self.placement)
            raise
        return self.register(role, pools)

    def shutdown_role(self, role: Role | str) -> None:
        role = Role(role)
        if role not in self._models:
            return
        if role is Role.ROLLOUT and self._rollout_pool is not None:
            self._rollout_pool.stop_monitors()
            self._rollout_pool = None
        errors = []
        for model_id in self.model_ids(role):
            try:
                self.shutdown(role, model_id)
            except Exception as exc:
                errors.append(exc)
        with self._lock:
            self._models.pop(role, None)
            self._routing.pop(role, None)
            self._revision.pop(role, None)
        if not self._models:
            from relax.distributed.ray.rollout import stop_launched_routers

            stop_launched_routers()
        if errors:
            raise RuntimeError(f"Failed to shut down inference role {role.value}") from errors[0]

    def shutdown_all(self) -> None:
        errors = []
        for role in tuple(self._models):
            try:
                self.shutdown_role(role)
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError("Failed to shut down one or more inference roles") from errors[0]


def _restarted(previous: ModelSnapshot, model: ModelSnapshot) -> bool:
    """Whether a replica came back READY after a restart.

    A restarted engine usually keeps its address, so the topology alone does
    not change; clients still have to drop what they cached for the old
    process.
    """
    restarting = {
        replica.engine_id
        for replica in (*previous.replicas, *(worker for _, worker in previous.pd_workers))
        if replica.state in (LifecycleState.STARTING, LifecycleState.DEAD)
    }
    return any(
        replica.engine_id in restarting and replica.state == LifecycleState.READY
        for replica in (*model.replicas, *(worker for _, worker in model.pd_workers))
    )


def _topology(model: ModelSnapshot) -> tuple:
    """The discovery fields that describe where a model can be reached."""
    return (
        model.router_url,
        tuple(sorted((replica.engine_id, replica.base_url) for replica in model.replicas)),
        tuple(sorted((worker.engine_id, worker.base_url) for _, worker in model.pd_workers)),
    )


# A drain blocks inside this actor until completions arrive as calls on this
# same actor, so it needs several execution slots; rollout engine operations get
# their own group because a scale-out runs for minutes.
_MANAGER_CONCURRENCY = 8
InferenceManagerActor = ray.remote(num_cpus=1, num_gpus=0, concurrency_groups={"rollout": 8})(InferenceManager)


def validate_task_layout(args: Any) -> None:
    """Plan every inference role of the task before its first engine starts.

    Each role is described exactly as it will start -- Teacher, then Rollout,
    then GenRM, in the placement groups the Controller gives them -- and
    planned into a scratch ledger over stand-in groups of the same size, so a
    layout error in a later role fails the task before an earlier role holds
    GPUs.
    """
    from relax.distributed.ray.genrm import genrm_role_models
    from relax.distributed.ray.rollout import placement_preview, rollout_role_models
    from relax.utils.opd.opd_utils import is_managed_opd_teacher_colocate, is_managed_opd_teacher_enabled

    resource = getattr(args, "resource", None) or {}
    engines = not getattr(args, "debug_train_only", False)

    def stand_in(name: str, role: str) -> tuple[str, tuple[int, ...], tuple[int, ...]] | None:
        num_gpus = int(resource[role][1]) if role in resource else 0
        return (f"preflight/{name}", tuple(range(num_gpus)), tuple(range(num_gpus))) if num_gpus else None

    # Sync colocate hands the actor placement group to rollout and GenRM; the
    # colocated teacher builds that same group first.
    shared = None
    if (
        getattr(args, "colocate", False)
        and not getattr(args, "hybrid", False)
        and {"actor", "rollout"} <= set(resource)
    ):
        shared = stand_in("actor", "actor")
    # One entry per start_servers call: (role, models, engine args, placement).
    calls: list[tuple[Role, list[ModelConfig], Any, dict[str, Any]]] = []
    if engines and is_managed_opd_teacher_enabled(args):
        from relax.utils.opd.opd_utils import managed_teacher_models

        teacher_pg = shared if is_managed_opd_teacher_colocate(args) else None
        for config, engine_args, placement in managed_teacher_models(args, teacher_pg):
            calls.append((Role.TEACHER, [config], engine_args, placement))
    # SFT starts a rollout only to predict.
    sft_without_rollout = (
        getattr(args, "loss_type", None) == "sft" and getattr(args, "sft_predict_interval", None) is None
    )
    if engines and "rollout" in resource and not sft_without_rollout:
        rollout_pg = shared or stand_in("rollout", "rollout")
        placement = {"pg": rollout_pg, "bundle_offset": 0 if rollout_pg is not None else None}
        calls.append((Role.ROLLOUT, rollout_role_models(args), args, placement))
    if getattr(args, "_genrm_instances_resolved", None) and "genrm" in resource:
        for config, engine_args, placement in genrm_role_models(args, shared or stand_in("genrm", "genrm")):
            calls.append((Role.GENRM, [config], engine_args, placement))

    ledger = PlacementPlanner()
    for role, models, engine_args, placement in calls:
        view, requests = placement_preview(engine_args, models, role=role, **placement)
        try:
            ledger.plan(requests, view)
        except ValueError as exc:
            raise ValueError(f"Invalid {role.value} placement; no inference engine was started: {exc}") from exc


def create_inference_manager(args: Any, runtime_env: dict[str, Any] | None = None) -> Any:
    """Create the task's inference manager actor, pinned to the head node.

    The Routers start in this actor's process and the rest of the job resolves
    a Router by the head node's address. An unsupported layout fails here,
    before any engine starts.
    """
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from relax.core.node_group_affinity import require_control_plane_resource_on_node
    from relax.distributed.ray.placement_group import _get_head_node_id

    reject_shared_co_resident(args)
    validate_task_layout(args)
    head_node_id = _get_head_node_id()
    require_control_plane_resource_on_node(args, head_node_id)
    return InferenceManagerActor.options(
        **with_control_plane_affinity(
            args,
            {
                "num_cpus": 1,
                "num_gpus": 0,
                "runtime_env": runtime_env,
                "max_concurrency": _MANAGER_CONCURRENCY,
                "scheduling_strategy": NodeAffinitySchedulingStrategy(node_id=head_node_id, soft=False),
            },
        )
    ).remote()
