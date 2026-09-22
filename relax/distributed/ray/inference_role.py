# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""One CPU owner per inference role, with legacy per-model actor facades."""

import asyncio
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from typing import Any, Callable
from uuid import uuid4

import ray

from relax.core.node_group_affinity import with_control_plane_affinity
from relax.engine.inference.config import ModelConfig
from relax.engine.inference.discovery import new_manager_epoch
from relax.engine.inference.manager import (
    EnginePoolRuntime,
    InferenceManager,
    ModelBusyError,
    OperationSnapshot,
    PreparationEvidence,
    RequestPermit,
)
from relax.engine.inference.placement import PlacementGroupView, PlacementPlanner, PlacementRequest, PlacementSlice
from relax.engine.inference.types import LifecycleState, Role, RoleSnapshot, RoutingSpec
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


class _ExternalPoolRuntime(EnginePoolRuntime):
    """CPU-side proxy for a GPU-owned role host.

    The task owner serializes lifecycle operations and admission. The host
    remains responsible for engine actors and returns the operation result.
    """

    def __init__(self, host: Any, model_id: str) -> None:
        self.host = host
        self.model_id = model_id
        self._onloaded = True

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        call = getattr(self.host, "call", None)
        if call is None:
            raise RuntimeError("External inference host has no call method")
        value = call.remote(self.model_id, method, *args, **kwargs)
        object_ref = getattr(ray, "ObjectRef", ())
        return ray.get(value) if object_ref and isinstance(value, object_ref) else value

    def health_check(self) -> bool:
        return bool(self._call("health_check"))

    def recover(self) -> set[int]:
        return set(self._call("recover"))

    def is_onloaded(self) -> bool:
        return self._onloaded

    def set_onloaded(self, value: bool) -> None:
        self._call("set_onloaded", value)
        self._onloaded = value

    def fanout(self, method: str, *, skip_ranks: set[int] | None = None, **kwargs: Any) -> list[int]:
        result = self._call(method, skip_ranks=skip_ranks, **kwargs)
        return list(result or [])

    def retire(self, ranks: list[int]) -> None:
        self._call("retire", ranks)

    def shutdown(self) -> None:
        if getattr(self.host, "call", None) is None:
            # Registration may precede backend binding; there is no worker to
            # close in that transitional state.
            return
        self._call("shutdown")


class TaskInferenceManager:
    """CPU control-plane owner for all inference role hosts.

    Runtime pools remain in their role-specific Serve/worker processes, but
    discovery is exposed through this single task-scoped handle.  The owner
    deliberately stores handles rather than GPU state so legacy role hosts
    remain replaceable during the migration.
    """

    def __init__(self) -> None:
        self._hosts: dict[Role, Any] = {}
        self.manager_epoch = new_manager_epoch()
        self._snapshots: dict[Role, RoleSnapshot] = {}
        self._permits: dict[str, RequestPermit] = {}
        self._placement_planner = PlacementPlanner()
        self._control_manager = InferenceManager()
        self._operations: dict[str, OperationSnapshot] = {}
        self._role_pools: dict[Role, dict[str, Any]] = {}
        self._external_roles: set[Role] = set()

    def register_role(self, role: Role | str, host: Any) -> None:
        role = Role(role)
        previous = self._hosts.get(role)
        if previous is not None:
            if previous != host:
                logger.debug("Keeping existing inference host for role %s", role.value)
            return
        self._hosts[role] = host
        if role not in self._role_pools and role not in self._external_roles:
            self._attach_external_role(role, host)

    def _attach_external_role(self, role: Role, host: Any) -> None:
        registered = self._control_manager.snapshot(role=role)
        if not registered.models:
            return
        manager = self._control_manager.for_role(role)
        runtimes = {model.model_id: _ExternalPoolRuntime(host, model.model_id) for model in registered.models}
        for model_id, runtime in runtimes.items():
            manager.bind_pool(model_id, runtime)
        manager.attach_host(runtimes)
        self._external_roles.add(role)

    def registered_roles(self) -> tuple[str, ...]:
        return tuple(role.value for role in self._hosts)

    def register_model(self, role: Role | str, spec: ModelConfig, *, operation_id: str) -> ModelConfig:
        role = Role(role)
        result = self._control_manager.register_model(spec, operation_id=operation_id, role=role)
        self._operations[operation_id] = OperationSnapshot(
            operation_id, self.manager_epoch, "completed", "register", result
        )
        return result

    def register_external_role(
        self,
        role: Role | str,
        host: Any | None,
        specs: Sequence[ModelConfig],
        routing: RoutingSpec,
    ) -> RoleSnapshot:
        """Register a backend-owned role without moving its GPU actors here.

        The task owner remains the authority for model identity, routes and
        operation history.  The host is only an execution/discovery adapter;
        its committed observations are normalized to the task epoch by
        :meth:`snapshot`.
        """
        role = Role(role)
        if host is not None:
            self.register_role(role, host)
        for spec in specs:
            self.register_model(role, spec, operation_id=f"register:{role.value}:{spec.model_id}")
        self.configure_routes(role, routing, operation_id=f"routes:{role.value}:startup")
        if host is not None and role not in self._external_roles:
            self._attach_external_role(role, host)
        if host is None:
            return replace(self._control_manager.snapshot(role=role), manager_epoch=self.manager_epoch)
        return self.snapshot(role=role)

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

    def _run_external_operation(self, role: Role, model_id: str, method: str, *args: Any, **kwargs: Any) -> Any:
        operation_id = f"{method}:{role.value}:{model_id}:{uuid4().hex}"
        operation = OperationSnapshot(operation_id, self.manager_epoch, "running", method)
        self._operations[operation_id] = operation
        try:
            result = self.call(role, model_id, method, *args, **kwargs)
            if role not in self._role_pools and method != "shutdown":
                self.sync_role_snapshot(role)
        except Exception as exc:
            self._operations[operation_id] = replace(operation, status="failed", error=str(exc))
            raise
        self._operations[operation_id] = replace(operation, status="completed", result=result)
        return result

    def sync_role_snapshot(self, role: Role | str) -> RoleSnapshot:
        return self.snapshot(role=role)

    def _publish_external_snapshot(self, role: Role, snapshot: RoleSnapshot) -> RoleSnapshot:
        """Commit backend observations into the task owner's publication
        fence."""
        registered = self._control_manager.snapshot(role=role)
        if not registered.models:
            return replace(snapshot, manager_epoch=self.manager_epoch, routing=registered.routing)
        if {model.model_id for model in snapshot.models} != {model.model_id for model in registered.models}:
            raise RuntimeError(f"External snapshot does not match registered models for role {role.value}")
        for model in snapshot.models:
            current = next(item for item in registered.models if item.model_id == model.model_id)
            candidate = replace(
                model,
                allow_defer=current.allow_defer,
                direct_eligible=current.direct_eligible,
                admission=model.admission and model.state == LifecycleState.READY,
            )
            if candidate.state == LifecycleState.READY:
                ready_replicas = [replica for replica in candidate.replicas if replica.state == LifecycleState.READY]
                evidence = PreparationEvidence(
                    initialized=bool(candidate.replicas),
                    health_checked=bool(ready_replicas),
                    router_updated=candidate.router_url is not None,
                    weights_synced=(
                        candidate.required_weight_version is not None
                        and all(
                            replica.weight_version == candidate.required_weight_version for replica in ready_replicas
                        )
                    ),
                )
                token = self._control_manager.begin_preparation(
                    candidate.model_id, required_weight_version=candidate.required_weight_version, role=role
                )
                self._control_manager.complete_preparation(token, candidate, evidence=evidence, role=role)
            else:
                self._control_manager.publish_model(candidate, role=role)
        committed = self._control_manager.snapshot(role=role)
        return replace(committed, manager_epoch=self.manager_epoch, phase=snapshot.phase)

    @staticmethod
    def _resolve(value: Any) -> Any:
        if isinstance(value, (RoleSnapshot, dict)):
            return value
        object_ref = getattr(ray, "ObjectRef", ())
        return ray.get(value) if object_ref and isinstance(value, object_ref) else value

    def snapshot(self, *, role: Role | str) -> RoleSnapshot:
        role = Role(role)
        if role in self._role_pools:
            return replace(self._control_manager.snapshot(role=role), manager_epoch=self.manager_epoch)
        host = self._hosts.get(role)
        if host is None:
            raise RuntimeError(f"Inference role is not registered: {role.value}")
        registered_snapshot = self._control_manager.snapshot(role=role)

        def normalize(snapshot: RoleSnapshot) -> RoleSnapshot:
            registered = {model.model_id for model in registered_snapshot.models}
            observed = {model.model_id for model in snapshot.models}
            if registered and observed != registered:
                raise RuntimeError(
                    f"Inference role snapshot models differ from owner registration: "
                    f"registered={sorted(registered)}, observed={sorted(observed)}"
                )
            return replace(
                snapshot,
                manager_epoch=self.manager_epoch,
                routing=registered_snapshot.routing,
                topology_revision=max(snapshot.topology_revision, registered_snapshot.topology_revision),
            )

        for method_name in ("get_role_snapshot", "get_discovery_snapshot", "snapshot"):
            method = getattr(host, method_name, None)
            if method is None:
                continue
            snapshot = self._resolve(method.remote())
            if not isinstance(snapshot, RoleSnapshot):
                from relax.engine.inference.discovery import role_snapshot_from_dict

                snapshot = role_snapshot_from_dict(snapshot)
            snapshot = normalize(snapshot)
            snapshot = self._publish_external_snapshot(role, snapshot)
            self._snapshots[role] = snapshot
            return snapshot
        method = getattr(host, "get_engines", None)
        if method is not None:
            from relax.engine.inference.discovery import role_snapshot_from_dict

            snapshot = role_snapshot_from_dict(self._resolve(method.remote(schema_version=2)))
            snapshot = normalize(snapshot)
            snapshot = self._publish_external_snapshot(role, snapshot)
            self._snapshots[role] = snapshot
            return snapshot
        raise RuntimeError(f"Inference role host has no snapshot method: {role.value}")

    def admit_request(
        self,
        model_id: str,
        request_id: str | None = None,
        *,
        role: Role | str,
        target: str | None = None,
    ) -> RequestPermit:
        role = Role(role)
        snapshot = self.snapshot(role=role)
        model = next((item for item in snapshot.models if item.model_id == model_id), None)
        if model is None:
            raise KeyError(f"Unknown model: {model_id}")
        if not model.admission or model.state is None or model.state.value != "ready":
            raise RuntimeError(f"Model {model_id} is not ready for inference")
        permit = RequestPermit(request_id or uuid4().hex, self.manager_epoch, role, model_id, target)
        previous = self._permits.get(permit.request_id)
        if previous is not None and previous != permit:
            raise ValueError(f"Request already exists: {permit.request_id}")
        self._permits[permit.request_id] = permit
        return permit

    def complete_request(self, permit: RequestPermit | str) -> None:
        request_id = permit if isinstance(permit, str) else permit.request_id
        current = self._permits.get(request_id)
        if current is None:
            return
        if not isinstance(permit, str) and current != permit:
            raise ValueError("Request permit does not belong to this manager epoch and target")
        self._permits.pop(request_id, None)

    def cancel_request(self, permit: RequestPermit | str) -> None:
        self.complete_request(permit)

    def get_request(self, request_id: str) -> RequestPermit | None:
        return self._permits.get(request_id)

    def plan_placement(
        self, requests: Sequence[PlacementRequest], placement_group: PlacementGroupView
    ) -> tuple[PlacementSlice, ...]:
        return self._placement_planner.plan(requests, placement_group)

    def cancel_placement(self, placement: PlacementSlice | PlacementGroupView, group_id: str | None = None) -> None:
        self._placement_planner.cancel(placement, group_id=group_id)

    def allocations(self, placement_group: PlacementGroupView) -> tuple[PlacementSlice, ...]:
        return self._placement_planner.allocations(placement_group)

    def ready(self, *, role: Role | str) -> bool:
        role = Role(role)
        if role in self._role_pools:
            return self._control_manager.ready(role=role)
        if role in self._external_roles:
            return self._control_manager.ready(role=role)
        host = self._hosts.get(role)
        if host is None:
            return False
        method = getattr(host, "ready", None)
        return bool(self._resolve(method.remote())) if method is not None else False

    def call(self, role: Role | str, model_id: str, method: str, /, *args: Any, **kwargs: Any) -> Any:
        role = Role(role)
        if role in self._role_pools or role in self._external_roles:
            return self._control_manager.dispatch(model_id, method, *args, wait=True, role=role, **kwargs)
        host = self._hosts.get(role)
        if host is None:
            raise RuntimeError(f"Inference role is not registered: {role.value}")
        call = getattr(host, "call", None)
        if call is None:
            raise RuntimeError(f"Inference role host has no call method: {role.value}")
        return self._resolve(call.remote(model_id, method, *args, **kwargs))

    def lifecycle(self, role: Role | str, model_id: str, method: str, /, *args: Any, **kwargs: Any) -> Any:
        """Run a serialized pool lifecycle operation through the task owner."""
        if method not in {"health_check", "recover", "onload", "offload", "shutdown"}:
            raise ValueError(f"Unsupported lifecycle method: {method}")
        role = Role(role)
        if role in self._role_pools:
            return self.call(role, model_id, method, *args, **kwargs)
        if role in self._external_roles:
            return self._run_external_operation(role, model_id, method, *args, **kwargs)
        return self.call(role, model_id, method, *args, **kwargs)

    def create_role(self, args: Any, role: Role | str, pool_configs: Mapping[str, Mapping[str, Any]]) -> None:
        role = Role(role)
        if role in self._role_pools:
            return
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
            manager.attach_host(pools, stop_routers=_stop_role_routers)
            self._role_pools[role] = pools
            self._operations[f"routes:{role.value}:startup"] = OperationSnapshot(
                f"routes:{role.value}:startup", self.manager_epoch, "completed", "routes", manager.snapshot().routing
            )
        except Exception:
            try:
                manager.close()
            finally:
                raise

    def shutdown_role(self, role: Role | str) -> None:
        role = Role(role)
        if role in self._role_pools:
            self._control_manager.shutdown(role=role)
            self._role_pools.pop(role, None)
            return
        if role in self._external_roles:
            self._control_manager.shutdown(role=role)
            self._external_roles.discard(role)
            return
        host = self._hosts.get(role)
        if host is None:
            return
        shutdown = getattr(host, "shutdown", None)
        if shutdown is not None:
            self._resolve(shutdown.remote())

    def shutdown_all(self) -> None:
        """Close every registered role through this task-scoped owner."""
        errors = []
        for role in tuple(self._role_pools) + tuple(role for role in self._hosts if role not in self._role_pools):
            try:
                self.shutdown_role(role)
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError("Failed to shut down one or more inference roles") from errors[0]


TaskInferenceManagerActor = ray.remote(num_cpus=1, num_gpus=0)(TaskInferenceManager)


def create_task_inference_manager(args: Any, runtime_env: dict[str, Any] | None = None) -> Any:
    """Create the single task-level CPU inference control-plane actor."""
    return TaskInferenceManagerActor.options(
        **with_control_plane_affinity(args, {"num_cpus": 1, "num_gpus": 0, "runtime_env": runtime_env})
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
        "get_genrm_engines_and_lock",
        "get_discovery_snapshot",
        "fanout",
        "retire",
        "set_onloaded",
    }
)
_OWNED_KWARGS = frozenset({"inference_manager", "model_id", "defer_init", "placement_manager_handle"})


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
    """Plain CPU host for initialized public pool adapters, not raw runtimes.

    Register models, bind backend runtimes and configure routes on the supplied
    manager first. Each adapter must expose shutdown and the public methods
    callers use (onload/offload/etc.); no methods or initialization are
    inferred. A manager accepts one host only. Construction never starts
    actors.
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

    @ray.method(concurrency_group="snapshot")
    def ready(self) -> bool:
        return self.inference_manager.ready()

    @ray.method(concurrency_group="snapshot")
    def snapshot(self, model_names: Sequence[str] | None = None) -> RoleSnapshot:
        return self.inference_manager.snapshot(tuple(model_names) if model_names is not None else None)

    @ray.method(concurrency_group="pool")
    def call(self, model_id: str, method: str, /, *args: Any, **kwargs: Any) -> Any:
        if method not in _POOL_METHODS:
            raise ValueError(f"Unsupported pool method: {method}")
        return self.inference_manager.dispatch(model_id, method, *args, wait=True, **kwargs)

    def call_wait(self, model_id: str, method: str, /, *args: Any, **kwargs: Any) -> Any:
        """Run a local host call while waiting for the model operation lock."""
        if method not in _POOL_METHODS:
            raise ValueError(f"Unsupported pool method: {method}")
        return self.inference_manager.dispatch(model_id, method, *args, wait=True, **kwargs)

    @ray.method(concurrency_group="control")
    def shutdown(self) -> None:
        self.inference_manager.close()


class InferenceRole(UnifiedServiceManager):
    """Compatibility host for the legacy GenRM/Teacher role actor API.

    This class is a migration bridge, not a second inference lifecycle
    implementation: the shared ``InferenceManager`` remains the source of
    truth for registration, routing, and lifecycle serialization. Keep this
    Ray actor wrapper until callers migrate to the unified inference host;
    remove it together with ``InferenceRoleManager`` and the per-model
    facade compatibility API afterward.

    Pool constructors must honor defer_init: no GPU process may be started
    before every model is registered and the routing table is committed.
    """

    def __init__(
        self,
        role: Role | str,
        pool_configs: Mapping[str, Mapping[str, Any]],
        inference_manager: InferenceManager | None = None,
        placement_manager_handle: Any | None = None,
    ) -> None:
        self.role = Role(role)
        _validate_configs(self.role, pool_configs)
        root_manager = inference_manager or InferenceManager()
        self.inference_manager = root_manager.for_role(self.role)
        self.pools: dict[str, Any] = {}
        try:
            for model_id, config in pool_configs.items():
                config = deepcopy(config)
                pool_kwargs = {
                    **config.get("kwargs", {}),
                    "inference_manager": self.inference_manager,
                    "model_id": model_id,
                    "defer_init": True,
                }
                if placement_manager_handle is not None:
                    pool_kwargs["placement_manager_handle"] = placement_manager_handle
                self.pools[model_id] = _create_model_pool(
                    self.role,
                    *config.get("args", ()),
                    **pool_kwargs,
                )
            self.inference_manager.configure_routes(
                RoutingSpec(
                    default_model=next(iter(self.pools)) if len(self.pools) == 1 else None,
                    route_key_to_model=tuple((model_id, model_id) for model_id in self.pools),
                ),
                operation_id="configure-role-routes",
            )
            for pool in self.pools.values():
                pool.initialize()
            self.inference_manager.attach_host(self.pools, stop_routers=_stop_role_routers)
        except Exception:
            self.inference_manager.attach_host(self.pools, stop_routers=_stop_role_routers)
            try:
                self.inference_manager.close()
            except Exception:
                logger.exception("Failed to clean up inference role %s", self.role.value)
            raise

    @ray.method(concurrency_group="pool")
    def call(self, model_id: str, method: str, /, *args: Any, **kwargs: Any) -> Any:
        if method not in _POOL_METHODS:
            raise ValueError(f"Unsupported pool method: {method}")
        return self.inference_manager.dispatch(model_id, method, *args, wait=False, **kwargs)


InferenceRoleManager = ray.remote(num_cpus=1, num_gpus=0, concurrency_groups={"snapshot": 1, "pool": 8, "control": 1})(
    InferenceRole
)


class ModelManagerFacade:
    """Legacy per-model handle forwarding to the unified role host.

    This facade exists only for callers that still expect one named Ray actor
    per model. It owns no model state or GPU resources and should be removed
    once those callers use the unified inference gateway directly.
    """

    def __init__(
        self,
        role_manager: Any,
        model_id: str,
        task_manager: Any | None = None,
        role: Role | str | None = None,
    ) -> None:
        self.role_manager = role_manager
        self.model_id = model_id
        self.task_manager = task_manager
        self.role = Role(role) if role is not None else None

    async def ready(self) -> bool:
        if self.task_manager is not None:
            return await self.task_manager.ready.remote(role=self.role)
        return await self.role_manager.ready.remote()

    async def get_role_snapshot(self) -> RoleSnapshot:
        if self.task_manager is not None:
            return await self.task_manager.snapshot.remote(role=self.role)
        return await self.role_manager.snapshot.remote()

    async def _forward(self, method: str, *args: Any, **kwargs: Any) -> Any:
        delay = 0.01
        while True:
            try:
                if self.task_manager is not None:
                    if method in {"health_check", "recover", "onload", "offload", "shutdown"}:
                        return await self.task_manager.lifecycle.remote(
                            self.role, self.model_id, method, *args, **kwargs
                        )
                    return await self.task_manager.call.remote(self.role, self.model_id, method, *args, **kwargs)
                return await self.role_manager.call.remote(self.model_id, method, *args, **kwargs)
            except ModelBusyError:
                # Ray preserves the cause type on RayTaskError. Only rejection
                # before dispatch is retryable; no pool RPC has started yet.
                await asyncio.sleep(delay)
                delay = min(delay * 2, 0.1)

    async def health_check(self, *args: Any, **kwargs: Any) -> Any:
        return await self._forward("health_check", *args, **kwargs)

    async def recover(self, *args: Any, **kwargs: Any) -> Any:
        return await self._forward("recover", *args, **kwargs)

    async def onload(self, *args: Any, **kwargs: Any) -> Any:
        return await self._forward("onload", *args, **kwargs)

    async def offload(self, *args: Any, **kwargs: Any) -> Any:
        return await self._forward("offload", *args, **kwargs)

    async def is_onloaded(self, *args: Any, **kwargs: Any) -> Any:
        return await self._forward("is_onloaded", *args, **kwargs)

    async def fanout(self, *args: Any, **kwargs: Any) -> Any:
        return await self._forward("fanout", *args, **kwargs)

    async def retire(self, *args: Any, **kwargs: Any) -> Any:
        return await self._forward("retire", *args, **kwargs)

    async def set_onloaded(self, *args: Any, **kwargs: Any) -> Any:
        return await self._forward("set_onloaded", *args, **kwargs)

    async def shutdown(self, *args: Any, **kwargs: Any) -> Any:
        return await self._forward("shutdown", *args, **kwargs)

    async def get_urls(self, *args: Any, **kwargs: Any) -> Any:
        return await self._forward("get_urls", *args, **kwargs)

    async def get_engine_hosts_ports(self, *args: Any, **kwargs: Any) -> Any:
        return await self._forward("get_engine_hosts_ports", *args, **kwargs)

    async def get_genrm_engines_and_lock(self, *args: Any, **kwargs: Any) -> Any:
        return await self._forward("get_genrm_engines_and_lock", *args, **kwargs)

    async def get_discovery_snapshot(self, *args: Any, **kwargs: Any) -> Any:
        return await self._forward("get_discovery_snapshot", *args, **kwargs)


InferenceManagerFacade = ray.remote(num_cpus=0, num_gpus=0)(ModelManagerFacade)


def create_role_managers(
    args: Any,
    role: Role | str,
    pool_configs: Mapping[str, Mapping[str, Any]],
    runtime_env: dict[str, Any] | None = None,
    actor_names: Mapping[str, str] | None = None,
    task_manager_handle: Any | None = None,
) -> dict[str, Any]:
    """Return initialized legacy handles, one per model, backed by one role
    actor.

    Each config contains ``args`` (pool positional arguments) and ``kwargs``.
    Names are optional model-ID-to-actor-name mappings. Offload-on-start
    remains the caller's responsibility; ready() confirms initialization, not
    admission.
    """
    role = Role(role)
    _validate_configs(role, pool_configs)
    names = dict(actor_names or {})
    if set(names) - set(pool_configs):
        raise ValueError("Actor names reference unknown models")
    if any(not isinstance(name, str) or not name for name in names.values()):
        raise ValueError("Actor names must be non-empty strings")
    if len(set(names.values())) != len(names):
        raise ValueError("Actor names must be unique")

    if task_manager_handle is not None:
        ray.get(task_manager_handle.create_role.remote(args, role, pool_configs))
        facades: dict[str, Any] = {}
        for model_id in pool_configs:
            options = {"num_cpus": 0, "num_gpus": 0, "runtime_env": runtime_env}
            if model_id in names:
                options["name"] = names[model_id]
            facades[model_id] = InferenceManagerFacade.options(**with_control_plane_affinity(args, options)).remote(
                None, model_id, task_manager_handle, role
            )
        return facades

    owner = None
    facades: dict[str, Any] = {}
    try:
        owner = InferenceRoleManager.options(
            **with_control_plane_affinity(args, {"num_cpus": 1, "num_gpus": 0, "runtime_env": runtime_env})
        ).remote(role, pool_configs)
        if ray.get(owner.ready.remote()) is not True:
            raise RuntimeError("Inference role initialization did not complete")
        if task_manager_handle is not None:
            ray.get(task_manager_handle.register_role.remote(role, owner))
            ray.get(task_manager_handle.sync_role_snapshot.remote(role))
        for model_id in pool_configs:
            options = {"num_cpus": 0, "num_gpus": 0, "runtime_env": runtime_env}
            if model_id in names:
                options["name"] = names[model_id]
            facade_args = (owner, model_id)
            facade_kwargs = {}
            if task_manager_handle is not None:
                facade_args += (task_manager_handle, role)
            facades[model_id] = InferenceManagerFacade.options(**with_control_plane_affinity(args, options)).remote(
                *facade_args, **facade_kwargs
            )
        if not all(result is True for result in ray.get([facade.ready.remote() for facade in facades.values()])):
            raise RuntimeError("Inference facade initialization did not complete")
        return facades
    except Exception:
        if owner is not None:
            try:
                ray.get(owner.shutdown.remote(), timeout=60)
            except Exception:
                logger.exception("Failed to clean up inference role %s", role.value)
        for actor in [*facades.values(), *([owner] if owner is not None else [])]:
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                logger.exception("Failed to terminate an inference role startup actor")
        raise
