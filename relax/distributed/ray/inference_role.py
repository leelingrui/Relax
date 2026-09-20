# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""One CPU owner per inference role, with legacy per-model actor facades."""

import asyncio
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Callable

import ray

from relax.core.node_group_affinity import with_control_plane_affinity
from relax.engine.inference.manager import InferenceManager, ModelBusyError
from relax.engine.inference.types import Role, RoleSnapshot, RoutingSpec
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

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
    }
)
_OWNED_KWARGS = frozenset({"inference_manager", "model_id", "defer_init"})


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

    def __init__(self, role: Role | str, pool_configs: Mapping[str, Mapping[str, Any]]) -> None:
        self.role = Role(role)
        _validate_configs(self.role, pool_configs)
        self.inference_manager = InferenceManager(self.role)
        self.pools: dict[str, Any] = {}
        try:
            for model_id, config in pool_configs.items():
                config = deepcopy(config)
                self.pools[model_id] = _create_model_pool(
                    self.role,
                    *config.get("args", ()),
                    **config.get("kwargs", {}),
                    inference_manager=self.inference_manager,
                    model_id=model_id,
                    defer_init=True,
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

    def __init__(self, role_manager: Any, model_id: str) -> None:
        self.role_manager = role_manager
        self.model_id = model_id

    async def ready(self) -> bool:
        return await self.role_manager.ready.remote()

    async def get_role_snapshot(self) -> RoleSnapshot:
        return await self.role_manager.snapshot.remote()

    async def _forward(self, method: str, *args: Any, **kwargs: Any) -> Any:
        delay = 0.01
        while True:
            try:
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

    owner = None
    facades: dict[str, Any] = {}
    try:
        owner = InferenceRoleManager.options(
            **with_control_plane_affinity(args, {"num_cpus": 1, "num_gpus": 0, "runtime_env": runtime_env})
        ).remote(role, pool_configs)
        if ray.get(owner.ready.remote()) is not True:
            raise RuntimeError("Inference role initialization did not complete")
        for model_id in pool_configs:
            options = {"num_cpus": 0, "num_gpus": 0, "runtime_env": runtime_env}
            if model_id in names:
                options["name"] = names[model_id]
            facades[model_id] = InferenceManagerFacade.options(**with_control_plane_affinity(args, options)).remote(
                owner, model_id
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
