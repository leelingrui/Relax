# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU-only model registry and atomic publication boundary for engine
owners."""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from threading import RLock, get_ident
from typing import Any, Callable, Iterator, Mapping, Protocol
from uuid import uuid4

from relax.engine.inference.capabilities import WeightSource
from relax.engine.inference.config import ModelConfig
from relax.engine.inference.discovery import new_manager_epoch
from relax.engine.inference.specs import validate_routes
from relax.engine.inference.types import LifecycleState, ModelSnapshot, Role, RoleSnapshot, RoutingSpec


@dataclass(frozen=True)
class PreparationToken:
    """Internal publication fence, not a request or GPU ownership permit."""

    manager_epoch: str
    model_id: str
    nonce: str


@dataclass(frozen=True)
class PreparationEvidence:
    initialized: bool
    health_checked: bool
    router_updated: bool
    weights_synced: bool = False


@dataclass(frozen=True)
class RequestPermit:
    """A request admission bound to one manager epoch and model target."""

    request_id: str
    manager_epoch: str
    role: Role
    model_id: str
    target: str | None = None


@dataclass(frozen=True)
class OperationSnapshot:
    """Stable observation of a control-plane operation."""

    operation_id: str
    owner_epoch: str
    status: str
    kind: str | None = None
    result: Any = None
    error: str | None = None


class EnginePoolRuntime(Protocol):
    """Execution adapter; implementations retain backend-specific RPC
    handling."""

    def health_check(self) -> bool: ...
    def recover(self) -> set[int]: ...
    def is_onloaded(self) -> bool: ...
    def set_onloaded(self, value: bool) -> None: ...
    def fanout(self, method: str, *, skip_ranks: set[int] | None = None, **kwargs: Any) -> list[int]: ...
    def retire(self, ranks: list[int]) -> None: ...
    def shutdown(self) -> None: ...


class ModelBusyError(RuntimeError):
    """Lock admission failed before execution; safe to retry the whole call."""


class _RoleInferenceState:
    """Registration never creates actors or grants admission.

    Engine owners submit complete observations after completing health, weight
    synchronization and Router updates. Resource planning is not owned here.
    """

    def __init__(self, role: Role) -> None:
        self._lock = RLock()
        self._models: dict[str, ModelConfig] = {}
        self._operations: dict[str, tuple[str, object]] = {}
        self._sealed = False
        self._preparations: dict[str, PreparationToken] = {}
        self._pools: dict[str, EnginePoolRuntime] = {}
        self._pool_locks: dict[str, RLock] = {}
        self._busy: set[str] = set()
        self._operation_owners: dict[str, tuple[int, int]] = {}
        self._dispatch_pools: dict[str, Any] = {}
        self._closing_models: set[str] = set()
        self._closed_models: dict[str, Any] = {}
        self._closing = False
        self._ready = False
        self._shutdown_lock = RLock()
        self._router_lock = RLock()
        self._stop_routers: Callable[[], None] | None = None
        self._routers_stopped = False
        self._snapshot = RoleSnapshot(role=Role(role), manager_epoch=new_manager_epoch())

    def attach_host(self, pools: Mapping[str, Any], *, stop_routers: Callable[[], None] | None = None) -> None:
        """Attach once, after registration/binding and initialization.

        Values are public adapters exposing shutdown and any dispatched
        methods, not necessarily the backend runtimes supplied to bind_pool.
        This does not initialize pools, configure routes or synthesize adapter
        methods.
        """
        with self._lock:
            if self._ready or self._closing:
                raise RuntimeError("Inference host already attached or shut down")
            for model_id in pools:
                if model_id not in self._pools:
                    raise KeyError(f"Unbound model: {model_id}")
            self._dispatch_pools = dict(pools)
            self._stop_routers = stop_routers
            self._ready = True

    def ready(self) -> bool:
        with self._lock:
            return self._ready

    @staticmethod
    def _topology_signature(snapshot: RoleSnapshot) -> tuple:
        """Return only discovery fields that describe inference topology."""
        models = tuple(
            sorted(
                (
                    model.model_id,
                    model.router_url,
                    tuple(sorted((replica.engine_id, replica.base_url) for replica in model.replicas)),
                )
                for model in snapshot.models
            )
        )
        routing = (
            snapshot.routing.default_model,
            tuple(sorted(snapshot.routing.route_key_to_model)),
            snapshot.routing.config_version,
        )
        return models, routing

    def dispatch(self, model_id: str, method: str, /, *args: Any, wait: bool = False, **kwargs: Any) -> Any:
        with self._lock:
            if model_id not in self._dispatch_pools:
                raise KeyError(f"Unknown model: {model_id}")
            if method != "shutdown" and (not self._ready or self._closing):
                raise RuntimeError("Inference role is not initialized or has been shut down")
        if method == "shutdown":
            result = self._close_pool(model_id, *args, **kwargs)
            with self._lock:
                all_closed = len(self._closed_models) == len(self._dispatch_pools)
                if all_closed:
                    self._closing = True
                    self._ready = False
            if all_closed:
                self._close_routers()
            return result
        with self._pool_operation(model_id, wait=wait):
            try:
                return getattr(self._dispatch_pools[model_id], method)(*args, **kwargs)
            except ModelBusyError as exc:
                # A nested/cross-model rejection may follow side effects. Do
                # not let the facade replay an operation that already started.
                raise RuntimeError("Pool operation raised busy after dispatch; not retryable") from exc

    def _close_pool(self, model_id: str, /, *args: Any, wait: bool = False, **kwargs: Any) -> Any:
        with self._pool_operation(model_id, wait=wait, closing=True):
            with self._lock:
                if model_id in self._closed_models:
                    return self._closed_models[model_id]
                self._closing_models.add(model_id)
            self._invalidate_pool(model_id, LifecycleState.DEAD)
            try:
                result = self._dispatch_pools[model_id].shutdown(*args, **kwargs)
            except ModelBusyError as exc:
                raise RuntimeError("Pool shutdown raised busy after dispatch; not retryable") from exc
            finally:
                self._invalidate_pool(model_id, LifecycleState.DEAD)
            with self._lock:
                self._closed_models[model_id] = result
            return result

    def _close_routers(self) -> None:
        with self._router_lock:
            if not self._routers_stopped:
                if self._stop_routers is not None:
                    self._stop_routers()
                self._routers_stopped = True

    def close(self) -> None:
        """Drain on the dedicated control thread, never under the snapshot
        lock."""
        with self._shutdown_lock:
            with self._lock:
                self._closing = True
                self._ready = False
            errors = []
            for model_id in reversed(tuple(self._dispatch_pools)):
                try:
                    self._close_pool(model_id, wait=True)
                except Exception as exc:
                    errors.append(exc)
            try:
                self._close_routers()
            except Exception as exc:
                errors.append(exc)
            if errors:
                raise RuntimeError("Failed to shut down one or more inference pools") from errors[0]

    def bind_pool(self, model_id: str, runtime: EnginePoolRuntime) -> None:
        """Attach a runtime adapter without creating or replacing any
        actors."""
        with self._lock:
            if model_id not in self._models:
                raise KeyError(f"Unregistered model: {model_id}")
            if model_id in self._pools:
                if self._pools[model_id] is not runtime:
                    raise ValueError(f"Pool already bound: {model_id}")
                return
            self._pools[model_id] = runtime
            self._pool_locks[model_id] = RLock()

    def _invalidate_pool(self, model_id: str, state: LifecycleState) -> None:
        if model_id in self._models:
            self.invalidate_model(model_id, state=state)

    @contextmanager
    def _pool_operation(self, model_id: str, *, wait: bool = False, closing: bool = False) -> Iterator[None]:
        lock = self._pool_locks[model_id]
        if not lock.acquire(blocking=wait):
            raise ModelBusyError(f"Model {model_id} is busy; retry after the active operation")
        try:
            with self._lock:
                owner, depth = self._operation_owners.get(model_id, (get_ident(), 0))
                if not depth and not closing and (self._closing or model_id in self._closing_models):
                    raise RuntimeError(f"Model {model_id} is closing or has been shut down")
                self._operation_owners[model_id] = (owner, depth + 1)
                self._busy.add(model_id)
            try:
                yield
            finally:
                with self._lock:
                    if depth:
                        self._operation_owners[model_id] = (owner, depth)
                    else:
                        self._operation_owners.pop(model_id)
                        self._busy.discard(model_id)
        finally:
            lock.release()

    def health_check(self, model_id: str) -> bool:
        with self._pool_operation(model_id):
            healthy = self._pools[model_id].health_check()
            if not healthy:
                self._invalidate_pool(model_id, LifecycleState.DEAD)
            return healthy

    def recover(self, model_id: str) -> set[int]:
        with self._pool_operation(model_id):
            self._invalidate_pool(model_id, LifecycleState.STARTING)
            try:
                rebuilt = self._pools[model_id].recover()
                if rebuilt:
                    self._pools[model_id].set_onloaded(True)
                return rebuilt
            except Exception:
                self._invalidate_pool(model_id, LifecycleState.DEAD)
                raise

    def onload(self, model_id: str, tags: list[str] | None = None) -> None:
        with self._pool_operation(model_id):
            pool = self._pools[model_id]
            self._invalidate_pool(model_id, LifecycleState.ONLOADING)
            try:
                rebuilt = pool.recover() if getattr(pool, "recover_on_onload", True) else set()
                if pool.is_onloaded() and tags is None and not getattr(pool, "always_resume", False):
                    return
                dead = pool.fanout("resume_memory_occupation", skip_ranks=rebuilt, tags=tags)
                if dead:
                    pool.retire(dead)
                    pool.recover()
                # Track occupied memory independently of admission: a partial
                # restore still needs releasing, but never publishes READY.
                pool.set_onloaded(True)
            except Exception:
                # Some actors may already have resumed before another fails.
                # Keep cleanup enabled; DEAD closes admission independently.
                pool.set_onloaded(True)
                self._invalidate_pool(model_id, LifecycleState.DEAD)
                raise

    def offload(self, model_id: str) -> None:
        with self._pool_operation(model_id):
            pool = self._pools[model_id]
            self._invalidate_pool(model_id, LifecycleState.DRAINING)
            try:
                if pool.is_onloaded():
                    pool.retire(pool.fanout("release_memory_occupation"))
                    pool.set_onloaded(False)
                self._invalidate_pool(model_id, LifecycleState.SLEEPING)
            except Exception:
                self._invalidate_pool(model_id, LifecycleState.DEAD)
                raise

    def shutdown(self, model_id: str) -> None:
        with self._pool_operation(model_id):
            self._invalidate_pool(model_id, LifecycleState.DEAD)
            try:
                self._pools[model_id].shutdown()
                self._pools[model_id].set_onloaded(False)
            finally:
                # Pool teardown callbacks may invalidate topology as STARTING.
                self._invalidate_pool(model_id, LifecycleState.DEAD)

    def _replayed(self, operation_id: str, kind: str, value: object) -> bool:
        if not operation_id:
            raise ValueError("An operation ID is required")
        previous = self._operations.get(operation_id)
        if previous is not None and previous != (kind, value):
            raise ValueError(f"Operation conflict: {operation_id}")
        return previous is not None

    def register_model(self, spec: ModelConfig, *, operation_id: str) -> ModelConfig:
        spec = deepcopy(spec)
        with self._lock:
            if self._replayed(operation_id, "register", spec):
                return deepcopy(self._models[spec.model_id])
            previous = self._models.get(spec.model_id)
            if previous is not None and previous != spec:
                raise ValueError(f"Model configuration conflict: {spec.model_id}")
            if previous is None:
                if self._sealed:
                    raise ValueError("The startup model set is already sealed")
                old_signature = self._topology_signature(self._snapshot)
                self._models[spec.model_id] = spec
                self._snapshot = replace(
                    self._snapshot,
                    models=self._snapshot.models
                    + (
                        ModelSnapshot(
                            spec.model_id, allow_defer=spec.allow_defer, direct_eligible=spec.direct_eligible
                        ),
                    ),
                )
                if self._topology_signature(self._snapshot) != old_signature:
                    self._snapshot = replace(self._snapshot, topology_revision=self._snapshot.topology_revision + 1)
            self._operations[operation_id] = ("register", deepcopy(spec))
            return deepcopy(spec)

    def configure_routes(self, routing: RoutingSpec, *, operation_id: str) -> RoutingSpec:
        with self._lock:
            if self._replayed(operation_id, "routes", routing):
                return routing
            validate_routes(routing, set(self._models))
            old_signature = self._topology_signature(self._snapshot)
            self._snapshot = replace(self._snapshot, routing=routing)
            self._sealed = True
            if self._topology_signature(self._snapshot) != old_signature:
                self._snapshot = replace(self._snapshot, topology_revision=self._snapshot.topology_revision + 1)
            self._operations[operation_id] = ("routes", routing)
            return routing

    def publish_model(self, model: ModelSnapshot) -> None:
        """Commit an observation without inferring readiness from actor
        existence."""
        with self._lock:
            owner = self._operation_owners.get(model.model_id)
            if owner is not None and owner[0] != get_ident():
                raise ModelBusyError(f"Model {model.model_id} is owned by another operation")
            if model.state == LifecycleState.READY and model.model_id in self._preparations:
                raise ValueError("An active preparation must complete through complete_preparation")
            spec = self._models[model.model_id]
            if (model.allow_defer, model.direct_eligible) != (spec.allow_defer, spec.direct_eligible):
                raise ValueError("Published capabilities differ from registered capabilities")
            ids = [replica.engine_id for replica in model.replicas]
            if len(set(ids)) != len(ids):
                raise ValueError("Duplicate logical replica identities")
            if model.admission and model.state != LifecycleState.READY:
                raise ValueError("Only READY models may admit requests")
            if model.state == LifecycleState.READY:
                ready = [
                    replica for replica in model.replicas if replica.state == LifecycleState.READY and replica.base_url
                ]
                if not model.router_url or not ready:
                    raise ValueError("READY requires a Router and a healthy logical replica")
                if spec.weight_source == WeightSource.POLICY:
                    if model.required_weight_version is None or any(
                        replica.weight_version != model.required_weight_version for replica in ready
                    ):
                        raise ValueError("READY policy replicas must have the required weight version")
            models = tuple(model if old.model_id == model.model_id else old for old in self._snapshot.models)
            candidate = replace(self._snapshot, models=models)
            if self._topology_signature(candidate) != self._topology_signature(self._snapshot):
                candidate = replace(candidate, topology_revision=candidate.topology_revision + 1)
            self._snapshot = candidate

    def commit_observation(
        self, expected: ModelSnapshot, model: ModelSnapshot, *, evidence: PreparationEvidence
    ) -> bool:
        """Discard RPC results collected before a concurrent lifecycle
        transition."""
        with self._lock:
            current = next(item for item in self._snapshot.models if item.model_id == model.model_id)
            owner = self._operation_owners.get(model.model_id)
            if current is not expected or (owner is not None and owner[0] != get_ident()):
                return False
            if current.state in (LifecycleState.DRAINING, LifecycleState.SLEEPING, LifecycleState.DEAD):
                return False
            if model.state == LifecycleState.READY:
                token = self._preparations.get(model.model_id)
                if token is None:
                    token = self.begin_preparation(
                        model.model_id, required_weight_version=model.required_weight_version
                    )
                self.complete_preparation(token, model, evidence=evidence)
            else:
                self.publish_model(model)
            return True

    def begin_preparation(self, model_id: str, *, required_weight_version: str | None = None) -> PreparationToken:
        """Close admission before initialization, recovery or weight
        replacement."""
        with self._lock:
            spec = self._models[model_id]
            if spec.weight_source == WeightSource.POLICY and required_weight_version is None:
                raise ValueError("Policy preparation requires a target weight version")
            if spec.weight_source != WeightSource.POLICY and required_weight_version is not None:
                raise ValueError("Static or external weights cannot request a policy weight version")
            current = next(model for model in self._snapshot.models if model.model_id == model_id)
            self.publish_model(
                replace(
                    current,
                    state=LifecycleState.STARTING,
                    admission=False,
                    replicas=tuple(replace(replica, state=LifecycleState.STARTING) for replica in current.replicas),
                    required_weight_version=required_weight_version,
                )
            )
            token = PreparationToken(self._snapshot.manager_epoch, model_id, uuid4().hex)
            self._preparations[model_id] = token
            return token

    def complete_preparation(
        self, token: PreparationToken, model: ModelSnapshot, *, evidence: PreparationEvidence
    ) -> None:
        """Publish only after the engine owner confirms all prerequisite
        operations.

        The owner must pass successful RPC results, not inferred actor
        presence. This interface does not perform those operations or drain
        requests.
        """
        with self._lock:
            if token.model_id != model.model_id or self._preparations.get(model.model_id) != token:
                raise ValueError("Stale or mismatched preparation token")
            if not (evidence.initialized and evidence.health_checked and evidence.router_updated):
                raise ValueError("Preparation lacks initialization, health or Router evidence")
            spec = self._models[model.model_id]
            if spec.weight_source != WeightSource.POLICY and (
                evidence.weights_synced or model.required_weight_version is not None
            ):
                raise ValueError("Static or external weights cannot publish dynamic weight synchronization")
            current = next(item for item in self._snapshot.models if item.model_id == model.model_id)
            if spec.weight_source == WeightSource.POLICY and (
                not evidence.weights_synced or model.required_weight_version != current.required_weight_version
            ):
                raise ValueError("Preparation lacks the requested policy weight synchronization")
            if model.state != LifecycleState.READY:
                raise ValueError("Completed preparation must publish READY")
            del self._preparations[model.model_id]
            try:
                self.publish_model(model)
            except Exception:
                self._preparations[model.model_id] = token
                raise

    def invalidate_model(self, model_id: str, *, state: LifecycleState) -> None:
        """Invalidate pending completion before offload, retirement or
        shutdown."""
        if state == LifecycleState.READY:
            raise ValueError("Invalidation cannot publish READY")
        with self._lock:
            current = next(model for model in self._snapshot.models if model.model_id == model_id)
            self.publish_model(
                replace(
                    current,
                    state=state,
                    admission=False,
                    replicas=tuple(replace(replica, state=state) for replica in current.replicas),
                )
            )
            self._preparations.pop(model_id, None)

    def snapshot(self, model_names: tuple[str, ...] | None = None) -> RoleSnapshot:
        with self._lock:
            if model_names is None:
                return self._snapshot
            unknown = set(model_names) - self._models.keys()
            if unknown:
                raise KeyError(f"Unknown models: {sorted(unknown)}")
            return replace(self._snapshot, models=tuple(m for m in self._snapshot.models if m.model_id in model_names))


class InferenceManager:
    """Task-scoped inference control plane shared by all inference roles.

    The role state object is an implementation detail.  Callers keep one
    manager handle and identify a model by ``(role, model_id)``; compatibility
    callers may still construct the manager with a default role and use the old
    role-local method signatures.
    """

    def __init__(self, role: Role | str | None = None) -> None:
        self._lock = RLock()
        self.manager_epoch = new_manager_epoch()
        self._default_role = Role(role) if role is not None else None
        self._roles: dict[Role, _RoleInferenceState] = {}
        self._permits: dict[str, RequestPermit] = {}
        if self._default_role is not None:
            self._state(self._default_role)

    def _state(self, role: Role | str | None = None) -> _RoleInferenceState:
        selected = Role(role) if role is not None else self._default_role
        if selected is None:
            raise ValueError("An inference role is required")
        with self._lock:
            state = self._roles.get(selected)
            if state is None:
                state = _RoleInferenceState(selected)
                state._snapshot = replace(state._snapshot, manager_epoch=self.manager_epoch)
                self._roles[selected] = state
            return state

    def for_role(self, role: Role | str) -> "InferenceManager":
        """Return a role-scoped compatibility view over this same owner."""
        view = object.__new__(InferenceManager)
        view._lock = self._lock
        view.manager_epoch = self.manager_epoch
        view._default_role = Role(role)
        view._roles = self._roles
        view._permits = self._permits
        view._state(view._default_role)
        return view

    @property
    def role(self) -> Role | None:
        return self._default_role

    def register_model(self, spec: ModelConfig, *, operation_id: str, role: Role | str | None = None) -> ModelConfig:
        return self._state(role).register_model(spec, operation_id=operation_id)

    def configure_routes(
        self, routing: RoutingSpec, *, operation_id: str, role: Role | str | None = None
    ) -> RoutingSpec:
        return self._state(role).configure_routes(routing, operation_id=operation_id)

    def bind_pool(self, model_id: str, runtime: EnginePoolRuntime, *, role: Role | str | None = None) -> None:
        self._state(role).bind_pool(model_id, runtime)

    def attach_host(
        self,
        pools: Mapping[str, Any],
        *,
        stop_routers: Callable[[], None] | None = None,
        role: Role | str | None = None,
    ) -> None:
        self._state(role).attach_host(pools, stop_routers=stop_routers)

    def ready(self, role: Role | str | None = None) -> bool:
        return self._state(role).ready()

    def dispatch(
        self,
        model_id: str,
        method: str,
        /,
        *args: Any,
        wait: bool = False,
        role: Role | str | None = None,
        **kwargs: Any,
    ) -> Any:
        return self._state(role).dispatch(model_id, method, *args, wait=wait, **kwargs)

    def health_check(self, model_id: str, *, role: Role | str | None = None) -> bool:
        return self._state(role).health_check(model_id)

    def recover(self, model_id: str, *, role: Role | str | None = None) -> set[int]:
        return self._state(role).recover(model_id)

    def onload(self, model_id: str, tags: list[str] | None = None, *, role: Role | str | None = None) -> None:
        self._state(role).onload(model_id, tags)

    def offload(self, model_id: str, *, role: Role | str | None = None) -> None:
        self._state(role).offload(model_id)

    def _close_pool(
        self, model_id: str, /, *args: Any, wait: bool = False, role: Role | str | None = None, **kwargs: Any
    ) -> Any:
        """Close one pool through the selected role state.

        Keep this manager-level hook for compatibility callers that wrap the
        old role-local shutdown boundary to observe drain ordering.
        """
        return self._state(role)._close_pool(model_id, *args, wait=wait, **kwargs)

    def shutdown(self, model_id: str | None = None, *, role: Role | str | None = None) -> None:
        if model_id is not None:
            self._state(role).shutdown(model_id)
            return
        selected = Role(role) if role is not None else self._default_role
        if selected is not None:
            state = self._state(selected)
            with state._shutdown_lock:
                with state._lock:
                    state._closing = True
                    state._ready = False
                errors = []
                for pool_model_id in reversed(tuple(state._dispatch_pools)):
                    try:
                        self._close_pool(pool_model_id, wait=True, role=selected)
                    except Exception as exc:
                        errors.append(exc)
                try:
                    state._close_routers()
                except Exception as exc:
                    errors.append(exc)
                if errors:
                    raise RuntimeError("Failed to shut down one or more inference pools") from errors[0]
            return
        for selected in tuple(self._roles):
            self._roles[selected].close()

    def close(self) -> None:
        self.shutdown()

    def publish_model(self, model: ModelSnapshot, *, role: Role | str | None = None) -> None:
        self._state(role).publish_model(model)

    def begin_preparation(
        self, model_id: str, *, required_weight_version: str | None = None, role: Role | str | None = None
    ) -> PreparationToken:
        return self._state(role).begin_preparation(model_id, required_weight_version=required_weight_version)

    def complete_preparation(
        self,
        token: PreparationToken,
        model: ModelSnapshot,
        *,
        evidence: PreparationEvidence,
        role: Role | str | None = None,
    ) -> None:
        self._state(role).complete_preparation(token, model, evidence=evidence)

    def invalidate_model(self, model_id: str, *, state: LifecycleState, role: Role | str | None = None) -> None:
        self._state(role).invalidate_model(model_id, state=state)

    def snapshot(self, model_names: tuple[str, ...] | None = None, *, role: Role | str | None = None) -> RoleSnapshot:
        return self._state(role).snapshot(model_names)

    def snapshots(self) -> dict[Role, RoleSnapshot]:
        return {role: state.snapshot() for role, state in self._roles.items()}

    def __getattr__(self, name: str) -> Any:
        """Keep the pre-unification role-local API available to old callers."""
        default_role = self.__dict__.get("_default_role")
        if default_role is None:
            raise AttributeError(name)
        return getattr(self._state(default_role), name)

    def get_discovery_snapshot(
        self,
        *,
        role: Role | str | None = None,
        model_id: str | None = None,
        allow_defer: bool | None = None,
        direct_eligible: bool | None = None,
    ) -> RoleSnapshot:
        snapshot = self.snapshot(role=role)
        if model_id is None:
            return snapshot
        model = next(item for item in snapshot.models if item.model_id == model_id)
        if allow_defer is not None and model.allow_defer != allow_defer:
            raise ValueError("Discovery capability allow_defer differs from registered model")
        if direct_eligible is not None and model.direct_eligible != direct_eligible:
            raise ValueError("Discovery capability direct_eligible differs from registered model")
        return replace(snapshot, models=(model,))

    def get_operation(self, operation_id: str, *, role: Role | str | None = None) -> OperationSnapshot:
        if not operation_id:
            raise ValueError("An operation ID is required")
        state = self._state(role)
        with state._lock:
            record = state._operations.get(operation_id)
            if record is None:
                raise KeyError(f"Unknown operation: {operation_id}")
            kind, result = record
            return OperationSnapshot(
                operation_id=operation_id,
                owner_epoch=self.manager_epoch,
                status="completed",
                kind=kind,
                result=deepcopy(result),
            )

    def admit_request(
        self,
        model_id: str,
        request_id: str | None = None,
        *,
        role: Role | str | None = None,
        phase_handle: str | None = None,
        target: str | None = None,
    ) -> RequestPermit:
        del phase_handle
        selected = Role(role) if role is not None else self._default_role
        if selected is None:
            raise ValueError("An inference role is required")
        with self._lock:
            snapshot = self._state(selected).snapshot()
            model = next((item for item in snapshot.models if item.model_id == model_id), None)
            if model is None:
                raise KeyError(f"Unknown model: {model_id}")
            if not model.admission or model.state != LifecycleState.READY:
                raise RuntimeError(f"Model {model_id} is not ready for inference")
            rid = request_id or uuid4().hex
            existing = self._permits.get(rid)
            permit = RequestPermit(rid, self.manager_epoch, selected, model_id, target)
            if existing is not None and existing != permit:
                raise ValueError(f"Request already exists: {rid}")
            self._permits[rid] = permit
            return permit

    def complete_request(self, permit: RequestPermit | str) -> None:
        request_id = permit if isinstance(permit, str) else permit.request_id
        with self._lock:
            current = self._permits.get(request_id)
            if current is None:
                return
            if not isinstance(permit, str) and current != permit:
                raise ValueError("Request permit does not belong to this manager epoch and target")
            self._permits.pop(request_id, None)

    def cancel_request(self, permit: RequestPermit | str) -> None:
        self.complete_request(permit)

    def get_request(self, request_id: str) -> RequestPermit | None:
        with self._lock:
            return self._permits.get(request_id)
