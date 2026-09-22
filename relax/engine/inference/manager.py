# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU-only model registry and atomic publication boundary for engine
owners."""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from threading import Condition, RLock, get_ident
from time import monotonic
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence
from uuid import uuid4

from relax.engine.inference.capabilities import WeightSource
from relax.engine.inference.config import ModelConfig
from relax.engine.inference.discovery import new_manager_epoch
from relax.engine.inference.lifecycle import (
    STEP_ACTIVATED,
    STEP_ADMISSION_CLOSED,
    STEP_DEACTIVATED,
    STEP_DRAINED,
    STEP_READY_PUBLISHED,
    STEP_RELEASE_CONFIRMED,
    ActivationToken,
    ErrorCode,
    LifecycleError,
    OperationError,
    OperationResult,
    OperationStatus,
)
from relax.engine.inference.specs import validate_routes
from relax.engine.inference.types import LifecycleState, ModelRef, ModelSnapshot, Role, RoleSnapshot, RoutingSpec


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
        # In-flight requests per model. Draining waits on this index rather than
        # on a timer: a deadline proves nothing about whether a request finished.
        self._inflight: dict[ModelRef, set[str]] = {}
        # Registrations whose abort was sent and whose termination is not
        # confirmed. They stay in flight: an abort ACK is not a completion.
        self._cancelling: set[str] = set()
        self._inflight_cv = Condition(self._lock)
        self._lifecycle_ops: dict[str, tuple[tuple, OperationResult]] = {}
        # Shared by reference so a role-scoped view sees the same authority.
        self._authority: dict[str, str | None] = {"coordinator_epoch": None}
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
        view._inflight = self._inflight
        view._cancelling = self._cancelling
        view._inflight_cv = self._inflight_cv
        view._lifecycle_ops = self._lifecycle_ops
        view._authority = self._authority
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
            self._inflight.setdefault(ModelRef(selected, model_id), set()).add(rid)
            return permit

    def complete_request(self, permit: RequestPermit | str) -> None:
        request_id = permit if isinstance(permit, str) else permit.request_id
        with self._inflight_cv:
            current = self._permits.get(request_id)
            if current is None:
                return
            if not isinstance(permit, str) and current != permit:
                raise ValueError("Request permit does not belong to this manager epoch and target")
            self._permits.pop(request_id, None)
            self._cancelling.discard(request_id)
            pending = self._inflight.get(ModelRef(current.role, current.model_id))
            if pending is not None:
                pending.discard(request_id)
                if not pending:
                    self._inflight.pop(ModelRef(current.role, current.model_id), None)
            self._inflight_cv.notify_all()

    def cancel_request(self, permit: RequestPermit | str, *, dispatched: bool = True) -> RequestPermit | None:
        """Record that an abort was started; the request is not yet complete.

        A cancel is not a completion: sending an abort only proves the engine's
        scheduler received it, never that the request left the running batch. So
        the registration is kept and reported as aborting -- this control plane
        never claims the request finished.

        It deliberately does *not* keep blocking a drain. The evidence that an
        aborted request really stopped belongs to the engine's release path,
        which pauses admission, aborts what is in flight and then waits for
        ``/flush_cache`` to return 200 -- and SGLang answers 400 while the
        scheduler still has pending or running requests, so that 200 *is* the
        confirmation. Second-guessing it here would turn a client disconnect into
        a stuck activation group, which is not how this behaved before the
        control plane existed.

        ``dispatched=False`` means the request never reached an engine, so there
        is nothing to abort or to track.

        Returns the registration still in flight, or ``None`` once it is gone.
        """
        request_id = permit if isinstance(permit, str) else permit.request_id
        if not dispatched:
            self.complete_request(permit)
            return None
        with self._inflight_cv:
            current = self._permits.get(request_id)
            if current is None:
                return None
            if not isinstance(permit, str) and current != permit:
                raise ValueError("Request permit does not belong to this manager epoch and target")
            self._cancelling.add(request_id)
            return current

    def get_request(self, request_id: str) -> RequestPermit | None:
        with self._lock:
            return self._permits.get(request_id)

    def inflight_requests(self, target: ModelRef) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._inflight.get(target, ())))

    def cancelling_requests(self, target: ModelRef) -> tuple[str, ...]:
        """Registrations whose abort was sent but whose end is unconfirmed."""
        with self._lock:
            return tuple(sorted(self._inflight.get(target, set()) & self._cancelling))

    def _waiting_requests(self, targets: Sequence[ModelRef]) -> dict[str, set[str]]:
        """In-flight registrations that are still expected to complete."""
        return {str(target): self._inflight.get(target, set()) - self._cancelling for target in targets}

    def _discard_aborted(self, target: ModelRef) -> tuple[str, ...]:
        """Drop aborted registrations once the release is confirmed.

        A confirmed release means the engine reported it no longer occupies
        device memory, which on the colocated paths is only true after its own
        pause/abort/flush sequence succeeded. The aborted request is therefore
        provably over, and keeping its registration would only leak.
        """
        with self._inflight_cv:
            pending = self._inflight.get(target)
            if not pending:
                return ()
            dropped = tuple(sorted(pending & self._cancelling))
            for request_id in dropped:
                pending.discard(request_id)
                self._permits.pop(request_id, None)
                self._cancelling.discard(request_id)
            if not pending:
                self._inflight.pop(target, None)
            if dropped:
                self._inflight_cv.notify_all()
            return dropped

    def _discard_registrations(self, target: ModelRef) -> tuple[str, ...]:
        """Drop registrations because the engine process is confirmed gone.

        Process exit is the one piece of evidence stronger than a drain:
        nothing of that request is running any more.
        """
        with self._inflight_cv:
            dropped = tuple(sorted(self._inflight.pop(target, set())))
            for request_id in dropped:
                self._permits.pop(request_id, None)
                self._cancelling.discard(request_id)
            if dropped:
                self._inflight_cv.notify_all()
            return dropped

    # ------------------------------------------------------------------
    # Unified lifecycle: the six states and the fixed transition order.
    # ------------------------------------------------------------------
    def bind_coordinator(self, coordinator_epoch: str) -> None:
        """Require an activation token for every shared-resource transition.

        Before a Coordinator exists, the legacy call sites move models
        directly; once one is bound, an untokened activate/deactivate is
        rejected so a user script cannot take a slice behind the Coordinator's
        back.
        """
        if not coordinator_epoch:
            raise ValueError("A coordinator epoch is required")
        with self._lock:
            current = self._authority["coordinator_epoch"]
            if current is not None and current != coordinator_epoch:
                raise LifecycleError(
                    OperationError(ErrorCode.CONFLICT, "Another coordinator already owns this control plane")
                )
            self._authority["coordinator_epoch"] = coordinator_epoch

    @property
    def coordinator_epoch(self) -> str | None:
        return self._authority["coordinator_epoch"]

    def _model_refs(
        self, model_names: str | ModelRef | Sequence[str | ModelRef], role: Role | str | None
    ) -> tuple[ModelRef, ...]:
        if isinstance(model_names, (str, ModelRef)):
            model_names = [model_names]
        refs: list[ModelRef] = []
        for item in model_names:
            if isinstance(item, ModelRef):
                refs.append(item)
                continue
            selected = Role(role) if role is not None else self._default_role
            if selected is None:
                raise LifecycleError(
                    OperationError(ErrorCode.INVALID_ARGUMENT, "A model name requires an inference role")
                )
            refs.append(ModelRef(selected, item))
        if len(set(refs)) != len(refs):
            raise LifecycleError(OperationError(ErrorCode.INVALID_ARGUMENT, "Duplicate lifecycle targets"))
        return tuple(refs)

    def _authorize(self, token: ActivationToken | None, targets: Sequence[ModelRef]) -> None:
        with self._lock:
            authority = self._authority["coordinator_epoch"]
        if authority is None:
            return
        if token is None:
            raise LifecycleError(
                OperationError(ErrorCode.INVALID_ARGUMENT, "A coordinator is bound; an activation token is required")
            )
        if token.coordinator_epoch != authority:
            raise LifecycleError(
                OperationError(ErrorCode.STALE_GENERATION, "Activation token belongs to an earlier coordinator")
            )
        missing = [str(target) for target in targets if not token.covers(target)]
        if missing:
            raise LifecycleError(
                OperationError(ErrorCode.INVALID_ARGUMENT, f"Activation token does not cover {sorted(missing)}")
            )

    def _replay_lifecycle(self, operation_id: str, signature: tuple) -> OperationResult | None:
        if not operation_id:
            raise LifecycleError(OperationError(ErrorCode.INVALID_ARGUMENT, "An operation ID is required"))
        with self._lock:
            record = self._lifecycle_ops.get(operation_id)
        if record is None:
            return None
        if record[0] != signature:
            raise LifecycleError(
                OperationError(ErrorCode.CONFLICT, f"Operation {operation_id} was used for other arguments")
            )
        return record[1]

    def _record_lifecycle(self, operation_id: str, signature: tuple, result: OperationResult) -> OperationResult:
        with self._lock:
            self._lifecycle_ops[operation_id] = (signature, result)
        return result

    def _result(
        self,
        operation_id: str,
        kind: str,
        targets: Sequence[ModelRef],
        *,
        status: OperationStatus,
        step: str | None = None,
        error: OperationError | None = None,
        release_confirmed: bool = False,
    ) -> OperationResult:
        return OperationResult(
            operation_id=operation_id,
            owner_epoch=self.manager_epoch,
            status=status,
            kind=kind,
            targets=tuple(targets),
            last_confirmed_step=step,
            error=error,
            release_confirmed=release_confirmed,
        )

    def _model_state(self, target: ModelRef) -> ModelSnapshot:
        snapshot = self._state(target.role).snapshot((target.model_id,))
        return snapshot.models[0]

    def _pool_of(self, target: ModelRef) -> EnginePoolRuntime | None:
        return self._state(target.role)._pools.get(target.model_id)

    def _dispatchable(self, target: ModelRef, method: str) -> bool:
        pool = self._state(target.role)._dispatch_pools.get(target.model_id)
        return pool is not None and callable(getattr(pool, method, None))

    def drain(
        self,
        model_names: str | ModelRef | Sequence[str | ModelRef],
        *,
        operation_id: str,
        activation_token: ActivationToken | None = None,
        timeout_s: float | None = None,
        role: Role | str | None = None,
    ) -> OperationResult:
        """Close model admission and wait for the registered requests to end.

        Success leaves the models in ``DRAINING``: draining does not release
        memory, and a timeout keeps them there with an error rather than
        handing the resource on.
        """
        targets = self._model_refs(model_names, role)
        signature = ("drain", targets, timeout_s)
        replay = self._replay_lifecycle(operation_id, signature)
        if replay is not None:
            return replay
        self._authorize(activation_token, targets)
        try:
            for target in targets:
                model = self._model_state(target)
                if model.state in (LifecycleState.SLEEPING, LifecycleState.DEAD):
                    continue
                self.invalidate_model(target.model_id, state=LifecycleState.DRAINING, role=target.role)
        except ModelBusyError as exc:
            return self._record_lifecycle(
                operation_id,
                signature,
                self._result(
                    operation_id,
                    "drain",
                    targets,
                    status=OperationStatus.FAILED,
                    error=OperationError(ErrorCode.BUSY, str(exc), retryable=True),
                ),
            )
        deadline = None if timeout_s is None else monotonic() + float(timeout_s)
        with self._inflight_cv:
            # Wait only for requests still expected to finish on their own. An
            # aborted one belongs to the engine release path, not to this wait;
            # see cancel_request.
            while any(self._waiting_requests(targets).values()):
                remaining = None if deadline is None else deadline - monotonic()
                if remaining is not None and remaining <= 0:
                    break
                # Completion notifies, so the bounded wait is only a liveness
                # re-check; it never concludes that a request finished.
                self._inflight_cv.wait(min(remaining, 1.0) if remaining is not None else 1.0)
            waiting = {name: sorted(requests) for name, requests in self._waiting_requests(targets).items()}
            aborting = {
                str(target): sorted(self._inflight.get(target, set()) & self._cancelling) for target in targets
            }
        outstanding = {name: requests for name, requests in waiting.items() if requests}
        aborted = {name: requests for name, requests in aborting.items() if requests}
        if outstanding:
            return self._record_lifecycle(
                operation_id,
                signature,
                self._result(
                    operation_id,
                    "drain",
                    targets,
                    status=OperationStatus.FAILED,
                    step=STEP_ADMISSION_CLOSED,
                    error=OperationError(
                        ErrorCode.TIMEOUT,
                        f"Requests are still in flight: {outstanding}"
                        + (f"; aborted, left to the engine release drain: {aborted}" if aborted else ""),
                        retryable=True,
                    ),
                ),
            )
        return self._record_lifecycle(
            operation_id,
            signature,
            self._result(operation_id, "drain", targets, status=OperationStatus.COMPLETED, step=STEP_DRAINED),
        )

    def deactivate(
        self,
        model_names: str | ModelRef | Sequence[str | ModelRef],
        *,
        operation_id: str,
        activation_token: ActivationToken | None = None,
        role: Role | str | None = None,
        observe: Callable[[Sequence[ModelRef]], None] | None = None,
    ) -> OperationResult:
        """Release device memory and confirm it, model by model.

        ``release_confirmed`` is only true when every target's pool reports
        that it no longer occupies memory. An exception, an unreachable actor
        or a pool that still reports occupation all leave it false.
        """
        targets = self._model_refs(model_names, role)
        signature = ("deactivate", targets)
        replay = self._replay_lifecycle(operation_id, signature)
        if replay is not None:
            return replay
        self._authorize(activation_token, targets)
        error: OperationError | None = None
        for target in targets:
            pool = self._pool_of(target)
            if pool is None:
                error = OperationError(
                    ErrorCode.NOT_FOUND, f"No engine pool is bound for {target}", model_id=target.model_id
                )
                break
            try:
                if self._dispatchable(target, "offload"):
                    self.dispatch(target.model_id, "offload", wait=True, role=target.role)
                else:
                    self.offload(target.model_id, role=target.role)
            except ModelBusyError as exc:
                error = OperationError(ErrorCode.BUSY, str(exc), retryable=True, model_id=target.model_id)
                break
            except Exception as exc:
                error = OperationError(ErrorCode.UNAVAILABLE, f"{type(exc).__name__}: {exc}", model_id=target.model_id)
                break
        if observe is not None:
            # Let an owner commit the backend's own observation before the
            # result is evaluated, so one operation produces one recorded state.
            try:
                observe(targets)
            except Exception as exc:
                error = error or OperationError(ErrorCode.UNAVAILABLE, f"{type(exc).__name__}: {exc}")
        confirmed = error is None and all(self._release_confirmed(target) for target in targets)
        if confirmed:
            for target in targets:
                self._discard_aborted(target)
        if error is None and not confirmed:
            error = OperationError(
                ErrorCode.UNKNOWN_COMPLETION, "A pool still reports occupied device memory after deactivation"
            )
        return self._record_lifecycle(
            operation_id,
            signature,
            self._result(
                operation_id,
                "deactivate",
                targets,
                status=OperationStatus.COMPLETED if error is None else OperationStatus.FAILED,
                step=STEP_RELEASE_CONFIRMED if confirmed else STEP_DEACTIVATED,
                error=error,
                release_confirmed=confirmed,
            ),
        )

    def _release_confirmed(self, target: ModelRef) -> bool:
        pool = self._pool_of(target)
        if pool is None:
            return False
        try:
            return not bool(pool.is_onloaded())
        except Exception:
            # An unreachable pool is not proof of a release.
            return False

    def activate(
        self,
        model_names: str | ModelRef | Sequence[str | ModelRef],
        *,
        operation_id: str,
        activation_token: ActivationToken | None = None,
        tags: list[str] | None = None,
        role: Role | str | None = None,
        observe: Callable[[Sequence[ModelRef]], None] | None = None,
    ) -> OperationResult:
        """Restore device memory for the targets and report what was confirmed.

        ``ready_published`` is only reported when every target actually reached
        ``READY``. A policy model whose weights the trainer has yet to
        synchronize legitimately stops at ``activated``; a target that ends up
        ``DEAD`` fails.
        """
        targets = self._model_refs(model_names, role)
        signature = ("activate", targets, tuple(tags or ()))
        replay = self._replay_lifecycle(operation_id, signature)
        if replay is not None:
            return replay
        self._authorize(activation_token, targets)
        error: OperationError | None = None
        for target in targets:
            model = self._model_state(target)
            if model.state == LifecycleState.READY and tags is None and self._pool_is_onloaded(target):
                continue
            try:
                if self._dispatchable(target, "onload"):
                    self.dispatch(target.model_id, "onload", tags, wait=True, role=target.role)
                else:
                    self.onload(target.model_id, tags, role=target.role)
            except ModelBusyError as exc:
                error = OperationError(ErrorCode.BUSY, str(exc), retryable=True, model_id=target.model_id)
                break
            except Exception as exc:
                error = OperationError(ErrorCode.UNAVAILABLE, f"{type(exc).__name__}: {exc}", model_id=target.model_id)
                break
        if observe is not None:
            try:
                observe(targets)
            except Exception as exc:
                error = error or OperationError(ErrorCode.UNAVAILABLE, f"{type(exc).__name__}: {exc}")
        states = {target: self._model_state(target).state for target in targets}
        dead = [str(target) for target, state in states.items() if state == LifecycleState.DEAD]
        if error is None and dead:
            error = OperationError(ErrorCode.UNAVAILABLE, f"Activation left models dead: {sorted(dead)}")
        ready = all(state == LifecycleState.READY for state in states.values())
        return self._record_lifecycle(
            operation_id,
            signature,
            self._result(
                operation_id,
                "activate",
                targets,
                status=OperationStatus.COMPLETED if error is None else OperationStatus.FAILED,
                step=STEP_READY_PUBLISHED if error is None and ready else STEP_ACTIVATED,
                error=error,
                release_confirmed=False,
            ),
        )

    def _pool_is_onloaded(self, target: ModelRef) -> bool:
        pool = self._pool_of(target)
        if pool is None:
            return False
        try:
            return bool(pool.is_onloaded())
        except Exception:
            return False

    def shutdown_models(
        self,
        model_names: str | ModelRef | Sequence[str | ModelRef],
        *,
        operation_id: str,
        activation_token: ActivationToken | None = None,
        timeout_s: float | None = None,
        role: Role | str | None = None,
    ) -> OperationResult:
        """Drain, then tear the pools down and confirm the release.

        A deliberate shutdown still drains first: closing a pool with requests
        in flight would report a completion this control plane never observed.
        """
        targets = self._model_refs(model_names, role)
        signature = ("shutdown", targets, timeout_s)
        replay = self._replay_lifecycle(operation_id, signature)
        if replay is not None:
            return replay
        self._authorize(activation_token, targets)
        drained = self.drain(
            targets, operation_id=f"{operation_id}:drain", activation_token=activation_token, timeout_s=timeout_s
        )
        error = None if drained.succeeded else drained.error
        teardown_error: OperationError | None = None
        for target in targets:
            try:
                self._close_pool(target.model_id, wait=True, role=target.role)
            except Exception as exc:
                teardown_error = teardown_error or OperationError(
                    ErrorCode.UNAVAILABLE, f"{type(exc).__name__}: {exc}", model_id=target.model_id
                )
        confirmed = all(self._release_confirmed(target) for target in targets)
        if teardown_error is None and confirmed:
            # The pool is down and the release is confirmed, so the requests it
            # was running are provably over. Process exit supersedes a drain
            # that could not confirm them; this is the deliberate-abort path,
            # not a way to pretend a drain succeeded.
            for target in targets:
                self._discard_registrations(target)
            error = None
        else:
            error = error or teardown_error
        return self._record_lifecycle(
            operation_id,
            signature,
            self._result(
                operation_id,
                "shutdown",
                targets,
                status=OperationStatus.COMPLETED if error is None and confirmed else OperationStatus.FAILED,
                step=STEP_RELEASE_CONFIRMED if confirmed else STEP_DEACTIVATED,
                error=error
                or (
                    None
                    if confirmed
                    else OperationError(ErrorCode.UNKNOWN_COMPLETION, "Shutdown did not confirm the memory release")
                ),
                release_confirmed=confirmed,
            ),
        )

    def get_lifecycle_operation(self, operation_id: str) -> OperationResult:
        with self._lock:
            record = self._lifecycle_ops.get(operation_id)
        if record is None:
            raise LifecycleError(OperationError(ErrorCode.NOT_FOUND, f"Unknown operation: {operation_id}"))
        return record[1]

    def model_states(self) -> dict[ModelRef, LifecycleState | None]:
        """Project every registered model's lifecycle state for a
        Coordinator."""
        with self._lock:
            roles = tuple(self._roles)
        return {
            ModelRef(role, model.model_id): model.state
            for role in roles
            for model in self._state(role).snapshot().models
        }
