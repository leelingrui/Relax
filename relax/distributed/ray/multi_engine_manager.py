# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Common lifecycle skeleton for managers that own a pool of SGLang engine
replicas (GenRM judges, OPD teachers, ...).

Concrete managers subclass ``MultiEngineManager`` (in addition to their own
``@ray.remote`` decorator) and implement the hooks below to plug in their
engine actor class, placement, GPU/port allocation, and env vars. The base
class owns: parallel engine bring-up, health checking, dead-engine
detection/retirement, recovery, and onload/offload with idempotency tracking.

Placement is resolved per engine (not once per manager): a manager may put
all engines on one shared placement group (e.g. GenRM colocated with
rollout), or give each engine its own dedicated placement group that it
creates and tears down itself (e.g. a non-colocated OPD teacher). Subclasses
express this via ``_resolve_placement``.
"""

from dataclasses import replace
from typing import Any, Optional

import ray
import requests
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from relax.engine.inference.capabilities import WeightSource
from relax.engine.inference.manager import InferenceManager, PreparationEvidence
from relax.engine.inference.placement import PlacementPlanner, PlacementSlice
from relax.engine.inference.specs import model_spec_from_pool, replicas_from_slots
from relax.engine.inference.types import (
    LifecycleState,
    ModelSnapshot,
    ReplicaSnapshot,
    Role,
    RoleSnapshot,
    RoutingSpec,
)
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

# An engine process can die on its own (e.g. SGLang's scheduler watchdog
# SIGQUITs the server after a CUDA-level hang). The next call into it then
# raises one of these. Everything else is a real bug and must propagate.
#   - ConnectionError / TimeoutError: raised by the engine's
#     release_memory_occupation when a drain loop hits its dead-server
#     fast-fail or its deadline.
#   - requests.exceptions.{ConnectionError,Timeout}: raised by _make_request,
#     i.e. the resume_memory_occupation path. These are OSError subclasses but
#     NOT builtin ConnectionError/TimeoutError, so they must be listed
#     explicitly -- otherwise an engine that died during the offloaded window
#     (only observable at onload) escalates to a global restart.
#   - RayActorError: the Ray actor itself is gone.
# ray.get re-raises as a class inheriting from BOTH RayTaskError and the
# original cause (ray/exceptions.py::as_instanceof_cause), so isinstance works.
_ENGINE_DEAD_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    ray.exceptions.RayActorError,
)

# Rebuilding one engine is ~1.5 min (weight load + cuda graph capture). Bound it
# so a dead *node* -- whose placement-group bundle can never be filled -- degrades
# to "run with N-1 engines" instead of hanging the training step forever.
_ENGINE_REBUILD_TIMEOUT_S = 900.0
_ENGINE_SHUTDOWN_TIMEOUT_S = 60.0


def _is_engine_dead(exc: BaseException) -> bool:
    return isinstance(exc, _ENGINE_DEAD_EXCEPTIONS)


class _PoolRuntime:
    def __init__(self, owner: "MultiEngineManager") -> None:
        self.owner = owner

    def health_check(self) -> bool:
        return self.owner._health_check_engines()

    def recover(self) -> set[int]:
        self.owner._check_recovery_allowed()
        return self.owner._recover_engines()

    def is_onloaded(self) -> bool:
        return self.owner._onloaded

    def set_onloaded(self, value: bool) -> None:
        self.owner._onloaded = value

    def fanout(self, method: str, *, skip_ranks: set[int] | None = None, **kwargs: Any) -> list[int]:
        return self.owner._fanout(method, skip_ranks=skip_ranks, **kwargs)

    def retire(self, ranks: list[int]) -> None:
        self.owner._retire_engines(ranks)

    def shutdown(self) -> None:
        self.owner._shutdown_engines()


class MultiEngineManager:
    """Base class for managers of a fixed-size pool of engine replicas.

    Not a Ray actor itself -- subclasses apply ``@ray.remote`` so this class
    can be unit-tested without a Ray runtime. Engine *slots* are tracked as a
    flat list indexed by rank; a ``None`` slot means "dead, needs rebuild".

    A slot is the unit of scheduling (one Ray actor, one placement-group
    bundle range). ``nodes_per_engine > 1`` lets one *logical* engine span
    multiple slots/nodes (e.g. a TP group larger than one node): ``num_slots``
    passed to ``__init__`` already accounts for this (it is node-count, not
    logical-engine-count), and only the head slot of each group
    (``rank % nodes_per_engine == 0``) runs the HTTP server, so ``engines``
    strides over the followers.
    """

    def __init__(
        self,
        args: Any,
        *,
        num_slots: int,
        nodes_per_engine: int = 1,
        engine_actor_cls: type,
        skip_init: bool = False,
        log_prefix: str = "",
        inference_manager: InferenceManager | None = None,
        model_id: str = "default",
        adapter: Any | None = None,
    ) -> None:
        self.args = args
        self.engine_actor_cls = engine_actor_cls
        self.replica_specs = replicas_from_slots("engine", num_slots, nodes_per_engine)
        self.nodes_per_engine = nodes_per_engine
        self._log_prefix = log_prefix
        self.adapter = adapter or self

        self.all_engines: list[Any] = [None] * num_slots
        self.num_new_engines = 0
        # Per-slot (pg_tuple, owns_pg) so shutdown()/_retire_engines() only
        # remove placement groups this manager itself created.
        self._engine_placements: dict[int, tuple] = {}
        self._placement_slices: dict[int, PlacementSlice] = {}
        self._engine_addr_and_ports: dict[int, dict] = {}
        # Track memory-occupation state so repeated onload/offload calls become
        # safe no-ops. Engines start onloaded; callers may immediately offload.
        self._onloaded = True
        self._memory_ready = True
        self._owns_inference_manager = inference_manager is None
        self.inference_manager = inference_manager or InferenceManager(
            getattr(self.adapter, "inference_role", Role.GENRM)
        )
        model_path = (
            getattr(self.adapter, "inference_model_path", None) or getattr(args, "hf_checkpoint", None) or "managed"
        )
        self.router_url = getattr(self.adapter, "router_url", None)
        self.inference_model_id = model_id
        self.model_spec = model_spec_from_pool(
            model_id,
            model_path,
            weight_source=getattr(self.adapter, "inference_weight_source", WeightSource.CHECKPOINT),
            num_slots=num_slots,
            nodes_per_engine=nodes_per_engine,
            num_gpus_per_engine=getattr(self.adapter, "inference_num_gpus_per_engine", 1),
            overrides=tuple(sorted(getattr(self.adapter, "inference_overrides", {}).items())),
            allow_defer=True,
        )
        self.replica_specs = self.model_spec.engine_groups[0].replicas
        self.inference_manager.register_model(
            self.model_spec,
            operation_id=f"register:{model_id}",
        )
        self.inference_manager.bind_pool(model_id, _PoolRuntime(self))
        if self._owns_inference_manager:
            self.inference_manager.configure_routes(RoutingSpec(default_model=model_id), operation_id="routes")
        self.manager_epoch = self.inference_manager.snapshot().manager_epoch
        if not skip_init:
            self.initialize()

    def initialize(self) -> None:
        """Materialize the registered pool after the role routing is sealed."""
        self.inference_manager.invalidate_model(self.inference_model_id, state=LifecycleState.STARTING)
        self._init_engines([rank for rank, engine in enumerate(self.all_engines) if engine is None])
        self.num_new_engines = len(self.engines)
        self._publish_engine_state()

    def _publish_engine_state(self) -> None:
        """Observe complete logical replicas after an operation, never in
        discovery."""
        expected = self.inference_manager.snapshot((self.inference_model_id,)).models[0]
        replicas = []
        router_url = getattr(self, "router_url", None)
        for spec in self.replica_specs:
            head_rank = spec.node_ranks[0]
            nodes = [self.all_engines[rank] for rank in spec.node_ranks]
            state = LifecycleState.DEAD
            observation = {}
            if all(node is not None for node in nodes):
                state = LifecycleState.SLEEPING if not self._onloaded else LifecycleState.STARTING
                if self._memory_ready:
                    try:
                        result = ray.get(nodes[0].get_inference_observation.remote(), timeout=15)
                        observation = result if isinstance(result, dict) else {}
                    except Exception as exc:
                        logger.warning("%s replica %s observation failed: %s", self._log_prefix, head_rank, exc)
                    if observation.get("healthy") and observation.get("router_registered") and router_url:
                        state = LifecycleState.READY
            replicas.append(ReplicaSnapshot(spec.replica_id, state, observation.get("base_url")))
        ready = any(replica.state == LifecycleState.READY for replica in replicas)
        state = (
            LifecycleState.READY
            if ready
            else (LifecycleState.SLEEPING if not self._onloaded else LifecycleState.STARTING)
        )
        if not any(engine is not None for engine in self.all_engines):
            state = LifecycleState.DEAD
        model = ModelSnapshot(self.inference_model_id, tuple(replicas), router_url, state, ready, allow_defer=True)
        self.inference_manager.commit_observation(expected, model, evidence=PreparationEvidence(True, ready, ready))

    def get_discovery_snapshot(
        self,
        *,
        role: Role,
        model_id: str,
        allow_defer: bool = False,
        direct_eligible: bool = False,
        phase: str | None = None,
        status_filter: str | None = None,
    ) -> RoleSnapshot:
        """Project the cached pool snapshot under the legacy model alias."""
        if status_filter not in (None, "active", "dead"):
            raise ValueError("status_filter must be one of: active, dead")
        snapshot = self.inference_manager.snapshot((self.inference_model_id,))
        model = snapshot.models[0]
        model = replace(
            model,
            model_id=model_id,
            replicas=tuple(
                replace(replica, engine_id=f"{model_id}/{replica.engine_id.split('/', 1)[1]}")
                for replica in model.replicas
                if status_filter is None
                or status_filter == ("dead" if replica.state == LifecycleState.DEAD else "active")
            ),
            allow_defer=allow_defer,
            direct_eligible=direct_eligible,
        )
        return replace(snapshot, role=role, models=(model,), phase=phase, routing=RoutingSpec(default_model=model_id))

    @property
    def engines(self) -> list[Any]:
        """Return the head-node slot of each logical engine."""
        return self.all_engines[:: self.nodes_per_engine]

    # ------------------------------------------------------------------
    # Hooks -- subclasses must implement these.
    # ------------------------------------------------------------------

    def _resolve_placement(self, rank: int) -> tuple[tuple, bool, int]:
        """Return ``(pg_tuple, owns_pg, gpu_index)`` for the slot at ``rank``.

        ``pg_tuple`` is ``(pg, reordered_bundle_indices, reordered_gpu_ids)``
        as returned by ``create_placement_group``. ``owns_pg`` marks whether
        this manager created ``pg_tuple`` itself (and must remove it on
        shutdown/retirement) or is borrowing a placement group it does not
        own. ``gpu_index`` is this slot's starting index into
        ``reordered_gpu_ids``/``reordered_bundle_indices``.
        """
        raise NotImplementedError

    def _resolve_planned_placement(self, rank: int) -> tuple[tuple, bool, int, PlacementSlice | None]:
        """Return the legacy placement tuple plus its static planner result.

        The tuple remains the compatibility contract used by existing manager
        implementations.  Adapters that participate in Phase 4 override this
        hook and provide a ``PlacementSlice``; PG lifecycle remains owned by
        the legacy ``owns_pg`` value until Phase 5.
        """
        pg_tuple, owns_pg, gpu_index = self.adapter._resolve_placement(rank)
        return pg_tuple, owns_pg, gpu_index, None

    def _ray_resource_kwargs(self, rank: int) -> dict:
        """Return the num_cpus/num_gpus fractional Ray resource request for one
        slot."""
        raise NotImplementedError

    def _allocate_engine_addr_and_ports(self, *, new_engines: list[tuple]) -> dict[int, dict]:
        """Allocate host/port/dist_init_addr for the given (rank, engine)
        pairs.

        Returns a dict keyed by rank; each value must contain at least
        ``host``, ``port``, ``nccl_port``, ``dist_init_addr``.
        """
        raise NotImplementedError

    def _build_engine_env_vars(self) -> dict[str, str]:
        """Return the runtime_env env vars for a new engine actor."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Hooks -- subclasses may override; sane defaults provided.
    # ------------------------------------------------------------------

    def _engine_ctor_args(self, rank: int) -> Any:
        """First positional argument passed to the engine actor constructor."""
        return self.args

    def _build_engine_ctor_kwargs(self, rank: int) -> dict:
        """Extra keyword arguments passed to the engine actor constructor
        (beyond rank/worker_type/base_gpu_id)."""
        return {}

    def _build_engine_init_kwargs(self, rank: int, addr_and_ports: dict) -> dict:
        """Keyword arguments passed to ``engine.init.remote(...)``."""
        return dict(addr_and_ports)

    # ------------------------------------------------------------------
    # Engine bring-up.
    # ------------------------------------------------------------------

    def _init_engines(self, ranks: list[int]) -> int:
        """Roll back only slots acquired by this attempt, including pre-init
        failures."""
        pending = [rank for rank in ranks if self.all_engines[rank] is None]
        try:
            return self._create_engines(pending)
        except Exception:
            for rank in pending:
                engine = self.all_engines[rank]
                if engine is not None:
                    try:
                        ray.get(engine.shutdown.remote(), timeout=_ENGINE_SHUTDOWN_TIMEOUT_S)
                    except Exception as exc:
                        logger.warning("%s partial startup shutdown failed: %s", self._log_prefix, exc)
                    try:
                        ray.kill(engine)
                    except Exception as exc:
                        logger.warning("%s partial startup actor cleanup failed: %s", self._log_prefix, exc)
                    self.all_engines[rank] = None
                self._remove_owned_pg(rank)
            raise

    def _create_engines(self, ranks: list[int]) -> int:
        """Create actors for the given slot ranks, fire init.remote() for all
        of them without blocking, then await everything in one ray.get so a
        large engine doesn't pay N x cold-load latency.

        On failure, kill any newly created engines and leave their slots None
        so the caller sees them as still-dead rather than silently healthy.
        """
        EngineActor = ray.remote(self.engine_actor_cls)
        new_engines: list[tuple[int, Any]] = []
        for rank in ranks:
            if self.all_engines[rank] is not None:
                continue

            pg_tuple, owns_pg, gpu_index, placement_slice = self.adapter._resolve_planned_placement(rank)
            self._engine_placements[rank] = (pg_tuple, owns_pg)
            if placement_slice is not None:
                self._placement_slices[rank] = placement_slice
            pg, reordered_bundle_indices, reordered_gpu_ids = pg_tuple
            base_gpu_id = int(reordered_gpu_ids[gpu_index])
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=reordered_bundle_indices[gpu_index],
            )

            engine = EngineActor.options(
                **self.adapter._ray_resource_kwargs(rank),
                scheduling_strategy=scheduling_strategy,
                runtime_env={"env_vars": self.adapter._build_engine_env_vars()},
            ).remote(
                self.adapter._engine_ctor_args(rank),
                rank=rank,
                worker_type="regular",
                base_gpu_id=base_gpu_id,
                **self.adapter._build_engine_ctor_kwargs(rank),
            )
            new_engines.append((rank, engine))
            self.all_engines[rank] = engine

        num_new_engines = len(new_engines)
        if num_new_engines == 0:
            return num_new_engines

        addr_and_ports = self.adapter._allocate_engine_addr_and_ports(new_engines=new_engines)
        for rank, _ in new_engines:
            self._engine_addr_and_ports[rank] = addr_and_ports[rank]

        init_handles = [
            engine.init.remote(**self.adapter._build_engine_init_kwargs(rank, addr_and_ports[rank]))
            for rank, engine in new_engines
        ]
        ray.get(init_handles, timeout=_ENGINE_REBUILD_TIMEOUT_S)

        return num_new_engines

    def _remove_owned_pg(self, rank: int) -> None:
        placement = self._engine_placements.pop(rank, None)
        placement_slice = self._placement_slices.pop(rank, None)
        if placement is None:
            return
        pg_tuple, owns_pg = placement
        if not owns_pg:
            return
        if any(other[0][0] == pg_tuple[0] for other in self._engine_placements.values()):
            return
        if placement_slice is not None:
            PlacementPlanner.cancel(placement_slice)
        try:
            from ray.util.placement_group import remove_placement_group

            remove_placement_group(pg_tuple[0])
        except Exception as exc:
            logger.warning(f"{self._log_prefix} remove placement group for rank={rank} failed: {exc}")

    # ------------------------------------------------------------------
    # Health / lifecycle.
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        healthy = self.inference_manager.health_check(self.inference_model_id)
        self._publish_engine_state()
        return healthy

    def _health_check_engines(self) -> bool:
        """Perform a health check on every engine."""
        health_results = []
        for engine in self.engines:
            if engine is not None:
                try:
                    health_results.append(ray.get(engine.health_generate.remote(), timeout=5.0))
                except Exception as e:
                    logger.warning(f"{self._log_prefix} engine health check failed: {e}")
                    health_results.append(False)
            else:
                health_results.append(False)
        return bool(health_results) and all(health_results)

    def onload(self, tags: Optional[list[str]] = None) -> None:
        if self._onloaded and self._memory_ready and tags is None and all(self.all_engines):
            return
        if not getattr(self, "_memory_ready", True) and tags is None:
            # Partial restores still occupy memory, but must not trigger the
            # common full-onload no-op path.
            self._onloaded = False
        self._memory_ready = False
        self.inference_manager.onload(self.inference_model_id, tags=tags)
        self._memory_ready = not tags
        self._publish_engine_state()

    def offload(self) -> None:
        self._memory_ready = False
        self.inference_manager.offload(self.inference_model_id)

    def _fanout(self, method: str, *, skip_ranks: Optional[set] = None, **kwargs) -> list[int]:
        """Call ``method`` on every live engine; return the ranks that are
        dead.

        Per-handle ray.get rather than one ray.get over the list: the batched
        form aborts on the first failure and loses which engine raised.
        """
        skip = skip_ranks or set()
        handles = {}
        for rank in range(0, len(self.all_engines), self.nodes_per_engine):
            engine = self.all_engines[rank]
            if engine is None or rank in skip:
                continue
            handles[rank] = getattr(engine, method).remote(**kwargs)

        dead = []
        for rank, handle in handles.items():
            try:
                ray.get(handle)
            except Exception as exc:
                if not _is_engine_dead(exc):
                    raise
                logger.warning(f"{self._log_prefix} engine rank={rank} died during {method}: {exc}")
                dead.append(rank)
        return dead

    def _retire_engines(self, ranks: list[int]) -> None:
        """Tear down dead engines and null their slots so recover() rebuilds
        them."""
        if ranks:
            self.inference_manager.invalidate_model(self.inference_model_id, state=LifecycleState.STARTING)
        for rank in ranks:
            for i in range(rank, rank + self.nodes_per_engine):
                engine = self.all_engines[i]
                if engine is None:
                    continue
                try:
                    # shutdown() kill_process_tree's the SGLang server. Must run
                    # before ray.kill or the scheduler subprocesses are orphaned
                    # and keep holding GPU memory, so the rebuild can't fit.
                    ray.get(engine.shutdown.remote(), timeout=_ENGINE_SHUTDOWN_TIMEOUT_S)
                except Exception as exc:
                    logger.warning(f"{self._log_prefix} engine rank={i} shutdown failed (killing anyway): {exc}")
                try:
                    ray.kill(engine)
                except Exception as exc:
                    logger.warning(f"{self._log_prefix} engine rank={i} ray.kill failed: {exc}")
                self.all_engines[i] = None
                self._remove_owned_pg(i)
                logger.info(f"{self._log_prefix} engine rank={i} retired")

    def recover(self) -> set:
        rebuilt = self.inference_manager.recover(self.inference_model_id)
        self._publish_engine_state()
        return rebuilt

    def _check_recovery_allowed(self) -> None:
        """Role-specific endpoint constraints are checked even during
        onload."""
        checker = getattr(type(self.adapter), "_check_recovery_allowed", None)
        if checker is not None and self.adapter is not self:
            checker(self.adapter)

    def _recover_engines(self) -> set:
        """Rebuild engines whose slot is None. Returns the ranks rebuilt.

        ``_init_engines`` already skips non-None slots, so it rebuilds exactly
        the holes, reusing the same placement-group bundles (or creating fresh
        dedicated ones) and probing fresh ports (surviving engines' ports are
        bound, so they're skipped).
        """
        incomplete = [
            replica
            for replica in self.replica_specs
            if any(self.all_engines[rank] is None for rank in replica.node_ranks)
        ]
        self._retire_engines([replica.node_ranks[0] for replica in incomplete])
        dead = [rank for replica in incomplete for rank in replica.node_ranks]
        if not dead:
            return set()

        logger.info(f"{self._log_prefix} recovering {len(dead)} engine(s): ranks={dead}")
        try:
            self._init_engines(dead)
        except Exception as exc:
            logger.exception(f"{self._log_prefix} engine rebuild failed for ranks={dead}: {exc}")

        rebuilt = {i for i in dead if self.all_engines[i] is not None}
        still_dead = [i for i in dead if i not in rebuilt]
        if still_dead:
            # Degrade rather than escalate: running on N-1 engines beats a
            # global restart. Only a total wipeout is unrecoverable here.
            if all(engine is None for engine in self.all_engines):
                raise RuntimeError(f"All engines are dead and could not be rebuilt (ranks={still_dead})")
            logger.error(
                f"{self._log_prefix} engines still dead after recovery, continuing degraded: ranks={still_dead}"
            )
        if rebuilt:
            logger.info(f"{self._log_prefix} recovered engine ranks={sorted(rebuilt)}")
        return rebuilt

    def is_onloaded(self) -> bool:
        return self._onloaded

    def shutdown(self) -> None:
        self._memory_ready = False
        self.inference_manager.shutdown(self.inference_model_id)

    def _shutdown_engines(self) -> None:
        """Tear down every engine and remove any placement group this manager
        created for it."""
        for rank in range(len(self.all_engines)):
            engine = self.all_engines[rank]
            if engine is not None:
                try:
                    ray.get(engine.shutdown.remote(), timeout=_ENGINE_SHUTDOWN_TIMEOUT_S)
                except Exception as exc:
                    logger.warning(f"{self._log_prefix} engine rank={rank} shutdown failed (killing anyway): {exc}")
                try:
                    ray.kill(engine)
                except Exception as exc:
                    logger.warning(f"{self._log_prefix} engine rank={rank} ray.kill failed: {exc}")
                self.all_engines[rank] = None
            self._remove_owned_pg(rank)
        logger.info(f"{self._log_prefix} shutdown complete.")
