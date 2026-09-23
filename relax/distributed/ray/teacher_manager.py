# Copyright (c) 2026 Relax Authors. All Rights Reserved.


import copy

import ray

from relax.backends.sglang.sglang_engine import SGLangEngine
from relax.core.service import create_placement_group
from relax.distributed.ray.multi_engine_manager import MultiEngineManager
from relax.distributed.ray.placement_ledger import plan_placement, release_placement
from relax.distributed.ray.rollout import _allocate_rollout_engine_addr_and_ports_normal
from relax.distributed.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from relax.engine.inference.capabilities import WeightSource
from relax.engine.inference.placement import (
    PlacementGroupView,
    PlacementOwner,
    PlacementRelease,
    PlacementRequest,
    PlacementSlice,
)
from relax.engine.inference.types import Role
from relax.utils.env import Envs
from relax.utils.http_utils import find_available_port
from relax.utils.logging_utils import get_logger
from relax.utils.opd.opd_utils import build_teacher_engine_args, build_teacher_overrides


logger = get_logger(__name__)


def _build_teacher_engine_env(args) -> dict[str, str]:
    env_vars = dict.fromkeys(NOSET_VISIBLE_DEVICES_ENV_VARS_LIST, "1") | {
        # OPD patches default off; enabled only when the corresponding env flag is
        # passed through from the driver. RELAX_OPD_PREEXPANDED_PATCH affects the
        # teacher engine only; RELAX_OPD_PER_POS_TOKEN_IDS affects teacher + student.
        "RELAX_OPD_PREEXPANDED_PATCH": str(int(Envs.RELAX_OPD_PREEXPANDED_PATCH)),
        "RELAX_OPD_PER_POS_TOKEN_IDS": str(int(Envs.RELAX_OPD_PER_POS_TOKEN_IDS)),
        "RELAX_OPD_TOKEN_IDS_LOGPROB_K": Envs.RELAX_OPD_TOKEN_IDS_LOGPROB_K,
        "SGL_JIT_DEEPGEMM_PRECOMPILE": "false",
        "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false",
        "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
        "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
        "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
        "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
        "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
        "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
    }
    if getattr(args, "fp16", False):
        env_vars["SGLANG_MAMBA_CONV_DTYPE"] = "float16"
    return env_vars


class TeacherEngineAdapter:
    """Teacher placement, ports, environment and endpoint recovery policy."""

    inference_role = Role.TEACHER
    inference_weight_source = WeightSource.CHECKPOINT

    def __init__(
        self,
        args,
        num_replicas: int,
        gpus_per_replica: int,
        pg: tuple | None = None,
        shared_pg: bool = False,
        bundle_offset: int = 0,
        *,
        inference_manager,
        placement_manager_handle,
        model_id: str = "default",
        defer_init: bool = False,
    ) -> None:
        self.args = args
        assert num_replicas >= 1, f"num_replicas must be >= 1, got {num_replicas}."
        assert gpus_per_replica > 0, f"gpus_per_replica must be > 0, got {gpus_per_replica}."
        if gpus_per_replica > args.num_gpus_per_node and gpus_per_replica % args.num_gpus_per_node:
            raise ValueError("Multi-node teacher replicas must occupy complete nodes")
        nodes_per_engine = max(1, gpus_per_replica // args.num_gpus_per_node)
        self.nodes_per_engine = nodes_per_engine
        if shared_pg:
            assert pg is not None, "shared_pg=True requires the full actor/rollout placement group."
            _pg, bundle_indices, gpu_ids = pg
            required = int(args.rollout_num_gpus) + bundle_offset + gpus_per_replica * num_replicas
            assert len(bundle_indices) >= required and len(gpu_ids) >= required, (
                f"shared teacher PG too small: bundles={len(bundle_indices)}, "
                f"gpu_ids={len(gpu_ids)}, required={required} (rollout_num_gpus={args.rollout_num_gpus} + "
                f"bundle_offset={bundle_offset} + gpus_per_replica={gpus_per_replica} * num_replicas={num_replicas})."
            )

        self.gpus_per_replica = gpus_per_replica
        self.num_replicas = num_replicas
        self._shared_pg = shared_pg
        self._shared_pg_tuple = pg
        self._bundle_offset = bundle_offset
        # The task owner's planner is the one ledger every role records in.
        self._placement_ledger = placement_manager_handle
        # Multiple teachers share one placement group, so the allocation
        # identity has to carry the model, not just the replica.
        self._placement_model_id = model_id

        overrides = build_teacher_overrides(args, colocate_sync=shared_pg)
        self._overrides = overrides
        self.inference_model_path = overrides["model_path"]
        self.inference_num_gpus_per_engine = gpus_per_replica
        self.inference_overrides = overrides
        self._teacher_args = build_teacher_engine_args(args, overrides)
        self._teacher_args.use_slime_router = False
        logger.info(
            f"[OPD teacher] launching {num_replicas} replica(s), "
            f"TP={gpus_per_replica}, model={overrides['model_path']}, "
            f"shared_pg={shared_pg}, mem_fraction_static={overrides.get('mem_fraction_static')}"
        )

        from relax.distributed.ray.rollout import _start_router

        router_args = copy.copy(args)
        router_args.use_slime_router = False
        self.router_ip, self.router_port = _start_router(router_args, force_new=True)
        self.router_url = f"http://{self.router_ip}:{self.router_port}"
        try:
            self.backend = MultiEngineManager(
                args,
                num_slots=num_replicas * nodes_per_engine,
                nodes_per_engine=nodes_per_engine,
                engine_actor_cls=SGLangEngine,
                log_prefix="[OPD teacher]",
                skip_init=True,
                inference_manager=inference_manager,
                model_id=model_id,
                adapter=self,
            )
            if not defer_init:
                self.backend.initialize()
        except Exception:
            if hasattr(self, "backend"):
                self.backend.shutdown()
            raise

    def shutdown(self) -> None:
        # The owner's manager stops the routers once every model has closed.
        self.backend.shutdown()

    def get_urls(self) -> list[str]:
        urls = []
        for engine in self.engines:
            if engine is None:
                continue
            base_url = ray.get(engine.get_url.remote())
            urls.append(f"{base_url}/generate")
        return urls

    def __getattr__(self, name):
        backend = self.__dict__.get("backend")
        if backend is None:
            raise AttributeError(name)
        return getattr(backend, name)

    def recover(self) -> set:
        self._check_recovery_allowed()
        return self.backend.recover()

    def _check_recovery_allowed(self) -> None:
        """Recover in place only when the teacher's endpoint can stay
        stable."""
        dead = [rank for rank, engine in enumerate(self.all_engines) if engine is None]
        if dead and not self._shared_pg:
            # A dedicated replacement PG may land on another node. OPD callers
            # hold URLs captured at startup, so rebuilding here could advertise
            # success while every caller keeps targeting the old host. Escalate
            # to Controller restart, which rebuilds and re-injects the routes.
            raise RuntimeError(
                f"Dedicated OPD teacher engines died at ranks={dead}; global restart is required to refresh URLs."
            )

    # ------------------------------------------------------------------
    # MultiEngineManager hooks.
    # ------------------------------------------------------------------

    def _dedicated_placement_group(self, replica: int) -> tuple:
        """Provision -- or reuse -- the placement group this replica owns.

        Offsets inside the group come from the planner, so this hook resolves
        the resource only and never derives a bundle offset.
        """
        nodes_per_engine = getattr(self, "nodes_per_engine", 1)
        existing = getattr(self, "_engine_placements", {}).get(replica * nodes_per_engine)
        if existing:
            return existing[0]
        return create_placement_group(
            num_gpus=self.gpus_per_replica,
            node_group_affinity=getattr(self.args, "enable_affinity", True),
        )

    def _resolve_planned_placement(self, rank: int):
        nodes_per_engine = getattr(self, "nodes_per_engine", 1)
        replica, node_rank = divmod(rank, nodes_per_engine)
        if self._shared_pg:
            pg_tuple = self._shared_pg_tuple
            owner = PlacementOwner.CONTROLLER
            requests = tuple(
                PlacementRequest(
                    group_id=f"teacher/{self._placement_model_id}/replica-{index}",
                    worker_type="regular",
                    num_gpus=self.gpus_per_replica,
                    num_gpus_per_engine=self.gpus_per_replica,
                    num_gpus_per_node=self.args.num_gpus_per_node,
                    phase="teacher",
                    bundle_offset=int(self.args.rollout_num_gpus)
                    + self._bundle_offset
                    + index * self.gpus_per_replica,
                )
                for index in range(self.num_replicas)
            )
        else:
            pg_tuple = self._dedicated_placement_group(replica)
            owner = PlacementOwner.MANAGER
            requests = (
                PlacementRequest(
                    group_id=f"teacher/{self._placement_model_id}/replica-{replica}",
                    worker_type="regular",
                    num_gpus=self.gpus_per_replica,
                    num_gpus_per_engine=self.gpus_per_replica,
                    num_gpus_per_node=self.args.num_gpus_per_node,
                    phase="teacher",
                    bundle_offset=0,
                ),
            )
        pg_view = PlacementGroupView(tuple(pg_tuple[1]), tuple(pg_tuple[2]), owner, identity=pg_tuple[0])
        planned = plan_placement(self._placement_ledger, requests, pg_view)[replica if self._shared_pg else 0]
        bundle = planned.bundle_indices[node_rank]
        return pg_tuple, self._shared_pg is False, tuple(pg_tuple[1]).index(bundle), planned

    def _release_placement(self, placement: PlacementSlice) -> PlacementRelease:
        return release_placement(self._placement_ledger, placement)

    def _ray_resource_kwargs(self, rank: int) -> dict:
        return {"num_cpus": 0.2, "num_gpus": 0.2}

    def _build_engine_env_vars(self) -> dict[str, str]:
        return _build_teacher_engine_env(self.args)

    def _engine_ctor_args(self, rank: int):
        return self._teacher_args

    def _build_engine_ctor_kwargs(self, rank: int) -> dict:
        return {
            "sglang_overrides": self._overrides,
            "num_gpus_per_engine": self.gpus_per_replica,
            "register_sigterm_handler": False,
            "weight_source": WeightSource.CHECKPOINT,
        }

    def _build_engine_init_kwargs(self, rank: int, addr_and_ports: dict) -> dict:
        # Static weights never join DCS; traffic uses this teacher's own Router.
        return {
            **addr_and_ports,
            "router_ip": self.router_ip,
            "router_port": self.router_port,
            "skip_dcs_registration": True,
            "skip_router_registration": False,
        }

    def _allocate_engine_addr_and_ports(self, *, new_engines: list[tuple]) -> dict[int, dict]:
        addr_and_ports: dict[int, dict] = {}
        pending = []
        for rank, engine in new_engines:
            # OPD consumers receive teacher URLs once during startup. Preserve
            # the original endpoint across recovery instead of silently moving
            # a rebuilt engine to a port those consumers never learn about.
            if self._shared_pg and rank in self._engine_addr_and_ports:
                addr_and_ports[rank] = dict(self._engine_addr_and_ports[rank])
                continue
            pending.append((rank, engine))
        if pending:
            allocated, _ = _allocate_rollout_engine_addr_and_ports_normal(
                args=self._teacher_args,
                rollout_engines=pending,
                worker_type="regular",
                num_gpus_per_engine=self.gpus_per_replica,
                rank_offset=0,
                base_port=find_available_port(15000),
            )
            addr_and_ports.update(allocated)
        return addr_and_ports
