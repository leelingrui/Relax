# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Managed OPD teacher models.

Each teacher is one static-weight model of the Teacher role; its engines start
through the shared :func:`~relax.distributed.ray.rollout.start_servers` path
inside the task's inference manager, either in the actor placement group
(colocate) or in a placement group of its own.
"""

from typing import Any

from relax.distributed.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from relax.engine.inference.config import EngineGroupConfig, ModelConfig
from relax.engine.inference.phase_plans import PHASE_TEACHER, placement_phase
from relax.engine.inference.types import WeightSource
from relax.utils.env import Envs
from relax.utils.opd.opd_utils import build_teacher_engine_args, build_teacher_overrides, teacher_region_offset


# Teachers probe ports from their own windows, apart from rollout and GenRM.
_TEACHER_PORT_BASE = 26000
_TEACHER_PORT_WINDOW_SIZE = 500


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


def teacher_role_model(
    args: Any,
    *,
    model_id: str,
    num_gpus: int,
    gpus_per_replica: int,
    pg: Any = None,
    bundle_offset: int = 0,
    index: int = 0,
) -> tuple[ModelConfig, Any, dict[str, Any]]:
    """Describe one teacher for ``InferenceManager.create_role``.

    With ``pg`` the teacher sits in the shared actor placement group at
    ``bundle_offset`` within the teacher region; without it the manager creates
    a placement group for this teacher alone.
    """
    if gpus_per_replica > args.num_gpus_per_node and gpus_per_replica % args.num_gpus_per_node:
        raise ValueError("Multi-node teacher replicas must occupy complete nodes")
    overrides = build_teacher_overrides(args, colocate_sync=pg is not None)
    engine_args = build_teacher_engine_args(args, overrides)
    engine_args.use_slime_router = False
    config = ModelConfig(
        model_id,
        overrides["model_path"],
        engine_groups=[EngineGroupConfig("regular", num_gpus, gpus_per_replica, dict(overrides))],
        weight_source=WeightSource.STATIC,
        env_vars=_build_teacher_engine_env(args),
        fault_tolerance_enabled=True,
    ).resolved(engine_args)
    placement = {
        "pg": pg,
        "bundle_offset": (teacher_region_offset(args) + bundle_offset) if pg is not None else 0,
        "phase": placement_phase(args, PHASE_TEACHER),
        "base_port": _TEACHER_PORT_BASE + index * _TEACHER_PORT_WINDOW_SIZE,
    }
    return config, engine_args, placement
