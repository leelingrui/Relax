# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""GenRM models for the Generative Reward Model service.

Every configured instance, including the single-instance ``__default__``
config, is one static-weight model of the GenRM role. Its engines start through
the shared :func:`~relax.distributed.ray.rollout.start_servers` path inside
the task's inference manager.
"""

import copy
from typing import Any

from relax.distributed.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from relax.engine.inference.config import EngineGroupConfig, ModelConfig
from relax.engine.inference.phase_plans import PHASE_GENRM, placement_phase
from relax.engine.inference.types import WeightSource


# Each instance probes ports from its own window, so instances starting side
# by side never race for the same free port (probe-then-bind).
_GENRM_PORT_BASE = 16000
_GENRM_PORT_WINDOW_SIZE = 1000


def genrm_engine_env(args: Any) -> dict[str, str]:
    env_vars = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST} | {
        "SGL_JIT_DEEPGEMM_PRECOMPILE": "false",
        "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false",
        # See rollout.py: recent SGLang reads SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK
        # (default True) and the deprecation shim value-copies SGL_DISABLE_* into it,
        # so the old DISABLE vars re-enable the check. Set ENABLE=false directly.
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false",
        "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
        "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
        "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
        "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
        # NOTE: disable custom all-reduce-v2, same as rollout.py — avoids
        # custom_all_reduce.cuh:37: CUDA error: invalid argument during CUDA graph capture.
        "SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2": "0",
    }
    if getattr(args, "fp16", False):
        env_vars["SGLANG_MAMBA_CONV_DTYPE"] = "float16"
    return env_vars


def genrm_role_models(args: Any, pg: Any) -> list[tuple[ModelConfig, Any, dict[str, Any]]]:
    """Describe every GenRM instance for ``InferenceManager.create_role``.

    Under sync colocate the instances sit behind the rollout region of the
    shared placement group, one after another; a GenRM that shares the rollout
    bundles starts at bundle 0 and is only legal when it defers.
    """
    region_offset = (
        0 if args.fully_async or getattr(args, "_genrm_colocate_with_rollout", False) else args.rollout_num_gpus
    )
    models = []
    bundle_offset = 0
    for index, (key, spec) in enumerate(args._genrm_instances_resolved.items()):
        engine_args = copy.copy(args)
        engine_args.genrm_model_path = spec["model_path"]
        engine_args.genrm_num_gpus_per_engine = spec["num_gpus_per_engine"]
        engine_args.genrm_engine_config = spec["engine_config"]
        sampling_config = spec["sampling_config"] or {}
        config = ModelConfig(
            key,
            spec["model_path"],
            engine_groups=[
                EngineGroupConfig(
                    "regular", spec["num_gpus"], spec["num_gpus_per_engine"], dict(spec["engine_config"] or {})
                )
            ],
            weight_source=WeightSource.STATIC,
            # The judge's request defaults live on its model, so the GenRM
            # service reads them from the manager instead of global arguments.
            sampling_defaults={
                "temperature": sampling_config.get("temperature", 0.2),
                "top_p": sampling_config.get("top_p", 1.0),
                "top_k": sampling_config.get("top_k", -1),
                "max_new_tokens": sampling_config.get("max_response_len", 1024),
            },
            chat_template_kwargs=dict(sampling_config.get("chat_template_kwargs") or {}),
            env_vars=genrm_engine_env(args),
            fault_tolerance_enabled=True,
        ).resolved(engine_args)
        placement = {
            "pg": pg,
            "bundle_offset": region_offset + bundle_offset,
            "phase": placement_phase(args, PHASE_GENRM),
            "base_port": _GENRM_PORT_BASE + index * _GENRM_PORT_WINDOW_SIZE,
            "ray_num_gpus": getattr(args, "genrm_ray_num_gpus", 0.2),
        }
        models.append((config, engine_args, placement))
        bundle_offset += spec["num_gpus"]
    return models
