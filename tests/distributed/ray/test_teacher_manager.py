# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _cleanup_teacher_manager_module():
    """Drop the stub-backed import so it cannot leak into other tests.

    Clearing ``sys.modules`` alone is not enough: ``importlib.import_module``
    also binds the submodule on its parent package, and ``from package import
    submodule`` prefers that attribute over a fresh import. A later test would
    then patch the stub-backed module while the code under test re-imports the
    real one.
    """
    import relax.distributed.ray as ray_pkg

    name = "relax.distributed.ray.teacher_manager"
    original = sys.modules.get(name)
    original_attr = getattr(ray_pkg, "teacher_manager", None)
    yield
    if original is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = original
    if original_attr is None:
        if hasattr(ray_pkg, "teacher_manager"):
            delattr(ray_pkg, "teacher_manager")
    else:
        ray_pkg.teacher_manager = original_attr


def _import_teacher_manager(monkeypatch):
    sys.modules.pop("relax.distributed.ray.teacher_manager", None)
    return importlib.import_module("relax.distributed.ray.teacher_manager")


def test_teacher_env_matches_rollout_genrm_stability_envs(monkeypatch):
    teacher_manager = _import_teacher_manager(monkeypatch)
    # RELAX_OPD_PREEXPANDED_PATCH is passed through from the driver env (default
    # "0"); set it so the test verifies the pass-through, not the default value.
    monkeypatch.setenv("RELAX_OPD_PREEXPANDED_PATCH", "1")
    args = SimpleNamespace(fp16=True)

    env = teacher_manager._build_teacher_engine_env(args)

    assert env["RELAX_OPD_PREEXPANDED_PATCH"] == "1"
    assert env["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] == "false"
    assert env["SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK"] == "true"
    assert env["SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK"] == "true"
    assert env["SGLANG_MEMORY_SAVER_CUDA_GRAPH"] == "true"
    assert env["SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT"] == "true"
    assert env["SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION"] == "false"
    assert env["SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE"] == "false"
    assert env["SGLANG_MAMBA_CONV_DTYPE"] == "float16"
