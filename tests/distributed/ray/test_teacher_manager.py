# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

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


def _install_teacher_manager_stubs(monkeypatch):
    sglang_engine = ModuleType("relax.backends.sglang.sglang_engine")
    sglang_engine.SGLangEngine = object

    service = ModuleType("relax.core.service")
    service.create_placement_group = MagicMock()

    rollout = ModuleType("relax.distributed.ray.rollout")
    rollout._allocate_rollout_engine_addr_and_ports_normal = MagicMock()
    rollout._start_router = MagicMock(return_value=("teacher-router", 3100))
    rollout.stop_launched_routers = MagicMock()

    ray_utils = ModuleType("relax.distributed.ray.utils")
    ray_utils.NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = []

    http_utils = ModuleType("relax.utils.http_utils")
    http_utils.find_available_port = MagicMock(return_value=15000)

    monkeypatch.setitem(sys.modules, "relax.backends.sglang.sglang_engine", sglang_engine)
    monkeypatch.setitem(sys.modules, "relax.core.service", service)
    monkeypatch.setitem(sys.modules, "relax.distributed.ray.rollout", rollout)
    monkeypatch.setitem(sys.modules, "relax.distributed.ray.utils", ray_utils)
    monkeypatch.setitem(sys.modules, "relax.utils.http_utils", http_utils)


def _import_teacher_manager(monkeypatch):
    _install_teacher_manager_stubs(monkeypatch)
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


def test_teacher_recovery_reuses_original_endpoint(monkeypatch):
    teacher_manager = _import_teacher_manager(monkeypatch)
    manager_cls = teacher_manager.TeacherEngineAdapter
    manager = object.__new__(manager_cls)
    manager._shared_pg = True
    original = {
        "host": "192.0.2.1",
        "port": 15001,
        "nccl_port": 15002,
        "dist_init_addr": "192.0.2.1:15003",
    }
    manager._engine_addr_and_ports = {0: original}

    result = manager._allocate_engine_addr_and_ports(new_engines=[(0, object())])

    assert result == {0: original}
    assert result[0] is not original
    sys.modules["relax.utils.http_utils"].find_available_port.assert_not_called()
    sys.modules["relax.distributed.ray.rollout"]._allocate_rollout_engine_addr_and_ports_normal.assert_not_called()


def test_dedicated_teacher_recovery_requires_global_restart(monkeypatch):
    teacher_manager = _import_teacher_manager(monkeypatch)
    manager_cls = teacher_manager.TeacherEngineAdapter
    manager = object.__new__(manager_cls)
    manager._shared_pg = False
    manager.all_engines = [None]

    with pytest.raises(RuntimeError, match="global restart"):
        manager.recover()


def test_dedicated_teacher_placement_creates_and_records_its_own_group(monkeypatch):
    from relax.engine.inference.placement import PlacementOwner, PlacementPlanner

    module = _import_teacher_manager(monkeypatch)
    manager = object.__new__(module.TeacherEngineAdapter)
    manager.args = SimpleNamespace(rollout_num_gpus=4, num_gpus_per_node=4, enable_affinity=False)
    manager.gpus_per_replica = 2
    manager.num_replicas = 2
    manager.nodes_per_engine = 1
    manager._shared_pg = False
    manager._engine_placements = {}
    manager._placement_ledger = PlacementPlanner()
    manager._placement_model_id = "math-teacher"
    dedicated = ("dedicated", [3, 1], [1, 0])
    module.create_placement_group.return_value = dedicated

    placement, owns_pg, gpu_index, planned = manager._resolve_planned_placement(rank=1)

    assert placement is dedicated
    assert owns_pg is True
    assert gpu_index == 0
    assert planned.owner is PlacementOwner.MANAGER
    assert planned.group_id == "teacher/math-teacher/replica-1"
    module.create_placement_group.assert_called_once_with(num_gpus=2, node_group_affinity=False)


def test_teacher_placement_uses_planner_slice_for_shared_pg(monkeypatch):
    from relax.engine.inference.placement import PlacementOwner, PlacementPlanner

    module = _import_teacher_manager(monkeypatch)
    manager = object.__new__(module.TeacherEngineAdapter)
    manager.args = SimpleNamespace(rollout_num_gpus=4, num_gpus_per_node=4, enable_affinity=False)
    manager.gpus_per_replica = 2
    manager.num_replicas = 2
    manager.nodes_per_engine = 1
    manager._shared_pg = True
    manager._bundle_offset = 2
    manager._shared_pg_tuple = ("shared", list(range(12)), list(range(12)))
    manager._placement_ledger = PlacementPlanner()
    manager._placement_model_id = "math-teacher"

    placement, owns_pg, gpu_index, planned = manager._resolve_planned_placement(rank=1)

    assert placement is manager._shared_pg_tuple
    assert owns_pg is False
    assert gpu_index == 8
    assert planned.reserved_offset == 8
    assert planned.owner is PlacementOwner.CONTROLLER
    # Teachers share one placement group, so the slice identity carries the
    # model and not just the replica index.
    assert planned.group_id == "teacher/math-teacher/replica-1"


def test_teacher_init_uses_own_router_without_dcs(monkeypatch):
    module = _import_teacher_manager(monkeypatch)
    cls = module.TeacherEngineAdapter
    manager = object.__new__(cls)
    manager.router_ip = "teacher-router"
    manager.router_port = 3100
    addr = {"host": "teacher.test", "port": 15000, "nccl_port": 15001, "dist_init_addr": "teacher.test:15002"}
    result = manager._build_engine_init_kwargs(0, addr)
    assert result == dict(
        addr, router_ip="teacher-router", router_port=3100, skip_dcs_registration=True, skip_router_registration=False
    )
    assert "skip_dcs_registration" not in addr


def test_teacher_constructor_declares_checkpoint_weight_source(monkeypatch):
    module = _import_teacher_manager(monkeypatch)
    cls = module.TeacherEngineAdapter
    manager = object.__new__(cls)
    manager._overrides = {}
    manager.gpus_per_replica = 2
    assert manager._build_engine_ctor_kwargs(0)["weight_source"] == "checkpoint"


def test_teacher_shutdown_leaves_router_cleanup_to_the_owner(monkeypatch):
    """The owner's manager stops the routers once every model has closed."""
    module = _import_teacher_manager(monkeypatch)
    manager = object.__new__(module.TeacherEngineAdapter)
    manager.backend = object.__new__(module.MultiEngineManager)
    monkeypatch.setattr(module.MultiEngineManager, "shutdown", MagicMock(side_effect=RuntimeError("cleanup")))
    with pytest.raises(RuntimeError, match="cleanup"):
        manager.shutdown()
    sys.modules["relax.distributed.ray.rollout"].stop_launched_routers.assert_not_called()
