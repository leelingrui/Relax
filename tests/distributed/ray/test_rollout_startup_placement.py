# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Rollout startup must not keep reserved GPU slices when it fails.

The initial layout is reserved in the task owner's ledger before any engine
exists, so a failed bring-up has to give the slices back while leaving the
Controller-owned placement group alone.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


try:
    from relax.distributed.ray import rollout as module
    from relax.engine.inference.config import EngineGroupConfig, ModelConfig
    from relax.engine.inference.placement import PlacementGroupView, PlacementOwner, PlacementPlanner
    from relax.engine.inference.types import Role, WeightSource

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="Missing ray/sglang dependencies")


def _args():
    return SimpleNamespace(
        rollout_num_gpus=4,
        rollout_num_gpus_per_engine=2,
        num_gpus_per_node=4,
        sglang_config=None,
        prefill_num_servers=None,
        rollout_engine_init_timeout=1.0,
        sglang_hf_checkpoint=None,
        hf_checkpoint="/tmp/checkpoint",
        sglang_router_ip=None,
        sglang_router_port=None,
        debug_rollout_only=False,
    )


@pytest.fixture
def startup(monkeypatch):
    state = SimpleNamespace(groups=[], fail=False)
    monkeypatch.setattr(module, "_start_router", lambda *a, **k: ("10.0.0.1", 3000))
    monkeypatch.setattr(module, "_wait_engine_init_with_progress", lambda *a, **k: None)

    def _engine_group(**kwargs):
        group = MagicMock()
        group.placement = kwargs["placement"]
        group.pg_owner = kwargs["pg_owner"]
        group.kwargs = kwargs
        if state.fail:
            group.start_engines.side_effect = RuntimeError("engine bring-up failed")
        else:
            group.start_engines.return_value = ([], {})
        state.groups.append(group)
        return group

    monkeypatch.setattr(module, "EngineGroup", _engine_group)
    return state


def _pg(num_gpus=4):
    return (MagicMock(), list(range(num_gpus)), list(range(num_gpus)))


def _view(pg):
    return PlacementGroupView(tuple(pg[1]), tuple(pg[2]), PlacementOwner.CONTROLLER, identity=pg[0])


def test_rollout_startup_records_its_layout_in_the_supplied_ledger(startup):
    ledger = PlacementPlanner()
    pg = _pg()

    module.start_rollout_servers(
        _args(),
        pg,
        planner=ledger,
    )

    (recorded,) = ledger.allocations(_view(pg))
    assert recorded.group_id == "rollout/default/group-0"
    assert recorded.reserved_offset == 0
    assert recorded.referenced_offsets == (0, 2)
    assert startup.groups[0].placement == recorded


def test_rollout_startup_failure_releases_the_reserved_slices(startup):
    startup.fail = True
    ledger = PlacementPlanner()
    pg = _pg()

    with pytest.raises(RuntimeError, match="engine bring-up failed"):
        module.start_rollout_servers(
            _args(),
            pg,
            planner=ledger,
        )

    assert ledger.allocations(_view(pg)) == ()


def test_rollout_startup_failure_keeps_the_controller_owned_group(startup):
    startup.fail = True
    ledger = PlacementPlanner()
    pg = _pg()

    with pytest.raises(RuntimeError):
        module.start_rollout_servers(
            _args(),
            pg,
            planner=ledger,
        )

    # Releasing a borrowed group never authorizes destroying it.
    assert ledger.release(_view(pg)).remove_placement_group is False


def _start_teacher(args, pg, ledger, *, bundle_offset, phase):
    teacher = ModelConfig(
        "default",
        "/tmp/teacher",
        engine_groups=[EngineGroupConfig("regular", 4, 2, {})],
        weight_source=WeightSource.STATIC,
    ).resolved(args)
    return module.start_servers(
        args, [teacher], planner=ledger, role=Role.TEACHER, pg=pg, bundle_offset=bundle_offset, phase=phase
    )


def test_rollout_startup_split_teacher_keeps_rollout_at_the_front(startup):
    args = _args()
    ledger = PlacementPlanner()
    pg = _pg(8)

    # A split teacher sits behind the rollout region and is recorded first.
    _start_teacher(args, pg, ledger, bundle_offset=4, phase=module.PHASE_GENERATE)
    module.start_rollout_servers(args, pg, planner=ledger)

    offsets = {item.group_id: item.reserved_offset for item in ledger.allocations(_view(pg))}
    assert offsets == {"teacher/default/group-0": 4, "rollout/default/group-0": 0}


def test_rollout_startup_shared_teacher_with_same_model_name_does_not_conflict(startup):
    args = _args()
    ledger = PlacementPlanner()
    pg = _pg(4)

    _start_teacher(args, pg, ledger, bundle_offset=0, phase="teacher")
    module.start_rollout_servers(args, pg, planner=ledger)

    assert {item.group_id for item in ledger.allocations(_view(pg))} == {
        "teacher/default/group-0",
        "rollout/default/group-0",
    }


def test_rollout_startup_debug_rollout_only_registers_engines_at_start(startup):
    args = _args()
    args.debug_rollout_only = True

    (server,) = module.start_rollout_servers(args, _pg(), planner=PlacementPlanner()).values()

    # No weight sync ever runs, so the engines must not wait for one.
    assert startup.groups[0].kwargs["skip_router_registration"] is False
    assert server.model_spec.needs_weight_update is False


def test_rollout_startup_planning_failure_removes_the_group_it_created(startup, monkeypatch):
    import importlib

    import relax.core.service as service

    # ``ray.util.placement_group`` the attribute is the function, not the module.
    ray_pg = importlib.import_module("ray.util.placement_group")

    pg = _pg()
    removed = []
    monkeypatch.setattr(service, "create_placement_group", lambda **kwargs: pg)
    monkeypatch.setattr(ray_pg, "remove_placement_group", removed.append)
    ledger = MagicMock()
    ledger.plan.side_effect = ValueError("bad layout")

    with pytest.raises(ValueError, match="bad layout"):
        _start_teacher(_args(), None, ledger, bundle_offset=0, phase="teacher")

    assert removed == [pg[0]] and startup.groups == []


def test_manager_create_role_checks_every_model_before_starting_any(monkeypatch):
    from relax.distributed.ray.inference_manager import InferenceManager

    started = []
    monkeypatch.setattr(module, "start_servers", lambda *a, **k: started.append(a) or {})
    args = _args()
    pg = _pg()

    def teacher(name, offset):
        config = ModelConfig(
            name,
            "/tmp/teacher",
            engine_groups=[EngineGroupConfig("regular", 2, 2, {})],
            weight_source=WeightSource.STATIC,
        ).resolved(args)
        return config, args, {"pg": pg, "bundle_offset": offset, "phase": "teacher", "base_port": 26000}

    manager = InferenceManager()
    # The second teacher overlaps the first one in the same phase.
    with pytest.raises(ValueError, match="overlap"):
        manager.create_role(Role.TEACHER, [teacher("a", 0), teacher("b", 1)])

    assert started == [] and manager.allocations() == ()


def _task_args(*, rollout_gpus, genrm_gpus, actor_gpus=4, **overrides):
    spec = {
        "model_path": "/tmp/judge",
        "num_gpus": genrm_gpus,
        "num_gpus_per_engine": 1,
        "engine_config": None,
        "sampling_config": None,
    }
    args = _args()
    args.__dict__.update(
        rollout_num_gpus=rollout_gpus,
        rollout_num_gpus_per_engine=1,
        colocate=True,
        hybrid=False,
        fully_async=False,
        resource={"actor": [1, actor_gpus], "rollout": [1, rollout_gpus], "genrm": [1, genrm_gpus]},
        _genrm_instances_resolved={"judge": spec},
        **overrides,
    )
    return args


def test_task_layout_accepts_split_rollout_and_genrm():
    from relax.distributed.ray.inference_manager import validate_task_layout

    validate_task_layout(_task_args(rollout_gpus=2, genrm_gpus=2))


def test_task_layout_rejects_a_later_role_before_any_engine_starts(monkeypatch):
    from relax.distributed.ray.inference_manager import validate_task_layout

    monkeypatch.setattr(module, "start_servers", MagicMock(side_effect=AssertionError("started an engine")))
    # GenRM sits behind four rollout bundles of a four-GPU actor group.
    with pytest.raises(ValueError, match="Invalid genrm placement"):
        validate_task_layout(_task_args(rollout_gpus=4, genrm_gpus=2))
