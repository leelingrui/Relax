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
