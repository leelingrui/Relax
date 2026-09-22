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
    from relax.engine.inference.manager import InferenceManager
    from relax.engine.inference.placement import PlacementGroupView, PlacementOwner, PlacementPlanner
    from relax.engine.inference.types import Role

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
        if state.fail:
            group.start_engines.side_effect = RuntimeError("engine bring-up failed")
        else:
            group.start_engines.return_value = ([], {})
        state.groups.append(group)
        return group

    monkeypatch.setattr(module, "EngineGroup", _engine_group)
    return state


def _pg():
    return (MagicMock(), [0, 1, 2, 3], [0, 1, 2, 3])


def _view(pg):
    return PlacementGroupView(tuple(pg[1]), tuple(pg[2]), PlacementOwner.CONTROLLER, identity=pg[0])


def test_rollout_startup_records_its_layout_in_the_supplied_ledger(startup):
    ledger = PlacementPlanner()
    pg = _pg()

    module.start_rollout_servers(
        _args(),
        pg,
        inference_manager=InferenceManager(Role.ROLLOUT),
        placement_manager_handle=ledger,
    )

    (recorded,) = ledger.allocations(_view(pg))
    assert recorded.group_id == "default/group-0"
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
            inference_manager=InferenceManager(Role.ROLLOUT),
            placement_manager_handle=ledger,
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
            inference_manager=InferenceManager(Role.ROLLOUT),
            placement_manager_handle=ledger,
        )

    # Releasing a borrowed group never authorizes destroying it.
    assert ledger.release(_view(pg)).remove_placement_group is False


def test_rollout_manager_uses_one_ledger_on_the_compatibility_path():
    from conftest import create_test_manager

    manager = create_test_manager()

    # No task owner injected: the manager keeps a single planner of its own
    # rather than a fresh ledger per call.
    ledger = manager._placement_ledger
    assert isinstance(ledger, PlacementPlanner)
    assert manager._placement_ledger is ledger


def test_rollout_manager_prefers_the_task_owner_ledger():
    from conftest import create_test_manager

    manager = create_test_manager()
    owner = object()
    manager.task_inference_manager = owner

    assert manager._placement_ledger is owner
