# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""A replica identity is issued once and never recomputed from a position.

Discovery publishes ``{model_id}/replica-{slot}``. That string is what a caller
correlates across weight versions, health transitions and scale events, so it
has to survive the group moving inside the server: scale-in removes a group and
shifts every ``rank_offset`` after it.
"""

from types import SimpleNamespace

import pytest


try:
    from conftest import make_engine_group, make_mock_args, make_mock_engine

    from relax.engine.inference.config import EngineGroupConfig, ModelConfig

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="Missing ray/sglang dependencies")


def _resolved(name: str, groups: list[EngineGroupConfig], gpus_per_node: int = 8) -> ModelConfig:
    args = SimpleNamespace(num_gpus_per_node=gpus_per_node, rollout_num_gpus_per_engine=groups[0].num_gpus_per_engine)
    return ModelConfig(name, "/ckpt", engine_groups=groups).resolved(args)


def _group(num_gpus: int, per_engine: int = 2, worker_type: str = "regular") -> EngineGroupConfig:
    return EngineGroupConfig(worker_type=worker_type, num_gpus=num_gpus, num_gpus_per_engine=per_engine)


# ======================== spec-side identity ===============================


def test_replicas_are_named_after_the_model_not_the_group():
    """Discovery's format is the only one; the group id stays separate."""
    model = _resolved("default", [_group(4)])
    assert model.engine_groups[0].topology.group_id == "default/group-0"
    assert [r.replica_id for r in model.engine_groups[0].topology.replicas] == [
        "default/replica-0",
        "default/replica-1",
    ]


def test_replica_numbering_continues_across_engine_groups():
    """Two groups of a model must not both start numbering at zero."""
    model = _resolved("default", [_group(4), _group(4)])
    identities = [r.replica_id for group in model.engine_groups for r in group.topology.replicas]
    assert identities == [
        "default/replica-0",
        "default/replica-1",
        "default/replica-2",
        "default/replica-3",
    ]
    assert len(set(identities)) == len(identities)


def test_a_placeholder_group_names_nothing_but_keeps_its_slot_range():
    """The runtime advances its engine offset over a placeholder too.

    Skipping the range here would renumber every later group relative to the
    ``rank_offset`` the engines actually run at.
    """
    model = _resolved("default", [_group(4, worker_type="placeholder"), _group(4)])
    assert model.engine_groups[0].topology.replicas == ()
    assert [r.replica_id for r in model.engine_groups[1].topology.replicas] == [
        "default/replica-2",
        "default/replica-3",
    ]


def test_a_multi_node_replica_is_named_once_for_all_its_nodes():
    """The identity belongs to the logical replica, not to each node actor."""
    model = _resolved("vl", [_group(32, per_engine=16)], gpus_per_node=8)
    replicas = model.engine_groups[0].topology.replicas
    assert [r.replica_id for r in replicas] == ["vl/replica-0", "vl/replica-2"]
    assert [r.node_ranks for r in replicas] == [(0, 1), (2, 3)]


# ======================== runtime-side identity ============================


def test_the_group_reports_the_identity_its_spec_was_created_with():
    model = _resolved("default", [_group(4)])
    group = make_engine_group(engines=[make_mock_engine(), make_mock_engine()])
    group.spec = model.engine_groups[0].topology

    assert group.replica_identity(0) == "default/replica-0"
    assert group.replica_identity(1) == "default/replica-1"


def test_an_identity_does_not_move_when_scale_in_shifts_the_group():
    """A group created at slot 2 keeps its names after an earlier group goes.

    Deriving the name from ``rank_offset`` at read time would renumber every
    surviving replica the moment a preceding group is removed.
    """
    group = make_engine_group(engines=[make_mock_engine(), make_mock_engine()], rank_offset=2)
    before = [group.replica_identity(slot) for slot in range(2)]
    assert before == ["default/replica-2", "default/replica-3"]

    group.rank_offset = 0  # what scale-in does to the groups behind the removed one
    assert [group.replica_identity(slot) for slot in range(2)] == before


def test_a_scaled_out_group_is_named_for_the_model_it_joins():
    group = make_engine_group(engines=[make_mock_engine()], rank_offset=4, is_scaled_out=True)
    group.model_id = "judge"
    group.spec = None
    group.__post_init__()

    assert group.replica_identity(0) == "judge/replica-4"


def test_an_engine_beyond_the_spec_still_gets_a_distinct_name():
    """Appending an engine without a respec must not reuse a recorded name."""
    group = make_engine_group(engines=[make_mock_engine(), make_mock_engine()], rank_offset=0)
    recorded = {group.replica_identity(slot) for slot in range(2)}
    group.all_engines.append(make_mock_engine())

    assert group.replica_identity(2) not in recorded


def test_default_args_produce_one_engine_per_replica():
    """Guard the fixture assumption the identity tests above rely on."""
    args = make_mock_args()
    group = make_engine_group(args=args, engines=[make_mock_engine()])
    assert group.nodes_per_engine == 1
