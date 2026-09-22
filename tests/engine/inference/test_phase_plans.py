# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Phase plans come from the flags plus the placement ledger, not from
guesswork.

The property under test is that a layout is only sequenced when it has to be: a
GenRM that shares the rollout bundles while staying resident must not be
serialized against generation, while a deferred scorer must be sequenced
whether or not it shares a slice.
"""

from types import SimpleNamespace

import pytest

from relax.engine.inference.phase_plans import (
    PHASE_GENERATE,
    PHASE_GENRM,
    PHASE_TEACHER,
    deferred_phases,
    phase_plans_from_contentions,
    phase_targets_from_args,
)
from relax.engine.inference.placement import (
    PlacementGroupView,
    PlacementOwner,
    PlacementPlanner,
    PlacementRequest,
)
from relax.engine.inference.types import ModelRef, Role


def build_args(**overrides):
    args = SimpleNamespace(
        colocate=True,
        hybrid=False,
        fully_async=False,
        rollout_num_gpus=4,
        sglang_config=None,
        prefill_num_servers=None,
        _genrm_instances_resolved={},
        opd_teacher_routes=None,
        teacher_hf_checkpoint=None,
        resource={},
        defer_reward_to_post_process=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def shared_slice_contentions(*phases: str):
    """Plan two phases onto the same bundles and report the real contention."""
    planner = PlacementPlanner()
    view = PlacementGroupView(tuple(range(4)), tuple(range(4)), PlacementOwner.CONTROLLER, identity="pg")
    for phase in phases:
        planner.plan(
            (
                PlacementRequest(
                    group_id=f"{phase}/model",
                    worker_type="regular",
                    num_gpus=4,
                    num_gpus_per_engine=4,
                    num_gpus_per_node=4,
                    phase=phase,
                    bundle_offset=0,
                ),
            ),
            view,
        )
    return planner.contended_phases()


def test_resident_genrm_on_shared_bundles_is_not_sequenced():
    args = build_args(_genrm_instances_resolved={"__default__": {}}, defer_reward_to_post_process=False)
    targets = phase_targets_from_args(args, roles=["rollout", "genrm"])
    assert set(targets) == {PHASE_GENERATE}
    assert deferred_phases(args) == ()
    # The ledger does report the shared slice; a plan would still be wrong.
    contentions = shared_slice_contentions(PHASE_GENERATE, PHASE_GENRM)
    assert contentions and sorted(contentions[0].phases) == sorted([PHASE_GENERATE, PHASE_GENRM])
    assert phase_plans_from_contentions(targets, contentions, deferred=deferred_phases(args)) == ()


def test_deferred_genrm_on_shared_bundles_becomes_one_plan():
    args = build_args(_genrm_instances_resolved={"__default__": {}}, defer_reward_to_post_process=True)
    targets = phase_targets_from_args(args, roles=["rollout", "genrm"])
    assert set(targets) == {PHASE_GENERATE, PHASE_GENRM}
    plans = phase_plans_from_contentions(
        targets, shared_slice_contentions(PHASE_GENERATE, PHASE_GENRM), deferred=deferred_phases(args)
    )
    assert len(plans) == 1
    plan = plans[0]
    assert sorted(phase.phase_id for phase in plan.phases) == sorted([PHASE_GENERATE, PHASE_GENRM])
    assert plan.phase(PHASE_GENRM).targets == (ModelRef(Role.GENRM, "__default__"),)
    assert plan.phase(PHASE_GENERATE).targets == (ModelRef(Role.ROLLOUT, "default"),)


def test_deferred_scorer_without_a_shared_slice_still_gets_its_own_plan():
    """Deferred means "asleep outside its stage", which needs sequencing
    too."""
    args = build_args(_genrm_instances_resolved={"a": {}}, defer_reward_to_post_process=True)
    targets = phase_targets_from_args(args, roles=["rollout", "genrm"])
    plans = phase_plans_from_contentions(targets, (), deferred=deferred_phases(args))
    assert [plan.plan_id for plan in plans] == [f"deferred-{PHASE_GENRM}"]
    assert plans[0].activation_group == f"deferred-{PHASE_GENRM}"
    assert [phase.phase_id for phase in plans[0].phases] == [PHASE_GENRM]


def test_a_phase_never_lands_in_two_activation_groups():
    args = build_args(_genrm_instances_resolved={"a": {}}, defer_reward_to_post_process=True)
    targets = phase_targets_from_args(args, roles=["rollout", "genrm"])

    class Contention:
        def __init__(self, key, offset):
            self.placement_group_key = key
            self.reserved_offsets = (offset,)
            self.phases = (PHASE_GENERATE, PHASE_GENRM)

    plans = phase_plans_from_contentions(
        targets, (Contention("pg:a", 0), Contention("pg:b", 0)), deferred=deferred_phases(args)
    )
    assert len(plans) == 1
    groups = {phase.phase_id: plan.activation_group for plan in plans for phase in plan.phases}
    assert len(set(groups.values())) == 1


def test_a_deployment_without_rollout_claims_no_generate_phase():
    args = build_args(_genrm_instances_resolved={"a": {}}, defer_reward_to_post_process=True)
    targets = phase_targets_from_args(args, roles=["genrm"])
    assert set(targets) == {PHASE_GENRM}


def test_managed_teacher_defaults_to_deferred_under_colocate_only():
    colocate = build_args(
        opd_teacher_routes=None,
        teacher_hf_checkpoint="/ckpt",
        resource={"teacher": [1, 4], "actor": [1, 8], "rollout": [1, 4]},
        use_opd=True,
        opd_type="sglang",
    )
    assert deferred_phases(colocate) == (PHASE_TEACHER,)
    assert set(phase_targets_from_args(colocate, roles=["rollout", "teacher"])) == {PHASE_GENERATE, PHASE_TEACHER}

    dedicated = build_args(
        colocate=False,
        hybrid=True,
        teacher_hf_checkpoint="/ckpt",
        resource={"teacher": [1, 4], "actor": [1, 8], "rollout": [1, 4]},
        use_opd=True,
        opd_type="sglang",
    )
    assert deferred_phases(dedicated) == ()

    overridden = build_args(
        opd_teacher_routes=None,
        teacher_hf_checkpoint="/ckpt",
        resource={"teacher": [1, 4], "actor": [1, 8], "rollout": [1, 4]},
        use_opd=True,
        opd_type="sglang",
        opd_deferred_scoring=False,
    )
    assert deferred_phases(overridden) == ()


def test_mopd_routes_name_the_teacher_models():
    args = build_args(
        opd_teacher_routes='{"math": "/math", "code": "/code"}',
        resource={"teacher": [1, 4], "actor": [1, 8], "rollout": [1, 4]},
        use_opd=True,
        opd_type="sglang",
    )
    targets = phase_targets_from_args(args, roles=["rollout", "teacher"])
    assert targets[PHASE_TEACHER] == (ModelRef(Role.TEACHER, "math"), ModelRef(Role.TEACHER, "code"))


def test_empty_teacher_routes_are_rejected_rather_than_treated_as_none():
    args = build_args(
        opd_teacher_routes="{}",
        resource={"teacher": [1, 4], "actor": [1, 8], "rollout": [1, 4]},
        use_opd=True,
        opd_type="sglang",
    )
    with pytest.raises(ValueError, match="non-empty JSON object"):
        phase_targets_from_args(args, roles=["teacher"])
