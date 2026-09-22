# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Derive phase plans from the existing CLI and the placement ledger.

There is no new user-facing configuration here. Which models exist comes from
the flags that already decide them (the resolved SGLang config, the resolved
GenRM instances, the managed OPD teacher routes). Which of them are *mutually
exclusive* is not guessed from those flags: it comes from the placement ledger,
which already knows exactly which allocations share GPUs across phases.

That distinction matters. Under colocate, GenRM either shares the rollout
bundles (the defer layout) or sits in its own region next to them (the split
layout), and a managed OPD teacher normally sits in its own region as well. Only
the sharing layouts may be sequenced; serializing a split layout would idle GPUs
that were deliberately given to a second role. Reading the answer off the ledger
means a layout change cannot silently invalidate the plan.

Phase identities are the planner's own phase labels (``inference``, ``genrm``,
``teacher``), which is what lets a plan be checked against the ledger at all.
"""

from typing import Any, Iterable, Mapping, Sequence

from relax.engine.inference.lifecycle import PhasePlan, PhaseSpec
from relax.engine.inference.types import ModelRef, Role


# The planner's default phase for rollout engines, and the labels the GenRM and
# teacher adapters already pass to it.
PHASE_GENERATE = "inference"
PHASE_GENRM = "genrm"
PHASE_TEACHER = "teacher"

PHASE_ROLES = {PHASE_GENERATE: Role.ROLLOUT, PHASE_GENRM: Role.GENRM, PHASE_TEACHER: Role.TEACHER}


def rollout_model_ids(args: Any) -> tuple[str, ...]:
    """Read the rollout model identities from the resolver rollout itself
    uses."""
    from relax.distributed.ray.rollout import _resolve_sglang_config

    return tuple(model.name for model in _resolve_sglang_config(args).models)


def genrm_model_ids(args: Any) -> tuple[str, ...]:
    """The resolved GenRM instance keys, including the single-instance
    sentinel.

    The keys are the model IDs: ``create_genrm_manager(s)`` passes them straight
    through as the pool identities.
    """
    return tuple(getattr(args, "_genrm_instances_resolved", None) or ())


def teacher_model_ids(args: Any) -> tuple[str, ...]:
    """MOPD route keys, or ``default`` for the single managed teacher."""
    from relax.utils.opd.opd_utils import is_managed_opd_teacher_enabled, resolve_opd_teacher_routes

    if not is_managed_opd_teacher_enabled(args):
        return ()
    return tuple(resolve_opd_teacher_routes(args)) or ("default",)


def deferred_genrm_enabled(args: Any) -> bool:
    """Whether GenRM scores in its own stage instead of inline with
    generation."""
    return bool(genrm_model_ids(args)) and bool(getattr(args, "defer_reward_to_post_process", False))


def deferred_opd_enabled(args: Any) -> bool:
    """Whether the managed OPD teacher scores in its own stage.

    Colocate defaults to the deferred stage, which is where moving the teacher
    prefill out of the generation loop pays off; a dedicated-GPU teacher keeps
    scoring inline because nothing is waiting for its memory.
    ``--opd-deferred-scoring`` overrides the default in either direction.
    """
    from relax.utils.opd.opd_utils import is_managed_opd_teacher_colocate

    if not teacher_model_ids(args):
        return False
    configured = getattr(args, "opd_deferred_scoring", None)
    if configured is not None:
        return bool(configured)
    return is_managed_opd_teacher_colocate(args)


def phase_targets_from_args(args: Any, *, roles: Iterable[str] | None = None) -> dict[str, tuple[ModelRef, ...]]:
    """Map each planner phase label to the models whose occupancy is sequenced.

    A role only appears when it actually scores in a separate stage. A GenRM
    that shares the rollout bundles while staying resident (split memory
    fractions, inline scoring) is *not* mutually exclusive with generation even
    though the ledger reports the shared slice, so including it would serialize a
    layout that was deliberately made co-resident.

    ``roles`` restricts the result to the roles a deployment created, so an SFT
    or debug run does not claim a rollout phase it never started.
    """
    deployed = None if roles is None else {str(role) for role in roles}
    targets: dict[str, tuple[ModelRef, ...]] = {}
    for phase_id, ids in (
        (PHASE_GENERATE, rollout_model_ids(args) if deployed is None or "rollout" in deployed else ()),
        (PHASE_GENRM, genrm_model_ids(args) if deferred_genrm_enabled(args) else ()),
        (PHASE_TEACHER, teacher_model_ids(args) if deferred_opd_enabled(args) else ()),
    ):
        if ids:
            targets[phase_id] = tuple(ModelRef(PHASE_ROLES[phase_id], model_id) for model_id in ids)
    return targets


def activation_group_name(contention: Any) -> str:
    """Name a group after the shared slice it protects.

    The placement-group key plus the lowest shared offset is stable across
    processes, unlike a Python object identity, and it stays meaningful when one
    placement group holds several independently contended regions.
    """
    offsets = tuple(getattr(contention, "reserved_offsets", ()) or (0,))
    return f"{getattr(contention, 'placement_group_key', 'pg')}@{min(offsets)}"


def deferred_phases(args: Any) -> tuple[str, ...]:
    """The phases whose models are asleep outside their own scoring stage."""
    phases = []
    if deferred_genrm_enabled(args):
        phases.append(PHASE_GENRM)
    if deferred_opd_enabled(args):
        phases.append(PHASE_TEACHER)
    return tuple(phases)


def phase_plans_from_contentions(
    phase_targets: Mapping[str, Sequence[ModelRef]],
    contentions: Sequence[Any],
    *,
    deferred: Sequence[str] = (),
) -> tuple[PhasePlan, ...]:
    """Build one plan per contended slice, plus one per unshared deferred phase.

    A phase that shares GPUs with nothing normally needs no plan: it can stay
    active alongside every other phase, and giving it one would only add a
    serialization point the layout does not require. A *deferred* phase is the
    exception -- being deferred means its models sleep outside their own stage,
    which is a sequencing statement whether or not the slice is shared -- so it
    gets a plan of its own and is activated only inside that stage.
    """
    plans: list[PhasePlan] = []
    planned: set[str] = set()
    for index, contention in enumerate(contentions):
        labels = tuple(getattr(contention, "phases", ()))
        phases = tuple(
            PhaseSpec(label, tuple(phase_targets[label]), (label,)) for label in labels if phase_targets.get(label)
        )
        if len(phases) < 2:
            # Either the contention is between phases this task does not own, or
            # only one side has models; there is nothing to sequence.
            continue
        group = activation_group_name(contention)
        if planned & {phase.phase_id for phase in phases}:
            # One phase cannot belong to two activation groups; the first
            # contended cluster that claimed it owns it.
            continue
        plans.append(PhasePlan(f"contended-{index}-{group}", group, phases=phases))
        planned.update(phase.phase_id for phase in phases)
    for phase_id in deferred:
        if phase_id in planned or not phase_targets.get(phase_id):
            continue
        group = f"deferred-{phase_id}"
        plans.append(
            PhasePlan(
                f"deferred-{phase_id}",
                group,
                phases=(PhaseSpec(phase_id, tuple(phase_targets[phase_id]), (phase_id,)),),
            )
        )
        planned.add(phase_id)
    return tuple(plans)


__all__ = [
    "PHASE_GENERATE",
    "PHASE_GENRM",
    "PHASE_ROLES",
    "PHASE_TEACHER",
    "activation_group_name",
    "deferred_genrm_enabled",
    "deferred_opd_enabled",
    "deferred_phases",
    "genrm_model_ids",
    "phase_plans_from_contentions",
    "phase_targets_from_args",
    "rollout_model_ids",
    "teacher_model_ids",
]
