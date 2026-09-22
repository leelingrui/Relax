# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Owner-level phase coordination over externally hosted roles.

The interesting property is the handover: entering a scoring phase must close
generation admission, offload it, *confirm* the release from the host rather than
from the fact that the call returned, and only then wake the scorer.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from relax.distributed.ray import inference_role as module
from relax.engine.inference.capabilities import WeightSource
from relax.engine.inference.lifecycle import ErrorCode, LifecycleError
from relax.engine.inference.phase_plans import PHASE_GENERATE, PHASE_GENRM
from relax.engine.inference.placement import PlacementGroupView, PlacementOwner, PlacementRequest
from relax.engine.inference.specs import ModelSpec
from relax.engine.inference.types import (
    LifecycleState,
    ModelRef,
    ModelSnapshot,
    ReplicaSnapshot,
    Role,
    RoleSnapshot,
    RoutingSpec,
)


class FakeHost:
    """A role host that only releases memory when actually asked to.

    ``leak`` models a backend that answers the RPC while keeping the memory; the
    owner must then refuse to hand the slice on.
    """

    def __init__(self, role: Role, model_id: str, events: list, *, leak: bool = False, onloaded: bool = True) -> None:
        self.role = role
        self.model_id = model_id
        self.events = events
        self.leak = leak
        self.onloaded = onloaded
        self.call = SimpleNamespace(remote=self._call)
        self.snapshot = SimpleNamespace(remote=self._snapshot)

    def _call(self, model_id, method, *args, **kwargs):
        assert model_id == self.model_id
        self.events.append((self.role.value, method))
        if method == "offload":
            self.onloaded = bool(self.leak)
        elif method == "onload":
            self.onloaded = True
        elif method == "is_onloaded":
            return self.onloaded
        elif method == "set_onloaded":
            self.onloaded = bool(args[0])
        return None

    def _snapshot(self):
        state = LifecycleState.READY if self.onloaded else LifecycleState.SLEEPING
        return RoleSnapshot(
            role=self.role,
            manager_epoch="host-epoch",
            models=(
                ModelSnapshot(
                    model_id=self.model_id,
                    state=state,
                    admission=state == LifecycleState.READY,
                    router_url="http://router:1",
                    replicas=(ReplicaSnapshot(f"{self.model_id}/replica-0", state, "http://engine:1"),),
                ),
            ),
            routing=RoutingSpec(default_model=self.model_id),
        )


def build_owner(*, leak_generation: bool = False, shared: bool = True):
    """One owner, two externally hosted roles, optionally on one slice."""
    events: list = []
    owner = module.TaskInferenceManager()
    view = PlacementGroupView(tuple(range(4)), tuple(range(4)), PlacementOwner.CONTROLLER, identity="pg")
    for phase, offset in ((PHASE_GENERATE, 0), (PHASE_GENRM, 0 if shared else 2)):
        owner.plan_placement(
            (
                PlacementRequest(
                    group_id=f"{phase}/model",
                    worker_type="regular",
                    num_gpus=4 if shared else 2,
                    num_gpus_per_engine=4 if shared else 2,
                    num_gpus_per_node=4,
                    phase=phase,
                    bundle_offset=offset,
                ),
            ),
            view,
        )
    hosts = {}
    # The scorer starts asleep, which is what the launch path does under
    # colocate: the deferred stage is what wakes it.
    for role, model_id, leak, onloaded in (
        (Role.ROLLOUT, "policy", leak_generation, True),
        (Role.GENRM, "judge", False, False),
    ):
        host = FakeHost(role, model_id, events, leak=leak, onloaded=onloaded)
        hosts[role] = host
        owner.register_external_role(
            role,
            host,
            (ModelSpec(model_id, "checkpoint", weight_source=WeightSource.CHECKPOINT, allow_defer=True),),
            RoutingSpec(default_model=model_id),
        )
    phase_targets = {
        PHASE_GENERATE: (ModelRef(Role.ROLLOUT, "policy"),),
        PHASE_GENRM: (ModelRef(Role.GENRM, "judge"),),
    }
    return owner, hosts, events, phase_targets


def test_owner_builds_one_plan_for_a_shared_slice_and_sequences_the_handover():
    owner, hosts, events, phase_targets = build_owner()
    epoch = owner.create_coordinator("job-1", phase_targets, deferred=(PHASE_GENRM,))
    assert epoch
    located = owner.locate_phase(PHASE_GENRM)
    assert located is not None
    plan_id, group = located
    assert owner.locate_phase(PHASE_GENERATE) == (plan_id, group)

    owner.adopt_phase(group, PHASE_GENERATE, operation_id="adopt-1")
    events.clear()
    handle = owner.enter_phase(plan_id, PHASE_GENRM, operation_id="score-1", timeout_s=1)
    lifecycle = [event for event in events if event[1] in {"offload", "onload"}]
    assert lifecycle == [("rollout", "offload"), ("genrm", "onload")]
    assert hosts[Role.ROLLOUT].onloaded is False and hosts[Role.GENRM].onloaded is True
    assert owner.get_activation_group(group).active == (ModelRef(Role.GENRM, "judge"),)

    result = owner.finish_phase(handle, operation_id="score-1-done", timeout_s=1)
    assert result.release_confirmed
    assert hosts[Role.GENRM].onloaded is False
    # Nothing is restored implicitly; the training path onloads generation.
    assert owner.get_activation_group(group).active == ()


def test_owner_refuses_to_wake_the_scorer_when_generation_keeps_its_memory():
    owner, hosts, events, phase_targets = build_owner(leak_generation=True)
    owner.create_coordinator("job-1", phase_targets, deferred=(PHASE_GENRM,))
    plan_id, group = owner.locate_phase(PHASE_GENRM)
    owner.adopt_phase(group, PHASE_GENERATE, operation_id="adopt-1")
    events.clear()
    with pytest.raises(LifecycleError):
        owner.enter_phase(plan_id, PHASE_GENRM, operation_id="score-1", timeout_s=1)
    assert ("genrm", "onload") not in events
    assert hosts[Role.GENRM].onloaded is False
    group_state = owner.get_activation_group(group)
    assert not group_state.release_confirmed and group_state.blocked_by == "inference"
    # A retry re-attempts the release; it still must not wake the scorer while
    # generation keeps the memory.
    with pytest.raises(LifecycleError):
        owner.enter_phase(plan_id, PHASE_GENRM, operation_id="score-2", timeout_s=1)
    assert ("genrm", "onload") not in events
    assert hosts[Role.GENRM].onloaded is False


def test_owner_drain_waits_for_a_registered_request_before_offloading():
    owner, hosts, events, phase_targets = build_owner()
    owner.create_coordinator("job-1", phase_targets, deferred=(PHASE_GENRM,))
    plan_id, group = owner.locate_phase(PHASE_GENRM)
    owner.adopt_phase(group, PHASE_GENERATE, operation_id="adopt-1")
    permit = owner.admit_request("policy", "request-1", role=Role.ROLLOUT)
    events.clear()
    with pytest.raises(LifecycleError):
        owner.enter_phase(plan_id, PHASE_GENRM, operation_id="score-1", timeout_s=0.05)
    assert ("rollout", "offload") not in events
    assert ("genrm", "onload") not in events
    owner.complete_request(permit)
    assert owner.enter_phase(plan_id, PHASE_GENRM, operation_id="score-2", timeout_s=2) is not None


def test_owner_without_a_shared_slice_only_sequences_the_deferred_scorer():
    owner, _hosts, _events, phase_targets = build_owner(shared=False)
    owner.create_coordinator("job-1", phase_targets, deferred=(PHASE_GENRM,))
    assert owner.coordinated_phases() == (PHASE_GENRM,)
    assert owner.locate_phase(PHASE_GENERATE) is None
    plan_id, group = owner.locate_phase(PHASE_GENRM)
    handle = owner.enter_phase(plan_id, PHASE_GENRM, operation_id="score-1", timeout_s=1)
    assert owner.finish_phase(handle, operation_id="score-1-done", timeout_s=1).release_confirmed


def test_owner_reports_no_coordinator_when_nothing_needs_sequencing():
    owner, _hosts, _events, phase_targets = build_owner(shared=False)
    assert owner.create_coordinator("job-1", phase_targets) == ""
    assert owner.coordinated_phases() == ()
    assert owner.activation_groups() == ()
    assert owner.locate_phase(PHASE_GENRM) is None


def test_owner_release_for_training_empties_every_group():
    owner, hosts, events, phase_targets = build_owner()
    owner.create_coordinator("job-1", phase_targets, deferred=(PHASE_GENRM,))
    _plan_id, group = owner.locate_phase(PHASE_GENERATE)
    owner.adopt_phase(group, PHASE_GENERATE, operation_id="adopt-1")
    events.clear()
    result = owner.switch_model(group, (), operation_id="train-release:1", timeout_s=1)
    assert result.succeeded and result.release_confirmed
    assert [event for event in events if event[1] == "offload"] == [("rollout", "offload")]
    assert owner.get_activation_group(group).active == ()


def test_owner_rejects_an_untokened_activation_once_the_coordinator_exists():
    """A rejected request raises; only a failed transition returns a result."""
    owner, _hosts, _events, phase_targets = build_owner()
    owner.create_coordinator("job-1", phase_targets, deferred=(PHASE_GENRM,))
    with pytest.raises(LifecycleError, match="activation token is required"):
        owner.activate((ModelRef(Role.GENRM, "judge"),), operation_id="sneaky")


def test_owner_refuses_a_managed_transition_for_an_unregistered_legacy_host():
    owner = module.TaskInferenceManager()
    host = SimpleNamespace(snapshot=SimpleNamespace(remote=lambda: RoleSnapshot(Role.TEACHER, "epoch")))
    owner.register_role(Role.TEACHER, host)
    result = owner.drain((ModelRef(Role.TEACHER, "default"),), operation_id="drain-1")
    assert not result.succeeded
    assert result.error.code is ErrorCode.UNSUPPORTED


def test_scoring_phase_is_a_no_op_without_a_coordinator(monkeypatch: pytest.MonkeyPatch):
    from relax.engine.rollout import scoring_phase as module_under_test

    monkeypatch.setattr(module_under_test, "_task_inference_manager", lambda: None)
    with module_under_test.scoring_phase(SimpleNamespace(), PHASE_GENRM) as handle:
        assert handle is None


def test_scoring_phase_enters_and_leaves_the_phase(monkeypatch: pytest.MonkeyPatch):
    from relax.engine.rollout import scoring_phase as module_under_test

    owner, hosts, events, phase_targets = build_owner()
    owner.create_coordinator("job-1", phase_targets, deferred=(PHASE_GENRM,))
    _plan_id, group = owner.locate_phase(PHASE_GENERATE)
    owner.adopt_phase(group, PHASE_GENERATE, operation_id="adopt-1")
    monkeypatch.setattr(module_under_test, "_task_inference_manager", lambda: owner)
    events.clear()
    with module_under_test.scoring_phase(SimpleNamespace(), PHASE_GENRM, batch_id="batch-1") as handle:
        assert handle is not None
        assert hosts[Role.GENRM].onloaded is True
        assert hosts[Role.ROLLOUT].onloaded is False
    assert hosts[Role.GENRM].onloaded is False
    assert [event for event in events if event[1] in {"offload", "onload"}] == [
        ("rollout", "offload"),
        ("genrm", "onload"),
        ("genrm", "offload"),
    ]


def test_scoring_phase_closes_the_phase_even_when_scoring_raises(monkeypatch: pytest.MonkeyPatch):
    from relax.engine.rollout import scoring_phase as module_under_test

    owner, hosts, _events, phase_targets = build_owner()
    owner.create_coordinator("job-1", phase_targets, deferred=(PHASE_GENRM,))
    monkeypatch.setattr(module_under_test, "_task_inference_manager", lambda: owner)
    with pytest.raises(RuntimeError, match="scoring blew up"):
        with module_under_test.scoring_phase(SimpleNamespace(), PHASE_GENRM, batch_id="batch-1"):
            raise RuntimeError("scoring blew up")
    assert hosts[Role.GENRM].onloaded is False


def test_training_release_evidence_gates_the_next_inference_phase():
    """Only training can speak for its ranks; unconfirmed keeps the slice
    closed."""
    from relax.engine.inference.lifecycle import ReleaseEvidence

    owner, hosts, _events, phase_targets = build_owner()
    owner.create_coordinator("job-1", phase_targets, deferred=(PHASE_GENRM,))
    plan_id, group = owner.locate_phase(PHASE_GENRM)
    partial = owner.confirm_training_release(
        group, ReleaseEvidence(False, confirmed_ranks=(0,), total_ranks=4), operation_id="release-1"
    )
    assert not partial.release_confirmed
    state = owner.get_activation_group(group)
    assert state.blocked_by == "training"
    with pytest.raises(LifecycleError, match="did not confirm its release"):
        owner.enter_phase(plan_id, PHASE_GENRM, operation_id="score-1", timeout_s=1)
    assert hosts[Role.GENRM].onloaded is False

    full = owner.confirm_training_release(
        group, ReleaseEvidence(True, confirmed_ranks=(0, 1, 2, 3), total_ranks=4), operation_id="release-2"
    )
    assert full.release_confirmed
    assert owner.get_activation_group(group).blocked_by is None
    assert owner.enter_phase(plan_id, PHASE_GENRM, operation_id="score-2", timeout_s=1) is not None


def test_owner_snapshot_keeps_the_task_epoch_after_a_phase_switch():
    owner, _hosts, _events, phase_targets = build_owner()
    owner.create_coordinator("job-1", phase_targets, deferred=(PHASE_GENRM,))
    plan_id, group = owner.locate_phase(PHASE_GENRM)
    owner.adopt_phase(group, PHASE_GENERATE, operation_id="adopt-1")
    owner.enter_phase(plan_id, PHASE_GENRM, operation_id="score-1", timeout_s=1)
    snapshot = owner.snapshot(role=Role.ROLLOUT)
    assert snapshot.manager_epoch == owner.manager_epoch
    assert snapshot.models[0].state == LifecycleState.SLEEPING
    assert not snapshot.models[0].admission
    assert replace(snapshot, models=()) == replace(owner.snapshot(role=Role.ROLLOUT), models=())
