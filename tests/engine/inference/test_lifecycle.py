# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unified six-state lifecycle on the Manager, plus the phase Coordinator.

These tests pin the two properties the deferred plans depend on: a transition
only reports a step it actually confirmed, and an unconfirmed release never
lets the next role onto the same GPUs.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event

import pytest

from relax.engine.inference.capabilities import WeightSource
from relax.engine.inference.lifecycle import (
    STEP_ACTIVATED,
    STEP_ADMISSION_CLOSED,
    STEP_DEACTIVATED,
    STEP_DRAINED,
    STEP_READY_PUBLISHED,
    STEP_RELEASE_CONFIRMED,
    ActivationToken,
    ErrorCode,
    HandoffResult,
    LifecycleCoordinator,
    LifecycleError,
    OperationError,
    OperationResult,
    OperationStatus,
    PhasePlan,
    PhaseSpec,
    ReleaseEvidence,
)
from relax.engine.inference.manager import InferenceManager, PreparationEvidence
from relax.engine.inference.specs import ModelSpec
from relax.engine.inference.types import LifecycleState, ModelRef, ReplicaSnapshot, Role, RoutingSpec


class FakeRuntime:
    """The engine-side runtime: it owns occupation, mirroring ``_PoolRuntime``."""

    def __init__(self, *, fail_onload: bool = False, fail_offload: bool = False, leak: bool = False) -> None:
        self.onloaded = True
        self.fail_onload = fail_onload
        self.fail_offload = fail_offload
        self.leak = leak
        self.calls: list[str] = []

    def health_check(self) -> bool:
        return True

    def recover(self) -> set[int]:
        return set()

    def is_onloaded(self) -> bool:
        return self.onloaded

    def set_onloaded(self, value: bool) -> None:
        # A backend that answers the RPC but keeps the memory must not be able
        # to report a release it did not perform.
        self.onloaded = True if (self.leak and not value) else value

    def fanout(self, method, *, skip_ranks=None, **kwargs):
        self.calls.append(method)
        if method == "resume_memory_occupation" and self.fail_onload:
            raise RuntimeError("resume failed")
        if method == "release_memory_occupation" and self.fail_offload:
            raise RuntimeError("release failed")
        return []

    def retire(self, ranks) -> None:
        pass

    def shutdown(self) -> None:
        self.calls.append("shutdown")


class FakePool:
    """The dispatch adapter: it routes lifecycle calls back through the Manager
    and publishes the resulting observation, exactly as a role pool does."""

    def __init__(self, manager, role: Role, model_id: str, runtime: FakeRuntime, *, ready_state=None) -> None:
        self.manager = manager
        self.role = role
        self.model_id = model_id
        self.runtime = runtime
        self._ready_state = ready_state or LifecycleState.READY

    def onload(self, tags=None):
        self.manager.onload(self.model_id, tags, role=self.role)
        self.publish(self._ready_state)

    def offload(self):
        self.manager.offload(self.model_id, role=self.role)

    def shutdown(self):
        self.manager.shutdown(self.model_id, role=self.role)

    def is_onloaded(self) -> bool:
        return self.runtime.is_onloaded()

    def publish(self, state: LifecycleState) -> None:
        current = self.manager.snapshot((self.model_id,), role=self.role).models[0]
        model = replace(
            current,
            state=state,
            admission=state == LifecycleState.READY,
            router_url="http://router:1",
            replicas=(ReplicaSnapshot(f"{self.model_id}/replica-0", state, "http://engine:1"),),
        )
        if state == LifecycleState.READY:
            token = self.manager.begin_preparation(self.model_id, role=self.role)
            self.manager.complete_preparation(
                token, model, evidence=PreparationEvidence(True, True, True), role=self.role
            )
        else:
            self.manager.publish_model(model, role=self.role)


def attach_model(
    manager: InferenceManager,
    role: Role,
    model_id: str,
    *,
    weight_source: WeightSource = WeightSource.CHECKPOINT,
    ready_state: LifecycleState | None = None,
    **runtime_kwargs,
) -> FakePool:
    view = manager.for_role(role)
    view.register_model(
        ModelSpec(model_id, "checkpoint", weight_source=weight_source, allow_defer=True),
        operation_id=f"register:{model_id}",
    )
    runtime = FakeRuntime(**runtime_kwargs)
    view.bind_pool(model_id, runtime)
    pool = FakePool(manager, role, model_id, runtime, ready_state=ready_state)
    view.configure_routes(RoutingSpec(default_model=model_id), operation_id=f"routes:{model_id}")
    view.attach_host({model_id: pool})
    return pool


def build_manager(role: Role = Role.GENRM, model_id: str = "judge", **runtime_kwargs):
    manager = InferenceManager()
    pool = attach_model(manager, role, model_id, **runtime_kwargs)
    pool.publish(LifecycleState.READY)
    return manager, pool


# ----------------------------------------------------------------------
# Manager: drain / deactivate / activate.
# ----------------------------------------------------------------------
def test_manager_drain_closes_admission_and_stays_draining():
    manager, _ = build_manager()
    target = ModelRef(Role.GENRM, "judge")
    result = manager.drain([target], operation_id="op-drain")
    assert result.status is OperationStatus.COMPLETED
    assert result.last_confirmed_step == STEP_DRAINED
    model = manager.snapshot(role=Role.GENRM).models[0]
    assert model.state == LifecycleState.DRAINING
    assert not model.admission
    with pytest.raises(RuntimeError, match="not ready"):
        manager.admit_request("judge", "r1", role=Role.GENRM)


def test_manager_drain_times_out_while_a_request_is_registered():
    manager, _ = build_manager()
    target = ModelRef(Role.GENRM, "judge")
    permit = manager.admit_request("judge", "r1", role=Role.GENRM)
    result = manager.drain([target], operation_id="op-drain", timeout_s=0.05)
    assert result.status is OperationStatus.FAILED
    assert result.error.code is ErrorCode.TIMEOUT
    # Admission is closed, but the resource is explicitly not handed on.
    assert result.last_confirmed_step == STEP_ADMISSION_CLOSED
    assert not result.release_confirmed
    assert manager.snapshot(role=Role.GENRM).models[0].state == LifecycleState.DRAINING
    manager.complete_request(permit)
    assert manager.drain([target], operation_id="op-drain-2").succeeded


def test_manager_drain_waits_for_a_request_that_finishes_concurrently():
    manager, _ = build_manager()
    target = ModelRef(Role.GENRM, "judge")
    permit = manager.admit_request("judge", "r1", role=Role.GENRM)
    started = Event()

    def finish() -> None:
        started.wait(5)
        manager.complete_request(permit)

    with ThreadPoolExecutor(2) as pool:
        completion = pool.submit(finish)
        started.set()
        result = manager.drain([target], operation_id="op-drain", timeout_s=5)
        completion.result(timeout=5)
    assert result.succeeded
    assert result.last_confirmed_step == STEP_DRAINED


def test_a_cancelled_request_is_recorded_but_left_to_the_engine_release_drain():
    """A cancel is not a completion, and not a reason to hold the slice.

    Matching the behaviour that predates this control plane: the engine's release
    path pauses admission, aborts what is in flight and only releases once
    ``flush_cache`` confirms the scheduler is empty. Blocking here instead would
    turn a client disconnect into a stuck activation group.
    """
    manager, _ = build_manager()
    target = ModelRef(Role.GENRM, "judge")
    permit = manager.admit_request("judge", "r1", role=Role.GENRM)
    assert manager.cancel_request(permit) == permit
    # Still registered and still not claimed complete.
    assert manager.inflight_requests(target) == ("r1",)
    assert manager.cancelling_requests(target) == ("r1",)
    assert manager.get_request("r1") == permit
    result = manager.drain([target], operation_id="op-drain", timeout_s=0.05)
    assert result.succeeded
    assert result.last_confirmed_step == STEP_DRAINED


def test_a_confirmed_release_clears_the_aborted_registration():
    manager, _ = build_manager()
    target = ModelRef(Role.GENRM, "judge")
    permit = manager.admit_request("judge", "r1", role=Role.GENRM)
    manager.cancel_request(permit)
    manager.drain([target], operation_id="op-drain", timeout_s=0.05)
    assert manager.deactivate([target], operation_id="op-deactivate").release_confirmed
    # The engine reported the memory is free, so the abort is provably over.
    assert manager.inflight_requests(target) == ()
    assert manager.cancelling_requests(target) == ()
    assert manager.get_request("r1") is None


def test_an_unconfirmed_release_keeps_the_aborted_registration():
    manager, _ = build_manager(leak=True)
    target = ModelRef(Role.GENRM, "judge")
    permit = manager.admit_request("judge", "r1", role=Role.GENRM)
    manager.cancel_request(permit)
    assert not manager.deactivate([target], operation_id="op-deactivate").release_confirmed
    assert manager.cancelling_requests(target) == ("r1",)


def test_a_cancel_before_the_request_reached_an_engine_clears_it():
    manager, _ = build_manager()
    target = ModelRef(Role.GENRM, "judge")
    permit = manager.admit_request("judge", "r1", role=Role.GENRM)
    assert manager.cancel_request(permit, dispatched=False) is None
    assert manager.inflight_requests(target) == ()
    assert manager.drain([target], operation_id="op-drain", timeout_s=0.05).succeeded


def test_a_cancelled_request_that_later_completes_unblocks_the_drain():
    manager, _ = build_manager()
    target = ModelRef(Role.GENRM, "judge")
    permit = manager.admit_request("judge", "r1", role=Role.GENRM)
    manager.cancel_request(permit)
    manager.complete_request(permit)
    assert manager.cancelling_requests(target) == ()
    assert manager.drain([target], operation_id="op-drain", timeout_s=0.05).succeeded


def test_only_a_request_still_expected_to_finish_fails_the_drain():
    manager, _ = build_manager()
    target = ModelRef(Role.GENRM, "judge")
    manager.admit_request("judge", "slow", role=Role.GENRM)
    aborted = manager.admit_request("judge", "aborted", role=Role.GENRM)
    manager.cancel_request(aborted)
    result = manager.drain([target], operation_id="op-drain", timeout_s=0.05)
    assert result.error.code is ErrorCode.TIMEOUT
    # The slow one is why it failed; the aborted one is reported, not waited on.
    assert "slow" in result.error.message
    assert "aborted, left to the engine release drain" in result.error.message


def test_shutdown_terminates_an_abort_that_was_never_confirmed():
    """A confirmed process exit is stronger evidence than any drain."""
    manager, pool = build_manager()
    target = ModelRef(Role.GENRM, "judge")
    permit = manager.admit_request("judge", "r1", role=Role.GENRM)
    manager.cancel_request(permit)
    result = manager.shutdown_models([target], operation_id="op-shutdown", timeout_s=0.05)
    assert result.succeeded and result.release_confirmed
    assert "shutdown" in pool.runtime.calls
    assert manager.inflight_requests(target) == ()
    assert manager.get_request("r1") is None


def test_shutdown_keeps_the_registration_when_the_teardown_fails():
    manager, pool = build_manager()
    target = ModelRef(Role.GENRM, "judge")
    permit = manager.admit_request("judge", "r1", role=Role.GENRM)
    manager.cancel_request(permit)

    def failing_shutdown() -> None:
        raise RuntimeError("engine will not die")

    pool.runtime.shutdown = failing_shutdown
    result = manager.shutdown_models([target], operation_id="op-shutdown", timeout_s=0.05)
    assert result.status is OperationStatus.FAILED
    # Nothing confirmed the engine stopped, so the registration survives.
    assert manager.inflight_requests(target) == ("r1",)


def test_manager_deactivate_confirms_the_release_and_sleeps():
    manager, pool = build_manager()
    target = ModelRef(Role.GENRM, "judge")
    manager.drain([target], operation_id="op-drain")
    result = manager.deactivate([target], operation_id="op-deactivate")
    assert result.succeeded and result.release_confirmed
    assert result.last_confirmed_step == STEP_RELEASE_CONFIRMED
    assert pool.is_onloaded() is False
    assert manager.snapshot(role=Role.GENRM).models[0].state == LifecycleState.SLEEPING


def test_manager_deactivate_does_not_confirm_a_release_the_pool_contradicts():
    manager, pool = build_manager(leak=True)
    target = ModelRef(Role.GENRM, "judge")
    result = manager.deactivate([target], operation_id="op-deactivate")
    assert result.status is OperationStatus.FAILED
    assert result.error.code is ErrorCode.UNKNOWN_COMPLETION
    assert result.last_confirmed_step == STEP_DEACTIVATED
    assert not result.release_confirmed
    assert pool.is_onloaded() is True


def test_manager_activate_publishes_ready_and_is_idempotent():
    manager, pool = build_manager()
    target = ModelRef(Role.GENRM, "judge")
    manager.deactivate([target], operation_id="op-deactivate")
    result = manager.activate([target], operation_id="op-activate")
    assert result.succeeded
    assert result.last_confirmed_step == STEP_READY_PUBLISHED
    assert manager.snapshot(role=Role.GENRM).models[0].admission
    assert manager.activate([target], operation_id="op-activate") == result
    with pytest.raises(LifecycleError, match="conflict"):
        manager.activate([target], operation_id="op-activate", tags=["weights"])


def test_manager_activate_reports_a_dead_pool_as_failed():
    manager, pool = build_manager(fail_onload=True)
    target = ModelRef(Role.GENRM, "judge")
    manager.deactivate([target], operation_id="op-deactivate")
    result = manager.activate([target], operation_id="op-activate")
    assert result.status is OperationStatus.FAILED
    assert result.last_confirmed_step == STEP_ACTIVATED
    assert manager.snapshot(role=Role.GENRM).models[0].state == LifecycleState.DEAD


def test_manager_policy_activation_stops_short_of_ready_until_weights_sync():
    manager = InferenceManager()
    # A policy model cannot be published READY without weight evidence, so its
    # pool adapter publishes ONLOADING until the trainer synchronizes weights.
    attach_model(
        manager,
        Role.ROLLOUT,
        "policy",
        weight_source=WeightSource.POLICY,
        ready_state=LifecycleState.ONLOADING,
    )
    target = ModelRef(Role.ROLLOUT, "policy")
    result = manager.activate([target], operation_id="op-activate")
    assert result.succeeded
    assert result.last_confirmed_step == STEP_ACTIVATED
    assert manager.snapshot(role=Role.ROLLOUT).models[0].state == LifecycleState.ONLOADING
    assert not manager.snapshot(role=Role.ROLLOUT).models[0].admission


def test_manager_rejects_untokened_transitions_once_a_coordinator_is_bound():
    manager, _ = build_manager()
    target = ModelRef(Role.GENRM, "judge")
    manager.bind_coordinator("epoch-1")
    with pytest.raises(LifecycleError, match="activation token is required"):
        manager.deactivate([target], operation_id="op-1")
    stale = ActivationToken("session", "epoch-0", "group", "op-2", (target,))
    with pytest.raises(LifecycleError, match="earlier coordinator"):
        manager.deactivate([target], operation_id="op-2", activation_token=stale)
    narrow = ActivationToken("session", "epoch-1", "group", "op-3", (ModelRef(Role.TEACHER, "other"),))
    with pytest.raises(LifecycleError, match="does not cover"):
        manager.deactivate([target], operation_id="op-3", activation_token=narrow)
    good = ActivationToken("session", "epoch-1", "group", "op-4", (target,))
    assert manager.deactivate([target], operation_id="op-4", activation_token=good).succeeded


def test_manager_shutdown_models_drains_before_teardown():
    manager, pool = build_manager()
    target = ModelRef(Role.GENRM, "judge")
    result = manager.shutdown_models([target], operation_id="op-shutdown", timeout_s=1)
    assert result.succeeded and result.release_confirmed
    assert "shutdown" in pool.runtime.calls
    assert manager.snapshot(role=Role.GENRM).models[0].state == LifecycleState.DEAD


# ----------------------------------------------------------------------
# Coordinator.
# ----------------------------------------------------------------------
class RecordingManager:
    """A Manager stand-in that records the order the Coordinator drives."""

    def __init__(self, *, drain_ok=True, release_confirmed=True, activate_ok=True) -> None:
        self.events: list[tuple[str, tuple[ModelRef, ...]]] = []
        self.drain_ok = drain_ok
        self.release_confirmed = release_confirmed
        self.activate_ok = activate_ok
        self.tokens: list[ActivationToken] = []

    def _result(self, kind, targets, *, ok, step, release_confirmed=False, code=ErrorCode.UNAVAILABLE):
        return OperationResult(
            operation_id=f"{kind}-op",
            owner_epoch="manager",
            status=OperationStatus.COMPLETED if ok else OperationStatus.FAILED,
            kind=kind,
            targets=tuple(targets),
            last_confirmed_step=step,
            error=None if ok else OperationError(code, f"{kind} failed"),
            release_confirmed=release_confirmed,
        )

    def drain(self, targets, *, operation_id, activation_token, timeout_s=None):
        self.events.append(("drain", tuple(targets)))
        self.tokens.append(activation_token)
        return self._result(
            "drain",
            targets,
            ok=self.drain_ok,
            step=STEP_DRAINED if self.drain_ok else STEP_ADMISSION_CLOSED,
            code=ErrorCode.TIMEOUT,
        )

    def deactivate(self, targets, *, operation_id, activation_token):
        self.events.append(("deactivate", tuple(targets)))
        self.tokens.append(activation_token)
        return self._result(
            "deactivate",
            targets,
            ok=self.release_confirmed,
            step=STEP_RELEASE_CONFIRMED if self.release_confirmed else STEP_DEACTIVATED,
            release_confirmed=self.release_confirmed,
            code=ErrorCode.UNKNOWN_COMPLETION,
        )

    def activate(self, targets, *, operation_id, activation_token, tags=None):
        self.events.append(("activate", tuple(targets)))
        self.tokens.append(activation_token)
        return self._result(
            "activate", targets, ok=self.activate_ok, step=STEP_READY_PUBLISHED if self.activate_ok else STEP_ACTIVATED
        )


ROLLOUT = ModelRef(Role.ROLLOUT, "policy")
JUDGE = ModelRef(Role.GENRM, "judge")
TEACHER_A = ModelRef(Role.TEACHER, "math")
TEACHER_B = ModelRef(Role.TEACHER, "code")


def defer_plan() -> PhasePlan:
    return PhasePlan(
        plan_id="colocate",
        activation_group="actor-bundles",
        phases=(
            PhaseSpec("inference", (ROLLOUT,)),
            PhaseSpec("genrm", (JUDGE,)),
            PhaseSpec("teacher", (TEACHER_A, TEACHER_B)),
        ),
    )


def build_coordinator(**kwargs) -> tuple[LifecycleCoordinator, RecordingManager]:
    manager = RecordingManager(**kwargs)
    coordinator = LifecycleCoordinator(manager, session_id="job-1", plans=(defer_plan(),))
    return coordinator, manager


def test_coordinator_switch_follows_the_fixed_order():
    coordinator, manager = build_coordinator()
    assert coordinator.switch_model("actor-bundles", ROLLOUT, operation_id="op-1").succeeded
    assert manager.events == [("activate", (ROLLOUT,))]
    result = coordinator.switch_model("actor-bundles", JUDGE, operation_id="op-2")
    assert result.succeeded
    assert manager.events[1:] == [
        ("drain", (ROLLOUT,)),
        ("deactivate", (ROLLOUT,)),
        ("activate", (JUDGE,)),
    ]
    assert coordinator.get_activation_group("actor-bundles").active == (JUDGE,)
    assert all(token.coordinator_epoch == coordinator.coordinator_epoch for token in manager.tokens)


def test_coordinator_replays_the_same_operation_and_conflicts_on_another_intent():
    coordinator, manager = build_coordinator()
    first = coordinator.switch_model("actor-bundles", ROLLOUT, operation_id="op-1")
    assert coordinator.switch_model("actor-bundles", ROLLOUT, operation_id="op-1") == first
    assert manager.events == [("activate", (ROLLOUT,))]
    with pytest.raises(LifecycleError, match="another transition"):
        coordinator.switch_model("actor-bundles", JUDGE, operation_id="op-1")
    assert coordinator.get_operation("op-1") == first


def test_coordinator_activates_a_whole_phase_together():
    coordinator, manager = build_coordinator()
    handle = coordinator.enter_phase("colocate", "teacher", operation_id="op-teacher")
    assert manager.events == [("activate", (TEACHER_A, TEACHER_B))]
    assert handle.token.targets == (TEACHER_A, TEACHER_B)
    with pytest.raises(LifecycleError, match="held exclusively"):
        coordinator.switch_model("actor-bundles", JUDGE, operation_id="op-other")
    result = coordinator.finish_phase(handle, operation_id="op-finish")
    assert result.succeeded and result.release_confirmed
    assert manager.events[1:] == [
        ("drain", (TEACHER_A, TEACHER_B)),
        ("deactivate", (TEACHER_A, TEACHER_B)),
    ]
    # No implicit restore of the previous occupant, and no next phase.
    assert coordinator.get_activation_group("actor-bundles").active == ()
    assert coordinator.finish_phase(handle, operation_id="op-finish") == result


def test_coordinator_refuses_the_next_role_after_an_unconfirmed_release():
    coordinator, manager = build_coordinator(release_confirmed=False)
    coordinator.switch_model("actor-bundles", ROLLOUT, operation_id="op-1")
    result = coordinator.switch_model("actor-bundles", JUDGE, operation_id="op-2")
    assert result.status is OperationStatus.FAILED
    assert result.last_confirmed_step == STEP_DEACTIVATED
    assert not result.release_confirmed
    # The conflicting role is never activated.
    assert ("activate", (JUDGE,)) not in manager.events
    group = coordinator.get_activation_group("actor-bundles")
    assert not group.release_confirmed and group.blocked_by == "inference"
    # Retrying re-attempts the release rather than locking the group forever,
    # but still refuses to activate the conflicting role while it is unproven.
    retry = coordinator.switch_model("actor-bundles", JUDGE, operation_id="op-3")
    assert retry.status is OperationStatus.FAILED
    assert ("activate", (JUDGE,)) not in manager.events
    assert manager.events[-2:] == [("drain", (ROLLOUT,)), ("deactivate", (ROLLOUT,))]


def test_coordinator_does_not_deactivate_when_the_drain_times_out():
    coordinator, manager = build_coordinator(drain_ok=False)
    coordinator.switch_model("actor-bundles", ROLLOUT, operation_id="op-1")
    result = coordinator.switch_model("actor-bundles", JUDGE, operation_id="op-2")
    assert result.status is OperationStatus.FAILED
    assert result.error.code is ErrorCode.TIMEOUT
    assert result.last_confirmed_step == STEP_ADMISSION_CLOSED
    assert [event for event, _ in manager.events] == ["activate", "drain"]


def test_coordinator_switch_is_a_no_op_only_when_every_target_is_ready():
    states = {ROLLOUT: LifecycleState.ONLOADING}
    manager = RecordingManager()
    coordinator = LifecycleCoordinator(
        manager, session_id="job-1", plans=(defer_plan(),), readiness=lambda target: states[target]
    )
    coordinator.switch_model("actor-bundles", ROLLOUT, operation_id="op-1")
    assert manager.events == [("activate", (ROLLOUT,))]
    # Still only partially restored: the switch must do real work again.
    coordinator.switch_model("actor-bundles", ROLLOUT, operation_id="op-2")
    assert manager.events == [("activate", (ROLLOUT,)), ("activate", (ROLLOUT,))]
    states[ROLLOUT] = LifecycleState.READY
    assert coordinator.switch_model("actor-bundles", ROLLOUT, operation_id="op-3").succeeded
    assert len(manager.events) == 2


def test_coordinator_keeps_independent_activation_groups_parallel():
    manager = RecordingManager()
    coordinator = LifecycleCoordinator(manager, session_id="job-1", plans=(defer_plan(),))
    coordinator.register_plan(
        PhasePlan(
            plan_id="teacher-cluster",
            activation_group="teacher-bundles",
            phases=(PhaseSpec("teacher_score", (TEACHER_A,), ("teacher",)),),
        )
    )
    handle = coordinator.enter_phase("colocate", "genrm", operation_id="op-genrm")
    # A different group is not serialized behind the exclusive handle.
    assert coordinator.transition("teacher-bundles", "teacher_score", operation_id="op-teacher").succeeded
    assert coordinator.get_activation_group("actor-bundles").exclusive_operation_id == handle.operation_id


def test_coordinator_validates_plans_against_placement_contention():
    coordinator, _ = build_coordinator()

    class Contention:
        phases = ("genrm", "inference")

    coordinator.validate_placement([Contention()])

    class Unplanned:
        phases = ("genrm", "critic")

    with pytest.raises(LifecycleError, match="no registered plan activates it"):
        coordinator.validate_placement([Unplanned()])

    manager = RecordingManager()
    split = LifecycleCoordinator(manager, session_id="job-1")
    split.register_plan(
        PhasePlan("a", "group-a", phases=(PhaseSpec("inference", (ROLLOUT,)),)),
    )
    split.register_plan(
        PhasePlan("b", "group-b", phases=(PhaseSpec("genrm", (JUDGE,)),)),
    )
    with pytest.raises(LifecycleError, match="span plans"):
        split.validate_placement([Contention()])

    together = LifecycleCoordinator(manager, session_id="job-1")
    together.register_plan(
        PhasePlan("c", "group-c", phases=(PhaseSpec("both", (ROLLOUT, JUDGE), ("inference", "genrm")),))
    )
    with pytest.raises(LifecycleError, match="One phase claims contending"):
        together.validate_placement([Contention()])


def test_coordinator_rejects_a_second_plan_for_one_activation_group():
    coordinator, _ = build_coordinator()
    with pytest.raises(LifecycleError, match="already belongs to plan"):
        coordinator.register_plan(PhasePlan("other", "actor-bundles", phases=(PhaseSpec("x", (JUDGE,)),)))
    assert coordinator.register_plan(defer_plan()) == defer_plan()


def test_coordinator_training_handoff_requires_a_confirmed_release():
    coordinator, _ = build_coordinator(release_confirmed=False)

    class Adapter:
        def __init__(self) -> None:
            self.granted = 0

        def prepare_training_handoff(self, batch_id, policy_version, *, operation_id, activation_token):
            self.granted += 1
            return HandoffResult(operation_id, batch_id, True, policy_version)

        def release_training_resources(self, *, operation_id, activation_token):
            return ReleaseEvidence(False, confirmed_ranks=(0,), total_ranks=4)

    adapter = Adapter()
    coordinator.switch_model("actor-bundles", ROLLOUT, operation_id="op-1")
    coordinator.switch_model("actor-bundles", JUDGE, operation_id="op-2")
    result = coordinator.prepare_training_handoff("actor-bundles", adapter, "batch-1", "v1", operation_id="op-handoff")
    assert not result.granted and adapter.granted == 0
    assert result.error.code is ErrorCode.UNAVAILABLE


def test_coordinator_release_evidence_blocks_the_next_activation_until_all_ranks_confirm():
    coordinator, _ = build_coordinator()

    class Adapter:
        def __init__(self, confirmed: bool) -> None:
            self.confirmed = confirmed

        def prepare_training_handoff(self, batch_id, policy_version, *, operation_id, activation_token):
            return HandoffResult(operation_id, batch_id, True, policy_version)

        def release_training_resources(self, *, operation_id, activation_token):
            return ReleaseEvidence(
                self.confirmed, confirmed_ranks=(0,) if not self.confirmed else (0, 1), total_ranks=2
            )

    coordinator.switch_model("actor-bundles", ROLLOUT, operation_id="op-1")
    evidence = coordinator.release_training_resources("actor-bundles", Adapter(False), operation_id="op-release")
    assert not evidence.release_confirmed
    with pytest.raises(LifecycleError, match="did not confirm its release"):
        coordinator.switch_model("actor-bundles", JUDGE, operation_id="op-2")
    assert coordinator.release_training_resources(
        "actor-bundles", Adapter(True), operation_id="op-release-2"
    ).release_confirmed
    assert coordinator.switch_model("actor-bundles", JUDGE, operation_id="op-3").succeeded


def test_coordinator_rejects_a_stale_phase_handle():
    coordinator, _ = build_coordinator()
    handle = coordinator.enter_phase("colocate", "genrm", operation_id="op-genrm")
    stale = replace(handle, token=replace(handle.token, coordinator_epoch="other-epoch"))
    with pytest.raises(LifecycleError, match="earlier coordinator"):
        coordinator.finish_phase(stale, operation_id="op-finish")


def test_coordinator_reports_an_unknown_group_and_operation():
    coordinator, _ = build_coordinator()
    with pytest.raises(LifecycleError, match="Unknown activation group"):
        coordinator.get_activation_group("missing")
    with pytest.raises(LifecycleError, match="Unknown operation"):
        coordinator.get_operation("missing")
    with pytest.raises(LifecycleError, match="has no phase"):
        coordinator.transition("actor-bundles", "missing", operation_id="op-1")


def test_coordinator_drives_the_manager_end_to_end_over_shared_gpus():
    """The Coordinator, the real Manager and two pools on one slice."""
    manager = InferenceManager()
    pools = {}
    for role, model_id in ((Role.ROLLOUT, "policy"), (Role.GENRM, "judge")):
        pool = attach_model(manager, role, model_id)
        pool.publish(LifecycleState.SLEEPING)
        pool.runtime.set_onloaded(False)
        pools[model_id] = pool
    states = manager.model_states()
    coordinator = LifecycleCoordinator(
        manager,
        session_id="job-1",
        plans=(
            PhasePlan(
                "colocate",
                "actor-bundles",
                phases=(
                    PhaseSpec("inference", (ModelRef(Role.ROLLOUT, "policy"),)),
                    PhaseSpec("genrm", (ModelRef(Role.GENRM, "judge"),)),
                ),
            ),
        ),
        readiness=lambda target: manager.model_states()[target] or LifecycleState.STARTING,
    )
    manager.bind_coordinator(coordinator.coordinator_epoch)
    assert set(states) == {ModelRef(Role.ROLLOUT, "policy"), ModelRef(Role.GENRM, "judge")}

    assert coordinator.transition("actor-bundles", "inference", operation_id="op-gen").succeeded
    assert pools["policy"].is_onloaded() and not pools["judge"].is_onloaded()

    handle = coordinator.enter_phase("colocate", "genrm", operation_id="op-score")
    assert not pools["policy"].is_onloaded() and pools["judge"].is_onloaded()
    assert manager.snapshot(role=Role.ROLLOUT).models[0].state == LifecycleState.SLEEPING
    assert manager.snapshot(role=Role.GENRM).models[0].admission

    assert coordinator.finish_phase(handle, operation_id="op-score-done").release_confirmed
    assert not pools["judge"].is_onloaded()
    # A direct, untokened switch is refused now that a coordinator owns the slice.
    with pytest.raises(LifecycleError, match="activation token is required"):
        manager.activate([ModelRef(Role.GENRM, "judge")], operation_id="sneaky")
