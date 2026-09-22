# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Phase coordination for roles that share inference GPUs.

Ownership stays split. The Manager is the only writer of model and replica
state; the Coordinator decides *who may occupy an activation group* and drives
the Manager through one fixed order:

``close admission -> drain -> deactivate -> confirm release -> grant next
activation -> activate -> publish READY``

Nothing in this module infers that GPU memory was released. A drain timeout, a
dead actor or an unreachable host leaves the group blocked, because handing the
slice to the next role without a confirmed release is how two engines end up on
the same GPU. The Coordinator therefore refuses to activate a conflicting role
after an unconfirmed release, and says so through ``release_confirmed``.

This module is pure CPU state: no Ray, no HTTP and no GPU access, so a control
plane can own a Coordinator instance without an extra actor process.
"""

from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any, Mapping, Protocol, Sequence

from relax.engine.inference.discovery import new_manager_epoch
from relax.engine.inference.types import LifecycleState, ModelRef, Role


# Confirmed steps, in the order a transition passes through them. A result
# reports the last step that was *confirmed*, never the step that was attempted.
STEP_ADMISSION_CLOSED = "admission_closed"
STEP_DRAINED = "drained"
STEP_DEACTIVATED = "deactivated"
STEP_RELEASE_CONFIRMED = "release_confirmed"
STEP_ACTIVATION_GRANTED = "activation_granted"
STEP_ACTIVATED = "activated"
STEP_READY_PUBLISHED = "ready_published"

LIFECYCLE_STEPS = (
    STEP_ADMISSION_CLOSED,
    STEP_DRAINED,
    STEP_DEACTIVATED,
    STEP_RELEASE_CONFIRMED,
    STEP_ACTIVATION_GRANTED,
    STEP_ACTIVATED,
    STEP_READY_PUBLISHED,
)


class OperationStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCEL_PENDING = "cancel_pending"
    CANCELLED = "cancelled"


class ErrorCode(str, Enum):
    INVALID_ARGUMENT = "invalid_argument"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    BUSY = "busy"
    STALE_GENERATION = "stale_generation"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    TIMEOUT = "timeout"
    UNKNOWN_COMPLETION = "unknown_completion"


@dataclass(frozen=True)
class OperationError:
    """A structured failure.

    ``retryable`` describes the control-plane operation only. It never promises
    that a request already sent upstream had no effect.
    """

    code: ErrorCode
    message: str
    retryable: bool = False
    model_id: str | None = None
    state: LifecycleState | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code.value, "message": self.message, "retryable": self.retryable}
        if self.model_id is not None:
            payload["model"] = self.model_id
        if self.state is not None:
            payload["state"] = self.state.value
        return payload


class LifecycleError(RuntimeError):
    """Raised for a rejected request, as opposed to a failed transition."""

    def __init__(self, error: OperationError) -> None:
        super().__init__(f"{error.code.value}: {error.message}")
        self.error = error


@dataclass(frozen=True)
class ActivationToken:
    """Authority to change the activation state of one activation group.

    The token binds the job/session, the Coordinator lifetime, the group, the
    operation and the exact target set, so possession of an ``operation_id`` is
    never enough to move a shared slice.
    """

    session_id: str
    coordinator_epoch: str
    activation_group: str
    operation_id: str
    targets: tuple[ModelRef, ...] = ()

    def covers(self, target: ModelRef) -> bool:
        return target in self.targets


@dataclass(frozen=True)
class PhaseHandle:
    """An operation-scoped exclusive claim on an activation group."""

    plan_id: str
    phase_id: str
    token: ActivationToken

    @property
    def activation_group(self) -> str:
        return self.token.activation_group

    @property
    def operation_id(self) -> str:
        return self.token.operation_id


@dataclass(frozen=True)
class OperationResult:
    """The shared observation of any control-plane operation."""

    operation_id: str
    owner_epoch: str
    status: OperationStatus
    kind: str
    targets: tuple[ModelRef, ...] = ()
    last_confirmed_step: str | None = None
    error: OperationError | None = None
    release_confirmed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", OperationStatus(self.status))
        if self.last_confirmed_step is not None and self.last_confirmed_step not in LIFECYCLE_STEPS:
            raise ValueError(f"Unknown lifecycle step: {self.last_confirmed_step}")

    @property
    def succeeded(self) -> bool:
        return self.status is OperationStatus.COMPLETED

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "owner_epoch": self.owner_epoch,
            "status": self.status.value,
            "kind": self.kind,
            "targets": [str(target) for target in self.targets],
            "last_confirmed_step": self.last_confirmed_step,
            "error": self.error.to_dict() if self.error is not None else None,
            "release_confirmed": self.release_confirmed,
        }


@dataclass(frozen=True)
class PhaseResult(OperationResult):
    """An ``OperationResult`` that also names the phase it belongs to."""

    plan_id: str | None = None
    phase_id: str | None = None
    activation_group: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update({"plan_id": self.plan_id, "phase_id": self.phase_id, "activation_group": self.activation_group})
        return payload


# ``switch_model`` shares the executor with phase transitions, so it shares the
# result type; only the ``kind`` differs.
SwitchResult = PhaseResult


@dataclass(frozen=True)
class PhaseSpec:
    """One mutually exclusive occupancy of an activation group.

    ``placement_phases`` are the planner's phase labels this phase occupies.
    They default to ``phase_id`` because the planner already labels requests
    with the phase that owns them, and they are what lets a plan be validated
    against the placement ledger instead of against a hand-written list.
    """

    phase_id: str
    targets: tuple[ModelRef, ...] = ()
    placement_phases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.phase_id:
            raise ValueError("A phase requires an identity")
        object.__setattr__(self, "targets", tuple(self.targets))
        if len(set(self.targets)) != len(self.targets):
            raise ValueError(f"Duplicate targets in phase {self.phase_id}")
        object.__setattr__(self, "placement_phases", tuple(self.placement_phases) or (self.phase_id,))


@dataclass(frozen=True)
class PhasePlan:
    """A fixed, versioned plan for one activation group.

    One plan owns one activation group: the first version deliberately has no
    cross-group joint transaction, so independent groups stay independent and
    parallel rather than being serialized behind a global phase.
    """

    plan_id: str
    activation_group: str
    phases: tuple[PhaseSpec, ...]
    version: int = 1

    def __post_init__(self) -> None:
        if not self.plan_id or not self.activation_group:
            raise ValueError("A phase plan requires a plan ID and an activation group")
        object.__setattr__(self, "phases", tuple(self.phases))
        ids = [phase.phase_id for phase in self.phases]
        if not ids:
            raise ValueError(f"Phase plan {self.plan_id} has no phases")
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate phase IDs in plan {self.plan_id}")

    def phase(self, phase_id: str) -> PhaseSpec:
        for phase in self.phases:
            if phase.phase_id == phase_id:
                return phase
        raise LifecycleError(OperationError(ErrorCode.NOT_FOUND, f"Plan {self.plan_id} has no phase {phase_id!r}"))

    def phase_for_target(self, target: ModelRef) -> PhaseSpec | None:
        for phase in self.phases:
            if target in phase.targets:
                return phase
        return None


@dataclass(frozen=True)
class ActivationGroupSnapshot:
    """What currently occupies a group, and who holds it exclusively."""

    activation_group: str
    coordinator_epoch: str
    plan_id: str | None = None
    phase_id: str | None = None
    active: tuple[ModelRef, ...] = ()
    exclusive_operation_id: str | None = None
    running_operation_id: str | None = None
    last_operation_id: str | None = None
    release_confirmed: bool = True
    blocked_reason: str | None = None
    blocked_by: str | None = None


@dataclass(frozen=True)
class ReleaseEvidence:
    """Per-rank proof that training released the shared resource.

    Rank 0 answering is not evidence: a single rank that still holds memory is
    enough to make the next activation fail with an allocation error.
    """

    release_confirmed: bool
    confirmed_ranks: tuple[int, ...] = ()
    total_ranks: int = 0
    error: OperationError | None = None


@dataclass(frozen=True)
class HandoffResult:
    """Training may take over and *wait for* the batch.

    Granting a handoff never claims the batch is already complete; the deferred
    publication path is what makes the data consumable, and conflating the two
    is how a trainer and a scorer end up waiting for each other.
    """

    operation_id: str
    batch_id: str
    granted: bool
    policy_version: str | None = None
    error: OperationError | None = None


class LifecycleManager(Protocol):
    """The Manager operations a Coordinator drives.

    Each call is expected to be idempotent per ``operation_id`` and to return a
    result rather than raise for an ordinary transition failure.
    """

    def drain(
        self,
        targets: Sequence[ModelRef],
        *,
        operation_id: str,
        activation_token: ActivationToken,
        timeout_s: float | None = None,
    ) -> OperationResult: ...

    def deactivate(
        self, targets: Sequence[ModelRef], *, operation_id: str, activation_token: ActivationToken
    ) -> OperationResult: ...

    def activate(
        self,
        targets: Sequence[ModelRef],
        *,
        operation_id: str,
        activation_token: ActivationToken,
        tags: list[str] | None = None,
    ) -> OperationResult: ...


class TrainingResourceAdapter(Protocol):
    """The training backend's side of a shared-GPU handoff."""

    def prepare_training_handoff(
        self, batch_id: str, policy_version: str | None, *, operation_id: str, activation_token: ActivationToken
    ) -> HandoffResult: ...

    def release_training_resources(
        self, *, operation_id: str, activation_token: ActivationToken
    ) -> ReleaseEvidence: ...


@dataclass
class _GroupState:
    activation_group: str
    plan_id: str | None = None
    phase_id: str | None = None
    active: tuple[ModelRef, ...] = ()
    exclusive_operation_id: str | None = None
    running_operation_id: str | None = None
    last_operation_id: str | None = None
    release_confirmed: bool = True
    blocked_reason: str | None = None
    # Which side failed to confirm: an inference deactivation can be retried by
    # draining and releasing again, while a training release that no rank
    # confirmed cannot -- re-releasing inference engines frees nothing there.
    blocked_by: str | None = None
    finished_phases: dict[str, PhaseResult] = field(default_factory=dict)


class LifecycleCoordinator:
    """Per-task phase coordination over one unified inference Manager."""

    def __init__(
        self,
        manager: LifecycleManager,
        *,
        session_id: str,
        plans: Sequence[PhasePlan] = (),
        readiness: Any = None,
    ) -> None:
        self._manager = manager
        self._session_id = session_id
        self.coordinator_epoch = new_manager_epoch()
        self._lock = RLock()
        self._plans: dict[str, PhasePlan] = {}
        self._groups: dict[str, _GroupState] = {}
        self._operations: dict[str, tuple[tuple, OperationResult]] = {}
        # Optional readiness probe: ``(role, model_id) -> LifecycleState``. It
        # only ever downgrades a no-op decision to a real transition, so a
        # missing probe costs a redundant activate, never a false READY.
        self._readiness = readiness
        for plan in plans:
            self.register_plan(plan)

    # ------------------------------------------------------------------
    # Plans and validation.
    # ------------------------------------------------------------------
    def register_plan(self, plan: PhasePlan) -> PhasePlan:
        """Register a fixed plan; re-registering an identical plan is a
        no-op."""
        with self._lock:
            previous = self._plans.get(plan.plan_id)
            if previous is not None:
                if previous != plan:
                    raise LifecycleError(
                        OperationError(ErrorCode.CONFLICT, f"Phase plan {plan.plan_id} is already registered")
                    )
                return previous
            for other in self._plans.values():
                if other.activation_group == plan.activation_group:
                    raise LifecycleError(
                        OperationError(
                            ErrorCode.CONFLICT,
                            f"Activation group {plan.activation_group!r} already belongs to plan {other.plan_id}",
                        )
                    )
            claimed: dict[ModelRef, str] = {}
            for phase in plan.phases:
                for target in phase.targets:
                    owner = claimed.setdefault(target, phase.phase_id)
                    if owner != phase.phase_id:
                        raise LifecycleError(
                            OperationError(
                                ErrorCode.CONFLICT,
                                f"Target {target} appears in phases {owner!r} and {phase.phase_id!r}",
                            )
                        )
            self._plans[plan.plan_id] = plan
            self._groups.setdefault(plan.activation_group, _GroupState(plan.activation_group))
            return plan

    def plans(self) -> tuple[PhasePlan, ...]:
        with self._lock:
            return tuple(self._plans.values())

    def validate_placement(self, contentions: Sequence[Any]) -> None:
        """Reject plans that cannot keep contending placement phases apart.

        ``contentions`` are ``PhaseContention`` records from the placement
        planner: allocations that share GPUs across phases. Every such set must
        map onto *distinct* phases of a *single* plan, otherwise two roles could
        legitimately be activated at once on the same slice.
        """
        with self._lock:
            index: dict[str, list[tuple[str, str]]] = {}
            for plan in self._plans.values():
                for phase in plan.phases:
                    for label in phase.placement_phases:
                        index.setdefault(label, []).append((plan.plan_id, phase.phase_id))
            for contention in contentions:
                labels = tuple(getattr(contention, "phases", ()))
                owners: set[str] = set()
                phase_ids: list[tuple[str, str]] = []
                for label in labels:
                    entries = index.get(label)
                    if not entries:
                        raise LifecycleError(
                            OperationError(
                                ErrorCode.INVALID_ARGUMENT,
                                f"Placement phase {label!r} shares GPUs but no registered plan activates it",
                            )
                        )
                    if len(entries) > 1:
                        raise LifecycleError(
                            OperationError(
                                ErrorCode.CONFLICT,
                                f"Placement phase {label!r} is claimed by several phases: {sorted(entries)}",
                            )
                        )
                    owners.add(entries[0][0])
                    phase_ids.append(entries[0])
                if len(owners) > 1:
                    raise LifecycleError(
                        OperationError(
                            ErrorCode.CONFLICT,
                            f"Contending placement phases {sorted(labels)} span plans {sorted(owners)}",
                        )
                    )
                if len({phase_id for _, phase_id in phase_ids}) != len(phase_ids):
                    raise LifecycleError(
                        OperationError(
                            ErrorCode.CONFLICT,
                            f"One phase claims contending placement phases {sorted(labels)}",
                        )
                    )

    # ------------------------------------------------------------------
    # Queries.
    # ------------------------------------------------------------------
    def get_activation_group(self, activation_group: str) -> ActivationGroupSnapshot:
        with self._lock:
            state = self._groups.get(activation_group)
            if state is None:
                raise LifecycleError(
                    OperationError(ErrorCode.NOT_FOUND, f"Unknown activation group: {activation_group}")
                )
            return ActivationGroupSnapshot(
                activation_group=state.activation_group,
                coordinator_epoch=self.coordinator_epoch,
                plan_id=state.plan_id,
                phase_id=state.phase_id,
                active=state.active,
                exclusive_operation_id=state.exclusive_operation_id,
                running_operation_id=state.running_operation_id,
                last_operation_id=state.last_operation_id,
                release_confirmed=state.release_confirmed,
                blocked_reason=state.blocked_reason,
                blocked_by=state.blocked_by,
            )

    def get_operation(self, operation_id: str) -> OperationResult:
        if not operation_id:
            raise LifecycleError(OperationError(ErrorCode.INVALID_ARGUMENT, "An operation ID is required"))
        with self._lock:
            record = self._operations.get(operation_id)
            if record is None:
                raise LifecycleError(OperationError(ErrorCode.NOT_FOUND, f"Unknown operation: {operation_id}"))
            return record[1]

    # ------------------------------------------------------------------
    # Public transitions.
    # ------------------------------------------------------------------
    def switch_model(
        self,
        activation_group: str,
        target: ModelRef | Sequence[ModelRef],
        *,
        operation_id: str,
        timeout_s: float | None = None,
        tags: list[str] | None = None,
    ) -> SwitchResult:
        """Make ``target`` the sole occupant of ``activation_group``.

        The current occupant comes from this Coordinator's own state, not from a
        caller-supplied ``source_model``: a stale source would deactivate the
        wrong role. Switching never edits routing or the default model.
        """
        targets = (target,) if isinstance(target, ModelRef) else tuple(target)
        return self._execute(
            activation_group,
            targets,
            kind="switch_model",
            operation_id=operation_id,
            timeout_s=timeout_s,
            tags=tags,
        )

    def transition(
        self,
        activation_group: str,
        target_phase: str,
        *,
        operation_id: str,
        timeout_s: float | None = None,
        tags: list[str] | None = None,
    ) -> PhaseResult:
        """Activate a whole phase of the plan that owns ``activation_group``."""
        plan, phase = self._resolve_phase_by_group(activation_group, target_phase)
        return self._execute(
            activation_group,
            phase.targets,
            kind="transition",
            operation_id=operation_id,
            timeout_s=timeout_s,
            tags=tags,
            plan_id=plan.plan_id,
            phase_id=phase.phase_id,
        )

    def enter_phase(
        self,
        plan_id: str,
        phase_id: str,
        *,
        operation_id: str,
        timeout_s: float | None = None,
        tags: list[str] | None = None,
    ) -> PhaseHandle:
        """Activate a phase and hold the group exclusively for this operation.

        The target set comes from the registered plan, so every model of a
        multi-teacher phase is activated together instead of switching them one
        by one and unloading each other.
        """
        plan = self._plan(plan_id)
        phase = plan.phase(phase_id)
        result = self._execute(
            plan.activation_group,
            phase.targets,
            kind="enter_phase",
            operation_id=operation_id,
            timeout_s=timeout_s,
            tags=tags,
            plan_id=plan.plan_id,
            phase_id=phase.phase_id,
            exclusive=True,
        )
        if not result.succeeded:
            raise LifecycleError(
                result.error
                or OperationError(ErrorCode.UNAVAILABLE, f"Phase {phase_id} of plan {plan_id} did not activate")
            )
        return PhaseHandle(
            plan_id=plan.plan_id,
            phase_id=phase.phase_id,
            token=self._token(plan.activation_group, operation_id, phase.targets),
        )

    def finish_phase(
        self,
        handle: PhaseHandle,
        *,
        operation_id: str,
        outcome: str = "completed",
        timeout_s: float | None = None,
    ) -> PhaseResult:
        """Close the phase: stop admission, drain, deactivate, confirm release.

        It never restores the previous occupant and never picks the next phase.
        A repeated call returns the original result.
        """
        if outcome not in ("completed", "failed", "cancelled"):
            raise LifecycleError(OperationError(ErrorCode.INVALID_ARGUMENT, f"Unknown phase outcome: {outcome}"))
        self._require_epoch(handle.token)
        with self._lock:
            state = self._group(handle.activation_group)
            previous = state.finished_phases.get(operation_id)
            if previous is not None:
                return previous
            if state.exclusive_operation_id not in (None, handle.operation_id):
                raise LifecycleError(
                    OperationError(
                        ErrorCode.BUSY,
                        f"Activation group {handle.activation_group} is held by "
                        f"operation {state.exclusive_operation_id}",
                    )
                )
            if state.running_operation_id not in (None, operation_id):
                raise LifecycleError(
                    OperationError(
                        ErrorCode.BUSY, f"Operation {state.running_operation_id} is still running on this group"
                    )
                )
            state.running_operation_id = operation_id
            targets = handle.token.targets
        token = self._token(handle.activation_group, operation_id, targets)
        step: str | None = None
        error: OperationError | None = None
        release_confirmed = False
        try:
            drained = self._manager.drain(
                targets, operation_id=f"{operation_id}:drain", activation_token=token, timeout_s=timeout_s
            )
            step = drained.last_confirmed_step or (STEP_DRAINED if drained.succeeded else STEP_ADMISSION_CLOSED)
            if not drained.succeeded:
                error = drained.error or OperationError(
                    ErrorCode.TIMEOUT, "Drain did not confirm in-flight completion"
                )
            else:
                deactivated = self._manager.deactivate(
                    targets, operation_id=f"{operation_id}:deactivate", activation_token=token
                )
                release_confirmed = deactivated.release_confirmed
                step = STEP_RELEASE_CONFIRMED if release_confirmed else STEP_DEACTIVATED
                if not deactivated.succeeded:
                    error = deactivated.error or OperationError(ErrorCode.UNAVAILABLE, "Deactivation did not complete")
                elif not release_confirmed:
                    error = OperationError(
                        ErrorCode.UNKNOWN_COMPLETION, "Deactivation did not confirm the memory release"
                    )
        except Exception as exc:  # A raising adapter must not lose the group.
            error = OperationError(ErrorCode.UNAVAILABLE, f"{type(exc).__name__}: {exc}")
        result = PhaseResult(
            operation_id=operation_id,
            owner_epoch=self.coordinator_epoch,
            status=OperationStatus.COMPLETED if error is None else OperationStatus.FAILED,
            kind="finish_phase",
            targets=targets,
            last_confirmed_step=step,
            error=error,
            release_confirmed=release_confirmed,
            plan_id=handle.plan_id,
            phase_id=handle.phase_id,
            activation_group=handle.activation_group,
        )
        with self._lock:
            state = self._group(handle.activation_group)
            state.running_operation_id = None
            state.last_operation_id = operation_id
            state.finished_phases[operation_id] = result
            state.release_confirmed = release_confirmed
            state.blocked_by = None if release_confirmed else "inference"
            if release_confirmed:
                state.active = ()
                state.phase_id = None
                state.exclusive_operation_id = None
                state.blocked_reason = None
            else:
                # Keep the claim: the slice is not proven free, so nothing else
                # may be activated on it.
                state.blocked_reason = error.message if error is not None else "release not confirmed"
            self._operations[operation_id] = ((handle.activation_group, "finish_phase", targets), result)
        return result

    def adopt_phase(self, activation_group: str, phase_id: str, *, operation_id: str) -> PhaseResult:
        """Record a phase whose engines an external owner already activated.

        Migration reality: a role whose runtime still lives outside this control
        plane brings its own engines up (rollout restores weights and KV in two
        stages through the training path). The Coordinator must still know the
        slice is occupied, or the next phase switch would leave those engines
        resident and put two roles on the same GPUs.

        This performs no activation and claims no release. It refuses when the
        group already has a different occupant, because that would record two
        occupants for one slice.
        """
        plan, phase = self._resolve_phase_by_group(activation_group, phase_id)
        with self._lock:
            state = self._group(activation_group)
            recorded = self._operations.get(operation_id)
            signature = (activation_group, "adopt_phase", phase.targets, plan.plan_id, phase.phase_id)
            if recorded is not None:
                if recorded[0] != signature:
                    raise LifecycleError(
                        OperationError(ErrorCode.CONFLICT, f"Operation {operation_id} was used for another transition")
                    )
                return recorded[1]  # type: ignore[return-value]
            if state.running_operation_id is not None:
                raise LifecycleError(
                    OperationError(
                        ErrorCode.BUSY,
                        f"Operation {state.running_operation_id} is running on activation group {activation_group}",
                    )
                )
            if not state.release_confirmed:
                raise LifecycleError(
                    OperationError(
                        ErrorCode.UNAVAILABLE,
                        f"Activation group {activation_group} did not confirm its release: {state.blocked_reason}",
                    )
                )
            if state.active and set(state.active) != set(phase.targets):
                raise LifecycleError(
                    OperationError(
                        ErrorCode.CONFLICT,
                        f"Activation group {activation_group} is occupied by "
                        f"{sorted(str(target) for target in state.active)}",
                    )
                )
            result = PhaseResult(
                operation_id=operation_id,
                owner_epoch=self.coordinator_epoch,
                status=OperationStatus.COMPLETED,
                kind="adopt_phase",
                targets=phase.targets,
                last_confirmed_step=STEP_ACTIVATED,
                plan_id=plan.plan_id,
                phase_id=phase.phase_id,
                activation_group=activation_group,
            )
            state.active = phase.targets
            state.plan_id = plan.plan_id
            state.phase_id = phase.phase_id
            state.release_confirmed = True
            state.blocked_reason = None
            state.blocked_by = None
            state.last_operation_id = operation_id
            self._operations[operation_id] = (signature, result)
            return result

    def activation_groups(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._groups))

    def phase_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted({phase.phase_id for plan in self._plans.values() for phase in plan.phases}))

    # ------------------------------------------------------------------
    # Training handoff.
    # ------------------------------------------------------------------
    def prepare_training_handoff(
        self,
        activation_group: str,
        adapter: TrainingResourceAdapter,
        batch_id: str,
        policy_version: str | None,
        *,
        operation_id: str,
    ) -> HandoffResult:
        """Grant training the group once the inference side released it."""
        with self._lock:
            state = self._group(activation_group)
            if not state.release_confirmed:
                return HandoffResult(
                    operation_id,
                    batch_id,
                    False,
                    policy_version,
                    OperationError(
                        ErrorCode.UNAVAILABLE,
                        f"Activation group {activation_group} has not confirmed its release: {state.blocked_reason}",
                    ),
                )
            token = self._token(activation_group, operation_id, state.active)
        return adapter.prepare_training_handoff(
            batch_id, policy_version, operation_id=operation_id, activation_token=token
        )

    def release_training_resources(
        self, activation_group: str, adapter: TrainingResourceAdapter, *, operation_id: str
    ) -> ReleaseEvidence:
        """Confirm every training rank released the group before the next
        activation."""
        with self._lock:
            state = self._group(activation_group)
            token = self._token(activation_group, operation_id, state.active)
        evidence = adapter.release_training_resources(operation_id=operation_id, activation_token=token)
        with self._lock:
            state = self._group(activation_group)
            state.release_confirmed = evidence.release_confirmed
            state.blocked_by = None if evidence.release_confirmed else "training"
            state.blocked_reason = (
                None
                if evidence.release_confirmed
                else (
                    evidence.error.message
                    if evidence.error is not None
                    else f"{len(evidence.confirmed_ranks)}/{evidence.total_ranks} training ranks confirmed"
                )
            )
        return evidence

    # ------------------------------------------------------------------
    # Internals.
    # ------------------------------------------------------------------
    def _token(self, activation_group: str, operation_id: str, targets: Sequence[ModelRef]) -> ActivationToken:
        return ActivationToken(
            session_id=self._session_id,
            coordinator_epoch=self.coordinator_epoch,
            activation_group=activation_group,
            operation_id=operation_id,
            targets=tuple(targets),
        )

    def _require_epoch(self, token: ActivationToken) -> None:
        if token.coordinator_epoch != self.coordinator_epoch:
            raise LifecycleError(
                OperationError(ErrorCode.STALE_GENERATION, "Activation token belongs to an earlier coordinator")
            )
        if token.session_id != self._session_id:
            raise LifecycleError(OperationError(ErrorCode.INVALID_ARGUMENT, "Activation token is for another session"))

    def _plan(self, plan_id: str) -> PhasePlan:
        with self._lock:
            plan = self._plans.get(plan_id)
        if plan is None:
            raise LifecycleError(OperationError(ErrorCode.NOT_FOUND, f"Unknown phase plan: {plan_id}"))
        return plan

    def _resolve_phase_by_group(self, activation_group: str, phase_id: str) -> tuple[PhasePlan, PhaseSpec]:
        with self._lock:
            plans = [plan for plan in self._plans.values() if plan.activation_group == activation_group]
        if not plans:
            raise LifecycleError(
                OperationError(ErrorCode.NOT_FOUND, f"No phase plan owns activation group {activation_group}")
            )
        return plans[0], plans[0].phase(phase_id)

    def _group(self, activation_group: str) -> _GroupState:
        state = self._groups.get(activation_group)
        if state is None:
            state = _GroupState(activation_group)
            self._groups[activation_group] = state
        return state

    def _all_ready(self, targets: Sequence[ModelRef]) -> bool:
        """Only a complete READY observation makes a switch a no-op."""
        if self._readiness is None:
            return False
        for target in targets:
            try:
                state = self._readiness(target) if callable(self._readiness) else self._readiness[target]
            except Exception:
                return False
            if LifecycleState(state) is not LifecycleState.READY:
                return False
        return True

    def _execute(
        self,
        activation_group: str,
        targets: Sequence[ModelRef],
        *,
        kind: str,
        operation_id: str,
        timeout_s: float | None,
        tags: list[str] | None,
        plan_id: str | None = None,
        phase_id: str | None = None,
        exclusive: bool = False,
    ) -> PhaseResult:
        if not operation_id:
            raise LifecycleError(OperationError(ErrorCode.INVALID_ARGUMENT, "An operation ID is required"))
        targets = tuple(targets)
        if len(set(targets)) != len(targets):
            raise LifecycleError(OperationError(ErrorCode.INVALID_ARGUMENT, "Duplicate activation targets"))
        signature = (activation_group, kind, targets, plan_id, phase_id)
        with self._lock:
            recorded = self._operations.get(operation_id)
            if recorded is not None:
                if recorded[0] != signature:
                    raise LifecycleError(
                        OperationError(ErrorCode.CONFLICT, f"Operation {operation_id} was used for another transition")
                    )
                return recorded[1]  # type: ignore[return-value]
            state = self._group(activation_group)
            if state.running_operation_id is not None:
                # A different intent does not queue behind the running one: a
                # stale intent must not be applied after it stopped being true.
                raise LifecycleError(
                    OperationError(
                        ErrorCode.BUSY,
                        f"Operation {state.running_operation_id} is running on activation group {activation_group}",
                    )
                )
            if state.exclusive_operation_id is not None and state.exclusive_operation_id != operation_id:
                raise LifecycleError(
                    OperationError(
                        ErrorCode.BUSY,
                        f"Activation group {activation_group} is held exclusively by "
                        f"operation {state.exclusive_operation_id}",
                    )
                )
            if not state.release_confirmed and state.blocked_by == "training":
                # Nothing this control plane can drain will free the training
                # ranks' memory, so the slice stays closed until the training
                # adapter confirms its own release.
                raise LifecycleError(
                    OperationError(
                        ErrorCode.UNAVAILABLE,
                        f"Activation group {activation_group} did not confirm its release: {state.blocked_reason}",
                    )
                )
            state.running_operation_id = operation_id
            current = state.active
            stale = tuple(target for target in current if target not in targets)
            if not state.release_confirmed:
                # A previous inference attempt left the slice unproven. Re-drain
                # and re-release everything recorded as occupying it, including a
                # target being re-activated: retrying the release is the recovery
                # path, and refusing outright would leave the group permanently
                # stuck once a late request finally finished.
                stale = tuple(current)
            blocked = not state.release_confirmed
        result = self._run_transition(
            activation_group,
            targets,
            stale,
            kind=kind,
            operation_id=operation_id,
            timeout_s=timeout_s,
            tags=tags,
            plan_id=plan_id,
            phase_id=phase_id,
            no_op=(not stale and not blocked and set(current) == set(targets) and self._all_ready(targets)),
        )
        with self._lock:
            state = self._group(activation_group)
            state.running_operation_id = None
            state.last_operation_id = operation_id
            self._operations[operation_id] = (signature, result)
            if result.succeeded:
                state.active = targets
                state.plan_id = plan_id or state.plan_id
                state.phase_id = phase_id
                state.release_confirmed = True
                state.blocked_reason = None
                state.blocked_by = None
                if exclusive:
                    state.exclusive_operation_id = operation_id
            else:
                state.release_confirmed = result.release_confirmed or not stale
                state.blocked_reason = result.error.message if result.error is not None else None
                state.blocked_by = None if state.release_confirmed else "inference"
                if result.last_confirmed_step in (STEP_RELEASE_CONFIRMED, STEP_ACTIVATION_GRANTED):
                    # The old occupant is gone; do not keep claiming it.
                    state.active = ()
                    state.phase_id = None
        return result

    def _run_transition(
        self,
        activation_group: str,
        targets: tuple[ModelRef, ...],
        stale: tuple[ModelRef, ...],
        *,
        kind: str,
        operation_id: str,
        timeout_s: float | None,
        tags: list[str] | None,
        plan_id: str | None,
        phase_id: str | None,
        no_op: bool,
    ) -> PhaseResult:
        def finish(
            status: OperationStatus,
            step: str | None,
            error: OperationError | None = None,
            release_confirmed: bool = False,
        ) -> PhaseResult:
            return PhaseResult(
                operation_id=operation_id,
                owner_epoch=self.coordinator_epoch,
                status=status,
                kind=kind,
                targets=targets,
                last_confirmed_step=step,
                error=error,
                release_confirmed=release_confirmed,
                plan_id=plan_id,
                phase_id=phase_id,
                activation_group=activation_group,
            )

        if no_op:
            return finish(OperationStatus.COMPLETED, STEP_READY_PUBLISHED, release_confirmed=True)
        # Each leg carries a token covering exactly the models it touches: the
        # models being evicted, then the models being activated.
        eviction_token = self._token(activation_group, operation_id, stale)
        token = self._token(activation_group, operation_id, targets)
        release_confirmed = not stale
        step: str | None = None
        try:
            if stale:
                drained = self._manager.drain(
                    stale, operation_id=f"{operation_id}:drain", activation_token=eviction_token, timeout_s=timeout_s
                )
                step = drained.last_confirmed_step or (STEP_DRAINED if drained.succeeded else STEP_ADMISSION_CLOSED)
                if not drained.succeeded:
                    return finish(
                        OperationStatus.FAILED,
                        step,
                        drained.error or OperationError(ErrorCode.TIMEOUT, "Drain did not confirm completion"),
                    )
                deactivated = self._manager.deactivate(
                    stale, operation_id=f"{operation_id}:deactivate", activation_token=eviction_token
                )
                release_confirmed = deactivated.release_confirmed
                step = STEP_RELEASE_CONFIRMED if release_confirmed else STEP_DEACTIVATED
                if not deactivated.succeeded or not release_confirmed:
                    return finish(
                        OperationStatus.FAILED,
                        step,
                        deactivated.error
                        or OperationError(
                            ErrorCode.UNKNOWN_COMPLETION, "Deactivation did not confirm the memory release"
                        ),
                        release_confirmed=release_confirmed,
                    )
            step = STEP_ACTIVATION_GRANTED
            if not targets:
                return finish(OperationStatus.COMPLETED, step, release_confirmed=release_confirmed)
            activated = self._manager.activate(
                targets, operation_id=f"{operation_id}:activate", activation_token=token, tags=tags
            )
            step = activated.last_confirmed_step or STEP_ACTIVATED
            if not activated.succeeded:
                return finish(
                    OperationStatus.FAILED,
                    step,
                    activated.error or OperationError(ErrorCode.UNAVAILABLE, "Activation did not complete"),
                    release_confirmed=release_confirmed,
                )
            return finish(OperationStatus.COMPLETED, step, release_confirmed=release_confirmed)
        except Exception as exc:
            return finish(
                OperationStatus.FAILED,
                step,
                OperationError(ErrorCode.UNAVAILABLE, f"{type(exc).__name__}: {exc}"),
                release_confirmed=release_confirmed,
            )


def phase_plan_from_placement(
    plan_id: str,
    activation_group: str,
    phase_targets: Mapping[str, Sequence[ModelRef]],
    *,
    version: int = 1,
) -> PhasePlan:
    """Build a plan whose phase IDs are the planner's phase labels."""
    return PhasePlan(
        plan_id=plan_id,
        activation_group=activation_group,
        phases=tuple(PhaseSpec(phase_id, tuple(targets), (phase_id,)) for phase_id, targets in phase_targets.items()),
        version=version,
    )


def targets_from_snapshots(snapshots: Mapping[Role, Any]) -> tuple[ModelRef, ...]:
    """Project role snapshots to the model references a plan addresses."""
    return tuple(ModelRef(role, model.model_id) for role, snapshot in snapshots.items() for model in snapshot.models)


__all__ = [
    "ActivationGroupSnapshot",
    "ActivationToken",
    "ErrorCode",
    "HandoffResult",
    "LIFECYCLE_STEPS",
    "LifecycleCoordinator",
    "LifecycleError",
    "LifecycleManager",
    "OperationError",
    "OperationResult",
    "OperationStatus",
    "PhaseHandle",
    "PhasePlan",
    "PhaseResult",
    "PhaseSpec",
    "ReleaseEvidence",
    "STEP_ACTIVATED",
    "STEP_ACTIVATION_GRANTED",
    "STEP_ADMISSION_CLOSED",
    "STEP_DEACTIVATED",
    "STEP_DRAINED",
    "STEP_READY_PUBLISHED",
    "STEP_RELEASE_CONFIRMED",
    "SwitchResult",
    "TrainingResourceAdapter",
    "phase_plan_from_placement",
    "targets_from_snapshots",
]
