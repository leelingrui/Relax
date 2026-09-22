# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Deferred scoring: move teacher prefill out of generation, gate publication.

The immediate path scores each sample inline while it is generated, which keeps
the teacher resident for the whole generation. Deferred scoring instead stages
whole batches, scores them once generation has finished, validates the result and
only then publishes -- so the same GPUs can run student, then teacher, then
trainer, one at a time.

Two rules shape everything here:

* Nothing half-scored is publishable. A batch either has every required training
  field for every eligible sample, or it fails. Publishing the successful part
  would put rows in the queue that the loss treats as complete distillation
  targets when they are not.
* Staging happens while generation is still in flight, scoring does not. The
  student cannot be offloaded for the teacher until the last request is done,
  which is why batches are staged during the step and flushed after it.

This executor belongs to the rollout workload: the Coordinator owns phases, the
Manager owns models, and this owns samples and the queue handoff.
"""

import asyncio
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Sequence

from relax.engine.inference.lifecycle import ErrorCode, OperationError
from relax.utils.logging_utils import get_logger
from relax.utils.types import Sample


logger = get_logger(__name__)


class DeferredState(str, Enum):
    """The fixed order a deferred batch passes through."""

    STAGED = "staged"
    WAITING_PHASE = "waiting_phase"
    SCORING = "scoring"
    VALIDATING = "validating"
    GPU_RELEASE_CONFIRMED = "gpu_release_confirmed"
    TRAIN_HANDOFF_GRANTED = "train_handoff_granted"
    PUBLISHING = "publishing"
    PRODUCTION_COMPLETE = "production_complete"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCEL_PENDING = "cancel_pending"
    CANCELLED = "cancelled"


DEFERRED_ORDER = (
    DeferredState.STAGED,
    DeferredState.WAITING_PHASE,
    DeferredState.SCORING,
    DeferredState.VALIDATING,
    DeferredState.GPU_RELEASE_CONFIRMED,
    DeferredState.TRAIN_HANDOFF_GRANTED,
    DeferredState.PUBLISHING,
    DeferredState.PRODUCTION_COMPLETE,
    DeferredState.COMPLETED,
)


@dataclass(frozen=True)
class SampleRef:
    """The sealed description of one sample's scoring contract.

    Sealed at submit time and kept until a terminal state: it is what results
    are correlated against, so a late or duplicated response cannot be mistaken
    for the sample it claims to be.
    """

    sample_index: Any
    group_index: int | None
    response_length: int
    prompt_length: int
    route_key: str | None = None
    has_multimodal: bool = False

    @property
    def eligible(self) -> bool:
        """An empty response has nothing to score and no fields to validate."""
        return self.response_length > 0


@dataclass(frozen=True)
class BatchRef:
    """A batch as sealed at submit time."""

    batch_id: str
    rollout_id: int
    policy_version: str | None
    token_selection: str
    required_fields: tuple[str, ...]
    samples: tuple[SampleRef, ...]

    @property
    def eligible_count(self) -> int:
        return sum(1 for sample in self.samples if sample.eligible)


@dataclass(frozen=True)
class DeferredHandle:
    operation_id: str
    batch_id: str


@dataclass(frozen=True)
class DeferredSnapshot:
    operation_id: str
    batch_id: str
    state: DeferredState
    last_confirmed_step: DeferredState | None = None
    scored: int = 0
    published: int = 0
    missing: tuple[Any, ...] = ()
    error: OperationError | None = None

    @property
    def terminal(self) -> bool:
        return self.state in (DeferredState.COMPLETED, DeferredState.FAILED, DeferredState.CANCELLED)


DeferredResult = DeferredSnapshot


@dataclass
class _Record:
    handle: DeferredHandle
    batch_ref: BatchRef
    plan_id: str | None
    # Flat samples for scoring and validation, and the caller's original nesting
    # for publication: the queue helper derives its ordering from the groups.
    samples: list[Sample]
    payload: Any
    is_last: bool
    snapshot: DeferredSnapshot
    task: asyncio.Task | None = None
    cancelled: bool = field(default=False)


def _leading_length(value: Any) -> int | None:
    """Rows in a scoring field, for list, tuple and array-like values."""
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    if shape is not None:
        return int(shape[0]) if len(shape) else 0
    try:
        return len(value)
    except TypeError:
        return None


def validate_scored_batch(batch_ref: BatchRef, samples: Sequence[Sample]) -> tuple[tuple[Any, ...], list[str]]:
    """Check the scored batch against the sealed reference.

    Returns the sample identities that are not publishable and the reasons. It
    verifies presence, order, per-field row counts against the response length,
    and that the loss mask still matches -- a field that is one row short would
    otherwise silently misalign every token after it.
    """
    problems: list[str] = []
    missing: list[Any] = []
    if len(samples) != len(batch_ref.samples):
        problems.append(f"batch size changed: sealed {len(batch_ref.samples)}, scored {len(samples)}")
        return tuple(ref.sample_index for ref in batch_ref.samples), problems

    seen: set[Any] = set()
    for ref, sample in zip(batch_ref.samples, samples, strict=True):
        identity = ref.sample_index
        if identity in seen:
            problems.append(f"sample {identity} appears twice in the scored batch")
            missing.append(identity)
            continue
        seen.add(identity)
        if sample.index != ref.sample_index:
            problems.append(f"sample order changed: sealed {ref.sample_index}, scored {sample.index}")
            missing.append(identity)
            continue
        response_length = int(sample.response_length or 0)
        if response_length != ref.response_length:
            problems.append(
                f"sample {identity} response length changed: sealed {ref.response_length}, now {response_length}"
            )
            missing.append(identity)
            continue
        if not ref.eligible:
            continue
        mask = getattr(sample, "loss_mask", None)
        if mask is not None and len(mask) != response_length:
            problems.append(f"sample {identity} loss mask covers {len(mask)} of {response_length} tokens")
            missing.append(identity)
            continue
        failed_field = None
        for name in batch_ref.required_fields:
            rows = _leading_length(getattr(sample, name, None))
            if rows is None:
                failed_field = f"sample {identity} is missing {name}"
                break
            if rows != response_length:
                failed_field = f"sample {identity} field {name} has {rows} rows, expected {response_length}"
                break
        if failed_field is not None:
            problems.append(failed_field)
            missing.append(identity)
    return tuple(missing), problems


def seal_batch(
    args: Any,
    samples: Sequence[Sample],
    *,
    batch_id: str,
    rollout_id: int,
    required_fields: Sequence[str],
    policy_version: str | None = None,
) -> BatchRef:
    """Seal what the scorer must produce for this batch."""
    route_key_field = getattr(args, "opd_teacher_key", None) or "data_source"
    refs = []
    for sample in samples:
        metadata = getattr(sample, "metadata", None) or {}
        multimodal = getattr(sample, "multimodal_inputs", None)
        response_length = int(sample.response_length or 0)
        refs.append(
            SampleRef(
                sample_index=sample.index,
                group_index=getattr(sample, "group_index", None),
                response_length=response_length,
                prompt_length=max(len(sample.tokens or ()) - response_length, 0),
                route_key=metadata.get(route_key_field) if isinstance(metadata, dict) else None,
                has_multimodal=bool(multimodal),
            )
        )
    return BatchRef(
        batch_id=batch_id,
        rollout_id=rollout_id,
        policy_version=policy_version,
        token_selection=str(getattr(args, "opd_token_selection", "")),
        required_fields=tuple(required_fields),
        samples=tuple(refs),
    )


class DeferredExecutor:
    """Run one deferred batch at a time, and never publish an incomplete
    one."""

    def __init__(self, args: Any) -> None:
        self.args = args
        self._records: dict[str, _Record] = {}
        # One batch at a time: the first version deliberately has no cross-batch
        # pipeline, because two batches would contend for the scoring phase.
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public contract.
    # ------------------------------------------------------------------
    def submit_deferred(
        self,
        batch_ref: BatchRef,
        plan_id: str | None,
        *,
        operation_id: str,
        samples: Sequence[Sample],
        payload: Any = None,
        is_last: bool = False,
    ) -> DeferredHandle:
        """Register a staged batch.

        This does not make it trainable.
        """
        if not operation_id:
            raise ValueError("A deferred operation ID is required")
        existing = self._records.get(operation_id)
        if existing is not None:
            if existing.batch_ref != batch_ref or existing.plan_id != plan_id:
                raise ValueError(f"Deferred operation {operation_id} was submitted with different inputs")
            return existing.handle
        handle = DeferredHandle(operation_id, batch_ref.batch_id)
        self._records[operation_id] = _Record(
            handle=handle,
            batch_ref=batch_ref,
            plan_id=plan_id,
            samples=list(samples),
            payload=payload if payload is not None else list(samples),
            is_last=is_last,
            snapshot=DeferredSnapshot(operation_id, batch_ref.batch_id, DeferredState.STAGED),
        )
        return handle

    def get_deferred(self, operation_id: str) -> DeferredSnapshot:
        record = self._records.get(operation_id)
        if record is None:
            raise KeyError(f"Unknown deferred operation: {operation_id}")
        return record.snapshot

    def cancel_deferred(self, operation_id: str) -> DeferredSnapshot:
        """Stop further scoring requests; stay ``cancel_pending`` until
        confirmed."""
        record = self._records.get(operation_id)
        if record is None:
            raise KeyError(f"Unknown deferred operation: {operation_id}")
        record.cancelled = True
        if record.snapshot.terminal:
            return record.snapshot
        if record.snapshot.state is DeferredState.STAGED:
            # Nothing was sent, so the cancellation is already confirmed.
            record.snapshot = replace(record.snapshot, state=DeferredState.CANCELLED)
            return record.snapshot
        record.snapshot = replace(record.snapshot, state=DeferredState.CANCEL_PENDING)
        return record.snapshot

    async def wait_deferred(self, handle: DeferredHandle, *, timeout_s: float | None = None) -> DeferredResult:
        """Wait for a terminal state; a timeout ends the wait, not the
        batch."""
        record = self._records[handle.operation_id]
        if record.task is None:
            return record.snapshot
        try:
            await asyncio.wait_for(asyncio.shield(record.task), timeout=timeout_s)
        except asyncio.TimeoutError:
            return record.snapshot
        return record.snapshot

    def operations(self) -> tuple[str, ...]:
        return tuple(self._records)

    # ------------------------------------------------------------------
    # Execution.
    # ------------------------------------------------------------------
    def start(
        self,
        handle: DeferredHandle,
        *,
        score: Callable[[list[Sample]], Any],
        publish: Callable[[Any, bool], Any],
    ) -> asyncio.Task:
        record = self._records[handle.operation_id]
        if record.task is None:
            record.task = asyncio.create_task(self._run(record, score, publish))
        return record.task

    def _advance(self, record: _Record, state: DeferredState, **changes: Any) -> None:
        record.snapshot = replace(record.snapshot, state=state, last_confirmed_step=state, **changes)

    def _fail(self, record: _Record, code: ErrorCode, message: str, **changes: Any) -> None:
        record.snapshot = replace(
            record.snapshot, state=DeferredState.FAILED, error=OperationError(code, message), **changes
        )

    async def _run(
        self,
        record: _Record,
        score: Callable[[list[Sample]], Any],
        publish: Callable[[Any, bool], Any],
    ) -> DeferredSnapshot:
        async with self._lock:
            if record.cancelled:
                self._advance(record, DeferredState.CANCELLED)
                return record.snapshot
            self._advance(record, DeferredState.WAITING_PHASE)
            try:
                self._advance(record, DeferredState.SCORING)
                failures = await score(record.samples)
            except Exception as exc:
                self._fail(record, ErrorCode.UNAVAILABLE, f"{type(exc).__name__}: {exc}")
                logger.exception(f"Deferred scoring failed for batch {record.batch_ref.batch_id}")
                return record.snapshot
            if record.cancelled:
                # The requests already sent were confirmed finished by ``score``.
                self._advance(record, DeferredState.CANCELLED)
                return record.snapshot
            self._advance(record, DeferredState.VALIDATING)
            missing, problems = validate_scored_batch(record.batch_ref, record.samples)
            unscored = tuple(dict.fromkeys(tuple(failures or ()) + missing))
            if unscored or problems:
                # Successful samples are kept in the record for diagnosis, but
                # the batch does not publish: a partially scored batch would
                # train rows whose distillation targets are absent.
                detail = "; ".join(problems[:5]) if problems else f"{len(unscored)} sample(s) were not scored"
                self._fail(
                    record,
                    ErrorCode.UNKNOWN_COMPLETION,
                    f"Deferred scoring is incomplete: {detail}",
                    scored=record.batch_ref.eligible_count - len(unscored),
                    missing=unscored,
                )
                logger.error(
                    f"Deferred batch {record.batch_ref.batch_id} not published: "
                    f"{len(unscored)} of {record.batch_ref.eligible_count} samples unscored; {problems[:5]}"
                )
                return record.snapshot
            self._advance(record, DeferredState.GPU_RELEASE_CONFIRMED, scored=record.batch_ref.eligible_count)
            self._advance(record, DeferredState.TRAIN_HANDOFF_GRANTED)
            self._advance(record, DeferredState.PUBLISHING)
            try:
                await publish(record.payload, record.is_last)
            except Exception as exc:
                self._fail(record, ErrorCode.UNAVAILABLE, f"publication failed: {type(exc).__name__}: {exc}")
                return record.snapshot
            self._advance(record, DeferredState.PRODUCTION_COMPLETE, published=len(record.samples))
            self._advance(record, DeferredState.COMPLETED, published=len(record.samples))
            return record.snapshot


__all__ = [
    "DEFERRED_ORDER",
    "BatchRef",
    "DeferredExecutor",
    "DeferredHandle",
    "DeferredResult",
    "DeferredSnapshot",
    "DeferredState",
    "SampleRef",
    "seal_batch",
    "validate_scored_batch",
]
