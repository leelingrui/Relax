# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Run a scoring stage inside a coordinated activation phase.

Deferred scoring needs the scorer's engines resident and the generation engines
asleep on a shared slice. That switch belongs to the framework: a user script
cannot close model admission, wait for in-flight generation or confirm that the
memory was really released, and a script that reaches for a well-known Ray actor
to do it bypasses the control plane entirely.

When the task has no coordinator -- because the layout gave every role its own
GPUs, or because scoring is inline -- the context manager does nothing and the
caller's existing path runs unchanged.
"""

import asyncio
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncIterator, Iterator
from uuid import uuid4

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

# Generation has finished before scoring starts, so the drain should be empty.
SCORING_DRAIN_TIMEOUT_S = 600.0


def _task_inference_manager() -> Any:
    """Find the task control plane from inside the rollout actor process."""
    try:
        from relax.distributed.ray.rollout import get_local_rollout_manager

        return getattr(get_local_rollout_manager(), "task_inference_manager", None)
    except Exception:
        return None


@contextmanager
def scoring_phase(args: Any, phase_id: str, *, batch_id: str | None = None) -> Iterator[Any]:
    """Hold ``phase_id`` exclusively for the duration of a scoring stage.

    Yields the ``PhaseHandle``, or ``None`` when this deployment does not
    sequence the phase. Entering drains and deactivates whatever occupied the
    slice and confirms the release before the scorer is activated; leaving drains
    and deactivates the scorer again. It never restores the previous occupant:
    the training path onloads generation when it synchronizes weights, and an
    extra restore here would cost a full weights and KV round trip.
    """
    del args
    from relax.distributed.ray.lifecycle_client import phase_client

    client = phase_client(_task_inference_manager(), phases=(phase_id,))
    if client is None:
        yield None
        return
    operation_id = f"score:{phase_id}:{batch_id or uuid4().hex}"
    handle = client.enter(phase_id, operation_id=operation_id, timeout_s=SCORING_DRAIN_TIMEOUT_S)
    if handle is None:
        yield None
        return
    logger.info(f"Entered scoring phase {phase_id} (operation={operation_id})")
    outcome = "completed"
    try:
        yield handle
    except Exception:
        outcome = "failed"
        raise
    finally:
        # The phase is closed on both paths: leaving the scorer resident would
        # deny the slice to training. A failed release is reported, not hidden,
        # because the next activation depends on it.
        result = client.finish(
            handle,
            operation_id=f"{operation_id}:finish",
            outcome=outcome,
            timeout_s=SCORING_DRAIN_TIMEOUT_S,
        )
        if not result.release_confirmed:
            raise RuntimeError(
                f"Scoring phase {phase_id} did not confirm its release: "
                f"step={result.last_confirmed_step} error={result.error}"
            )
        logger.info(f"Left scoring phase {phase_id} (operation={operation_id})")


@asynccontextmanager
async def async_scoring_phase(args: Any, phase_id: str, *, batch_id: str | None = None) -> AsyncIterator[Any]:
    """``scoring_phase`` for async callers.

    Entering and leaving block on Ray, so they run on a worker thread: holding
    the event loop through a drain would stop the very completion reports the
    drain is waiting for.
    """
    context = scoring_phase(args, phase_id, batch_id=batch_id)
    handle = await asyncio.to_thread(context.__enter__)
    try:
        yield handle
    except BaseException as exc:
        if await asyncio.to_thread(context.__exit__, type(exc), exc, exc.__traceback__):
            return
        raise
    else:
        await asyncio.to_thread(context.__exit__, None, None, None)


async def async_activate_phase(phase_id: str, *, operation_id: str, timeout_s: float | None = None) -> Any:
    """Activate ``phase_id`` from async code, or return ``None`` if unsequenced.

    Used for the follow-up stage of a token selection that needs a second pass on
    the student after the teacher scored: the student was deliberately offloaded
    for the teacher, so it has to be brought back explicitly.
    """
    from relax.distributed.ray.lifecycle_client import phase_client

    client = phase_client(_task_inference_manager(), phases=(phase_id,))
    if client is None:
        return None
    return await asyncio.to_thread(client.activate_phase, phase_id, operation_id=operation_id, timeout_s=timeout_s)


__all__ = ["SCORING_DRAIN_TIMEOUT_S", "async_activate_phase", "async_scoring_phase", "scoring_phase"]
