# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Run a deferred scoring stage on GPUs shared with generation.

Deferred scoring needs the scorer resident and the generation engines asleep on
a shared slice. The task InferenceManager performs that switch: it closes
admission, drains in-flight requests and releases memory before loading the
next role. Without a deferred layout these helpers do nothing.
"""

import asyncio
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncIterator, Iterator

import ray

from relax.engine.inference.phase_plans import (
    PHASE_GENRM,
    PHASE_TEACHER,
    deferred_genrm_enabled,
    deferred_opd_enabled,
)
from relax.engine.inference.types import Role
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

# Generation has finished before scoring starts, so the drain should be empty.
SCORING_DRAIN_TIMEOUT_S = 600.0

_SCORER_ROLES = {
    PHASE_GENRM: (Role.GENRM, deferred_genrm_enabled),
    PHASE_TEACHER: (Role.TEACHER, deferred_opd_enabled),
}


def _task_inference_manager() -> Any:
    """Find the task InferenceManager from inside the rollout actor process."""
    from relax.distributed.ray.rollout import get_local_rollout_manager

    return getattr(get_local_rollout_manager(), "task_inference_manager", None)


def _switch(deactivate: list[Role], activate: list[Role]) -> None:
    manager = _task_inference_manager()
    if manager is None:
        raise RuntimeError("Deferred scoring requires the task InferenceManager")
    ray.get(manager.switch.remote(deactivate, activate, SCORING_DRAIN_TIMEOUT_S))


@contextmanager
def scoring_phase(args: Any, phase_id: str) -> Iterator[bool]:
    """Hold the GPUs for ``phase_id``'s scorer while the block runs.

    Yields whether a switch happened. Leaving puts the scorer back to sleep but
    does not restore generation: the next weight sync onloads it anyway.
    """
    scorer, enabled = _SCORER_ROLES[phase_id]
    if not enabled(args):
        yield False
        return
    _switch([Role.ROLLOUT], [scorer])
    logger.info(f"Entered scoring phase {phase_id}")
    try:
        yield True
    finally:
        _switch([scorer], [])
        logger.info(f"Left scoring phase {phase_id}")


@asynccontextmanager
async def async_scoring_phase(args: Any, phase_id: str) -> AsyncIterator[bool]:
    """``scoring_phase`` for async callers; switches run on a worker thread."""
    scorer, enabled = _SCORER_ROLES[phase_id]
    if not enabled(args):
        yield False
        return
    await asyncio.to_thread(_switch, [Role.ROLLOUT], [scorer])
    try:
        yield True
    finally:
        await asyncio.to_thread(_switch, [scorer], [])


async def async_reactivate_generation(args: Any) -> None:
    """Bring the student back after the teacher stage, for a second pass."""
    if deferred_opd_enabled(args):
        await asyncio.to_thread(_switch, [Role.TEACHER], [Role.ROLLOUT])


__all__ = ["SCORING_DRAIN_TIMEOUT_S", "async_reactivate_generation", "async_scoring_phase", "scoring_phase"]
