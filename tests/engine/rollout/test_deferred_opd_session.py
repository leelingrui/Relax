# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""The deferred session stages during the step and scores after it.

Ordering is the point: batches are staged while generation may still have
requests in flight, and only scored once the step is done, because the student
has to be offloaded before the teacher can take the same GPUs.
"""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from relax.engine.rollout import deferred_opd as module
from relax.engine.rollout import scoring_phase as phase_module
from relax.utils.types import Sample


def build_args(**overrides):
    args = SimpleNamespace(
        use_opd=True,
        opd_type="sglang",
        use_agentic_rollout=False,
        colocate=True,
        hybrid=False,
        resource={"teacher": [1, 4], "actor": [1, 8], "rollout": [1, 4]},
        teacher_hf_checkpoint="/ckpt",
        opd_teacher_routes=None,
        opd_token_selection="student_topk",
        opd_teacher_key="data_source",
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def build_group(start: int, size: int = 2, response_length: int = 3) -> list[Sample]:
    return [
        Sample(
            index=start + offset,
            group_index=start,
            tokens=list(range(10 + response_length)),
            response_length=response_length,
            loss_mask=[1] * response_length,
            metadata={"data_source": "math"},
        )
        for offset in range(size)
    ]


class FakeOpd:
    """An OpdManager stand-in that records which stage ran."""

    def __init__(self, *, needs_student: bool = False, fail_indices: tuple[int, ...] = ()) -> None:
        self.needs_student_prefill = needs_student
        self.fail_indices = fail_indices
        self.events: list[str] = []

    def schema_opd_transfer_data(self):
        return ["opd_topk_token_ids", "opd_topk_teacher_log_probs", "rollout_log_probs"]

    async def prepare_teacher_inputs(self, samples):
        self.events.append("prepare")

    async def score_teacher(self, samples, session=None):
        self.events.append("teacher")
        results = []
        for sample in samples:
            ok = sample.index not in self.fail_indices
            if ok and sample.response_length:
                sample.opd_topk_token_ids = np.zeros((sample.response_length, 2), dtype=np.int64)
                sample.opd_topk_teacher_log_probs = np.zeros((sample.response_length, 2), dtype=np.float32)
            results.append(ok)
        return results

    async def score_student_at_teacher(self, samples, session=None, encode=None):
        self.events.append("student")

    def assemble_transfer(self, samples):
        self.events.append("assemble")


@pytest.fixture
def phases(monkeypatch: pytest.MonkeyPatch):
    state = SimpleNamespace(events=[], activate_result=None)

    @asynccontextmanager
    async def fake_scoring_phase(args, phase_id, *, batch_id=None):
        state.events.append(("enter", phase_id))
        try:
            yield object()
        finally:
            state.events.append(("leave", phase_id))

    async def fake_activate(phase_id, *, operation_id, timeout_s=None):
        state.events.append(("activate", phase_id))
        return state.activate_result

    monkeypatch.setattr(phase_module, "async_scoring_phase", fake_scoring_phase)
    monkeypatch.setattr(phase_module, "async_activate_phase", fake_activate)
    return state


def build_session(args, opd, published: list):
    async def publish(call_args, payload, count, rollout_id, client, *, is_last=False):
        published.append((payload, count, rollout_id, is_last))

    return module.DeferredOpdSession.maybe_create(
        args, 7, object(), opd, publish=publish, encode_multimodal_inputs=None
    )


def test_staging_publishes_nothing_until_the_flush(phases):
    args = build_args()
    opd = FakeOpd()
    published: list = []
    session = build_session(args, opd, published)
    assert session is not None
    groups = [build_group(0), build_group(2)]

    async def main():
        await session.transfer(args, groups, 2, 7, object(), is_last=True)
        assert session.staged_batches == 1
        assert published == []
        assert opd.events == []
        await session.flush()

    asyncio.run(main())
    assert opd.events == ["prepare", "teacher", "assemble"]
    assert phases.events == [("enter", "teacher"), ("leave", "teacher")]
    assert len(published) == 1
    payload, count, rollout_id, is_last = published[0]
    # The caller's grouping is preserved, because the queue helper orders by it.
    assert payload == groups and count == 2 and rollout_id == 7 and is_last is True


def test_a_second_student_pass_gets_its_own_activation_stage(phases):
    args = build_args(opd_token_selection="teacher_topk")
    opd = FakeOpd(needs_student=True)
    published: list = []
    session = build_session(args, opd, published)

    async def main():
        await session.transfer(args, [build_group(0)], 1, 7, object())
        await session.flush()

    asyncio.run(main())
    # The student is woken only after the teacher phase closed.
    assert opd.events == ["prepare", "teacher", "student", "assemble"]
    assert phases.events == [("enter", "teacher"), ("leave", "teacher"), ("activate", "inference")]
    assert len(published) == 1


def test_a_failed_student_reactivation_fails_the_batch(phases):
    args = build_args()
    opd = FakeOpd(needs_student=True)
    published: list = []
    session = build_session(args, opd, published)
    phases.activate_result = SimpleNamespace(
        succeeded=False, last_confirmed_step="drained", error="release not confirmed"
    )

    async def main():
        await session.transfer(args, [build_group(0)], 1, 7, object())
        await session.flush()

    with pytest.raises(RuntimeError, match="did not publish"):
        asyncio.run(main())
    assert published == []
    assert "student" not in opd.events


def test_an_unscored_sample_fails_the_step_instead_of_publishing(phases):
    args = build_args()
    opd = FakeOpd(fail_indices=(1,))
    published: list = []
    session = build_session(args, opd, published)

    async def main():
        await session.transfer(args, [build_group(0)], 1, 7, object())
        await session.flush()

    with pytest.raises(RuntimeError, match="did not publish"):
        asyncio.run(main())
    assert published == []
    # No student stage runs once the teacher result is already incomplete.
    assert "student" not in opd.events


def test_batches_publish_in_the_order_they_were_staged(phases):
    args = build_args()
    opd = FakeOpd()
    published: list = []
    session = build_session(args, opd, published)

    async def main():
        await session.transfer(args, [build_group(0)], 1, 7, object())
        await session.transfer(args, [build_group(2)], 1, 7, object(), is_last=True)
        assert session.staged_batches == 2
        await session.flush()

    asyncio.run(main())
    assert [payload[0][0].index for payload, *_ in published] == [0, 2]
    assert [entry[-1] for entry in published] == [False, True]
    assert session.staged_batches == 0


def test_an_empty_batch_is_not_staged(phases):
    args = build_args()
    session = build_session(args, FakeOpd(), [])

    async def main():
        await session.transfer(args, [], 0, 7, object())
        await session.transfer(args, [[]], 0, 7, object())
        assert session.staged_batches == 0

    asyncio.run(main())


def test_no_session_when_scoring_is_immediate():
    published: list = []
    # Dedicated teacher GPUs: nothing is waiting for the teacher's memory.
    assert build_session(build_args(colocate=False, hybrid=True), FakeOpd(), published) is None
    # Explicit opt-out.
    assert build_session(build_args(opd_deferred_scoring=False), FakeOpd(), published) is None
    # OPD disabled.
    assert build_session(build_args(use_opd=False), FakeOpd(), published) is None
    # The resident Agentic pipeline keeps its own inline scoring.
    assert build_session(build_args(use_agentic_rollout=True), FakeOpd(), published) is None
    # No OPD manager at all.
    assert build_session(build_args(), None, published) is None


def test_generation_owned_fields_are_not_scoring_requirements():
    assert module.required_scoring_fields(FakeOpd()) == (
        "opd_topk_token_ids",
        "opd_topk_teacher_log_probs",
    )
