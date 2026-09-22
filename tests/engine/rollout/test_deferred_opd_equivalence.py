# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Deferred scoring must produce exactly the immediate path's training fields.

Same samples, same teacher responses: the only difference is *when* the teacher
runs. If the deferred fields differed, the loss would see different distillation
targets depending on the GPU layout, which is the one thing the deferred pipeline
must not change. Out-of-order completion is included, because the whole point of
correlating by sample identity is that arrival order must not matter.
"""

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pybase64
import pytest

from relax.engine.rollout import deferred_opd as module
from relax.engine.rollout import scoring_phase as phase_module
from relax.engine.rollout.on_policy_distillation import OpdManager
from relax.utils.opd.opd_main_worker import LogprobResponse
from relax.utils.types import Sample


RESPONSE_LENGTH = 3
TOP_K = 2


def build_args(token_selection: str = "student_topk", **overrides):
    args = SimpleNamespace(
        use_opd=True,
        opd_type="sglang",
        use_agentic_rollout=False,
        colocate=True,
        hybrid=False,
        resource={"teacher": [1, 4], "actor": [1, 8], "rollout": [1, 4]},
        teacher_hf_checkpoint="/ckpt",
        opd_teacher_routes=None,
        opd_token_selection=token_selection,
        opd_log_prob_top_k=TOP_K,
        opd_kl_coef=0.0,
        opd_loss_coef=1.0,
        opd_teacher_key="data_source",
        opd_teacher_url="http://teacher:1/generate",
        opd_teacher_urls=["http://teacher:1/generate"],
        opd_teacher_gateway_url=None,
        opd_teacher_timeout_s=30,
        opd_teacher_connector_limit=8,
        opd_teacher_prompt_key=None,
        opd_teacher_image_key=None,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=1234,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def build_samples(count: int = 4, *, multimodal: bool = False, routes: bool = False) -> list[Sample]:
    samples = []
    for index in range(count):
        sample = Sample(
            index=index,
            group_index=index // 2,
            tokens=list(range(10 + RESPONSE_LENGTH)),
            rollout_tokens=list(range(10 + RESPONSE_LENGTH)),
            response_length=RESPONSE_LENGTH,
            loss_mask=[1] * RESPONSE_LENGTH,
            metadata={"data_source": "vl" if (routes and index % 2) else "math"},
        )
        # Student self top-k from generation, the same on both paths.
        sample.student_topk_token_ids = np.arange(
            index * 100, index * 100 + RESPONSE_LENGTH * TOP_K, dtype=np.int64
        ).reshape(RESPONSE_LENGTH, TOP_K)
        sample.student_topk_log_probs = np.full((RESPONSE_LENGTH, TOP_K), -float(index) - 1.0, dtype=np.float32)
        if multimodal:
            sample.multimodal_inputs = {"images": [f"image-{index}"]}
        samples.append(sample)
    return samples


def teacher_response(sample: Sample) -> LogprobResponse:
    """A deterministic teacher answer derived from the sample identity."""
    base = np.full(RESPONSE_LENGTH + 1, -0.5 - sample.index, dtype=np.float32)
    rows = RESPONSE_LENGTH + 1
    vals = np.arange(rows * TOP_K, dtype=np.float32) * -0.01 - sample.index
    ids = (np.arange(rows * TOP_K, dtype=np.int32) + sample.index * 7).astype(np.int32)
    return LogprobResponse(
        {
            "meta_info": {
                # Base per-token log-probs, the engine's own top-k, and the
                # log-probs at the token ids the payload queried.
                "input_token_logprobs_val_b64": pybase64.b64encode(base.tobytes()).decode(),
                "input_top_logprobs_val_b64": pybase64.b64encode(vals.tobytes()).decode(),
                "input_top_logprobs_idx_b64": pybase64.b64encode(ids.tobytes()).decode(),
                "input_token_ids_logprobs_val_b64": pybase64.b64encode((vals * 2).tobytes()).decode(),
                "input_token_ids_logprobs_idx_b64": pybase64.b64encode(ids.tobytes()).decode(),
            }
        }
    )


def install_fake_teacher(opd: OpdManager, *, reverse_order: bool = False) -> list[int]:
    """Answer every request from the sample identity, optionally out of order."""
    completed: list[int] = []

    async def fake_post(session, url, payload, sample, err_tag):
        if reverse_order:
            # Later samples answer first, so completion order differs from
            # submission order.
            await asyncio.sleep(0.005 * (4 - sample.index))
        completed.append(sample.index)
        return teacher_response(sample)

    opd._post_logprob = fake_post
    return completed


@asynccontextmanager
async def _noop_phase(args, phase_id, *, batch_id=None):
    yield object()


async def _noop_activate(phase_id, *, operation_id, timeout_s=None):
    return None


TRAINING_FIELDS = (
    "opd_topk_token_ids",
    "opd_topk_teacher_log_probs",
    "opd_topk_student_log_probs",
    "opd_topk_ksz",
    "teacher_log_probs",
    "teacher_topk_token_ids",
    "teacher_topk_log_probs",
    "teacher_at_student_topk_log_probs",
)


def snapshot_fields(samples: list[Sample]) -> list[dict]:
    captured = []
    for sample in samples:
        row = {}
        for name in TRAINING_FIELDS:
            value = getattr(sample, name, None)
            row[name] = None if value is None else np.asarray(value).tolist()
        captured.append(row)
    return captured


def run_immediate(args, samples, *, reverse_order: bool = False) -> list[dict]:
    opd = OpdManager(args)
    install_fake_teacher(opd, reverse_order=reverse_order)
    asyncio.run(opd.prefill(samples))
    return snapshot_fields(samples)


def run_deferred(args, samples, *, reverse_order: bool = False, monkeypatch=None) -> tuple[list[dict], list]:
    opd = OpdManager(args)
    install_fake_teacher(opd, reverse_order=reverse_order)
    published: list = []

    async def publish(call_args, payload, count, rollout_id, client, *, is_last=False):
        published.append(payload)

    monkeypatch.setattr(phase_module, "async_scoring_phase", _noop_phase)
    monkeypatch.setattr(phase_module, "async_activate_phase", _noop_activate)
    session = module.DeferredOpdSession.maybe_create(args, 7, object(), opd, publish=publish)
    assert session is not None

    async def main():
        await session.transfer(args, [samples], len(samples), 7, object(), is_last=True)
        await session.flush()

    asyncio.run(main())
    return snapshot_fields(samples), published


@pytest.mark.parametrize("token_selection", ["student_topk", "student_sampled"])
def test_deferred_fields_match_the_immediate_path(monkeypatch, token_selection):
    args = build_args(token_selection)
    immediate = run_immediate(args, build_samples())
    deferred, published = run_deferred(args, build_samples(), monkeypatch=monkeypatch)
    assert deferred == immediate
    assert len(published) == 1


def test_out_of_order_teacher_completion_does_not_change_the_fields(monkeypatch):
    args = build_args()
    immediate = run_immediate(args, build_samples())
    deferred, _published = run_deferred(args, build_samples(), reverse_order=True, monkeypatch=monkeypatch)
    assert deferred == immediate


def test_multimodal_and_multi_teacher_routing_keep_the_fields_identical(monkeypatch):
    args = build_args(opd_teacher_routes='{"math": "/math", "vl": "/vl"}')
    args.opd_teacher_routes_map = {"math": ["http://math:1/generate"], "vl": ["http://vl:1/generate"]}
    immediate = run_immediate(args, build_samples(multimodal=True, routes=True))
    deferred, _published = run_deferred(
        args, build_samples(multimodal=True, routes=True), monkeypatch=monkeypatch
    )
    assert deferred == immediate


def test_the_train_data_projection_is_identical(monkeypatch):
    args = build_args()
    immediate_samples = build_samples()
    run_immediate(args, immediate_samples)
    deferred_samples = build_samples()
    run_deferred(args, deferred_samples, monkeypatch=monkeypatch)

    immediate_train: dict = {}
    deferred_train: dict = {}
    OpdManager(args).produce_opd_transfer_data(immediate_samples, immediate_train)
    OpdManager(args).produce_opd_transfer_data(deferred_samples, deferred_train)
    assert set(immediate_train) == set(deferred_train)
    for key, value in immediate_train.items():
        assert deepcopy(deferred_train[key]) == value, key
