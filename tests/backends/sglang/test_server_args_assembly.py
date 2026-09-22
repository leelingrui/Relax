# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Every role assembles its ServerArgs the same way.

The model, the parallel sizes and the memory policy differ per role and stay in
each role's own base arguments. What follows -- inheriting the global
``--sglang-*`` defaults, applying the role's overrides, and dropping what the
installed SGLang does not recognise -- is one shared tail, so a fix to it
cannot reach only one role.
"""

import dataclasses
from types import SimpleNamespace

import pytest


pytest.importorskip("sglang.srt.server_args", exc_type=ImportError)

from sglang.srt.server_args import ServerArgs  # noqa: E402

from relax.backends.sglang.sglang_engine import _finalize_server_args  # noqa: E402


SERVER_ARG_NAMES = {field.name for field in dataclasses.fields(ServerArgs)}
UNKNOWN_FIELD = "definitely_not_a_server_args_field"


def _args(**overrides):
    return SimpleNamespace(**overrides)


def _finalize(args, kwargs, *, overrides=None, worker_type="regular"):
    return _finalize_server_args(
        args,
        kwargs,
        rank=0,
        worker_type=worker_type,
        overrides=overrides,
        overrides_label="--test-config",
    )


def test_an_override_naming_an_unknown_field_is_dropped_not_passed_on():
    """It would otherwise be a TypeError inside ServerArgs at engine start."""
    kwargs, _ = _finalize(_args(), {"tp_size": 2}, overrides={UNKNOWN_FIELD: "x"})

    assert UNKNOWN_FIELD not in kwargs
    assert kwargs["tp_size"] == 2


def test_an_override_wins_over_the_role_base_and_the_global_default():
    kwargs, _ = _finalize(_args(sglang_tp_size=8), {"tp_size": 2}, overrides={"tp_size": 4})

    assert kwargs["tp_size"] == 4


def test_a_global_default_only_fills_what_the_role_left_open():
    kwargs, _ = _finalize(_args(sglang_tp_size=8, sglang_load_format="dummy"), {"tp_size": 2})

    assert kwargs["tp_size"] == 2
    assert kwargs["load_format"] == "dummy"


def test_a_base_key_this_sglang_does_not_know_is_dropped():
    """Roles carry keys for SGLang versions other than the installed one."""
    kwargs, _ = _finalize(_args(), {"tp_size": 2, UNKNOWN_FIELD: 1})

    assert UNKNOWN_FIELD not in kwargs


def test_the_external_engine_check_list_describes_the_role_base_only():
    """Inherited defaults and overrides are not part of the contract an
    external engine is checked against."""
    _, checked = _finalize(
        _args(sglang_load_format="dummy"),
        {"tp_size": 2, "model_path": "/ckpt"},
        overrides={"mem_fraction_static": 0.5},
    )

    assert "tp_size" in checked
    # model_path is on the skip list, and nothing inherited or overridden joins.
    assert "model_path" not in checked
    assert "load_format" not in checked
    assert "mem_fraction_static" not in checked


@pytest.mark.skipif(
    "cuda_graph_backend_prefill" not in SERVER_ARG_NAMES,
    reason="installed SGLang has no cuda_graph_backend_prefill",
)
def test_memory_saver_disables_the_incompatible_prefill_graph_backend():
    kwargs, _ = _finalize(_args(), {"enable_memory_saver": True})

    assert kwargs["cuda_graph_backend_prefill"] == "disabled"


def test_a_decode_worker_keeps_its_own_hierarchical_cache_setting():
    """The global default must not reach a decode worker for this one field."""
    kwargs, _ = _finalize(
        _args(sglang_enable_hierarchical_cache=True),
        {"tp_size": 2},
        worker_type="decode",
    )

    assert "enable_hierarchical_cache" not in kwargs


# ======================== both roles use the tail ==========================


def test_rollout_and_genrm_both_assemble_through_the_shared_tail(monkeypatch):
    """A fix to the tail has to reach every role, so both must call it."""
    from relax.backends.sglang import sglang_engine

    seen: list[str] = []

    def _spy(args, kwargs, *, rank, worker_type, overrides, overrides_label):
        seen.append(overrides_label)
        return kwargs, []

    monkeypatch.setattr(sglang_engine, "_finalize_server_args", _spy)

    class _Args(SimpleNamespace):
        """Unset numbers and flags read as 0; the spy skips inheritance."""

        def __getattr__(self, name):
            return 0

    base = dict(seed=0, num_gpus_per_node=8, actor_num_gpus_per_node=8, actor_num_nodes=1)
    rollout_args = _Args(hf_checkpoint="/ckpt", rollout_num_gpus_per_engine=2, sglang_pp_size=1, **base)
    genrm_args = _Args(
        genrm_model_path="/judge",
        genrm_num_gpus_per_engine=2,
        genrm_engine_config={},
        **base,
    )
    common = dict(dist_init_addr="127.0.0.1:1", nccl_port=2, host="127.0.0.1", port=3)

    sglang_engine._compute_server_args(rollout_args, 0, **common)
    sglang_engine._compute_genrm_server_args(genrm_args, 0, **common)

    assert seen == ["sglang_overrides", "--genrm-engine-config"]
