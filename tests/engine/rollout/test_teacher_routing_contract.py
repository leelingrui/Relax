# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Characterize existing Teacher route selection before discovery migration."""

from types import SimpleNamespace

import pytest

from relax.engine.rollout import on_policy_distillation as opd


@pytest.fixture(autouse=True)
def isolated_teacher_round_robin(monkeypatch) -> None:
    monkeypatch.setattr(opd, "_TEACHER_URL_RR", {})


def test_teacher_routing_round_robin_is_per_route() -> None:
    args = SimpleNamespace(opd_teacher_routes_map={"math": ["m0", "m1"], "code": ["c0", "c1"]})
    math = SimpleNamespace(metadata={"data_source": "math"})
    code = SimpleNamespace(metadata={"data_source": "code"})
    assert [opd._pick_teacher_url(args, s) for s in [math, code, math, math, code]] == ["m0", "c0", "m1", "m0", "c1"]


@pytest.mark.parametrize("metadata,error", [({}, ValueError), ({"data_source": "unknown"}, KeyError)])
def test_teacher_routing_missing_route_never_uses_default(metadata: dict, error: type[Exception]) -> None:
    args = SimpleNamespace(opd_teacher_routes_map={"math": ["m0"]}, opd_teacher_url="fallback")
    with pytest.raises(error):
        opd._pick_teacher_url(args, SimpleNamespace(metadata=metadata))


def test_teacher_routing_single_teacher_replica_fallback() -> None:
    args = SimpleNamespace(opd_teacher_urls=["a", "b"], opd_teacher_url="legacy")
    assert [opd._pick_teacher_url(args) for _ in range(3)] == ["a", "b", "a"]
    args.opd_teacher_urls = ["a"]
    assert opd._pick_teacher_url(args) == "legacy"
