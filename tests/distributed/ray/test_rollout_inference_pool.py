# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""RolloutServer observations are the only evidence a model is READY."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from conftest import (
    HAS_DEPS,
    AwaitableValue,
    create_test_manager,
    make_engine_group,
    make_mock_engine,
    make_rollout_server,
)


if HAS_DEPS:
    from relax.engine.inference.config import ModelConfig
    from relax.engine.inference.types import LifecycleState, Role, WeightSource


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="Missing ray/sglang dependencies")


def _observed_engine(version: str | None = "v1", **overrides: Any) -> Any:
    engine = make_mock_engine(weight_version=version)
    observation = {"healthy": True, "router_registered": True, "weight_version": version, "base_url": "http://w:1"}
    observation.update(overrides)
    engine.get_inference_observation.remote.return_value = AwaitableValue(observation)
    return engine


def _pool(groups: list[Any]) -> Any:
    pool = create_test_manager(servers={"default": make_rollout_server(engine_groups=groups)})
    pool.status = "onload"
    pool.rollout_engine_lock = MagicMock()
    return pool


def _model(pool: Any) -> Any:
    pool.refresh_inference_state()
    return pool.inference_manager.snapshot(Role.ROLLOUT).models[0]


@pytest.mark.parametrize("version", [None, "", "default"])
def test_rollout_observe_unknown_policy_version_never_ready(patch_ray_get: Any, version: str | None) -> None:
    model = _model(_pool([make_engine_group(engines=[_observed_engine(version)])]))
    assert model.state == LifecycleState.STARTING
    assert not model.admission
    assert model.required_weight_version is None


@pytest.mark.parametrize("missing", ["healthy", "router_registered"])
def test_rollout_observe_missing_evidence_never_ready(patch_ray_get: Any, missing: str) -> None:
    engine = _observed_engine()
    del engine.get_inference_observation.remote.return_value.value[missing]
    assert not _model(_pool([make_engine_group(engines=[engine])])).admission


def test_rollout_observe_mixed_versions_block_admission(patch_ray_get: Any) -> None:
    groups = [make_engine_group(engines=[_observed_engine("v1")]), make_engine_group(engines=[_observed_engine("v2")])]
    assert not _model(_pool(groups)).admission


def test_rollout_weight_update_closes_admission_until_completion(patch_ray_get: Any) -> None:
    pool = _pool([make_engine_group(engines=[_observed_engine("v1")])])
    assert _model(pool).admission
    pool.invalidate_inference_state()
    assert not _model(pool).admission
    pool.complete_inference_weight_update()
    assert pool.inference_manager.snapshot(Role.ROLLOUT).models[0].admission


@pytest.mark.parametrize("decode_version,ready", [("v1", True), ("v2", False)])
def test_rollout_observe_pd_exposes_only_router_service(patch_ray_get: Any, decode_version: str, ready: bool) -> None:
    groups = [
        make_engine_group(engines=[_observed_engine("v1")], worker_type="prefill"),
        make_engine_group(engines=[_observed_engine(decode_version)], worker_type="decode", rank_offset=1),
    ]
    model = _model(_pool(groups))
    assert [replica.engine_id for replica in model.replicas] == ["default/pd-service"]
    assert model.replicas[0].base_url == model.router_url
    assert sorted(kind for kind, _ in model.pd_workers) == ["decode", "prefill"]
    assert model.admission is ready


def test_static_server_offload_and_onload_are_idempotent(patch_ray_get: Any) -> None:
    engine = _observed_engine(None)
    server = make_rollout_server(engine_groups=[make_engine_group(engines=[engine])])
    server.static = True
    server.model_spec = ModelConfig("default", "ckpt", weight_source=WeightSource.STATIC)
    engine.release_memory_occupation.remote.return_value = AwaitableValue(None)
    engine.resume_memory_occupation.remote.return_value = AwaitableValue(None)

    server.onload()
    server.offload()
    server.offload()
    server.onload()
    server.onload()

    assert engine.release_memory_occupation.remote.call_count == 1
    assert engine.resume_memory_occupation.remote.call_count == 1
    assert server.observe().state == LifecycleState.READY
