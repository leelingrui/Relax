# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Rollout's common inference pool must preserve its training-side contract."""

import asyncio
from typing import Any
from unittest.mock import MagicMock, call, patch

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
    from relax.distributed.ray.rollout import (
        GPU_MEMORY_TYPE_CUDA_GRAPH,
        GPU_MEMORY_TYPE_KV_CACHE,
        GPU_MEMORY_TYPE_WEIGHTS,
        _RolloutPoolRuntime,
    )
    from relax.engine.inference.capabilities import WeightSource
    from relax.engine.inference.manager import InferenceManager
    from relax.engine.inference.specs import ModelSpec
    from relax.engine.inference.types import LifecycleState, Role, RoutingSpec


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="Missing ray/sglang dependencies")


def _observed_engine(version: str | None = "v1", **overrides: Any) -> Any:
    engine = make_mock_engine(weight_version=version)
    observation = {
        "healthy": True,
        "router_registered": True,
        "weight_version": version,
        "base_url": "http://localhost:30000",
    }
    observation.update(overrides)
    engine.get_inference_observation.remote.return_value = AwaitableValue(observation)
    return engine


def _manager(groups: list[Any]) -> Any:
    from relax.distributed.ray.inference_role import UnifiedServiceManager
    from relax.distributed.ray.model_pool import ModelPool

    server = make_rollout_server(engine_groups=groups)
    manager = create_test_manager(servers={"default": server})
    manager.status = "onload"
    manager.rollout_engine_lock = MagicMock()
    manager.inference_manager = InferenceManager(Role.ROLLOUT)
    manager.inference_manager.register_model(
        ModelSpec("default", "/tmp/test", weight_source=WeightSource.POLICY, allow_defer=True), operation_id="register"
    )
    pool = ModelPool.from_runtime(manager.inference_manager, "default", _RolloutPoolRuntime(server))
    manager.inference_manager.configure_routes(RoutingSpec(default_model="default"), operation_id="routes")
    manager.service_manager = UnifiedServiceManager(
        Role.ROLLOUT, inference_manager=manager.inference_manager, pools={"default": pool}
    )
    return manager


@pytest.mark.parametrize("version", [None, "", "default"])
def test_rollout_refresh_unknown_policy_version_never_ready(patch_ray_get: Any, version: str | None) -> None:
    manager = _manager([make_engine_group(engines=[_observed_engine(version)])])
    manager.refresh_inference_state()
    model = manager.get_discovery_snapshot().models[0]
    assert model.state == LifecycleState.STARTING
    assert not model.admission
    assert model.required_weight_version is None
    assert model.replicas[0].state != LifecycleState.READY


@pytest.mark.parametrize("version", [None, "default", "v2"])
def test_rollout_router_member_with_unready_weights_blocks_model(patch_ray_get: Any, version: str | None) -> None:
    manager = _manager([make_engine_group(engines=[_observed_engine("v1"), _observed_engine(version)])])
    manager.refresh_inference_state()
    assert not manager.get_discovery_snapshot().models[0].admission


def test_rollout_backend_weight_failure_stays_closed_until_completion(patch_ray_get: Any) -> None:
    manager = _manager([make_engine_group(engines=[_observed_engine("v1")])])
    manager.refresh_inference_state()
    assert manager.get_discovery_snapshot().models[0].admission
    manager.invalidate_inference_state()
    manager.refresh_inference_state()
    assert not manager.get_discovery_snapshot().models[0].admission
    manager.complete_inference_weight_update()
    assert manager.get_discovery_snapshot().models[0].admission


@pytest.mark.parametrize("fails", [False, True])
def test_train_group_weight_sync_publishes_only_after_all_ranks(patch_ray_get: Any, fails: bool) -> None:
    from relax.distributed.ray.actor_group import RayTrainGroup

    group = RayTrainGroup.__new__(RayTrainGroup)
    events = []
    manager = MagicMock()
    group._rollout_manager = manager
    manager.set_weight_updating.remote.side_effect = lambda value: events.append(("updating", value)) or True
    manager.refresh_inference_state.remote.side_effect = lambda: events.append(("refresh",))
    actor = MagicMock()
    group._actor_handlers = [actor]

    def update():
        events.append(("sync",))
        if fails:
            raise RuntimeError("sync failed")

    actor.update_weights.remote.side_effect = update
    if fails:
        with pytest.raises(RuntimeError, match="sync failed"):
            group.update_weights()
        assert events == [("updating", True), ("sync",)]
    else:
        group.update_weights()
        assert events == [("updating", True), ("sync",), ("updating", False), ("refresh",)]


@pytest.mark.parametrize("missing", ["healthy", "router_registered", "weight_version"])
def test_rollout_refresh_missing_evidence_never_ready(patch_ray_get: Any, missing: str) -> None:
    engine = _observed_engine()
    del engine.get_inference_observation.remote.return_value.value[missing]
    manager = _manager([make_engine_group(engines=[engine])])
    manager.refresh_inference_state()
    model = manager.get_discovery_snapshot().models[0]
    assert model.state != LifecycleState.READY
    assert not model.admission


@pytest.mark.parametrize("decode_version,ready", [("v1", True), ("v2", False), ("default", False), (None, False)])
def test_rollout_refresh_pd_exposes_only_router_service(
    patch_ray_get: Any, decode_version: str | None, ready: bool
) -> None:
    prefill = _observed_engine(base_url="http://prefill:30000")
    decode = _observed_engine(decode_version, base_url="http://decode:30000")
    manager = _manager(
        [
            make_engine_group(engines=[prefill], worker_type="prefill"),
            make_engine_group(engines=[decode], worker_type="decode", rank_offset=1),
        ]
    )
    manager.refresh_inference_state()
    model = manager.get_discovery_snapshot().models[0]
    assert model.admission is ready
    assert (model.state == LifecycleState.READY) is ready
    assert len(model.replicas) == 1
    service = model.replicas[0]
    assert service.engine_id == "default/pd-service"
    assert service.base_url == model.router_url == "http://127.0.0.1:3000"
    assert (service.state == LifecycleState.READY) is ready
    assert not model.direct_eligible


@pytest.mark.parametrize("worker_type", ["prefill", "decode"])
def test_rollout_refresh_incomplete_pd_pair_never_ready(patch_ray_get: Any, worker_type: str) -> None:
    manager = _manager([make_engine_group(engines=[_observed_engine()], worker_type=worker_type)])
    manager.refresh_inference_state()
    model = manager.get_discovery_snapshot().models[0]
    assert not model.admission
    assert model.replicas[0].state != LifecycleState.READY


@pytest.mark.parametrize("follower", ["present", "missing", "truncated"])
def test_rollout_refresh_multinode_requires_complete_replica(patch_ray_get: Any, follower: str) -> None:
    head = _observed_engine()
    nodes = [head, None if follower == "missing" else _observed_engine()]
    group = make_engine_group(engines=nodes, num_gpus_per_engine=16)
    if follower == "truncated":
        group.all_engines.pop()
    manager = _manager([group])
    manager.refresh_inference_state()
    model = manager.get_discovery_snapshot().models[0]
    assert model.admission is (follower == "present")
    assert len(model.replicas) == 1
    assert (model.replicas[0].state == LifecycleState.READY) is (follower == "present")
    if follower == "present":
        nodes[1].get_inference_observation.remote.assert_not_called()
    else:
        head.get_inference_observation.remote.assert_not_called()


def test_rollout_discovery_filters_do_not_mutate_or_refresh_state(patch_ray_get: Any) -> None:
    engine = _observed_engine()
    manager = _manager([make_engine_group(engines=[engine, None])])
    manager.refresh_inference_state()
    before = manager.inference_manager.snapshot()
    engine.get_inference_observation.remote.reset_mock()
    for status_filter, state in (("active", LifecycleState.READY), ("dead", LifecycleState.DEAD)):
        snapshot = manager.get_discovery_snapshot("default", status_filter=status_filter)
        model = snapshot.models[0]
        assert len(model.replicas) == 1
        assert model.replicas[0].state == state
        assert model.state == before.models[0].state
        assert model.admission == before.models[0].admission
        assert snapshot.phase == "onload"
        assert manager.inference_manager.snapshot() is before
    assert manager.get_discovery_snapshot().models == before.models
    engine.get_inference_observation.remote.assert_not_called()
    with pytest.raises(ValueError, match="status_filter"):
        manager.get_discovery_snapshot(status_filter="ready")
    assert manager.inference_manager.snapshot() is before


def test_rollout_pool_runtime_preserves_explicit_recovery_and_retirement() -> None:
    server = MagicMock()
    pool = _RolloutPoolRuntime(server)
    assert pool.recover_on_onload is False
    assert pool.always_resume is True
    assert pool.recover() == set()
    server.recover.assert_called_once_with()
    pool.retire([])
    with pytest.raises(ValueError, match="weight-update fence"):
        pool.retire([0])
    with pytest.raises(ValueError, match="Unsupported"):
        pool.fanout("unknown")


def test_rollout_pool_lifecycle_delegates_without_changing_five_tuple(patch_ray_get: Any) -> None:
    placeholder = make_engine_group(engines=[], worker_type="placeholder")
    head, follower = _observed_engine(), _observed_engine()
    group = make_engine_group(engines=[head, follower], num_gpus_per_engine=16, rank_offset=1)
    group.gpu_offset = 8
    group.num_new_engines = 2
    manager = _manager([placeholder, group])
    server = manager.servers["default"]
    expected = ([head], manager.rollout_engine_lock, 2, [16], [8])
    assert manager.get_rollout_engines_and_lock() == expected
    common = manager.inference_manager
    with (
        patch.object(server, "offload") as offload,
        patch.object(server, "onload") as onload,
        patch.object(server, "recover") as recover,
        patch.object(common, "offload", wraps=common.offload) as common_offload,
        patch.object(common, "onload", wraps=common.onload) as common_onload,
        patch.object(common, "recover", wraps=common.recover) as common_recover,
    ):
        asyncio.run(manager.offload())
        asyncio.run(manager.offload())
        common_offload.assert_called_once_with("default")
        offload.assert_called_once_with()
        assert common.snapshot().models[0].state == LifecycleState.SLEEPING
        assert manager.get_rollout_engines_and_lock() == expected

        asyncio.run(manager.onload_weights())
        assert manager.status == "onloading"
        assert not manager.get_discovery_snapshot().models[0].admission
        recover.assert_not_called()
        assert manager.get_rollout_engines_and_lock() == expected

        asyncio.run(manager.onload_kv())
        assert manager.status == "onload"
        assert manager.get_discovery_snapshot().models[0].admission
        tags = [[GPU_MEMORY_TYPE_WEIGHTS], [GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH]]
        assert common_onload.call_args_list == [call("default", value) for value in tags]
        assert onload.call_args_list == [call(value) for value in tags]
        recover.assert_not_called()
        assert manager.get_rollout_engines_and_lock() == expected

        assert manager.recover_rollout_engines() == expected
        common_recover.assert_called_once_with("default")
        recover.assert_called_once_with()
        assert not manager.get_discovery_snapshot().models[0].admission

        asyncio.run(manager.onload())
        onload.assert_called_with(None)
        recover.assert_called_once_with()
        assert manager.get_rollout_engines_and_lock() == expected
        manager.clear_num_new_engines()
        assert manager.get_rollout_engines_and_lock() == (expected[0], expected[1], 0, expected[3], expected[4])


def test_rollout_uses_shared_host_and_generic_pool(patch_ray_get: Any) -> None:
    from relax.distributed.ray.inference_role import UnifiedServiceManager
    from relax.distributed.ray.model_pool import ModelPool

    manager = _manager([make_engine_group(engines=[_observed_engine()])])
    assert type(manager.service_manager) is UnifiedServiceManager
    assert type(manager.service_manager.pools["default"]) is ModelPool
    assert manager.service_manager.inference_manager is manager.inference_manager
    assert manager.service_manager.snapshot() is manager.inference_manager.snapshot()
    manager.service_manager.shutdown()
    assert not manager.service_manager.ready()
    assert manager.service_manager.snapshot().models[0].state == LifecycleState.DEAD
