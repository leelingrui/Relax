# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Lock ownership and publication boundary regressions."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest

from relax.engine.inference.capabilities import WeightSource
from relax.engine.inference.manager import InferenceManager, ModelBusyError
from relax.engine.inference.specs import ModelSpec
from relax.engine.inference.types import LifecycleState, ModelSnapshot, Role


class _Runtime:
    def health_check(self) -> bool:
        return True


def _manager() -> InferenceManager:
    manager = InferenceManager(Role.TEACHER)
    manager.register_model(
        ModelSpec("model", "checkpoint", weight_source=WeightSource.CHECKPOINT), operation_id="register"
    )
    manager.bind_pool("model", _Runtime())
    return manager


def test_lock_owner_can_publish_but_other_thread_cannot() -> None:
    manager = _manager()
    model = ModelSnapshot("model", state=LifecycleState.SLEEPING)
    entered, release = Event(), Event()

    def owner_publish() -> None:
        with manager._pool_operation("model"):
            entered.set()
            assert release.wait(5)
            manager.publish_model(model)

    with ThreadPoolExecutor(max_workers=2) as executor:
        owner = executor.submit(owner_publish)
        assert entered.wait(2)
        with pytest.raises(ModelBusyError):
            manager.publish_model(model)
        release.set()
        owner.result(timeout=2)

    assert manager.snapshot().models[0].state is LifecycleState.SLEEPING


def test_real_pool_dispatch_can_publish_ready_after_onload(monkeypatch) -> None:
    from relax.distributed.ray import multi_engine_manager as module
    from relax.distributed.ray.inference_role import UnifiedServiceManager
    from relax.distributed.ray.model_pool import ModelPool

    manager = InferenceManager(Role.TEACHER)
    backend = module.MultiEngineManager(
        SimpleNamespace(hf_checkpoint="checkpoint"),
        num_slots=1,
        engine_actor_cls=object,
        skip_init=True,
        inference_manager=manager,
        model_id="model",
    )
    backend.router_url = "http://router"
    observation = {"healthy": True, "router_registered": True, "base_url": "http://engine"}
    backend.all_engines = [SimpleNamespace(get_inference_observation=SimpleNamespace(remote=lambda: observation))]
    monkeypatch.setattr(module.ray, "get", lambda value, **kwargs: value)
    monkeypatch.setattr(backend, "_recover_engines", lambda: set())
    monkeypatch.setattr(backend, "_fanout", lambda *args, **kwargs: [])
    backend._onloaded = False
    backend._memory_ready = False
    pool = ModelPool.from_backend(backend, inference_manager=manager, model_id="model")
    host = UnifiedServiceManager(Role.TEACHER, inference_manager=manager, pools={"model": pool})
    host.call_wait("model", "onload")
    assert host.snapshot().models[0].admission
    host.call_wait("model", "offload")
    assert not host.snapshot().models[0].admission
