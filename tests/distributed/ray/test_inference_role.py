# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU-only role ownership and compatibility forwarding tests."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from relax.distributed.ray import inference_role as module
from relax.engine.inference.capabilities import WeightSource
from relax.engine.inference.manager import ModelBusyError
from relax.engine.inference.specs import ModelSpec
from relax.engine.inference.types import Role


@pytest.fixture
def pools(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    state = SimpleNamespace(
        events=[], instances={}, fail_init=None, fail_construct=None, fail_shutdown=None, router_cleanup=MagicMock()
    )
    state.router_cleanup.side_effect = lambda: state.events.append(("routers", "shutdown"))
    monkeypatch.setattr(module, "_stop_role_routers", state.router_cleanup)

    class Pool:
        def __init__(self, value: Any, *, inference_manager: Any, model_id: str, defer_init: bool, **kwargs: Any):
            assert defer_init is True
            if model_id == state.fail_construct:
                raise RuntimeError("constructor failed")
            self.manager = inference_manager
            self.model_id = model_id
            self.value = value
            self.kwargs = kwargs
            self.manager.register_model(
                ModelSpec(model_id, "checkpoint", weight_source=WeightSource.CHECKPOINT),
                operation_id=f"register:{model_id}",
            )
            self.manager.bind_pool(model_id, self)
            state.instances[model_id] = self
            state.events.append(("construct", model_id))

        def initialize(self) -> None:
            snapshot = self.manager.snapshot()
            assert {model.model_id for model in snapshot.models} == set(state.instances)
            assert dict(snapshot.routing.route_key_to_model) == {key: key for key in state.instances}
            state.events.append(("initialize", self.model_id))
            if self.model_id == state.fail_init:
                raise RuntimeError("initialize failed")

        def shutdown(self) -> None:
            state.events.append(("shutdown", self.model_id))
            if self.model_id == state.fail_shutdown:
                raise RuntimeError("shutdown failed")

    state.factory = MagicMock(side_effect=lambda role, *args, **kwargs: Pool(*args, **kwargs))
    monkeypatch.setattr(module, "_create_model_pool", state.factory)
    return state


def _configs() -> dict[str, dict[str, Any]]:
    return {key: {"args": (key,), "kwargs": {"bundle_offset": index}} for index, key in enumerate(("a", "b"))}


@pytest.mark.parametrize("role", [Role.GENRM, Role.TEACHER])
def test_role_registers_all_models_before_initialization(pools: SimpleNamespace, role: Role) -> None:
    configs = _configs()
    owner = module.InferenceRole(role, configs)
    assert owner.ready() is True
    assert pools.events == [("construct", "a"), ("construct", "b"), ("initialize", "a"), ("initialize", "b")]
    assert pools.instances["a"].manager is pools.instances["b"].manager is owner.inference_manager
    assert pools.instances["b"].value == "b"
    assert pools.instances["b"].kwargs == {"bundle_offset": 1}
    assert configs == _configs()
    assert owner.snapshot() is owner.inference_manager.snapshot()
    assert owner.snapshot().routing.default_model is None
    assert [model.model_id for model in owner.snapshot(["b"]).models] == ["b"]
    assert all(not model.admission for model in owner.snapshot().models)
    assert [call.args for call in pools.factory.call_args_list] == [(role, "a"), (role, "b")]


def test_role_single_model_has_default_route(pools: SimpleNamespace) -> None:
    owner = module.InferenceRole("teacher", {"a": _configs()["a"]})
    assert owner.snapshot().routing.default_model == "a"


@pytest.mark.parametrize("failing_model", ["a", "b"])
def test_role_initialization_failure_cleans_even_uninitialized_pools(
    pools: SimpleNamespace, failing_model: str
) -> None:
    pools.fail_init = failing_model
    pools.fail_shutdown = "b"
    with pytest.raises(RuntimeError, match="initialize failed"):
        module.InferenceRole("genrm", _configs())
    assert pools.events[-3:] == [("shutdown", "b"), ("shutdown", "a"), ("routers", "shutdown")]
    pools.router_cleanup.assert_called_once_with()


def test_role_constructor_failure_cleans_previously_constructed_pools(pools: SimpleNamespace) -> None:
    pools.fail_construct = "b"
    with pytest.raises(RuntimeError, match="constructor failed"):
        module.InferenceRole("teacher", _configs())
    assert pools.events == [("construct", "a"), ("shutdown", "a"), ("routers", "shutdown")]


@pytest.mark.parametrize("method", sorted(module._POOL_METHODS))
def test_role_dispatch_preserves_arguments_and_return_value(pools: SimpleNamespace, method: str) -> None:
    owner = module.InferenceRole("genrm", _configs())
    result = ([object()], object(), 2)
    implementation = MagicMock(return_value=result)
    setattr(pools.instances["b"], method, implementation)
    assert owner.call("b", method, 7, model_id="legacy-alias", tags=["weights"]) is result
    implementation.assert_called_once_with(7, model_id="legacy-alias", tags=["weights"])


def test_role_dispatch_rejects_private_methods_and_unknown_models(pools: SimpleNamespace) -> None:
    owner = module.InferenceRole("teacher", _configs())
    for method in ("initialize", "_init_engines", "__dict__", "snapshot"):
        with pytest.raises(ValueError, match="Unsupported pool method"):
            owner.call("a", method)
    with pytest.raises(KeyError, match="Unknown model"):
        owner.call("missing", "health_check")
    owner.shutdown()
    assert owner.ready() is False
    with pytest.raises(RuntimeError, match="shut down"):
        owner.call("a", "recover")


def test_role_shutdown_attempts_all_pools_on_error(pools: SimpleNamespace) -> None:
    owner = module.InferenceRole("teacher", _configs())
    pools.fail_shutdown = "b"
    with pytest.raises(RuntimeError, match="one or more"):
        owner.shutdown()
    assert pools.events[-3:] == [("shutdown", "b"), ("shutdown", "a"), ("routers", "shutdown")]
    assert owner.ready() is False


@pytest.mark.parametrize("method", sorted(module._POOL_METHODS))
def test_role_facade_forwards_explicit_methods(pools: SimpleNamespace, method: str) -> None:
    owner = module.InferenceRole("genrm", _configs())
    result = ([object()], object(), 3)
    implementation = MagicMock(return_value=result)
    setattr(pools.instances["a"], method, implementation)
    remote = AsyncMock(side_effect=owner.call)
    facade = module.ModelManagerFacade(SimpleNamespace(call=SimpleNamespace(remote=remote)), "a")
    actual = asyncio.run(getattr(facade, method)(model_id="a", flag=True))
    assert actual is result
    remote.assert_awaited_once_with("a", method, model_id="a", flag=True)
    implementation.assert_called_once_with(model_id="a", flag=True)


def test_role_facade_propagates_failure() -> None:
    remote = AsyncMock(side_effect=RuntimeError("pool failed"))
    facade = module.ModelManagerFacade(SimpleNamespace(call=SimpleNamespace(remote=remote)), "a")
    with pytest.raises(RuntimeError, match="pool failed"):
        asyncio.run(facade.onload())


def test_role_facade_is_onloaded_reads_current_pool_state(pools: SimpleNamespace) -> None:
    owner = module.InferenceRole("teacher", _configs())
    implementation = MagicMock(side_effect=[True, False, True])
    pools.instances["a"].is_onloaded = implementation
    remote = AsyncMock(side_effect=owner.call)
    facade = module.ModelManagerFacade(SimpleNamespace(call=SimpleNamespace(remote=remote)), "a")

    for expected in (True, False, True):
        assert asyncio.run(facade.is_onloaded()) is expected
    assert implementation.call_count == 3
    implementation.assert_called_with()
    remote.assert_awaited_with("a", "is_onloaded")


def test_role_model_shutdown_keeps_other_model_routers(pools: SimpleNamespace) -> None:
    owner = module.InferenceRole("teacher", _configs())
    owner.call("a", "shutdown")
    pools.router_cleanup.assert_not_called()
    assert owner.ready() is True
    assert pools.events[-1] == ("shutdown", "a")
    owner.shutdown()
    pools.router_cleanup.assert_called_once_with()
    assert pools.events.count(("shutdown", "a")) == 1


def test_role_last_model_shutdown_stops_routers_once(pools: SimpleNamespace) -> None:
    owner = module.InferenceRole("teacher", _configs())
    owner.call("a", "shutdown")
    owner.call("a", "shutdown")
    pools.router_cleanup.assert_not_called()
    owner.call("b", "shutdown")
    assert set(owner.inference_manager._closed_models) == {"a", "b"}
    assert owner.ready() is False
    owner.call("b", "shutdown")
    owner.shutdown()
    pools.router_cleanup.assert_called_once_with()
    assert pools.events.count(("shutdown", "a")) == 1
    assert pools.events.count(("shutdown", "b")) == 1
    with pytest.raises(RuntimeError, match="shut down"):
        owner.call("a", "onload")


def test_role_failed_model_shutdown_blocks_revival_but_allows_retry(pools: SimpleNamespace) -> None:
    owner = module.InferenceRole("teacher", _configs())
    pools.fail_shutdown = "a"
    with pytest.raises(RuntimeError, match="shutdown failed"):
        owner.call("a", "shutdown")
    assert "a" not in owner.inference_manager._closed_models
    with pytest.raises(RuntimeError, match="closing"):
        owner.call("a", "recover")
    owner.call("b", "shutdown")
    pools.router_cleanup.assert_not_called()
    pools.fail_shutdown = None
    owner.call("a", "shutdown")
    pools.router_cleanup.assert_called_once_with()


def test_role_last_model_shutdown_retries_failed_router_cleanup(pools: SimpleNamespace) -> None:
    owner = module.InferenceRole("teacher", {"a": _configs()["a"]})
    pools.router_cleanup.side_effect = RuntimeError("router cleanup failed")
    with pytest.raises(RuntimeError, match="router cleanup failed"):
        owner.call("a", "shutdown")
    assert owner.ready() is False
    pools.router_cleanup.side_effect = None
    owner.call("a", "shutdown")
    assert pools.router_cleanup.call_count == 2
    assert pools.events.count(("shutdown", "a")) == 1


def test_role_concurrency_groups_reserve_snapshot_and_shutdown_workers() -> None:
    groups = module.InferenceRoleManager.__ray_metadata__.concurrency_groups
    assert groups["snapshot"] == 1
    assert groups["pool"] > 1
    assert groups["control"] == 1
    assert module.InferenceRole.snapshot.__ray_concurrency_group__ == "snapshot"
    assert module.InferenceRole.ready.__ray_concurrency_group__ == "snapshot"
    assert module.InferenceRole.call.__ray_concurrency_group__ == "pool"
    assert module.InferenceRole.shutdown.__ray_concurrency_group__ == "control"


def test_role_same_pool_calls_serialize_without_blocking_snapshot_or_other_pool(pools: SimpleNamespace) -> None:
    owner = module.InferenceRole("teacher", _configs())
    entered, release, queued, second_entered = Event(), Event(), Event(), Event()

    def onload() -> str:
        entered.set()
        assert release.wait(5)
        return "onloaded"

    def offload() -> str:
        second_entered.set()
        return "offloaded"

    def queued_call() -> Any:
        queued.set()
        return owner.call("a", "offload")

    pools.instances["a"].onload = onload
    pools.instances["a"].offload = offload
    pools.instances["b"].health_check = lambda: True
    with ThreadPoolExecutor(max_workers=4) as executor:
        first = executor.submit(owner.call, "a", "onload")
        try:
            assert entered.wait(2)
            second = executor.submit(queued_call)
            assert queued.wait(2)
            assert executor.submit(owner.snapshot).result(timeout=2) is owner.inference_manager.snapshot()
            assert executor.submit(owner.call, "b", "health_check").result(timeout=2) is True
            with pytest.raises(ModelBusyError):
                second.result(timeout=2)
            assert not second_entered.is_set()
        finally:
            release.set()
        assert first.result(timeout=2) == "onloaded"
        assert owner.call("a", "offload") == "offloaded"


def test_local_call_wait_preserves_synchronous_pool_semantics(pools: SimpleNamespace) -> None:
    owner = module.InferenceRole("teacher", {"a": _configs()["a"]})
    entered, release, finished = Event(), Event(), Event()

    def onload() -> str:
        entered.set()
        assert release.wait(5)
        finished.set()
        return "onloaded"

    pools.instances["a"].onload = onload
    with ThreadPoolExecutor(max_workers=2) as executor:
        active = executor.submit(owner.call, "a", "onload")
        assert entered.wait(2)
        waiting = executor.submit(owner.call_wait, "a", "onload")
        assert not finished.wait(0.05)
        release.set()
        assert active.result(timeout=2) == "onloaded"
        assert waiting.result(timeout=2) == "onloaded"


def test_role_shutdown_waits_for_calls_and_rejects_new_operations(
    pools: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = module.InferenceRole("teacher", _configs())
    entered, release, cleanup_entered = Event(), Event(), Event()
    cleanup = owner.inference_manager._close_pool

    def onload() -> None:
        entered.set()
        assert release.wait(5)
        pools.events.append(("onload", "finished"))

    def shutdown_cleanup(*args: Any, **kwargs: Any) -> Any:
        cleanup_entered.set()
        return cleanup(*args, **kwargs)

    pools.instances["a"].onload = onload
    monkeypatch.setattr(owner.inference_manager, "_close_pool", shutdown_cleanup)
    with ThreadPoolExecutor(max_workers=4) as executor:
        active = executor.submit(owner.call, "a", "onload")
        try:
            assert entered.wait(2)
            closing = executor.submit(owner.shutdown)
            assert cleanup_entered.wait(2)
            assert owner.ready() is False
            assert executor.submit(owner.snapshot).result(timeout=2) is owner.inference_manager.snapshot()
            with pytest.raises(RuntimeError, match="shut down"):
                executor.submit(owner.call, "b", "recover").result(timeout=2)
            pools.router_cleanup.assert_not_called()
            assert ("shutdown", "a") not in pools.events
        finally:
            release.set()
        active.result(timeout=2)
        closing.result(timeout=2)
    assert pools.events.index(("onload", "finished")) < pools.events.index(("shutdown", "a"))
    assert pools.events[-1] == ("routers", "shutdown")


def test_role_concurrent_model_shutdown_cleans_routers_once(pools: SimpleNamespace) -> None:
    owner = module.InferenceRole("teacher", _configs())
    barrier = Barrier(2, timeout=5)

    def shutdown(model_id: str) -> str:
        barrier.wait()
        pools.events.append(("shutdown", model_id))
        return model_id

    pools.instances["a"].shutdown = lambda: shutdown("a")
    pools.instances["b"].shutdown = lambda: shutdown("b")
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(owner.call, "a", "shutdown")
        second = executor.submit(owner.call, "b", "shutdown")
        assert first.result(timeout=6) == "a"
        assert second.result(timeout=6) == "b"
    assert owner.ready() is False
    assert pools.events[-1] == ("routers", "shutdown")
    pools.router_cleanup.assert_called_once_with()
    assert owner.call("a", "shutdown") == "a"
    owner.shutdown()
    pools.router_cleanup.assert_called_once_with()


def test_role_concurrent_role_and_model_shutdown_do_not_double_close(
    pools: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = module.InferenceRole("teacher", {"a": _configs()["a"]})
    entered, release, cleanup_entered = Event(), Event(), Event()
    cleanup = owner.inference_manager._close_pool

    def shutdown_pool() -> None:
        entered.set()
        assert release.wait(5)
        pools.events.append(("shutdown", "a"))

    def shutdown_cleanup(*args: Any, **kwargs: Any) -> Any:
        cleanup_entered.set()
        return cleanup(*args, **kwargs)

    pools.instances["a"].shutdown = shutdown_pool
    monkeypatch.setattr(owner.inference_manager, "_close_pool", shutdown_cleanup)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(owner.call, "a", "shutdown")
        try:
            assert entered.wait(2)
            second = executor.submit(owner.shutdown)
            assert cleanup_entered.wait(2)
            pools.router_cleanup.assert_not_called()
        finally:
            release.set()
        first.result(timeout=2)
        second.result(timeout=2)
    assert pools.events.count(("shutdown", "a")) == 1
    pools.router_cleanup.assert_called_once_with()


def test_role_facade_exposes_full_authoritative_snapshot(pools: SimpleNamespace) -> None:
    owner = module.InferenceRole("teacher", _configs())
    remote = AsyncMock(side_effect=owner.snapshot)
    facade = module.ModelManagerFacade(SimpleNamespace(snapshot=SimpleNamespace(remote=remote)), "a")
    snapshot = asyncio.run(facade.get_role_snapshot())
    assert snapshot is owner.inference_manager.snapshot()
    assert [model.model_id for model in snapshot.models] == ["a", "b"]
    remote.assert_awaited_once_with()


def test_role_constructor_error_survives_router_cleanup_failure(pools: SimpleNamespace) -> None:
    pools.fail_construct = "a"
    pools.router_cleanup.side_effect = RuntimeError("router cleanup failed")
    with pytest.raises(RuntimeError, match="constructor failed"):
        module.InferenceRole("teacher", _configs())
    pools.router_cleanup.assert_called_once_with()


@pytest.fixture
def actors(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    owner = MagicMock()
    owner.ready.remote.return_value = True
    facades = [MagicMock(), MagicMock()]
    for facade in facades:
        facade.ready.remote.return_value = True
    role_cls, facade_cls = MagicMock(), MagicMock()
    role_cls.options.return_value.remote.return_value = owner
    facade_cls.options.return_value.remote.side_effect = facades
    monkeypatch.setattr(module, "InferenceRoleManager", role_cls)
    monkeypatch.setattr(module, "InferenceManagerFacade", facade_cls)
    get = MagicMock(side_effect=lambda refs, **kwargs: refs)
    kill = MagicMock()
    monkeypatch.setattr(module.ray, "get", get)
    monkeypatch.setattr(module.ray, "kill", kill)
    return SimpleNamespace(owner=owner, facades=facades, role_cls=role_cls, facade_cls=facade_cls, get=get, kill=kill)


def test_role_factory_creates_one_cpu_owner_and_named_facades(actors: SimpleNamespace) -> None:
    runtime_env = {"env_vars": {"TEST_ROLE": "1"}}
    names = {"a": "legacy-a", "b": "legacy-b"}
    result = module.create_role_managers(SimpleNamespace(), "genrm", _configs(), runtime_env, names)
    assert result == dict(zip(("a", "b"), actors.facades, strict=True))
    actors.role_cls.options.assert_called_once_with(num_cpus=1, num_gpus=0, runtime_env=runtime_env)
    actors.role_cls.options.return_value.remote.assert_called_once_with(Role.GENRM, _configs())
    assert actors.facade_cls.options.call_count == 2
    for index, model_id in enumerate(("a", "b")):
        assert actors.facade_cls.options.call_args_list[index].kwargs == {
            "num_cpus": 0,
            "num_gpus": 0,
            "runtime_env": runtime_env,
            "name": names[model_id],
        }
        assert actors.facade_cls.options.return_value.remote.call_args_list[index].args == (actors.owner, model_id)
        actors.facades[index].ready.remote.assert_called_once_with()
    actors.owner.ready.remote.assert_called_once_with()
    assert actors.get.call_args_list[0].args == (True,)
    actors.kill.assert_not_called()


@pytest.mark.parametrize("failure", ["owner", "spawn", "facade_ready"])
def test_role_factory_failure_cleans_owner_and_created_facades(actors: SimpleNamespace, failure: str) -> None:
    if failure == "owner":
        actors.owner.ready.remote.return_value = False
    elif failure == "spawn":
        actors.facade_cls.options.return_value.remote.side_effect = [actors.facades[0], RuntimeError("spawn failed")]
    else:
        actors.facades[1].ready.remote.return_value = False
    with pytest.raises(RuntimeError):
        module.create_role_managers(SimpleNamespace(), "teacher", _configs())
    actors.owner.shutdown.remote.assert_called_once_with()
    actors.kill.assert_any_call(actors.owner, no_restart=True)
    expected_count = {"owner": 1, "spawn": 2, "facade_ready": 3}[failure]
    assert actors.kill.call_count == expected_count


@pytest.mark.parametrize(
    "configs,names",
    [
        ({}, None),
        (_configs(), {"missing": "name"}),
        (_configs(), {"a": "same", "b": "same"}),
        ({"a": {"args": (), "kwargs": {"defer_init": False}}}, None),
    ],
)
def test_role_factory_rejects_invalid_config_before_spawning(
    actors: SimpleNamespace, configs: dict, names: dict | None
) -> None:
    with pytest.raises(ValueError):
        module.create_role_managers(SimpleNamespace(), "teacher", configs, actor_names=names)
    actors.role_cls.options.assert_not_called()


def test_role_rejects_rollout_pool_before_construction(pools: SimpleNamespace) -> None:
    with pytest.raises(ValueError, match="Unsupported inference pool role"):
        module.InferenceRole("rollout", _configs())
    assert pools.events == []
