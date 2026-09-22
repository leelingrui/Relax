# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""The task owner creates and holds the rollout engine pool."""

from types import SimpleNamespace
from typing import Any

import pytest
from conftest import AwaitableValue


try:
    import ray  # noqa: F401

    from relax.distributed.ray import inference_role
    from relax.distributed.ray import rollout as rollout_module
    from relax.distributed.ray.inference_role import TaskInferenceManager
    from relax.engine.inference.types import Role

    # Ray stores the undecorated class on the actor wrapper, so the entry
    # point's forwarding can be exercised without starting an actor.
    _OriginalRolloutManager = rollout_module.RolloutManager.__ray_metadata__.modified_class

    HAS_DEPS = True
except ImportError:  # pragma: no cover - exercised only without ray/sglang
    HAS_DEPS = False


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="Missing ray/sglang dependencies")


class _FakePool:
    """Records how the owner built it; starts no engine."""

    instances: list["_FakePool"] = []

    def __init__(self, args, pg, *, inference_manager=None, placement_ledger=None, task_inference_manager=None):
        self.args = args
        self.pg = pg
        self.inference_manager = inference_manager
        self.placement_ledger = placement_ledger
        self.task_inference_manager = task_inference_manager
        self.model_pools = {"default": object()}
        self.disposed = False
        self.calls: list[tuple] = []
        _FakePool.instances.append(self)

    def get_primary_router_address(self) -> dict:
        return {"router_ip": "10.0.0.1", "router_port": 31000}

    def get_status(self):
        self.calls.append(("get_status",))
        return "onload"

    async def offload(self):
        self.calls.append(("offload",))
        return "offloaded"

    def dispose(self) -> None:
        self.disposed = True


@pytest.fixture
def owner(monkeypatch: Any) -> TaskInferenceManager:
    _FakePool.instances = []
    monkeypatch.setattr(rollout_module, "RolloutEnginePool", _FakePool)
    return TaskInferenceManager()


def test_create_rollout_role_builds_the_pool_on_the_owner_manager_and_ledger(owner: TaskInferenceManager) -> None:
    router = owner.create_rollout_role(SimpleNamespace(tag="args"), "pg-handle")

    assert router == {"router_ip": "10.0.0.1", "router_port": 31000}
    pool = _FakePool.instances[0]
    assert pool.pg == "pg-handle"
    assert pool.inference_manager.role is Role.ROLLOUT
    assert pool.inference_manager.manager_epoch == owner._control_manager.manager_epoch
    assert pool.placement_ledger is owner._placement_planner
    # The owner holds the pool directly, so there is no host to call back into.
    assert pool.task_inference_manager is None
    assert owner._role_pools[Role.ROLLOUT] is pool.model_pools


def test_create_rollout_role_is_idempotent(owner: TaskInferenceManager) -> None:
    first = owner.create_rollout_role(SimpleNamespace(), "pg-handle")
    second = owner.create_rollout_role(SimpleNamespace(), "pg-handle")

    assert first == second
    assert len(_FakePool.instances) == 1


def test_create_rollout_role_records_the_route_operation(owner: TaskInferenceManager) -> None:
    owner.create_rollout_role(SimpleNamespace(), "pg-handle")

    operation = owner.get_operation("routes:rollout:startup")
    assert operation.status == "completed"
    assert operation.owner_epoch == owner.manager_epoch


def test_rollout_operation_runs_sync_and_async_pool_methods(owner: TaskInferenceManager) -> None:
    owner.create_rollout_role(SimpleNamespace(), "pg-handle")

    assert owner.rollout_operation("get_status") == "onload"
    assert owner.rollout_operation("offload") == "offloaded"
    assert _FakePool.instances[0].calls == [("get_status",), ("offload",)]


def test_rollout_operation_rejects_an_unlisted_method(owner: TaskInferenceManager) -> None:
    owner.create_rollout_role(SimpleNamespace(), "pg-handle")

    with pytest.raises(ValueError, match="Unsupported rollout pool method"):
        owner.rollout_operation("dispose")


def test_rollout_operation_without_a_pool_reports_the_missing_owner(owner: TaskInferenceManager) -> None:
    with pytest.raises(RuntimeError, match="has not been created"):
        owner.rollout_operation("get_status")


def test_shutdown_role_disposes_the_rollout_pool(owner: TaskInferenceManager) -> None:
    owner.create_rollout_role(SimpleNamespace(), "pg-handle")

    owner.shutdown_role(Role.ROLLOUT)

    assert _FakePool.instances[0].disposed
    assert owner._rollout_pool is None
    assert Role.ROLLOUT not in owner._role_pools


def test_shutdown_all_covers_the_rollout_pool(owner: TaskInferenceManager) -> None:
    owner.create_rollout_role(SimpleNamespace(), "pg-handle")

    owner.shutdown_all()

    assert _FakePool.instances[0].disposed


def test_every_listed_rollout_method_exists_on_the_pool() -> None:
    for name in inference_role._ROLLOUT_POOL_METHODS:
        assert callable(getattr(rollout_module.RolloutEnginePool, name, None)), name


# ---------------------------------------------------------------------------
# The rollout Ray entry point holds no engine of its own.
# ---------------------------------------------------------------------------
class _FakeOwnerHandle:
    """Stands in for the task owner's Ray handle."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.rollout_operation = SimpleNamespace(remote=self._record("rollout_operation"))
        self.shutdown_role = SimpleNamespace(remote=self._record("shutdown_role"))

    def _record(self, name: str):
        def remote(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return AwaitableValue(f"{name}:{args[0] if args else ''}")

        return remote


def _shell(owner_handle: Any) -> Any:
    manager = object.__new__(_OriginalRolloutManager)
    manager.pool = None
    manager.task_inference_manager = owner_handle
    manager.workload = SimpleNamespace(rollout_id=7)
    return manager


def test_the_rollout_entry_point_forwards_engine_calls_to_the_owner(patch_ray_get: Any) -> None:
    handle = _FakeOwnerHandle()
    manager = _shell(handle)

    assert manager.get_status() == "rollout_operation:get_status"
    manager.recover_rollout_engines("default")

    assert handle.calls[0] == ("rollout_operation", ("get_status",), {})
    assert handle.calls[1] == ("rollout_operation", ("recover_rollout_engines", "default"), {"rollout_started": True})


@pytest.mark.asyncio
async def test_the_rollout_entry_point_awaits_owner_side_coroutines() -> None:
    handle = _FakeOwnerHandle()
    manager = _shell(handle)

    assert await manager.offload() == "rollout_operation:offload"
    assert handle.calls == [("rollout_operation", ("offload",), {})]


def test_dispose_shuts_the_role_down_through_the_owner(patch_ray_get: Any) -> None:
    handle = _FakeOwnerHandle()
    manager = _shell(handle)

    manager.dispose()

    assert handle.calls == [("shutdown_role", (Role.ROLLOUT,), {})]


def test_startup_writes_the_owner_side_router_endpoint_back_into_args(monkeypatch: Any, patch_ray_get: Any) -> None:
    """The router starts in the owner's process, so its args copy is not
    ours."""
    monkeypatch.setattr(rollout_module, "init_tracking", lambda *a, **k: None)
    monkeypatch.setattr(rollout_module, "tq", SimpleNamespace(init=lambda *a: None, get_client=lambda: None))
    monkeypatch.setattr(rollout_module, "RolloutWorkload", lambda *a, **k: SimpleNamespace(rollout_id=-1))

    handle = SimpleNamespace(
        create_rollout_role=SimpleNamespace(
            remote=lambda *a, **k: AwaitableValue({"router_ip": "10.0.0.1", "router_port": 31000})
        )
    )
    args = SimpleNamespace(
        tq_config=None,
        use_agentic_rollout=False,
        sglang_router_ip="unset",
        sglang_router_port=0,
        # Generation runs in this process, so the shell initializes its own
        # HTTP client; 0 GPUs makes that a no-op here.
        rollout_num_gpus=0,
    )

    manager = object.__new__(_OriginalRolloutManager)
    _OriginalRolloutManager.__init__(manager, args, "pg-handle", inference_manager_handle=handle)

    assert (args.sglang_router_ip, args.sglang_router_port) == ("10.0.0.1", 31000)


def test_the_rollout_entry_point_refuses_to_start_without_an_owner(monkeypatch: Any) -> None:
    """There is no local-pool fallback any more: no owner, no rollout role."""
    monkeypatch.setattr(rollout_module, "init_tracking", lambda *a, **k: None)
    monkeypatch.setattr(rollout_module, "init_http_client", lambda *a, **k: None)
    monkeypatch.setattr(rollout_module, "tq", SimpleNamespace(init=lambda *a: None, get_client=lambda: None))
    monkeypatch.setattr(rollout_module, "RolloutWorkload", lambda *a, **k: SimpleNamespace(rollout_id=-1))

    args = SimpleNamespace(tq_config=None, use_agentic_rollout=False, rollout_num_gpus=0)
    manager = object.__new__(_OriginalRolloutManager)
    with pytest.raises(ValueError, match="task inference manager handle"):
        _OriginalRolloutManager.__init__(manager, args, "pg-handle")


def test_every_declared_concurrency_group_exists() -> None:
    """Ray asserts on an undefined group when the actor is created, which no
    CPU test would otherwise reach."""
    metadata = rollout_module.RolloutManager.__ray_metadata__
    declared = set(metadata.concurrency_groups)
    for name in dir(metadata.modified_class):
        group = getattr(getattr(metadata.modified_class, name), "__ray_concurrency_group__", None)
        assert group is None or group in declared, f"{name} -> {group}"
