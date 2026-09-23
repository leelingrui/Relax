# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Core lifecycle behavior of ``MultiEngineManager``, exercised without a Ray
cluster: fake engine handles stand in for Ray ObjectRefs, and ``ray.get``/
``ray.kill`` are patched onto the module directly.

This is the shared skeleton behind both the GenRM and the Teacher engine
adapters, so a regression here silently breaks both judge serving and OPD
teacher recovery/offload-onload.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


try:
    import ray  # noqa: F401

    from relax.distributed.ray.multi_engine_manager import MultiEngineManager
    from relax.engine.inference.manager import InferenceManager
    from relax.engine.inference.types import Role, RoutingSpec

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="requires ray")


class _RemoteCall:
    """Fakes ``engine.method.remote()``: returns a token that the patched
    ``ray.get`` resolves to whatever the engine was configured to return (or
    raises, for a dead engine)."""

    def __init__(self, engine: "_FakeEngine", method: str):
        self._engine = engine
        self._method = method

    def remote(self, **kwargs):
        return (self._engine, self._method)


class _FakeEngine:
    def __init__(self, name: str, *, dead_methods: frozenset = frozenset()):
        self.name = name
        self.dead_methods = dead_methods
        self.calls: list[str] = []

    def __getattr__(self, method: str):
        return _RemoteCall(self, method)


class _FakeManager(MultiEngineManager):
    """A minimal concrete manager: one engine per rank, no placement group
    (hooks return sentinel values that the test never inspects)."""

    def __init__(self, num_slots: int, *, owns_pg: bool = False, log_prefix: str = "[fake]"):
        self._made: list[_FakeEngine] = []
        self._dead_at_init: set[int] = set()
        self._removed_pg = False
        self._instance_owns_pg = owns_pg
        manager = InferenceManager(Role.GENRM)
        super().__init__(
            SimpleNamespace(),
            num_slots=num_slots,
            engine_actor_cls=_FakeEngineActorCls,
            log_prefix=log_prefix,
            inference_manager=manager,
            skip_init=True,
        )
        manager.configure_routes(RoutingSpec(default_model="default"), operation_id="routes")
        self.initialize()

    def _resolve_planned_placement(self, rank):
        return ((f"pg-{rank}", [0], [0]), self._instance_owns_pg, 0, None)

    def _ray_resource_kwargs(self, rank):
        return {}

    def _allocate_engine_addr_and_ports(self, *, new_engines):
        return {rank: {"host": "h", "port": 1} for rank, _ in new_engines}

    def _build_engine_env_vars(self):
        return {}


class _FakeEngineActorCls:
    """Stand-in engine "actor class"; ``ray.remote(cls)`` in the base class
    just needs something ``.options(...).remote(...)`` works on."""


@pytest.fixture(autouse=True)
def _patch_ray(monkeypatch):
    """Patch the ray module used by multi_engine_manager: ``ray.remote`` wraps
    our fake class into something whose ``.options().remote()`` returns a
    _FakeEngine; ``ray.get`` resolves _RemoteCall tokens; ``ray.kill`` is a no-
    op recorder."""
    import relax.distributed.ray.multi_engine_manager as mem

    created: list[_FakeEngine] = []
    killed: list[_FakeEngine] = []

    def fake_remote(cls):
        if cls is not _FakeEngineActorCls:
            return cls  # pass through decorators applied to real classes elsewhere

        class _Options:
            @staticmethod
            def options(**kwargs):
                class _Ctor:
                    @staticmethod
                    def remote(args, *, rank, worker_type, base_gpu_id, **ctor_kwargs):
                        engine = _FakeEngine(f"engine-{rank}")
                        created.append(engine)
                        return engine

                return _Ctor

        return _Options

    def fake_get(handle_or_list, timeout=None):
        if isinstance(handle_or_list, list):
            return [fake_get(h) for h in handle_or_list]
        engine, method = handle_or_list
        if method in engine.dead_methods:
            raise ConnectionError(f"{engine.name} is dead for {method}")
        engine.calls.append(method)
        return True

    def fake_kill(engine):
        killed.append(engine)

    monkeypatch.setattr(mem.ray, "remote", fake_remote)
    monkeypatch.setattr(mem.ray, "get", fake_get)
    monkeypatch.setattr(mem.ray, "kill", fake_kill)
    monkeypatch.setattr(mem.ray.exceptions, "RayActorError", RuntimeError, raising=False)

    yield SimpleNamespace(created=created, killed=killed)


def test_fanout_isolates_one_dead_engine_from_the_rest(_patch_ray):
    manager = _FakeManager(num_slots=3)
    for engine in manager.all_engines:
        engine.calls.clear()  # drop the init() calls made during construction
    dead_engine = manager.all_engines[1]
    dead_engine.dead_methods = frozenset({"release_memory_occupation"})

    dead_ranks = manager._fanout("release_memory_occupation")

    assert dead_ranks == [1]
    # The other two engines still got the call.
    assert manager.all_engines[0].calls == ["release_memory_occupation"]
    assert manager.all_engines[2].calls == ["release_memory_occupation"]


def test_fanout_reraises_non_dead_exceptions(_patch_ray, monkeypatch):
    import relax.distributed.ray.multi_engine_manager as mem

    manager = _FakeManager(num_slots=1)

    def raise_value_error(handle_or_list, timeout=None):
        raise ValueError("a real bug, not a dead engine")

    monkeypatch.setattr(mem.ray, "get", raise_value_error)

    with pytest.raises(ValueError, match="a real bug"):
        manager._fanout("release_memory_occupation")


def test_offload_onload_are_idempotent_when_state_unchanged(_patch_ray):
    manager = _FakeManager(num_slots=2)
    for engine in manager.all_engines:
        engine.calls.clear()  # drop the init() calls made during construction
    assert manager.is_onloaded()

    manager.offload()
    assert not manager.is_onloaded()
    for engine in manager.all_engines:
        assert engine.calls == ["release_memory_occupation"]

    # A second offload while already offloaded must not re-fire the RPC.
    manager.offload()
    for engine in manager.all_engines:
        assert engine.calls == ["release_memory_occupation"]

    manager.onload()
    assert manager.is_onloaded()
    for engine in manager.all_engines:
        assert engine.calls == ["release_memory_occupation", "resume_memory_occupation", "get_inference_observation"]

    # A second onload (no tags) while already onloaded must not re-fire the RPC.
    manager.onload()
    for engine in manager.all_engines:
        assert engine.calls == ["release_memory_occupation", "resume_memory_occupation", "get_inference_observation"]


def test_partial_onload_keeps_offload_enabled_without_marking_memory_ready(_patch_ray):
    manager = _FakeManager(num_slots=1)
    manager.offload()
    manager.onload(tags=["weights"])
    assert manager.is_onloaded()
    assert not manager._memory_ready
    engine = manager.all_engines[0]
    engine.calls.clear()
    manager.offload()
    assert engine.calls == ["release_memory_occupation"]
    manager.onload()
    assert manager._memory_ready


def test_retire_engines_kills_and_nulls_the_slot(_patch_ray):
    manager = _FakeManager(num_slots=2)
    dead_engine = manager.all_engines[0]

    manager._retire_engines([0])

    assert manager.all_engines[0] is None
    assert manager.all_engines[1] is not None  # untouched
    assert dead_engine in _patch_ray.killed


def test_recover_rebuilds_only_the_dead_slot(_patch_ray):
    manager = _FakeManager(num_slots=2)
    original_engine_1 = manager.all_engines[1]
    manager.all_engines[0] = None  # simulate a prior retirement

    rebuilt = manager.recover()

    assert rebuilt == {0}
    assert manager.all_engines[0] is not None
    assert manager.all_engines[0] is not original_engine_1
    assert manager.all_engines[1] is original_engine_1  # untouched


def test_recover_raises_on_total_wipeout(_patch_ray, monkeypatch):
    import relax.distributed.ray.multi_engine_manager as mem

    manager = _FakeManager(num_slots=1)
    manager.all_engines[0] = None

    def failing_options(**kwargs):
        class _Ctor:
            @staticmethod
            def remote(*args, **kwargs2):
                raise RuntimeError("scheduling failed, node is gone")

        return _Ctor

    class _FailingActor:
        options = staticmethod(failing_options)

    monkeypatch.setattr(mem.ray, "remote", lambda cls: _FailingActor)

    with pytest.raises(RuntimeError, match="could not be rebuilt"):
        manager.recover()


def test_shutdown_removes_owned_placement_group_but_not_borrowed_one(_patch_ray, monkeypatch):
    # ray.util.placement_group is shadowed as an attribute by a same-named
    # function on the ray.util package, so it must be patched via sys.modules
    # (where the actual submodule lives) rather than a dotted monkeypatch path.
    placement_group_submodule = sys.modules["ray.util.placement_group"]
    removed_pgs = []
    monkeypatch.setattr(placement_group_submodule, "remove_placement_group", lambda pg: removed_pgs.append(pg))

    owning_manager = _FakeManager(num_slots=1, owns_pg=True)
    owning_manager.shutdown()
    assert removed_pgs == ["pg-0"]

    removed_pgs.clear()
    borrowing_manager = _FakeManager(num_slots=1, owns_pg=False)
    borrowing_manager.shutdown()
    assert removed_pgs == []


def test_shutdown_releases_the_planned_slice_before_removing_its_group(_patch_ray, monkeypatch):
    from relax.engine.inference.placement import (
        PlacementGroupView,
        PlacementOwner,
        PlacementPlanner,
        PlacementRequest,
    )

    placement_group_submodule = sys.modules["ray.util.placement_group"]
    removed_pgs = []
    monkeypatch.setattr(placement_group_submodule, "remove_placement_group", lambda pg: removed_pgs.append(pg))

    ledger = PlacementPlanner()
    view = PlacementGroupView((0,), (0,), PlacementOwner.MANAGER, identity="own-pg")
    request = PlacementRequest(
        group_id="teacher/math/replica-0",
        worker_type="regular",
        num_gpus=1,
        num_gpus_per_engine=1,
        num_gpus_per_node=8,
    )

    class _PlannedManager(_FakeManager):
        """A manager whose slot comes from the shared ledger, as the migrated
        adapters' slots do."""

        def _resolve_planned_placement(self, rank):
            (planned,) = ledger.plan((request,), view)
            return ("own-pg", [0], [0]), True, 0, planned

        def _release_placement(self, placement):
            return ledger.release(placement)

    manager = _PlannedManager(num_slots=1, owns_pg=True)
    assert len(ledger.allocations(view)) == 1

    manager.shutdown()

    assert removed_pgs == ["own-pg"]
    assert ledger.allocations(view) == ()


def test_manager_onload_skips_resume_for_newly_rebuilt_engine(_patch_ray):
    manager = _FakeManager(num_slots=2)
    manager.offload()
    manager._retire_engines([0])
    survivor = manager.all_engines[1]
    survivor.calls.clear()

    manager.onload()

    assert manager.all_engines[0].calls == ["init", "get_inference_observation"]
    assert survivor.calls == ["resume_memory_occupation", "get_inference_observation"]
    assert manager.is_onloaded()


def test_manager_onload_rebuilds_engine_that_died_while_sleeping(_patch_ray):
    manager = _FakeManager(num_slots=1)
    manager.offload()
    old = manager.all_engines[0]
    old.dead_methods = frozenset({"resume_memory_occupation"})

    manager.onload()

    assert old in _patch_ray.killed
    assert manager.all_engines[0] is not old
    assert manager.all_engines[0].calls == ["init", "get_inference_observation"]
    assert manager.is_onloaded()


def test_pool_snapshot_keeps_replica_identity_and_surviving_admission(_patch_ray, monkeypatch):
    from relax.distributed.ray import multi_engine_manager as module
    from relax.engine.inference.types import LifecycleState

    manager = _FakeManager(num_slots=2)
    manager.router_url = "http://router"
    manager._retire_engines([0])
    monkeypatch.setattr(
        module.ray,
        "get",
        lambda *args, **kwargs: {"healthy": True, "router_registered": True, "base_url": "http://survivor"},
    )
    manager._publish_engine_state()
    snapshot = manager.inference_manager.snapshot(role=Role.GENRM)
    assert snapshot.models[0].admission
    assert snapshot.models[0].replicas[0].state == LifecycleState.DEAD
    assert snapshot.models[0].replicas[1].engine_id == "default/replica-1"
    monkeypatch.setattr(module.ray, "get", lambda *args, **kwargs: pytest.fail("Discovery must not issue RPCs"))
    assert snapshot.to_dict("dead")["models"]["default"]["engines"][0]["engine_id"] == "default/replica-0"
    assert manager.inference_manager.snapshot(role=Role.GENRM) == snapshot


def test_pool_address_allocation_failure_rolls_back_created_actors(_patch_ray, monkeypatch):
    manager = _FakeManager(num_slots=2)
    manager._retire_engines([0, 1])

    def fail_ports(**kwargs):
        raise RuntimeError("port allocation failed")

    monkeypatch.setattr(manager, "_allocate_engine_addr_and_ports", fail_ports)
    with pytest.raises(RuntimeError, match="port allocation failed"):
        manager._init_engines([0, 1])
    assert manager.all_engines == [None, None]
    assert not manager._engine_placements
    assert all(engine in _patch_ray.killed for engine in _patch_ray.created)


def test_pool_recovery_retires_surviving_multinode_followers(_patch_ray):
    from relax.engine.inference.specs import replicas_from_slots

    manager = _FakeManager(num_slots=2)
    manager.nodes_per_engine = 2
    manager.replica_specs = replicas_from_slots("default", 2, 2)
    follower = manager.all_engines[1]
    manager.all_engines[0] = None
    rebuilt = manager.recover()
    assert rebuilt == {0, 1}
    assert follower in _patch_ray.killed
    assert all(engine is not None for engine in manager.all_engines)


@pytest.mark.parametrize("role", ["teacher", "genrm"])
@pytest.mark.parametrize("defer_init", [False, True])
def test_model_pool_real_constructor_with_fake_engines(monkeypatch, role, defer_init):
    from unittest.mock import MagicMock

    from relax.distributed.ray import genrm, teacher_manager
    from relax.distributed.ray.model_pool import ModelPool, create_model_pool
    from relax.engine.inference.placement import PlacementPlanner

    args = SimpleNamespace(
        num_gpus_per_node=4,
        rollout_num_gpus=0,
        fully_async=False,
        genrm_num_gpus=2,
        genrm_num_gpus_per_engine=2,
        genrm_model_path="judge-checkpoint",
        genrm_engine_config={"context_length": 1024},
        debug_train_only=False,
    )
    pg = ("pg", [0, 1], [0, 1])
    rollout = sys.modules["relax.distributed.ray.rollout"]
    monkeypatch.setattr(rollout, "_start_router", lambda *a, **k: ("router", 3100))
    monkeypatch.setattr(rollout, "stop_launched_routers", MagicMock())
    monkeypatch.setattr(genrm, "init_http_client", lambda args: None)
    monkeypatch.setattr(genrm, "GenRMEngine", _FakeEngineActorCls)
    monkeypatch.setattr(
        genrm,
        "_allocate_genrm_engine_addr_and_ports",
        lambda **kw: {rank: {"host": "h", "port": 1} for rank, _ in kw["new_engines"]},
    )
    monkeypatch.setattr(teacher_manager, "SGLangEngine", _FakeEngineActorCls)
    monkeypatch.setattr(
        teacher_manager,
        "build_teacher_overrides",
        lambda *a, **k: {"model_path": "teacher-checkpoint", "context_length": 2048},
    )
    monkeypatch.setattr(teacher_manager, "build_teacher_engine_args", lambda args, overrides: SimpleNamespace())
    monkeypatch.setattr(teacher_manager, "find_available_port", lambda port: port)
    monkeypatch.setattr(
        teacher_manager,
        "_allocate_rollout_engine_addr_and_ports_normal",
        lambda **kw: ({rank: {"host": "h", "port": 1} for rank, _ in kw["rollout_engines"]}, None),
    )
    kwargs = (
        {"pg": pg}
        if role == "genrm"
        else {
            "num_replicas": 1,
            "gpus_per_replica": 2,
            "pg": pg,
            "shared_pg": True,
        }
    )
    manager = InferenceManager(Role(role))
    ledger = PlacementPlanner()
    pool = create_model_pool(
        role, args, inference_manager=manager, placement_manager_handle=ledger, defer_init=defer_init, **kwargs
    )
    assert pool.inference_manager is manager
    manager.configure_routes(RoutingSpec(default_model="default"), operation_id="routes")
    assert type(pool) is ModelPool
    adapter = pool.backend
    assert not isinstance(adapter, MultiEngineManager)
    assert "all_engines" not in vars(adapter)
    assert "num_new_engines" not in vars(adapter)
    assert pool.inference_manager.snapshot().role == role
    assert adapter.model_spec.model_path == ("judge-checkpoint" if role == "genrm" else "teacher-checkpoint")
    assert adapter.model_spec.engine_groups[0].num_gpus_per_engine == 2
    assert adapter.model_spec.engine_groups[0].overrides["context_length"] == (1024 if role == "genrm" else 2048)
    if defer_init:
        assert adapter.num_new_engines == 0
        pool.initialize()
    assert adapter.num_new_engines == 1
    assert len(adapter.engines) == 1
    pool.offload()
    assert not pool.is_onloaded()
    pool.onload()
    assert pool.is_onloaded()
    pool.shutdown()
    # The replica's slice was recorded in, and returned to, the one ledger.
    assert ledger.allocations() == ()


def test_model_pool_runtime_is_onloaded_uses_runtime():
    from unittest.mock import MagicMock

    from relax.distributed.ray.model_pool import ModelPool

    manager = MagicMock(spec=["bind_pool", "onload", "offload", "recover", "health_check", "shutdown"])
    runtime = MagicMock(spec=["is_onloaded"])
    runtime.is_onloaded.return_value = False
    pool = ModelPool.from_runtime(manager, "model", runtime)
    manager.bind_pool.assert_called_once_with("model", runtime)
    assert pool.is_onloaded() is False
    for method in ("onload", "offload", "recover", "health_check", "shutdown"):
        getattr(pool, method)()
        getattr(manager, method).assert_called_once_with("model")


def test_multi_instance_genrm_slices_do_not_collide_in_one_ledger():
    """Instances of a multi-instance role share one placement group, so their
    slices must stay distinct in the single task-level ledger."""
    from relax.distributed.ray.genrm import GenRMEngineAdapter
    from relax.engine.inference.placement import PlacementOwner, PlacementPlanner

    ledger = PlacementPlanner()
    pg = ("shared", list(range(8)), list(range(8)))
    planned = []
    for index, model_id in enumerate(("judge-a", "judge-b")):
        adapter = object.__new__(GenRMEngineAdapter)
        adapter.args = SimpleNamespace(
            num_gpus_per_node=4,
            rollout_num_gpus=4,
            fully_async=False,
            genrm_num_gpus=2,
            genrm_num_gpus_per_engine=2,
        )
        adapter.pg = pg
        adapter.nodes_per_engine = 1
        adapter.num_gpu_per_engine = 2
        adapter.bundle_offset = index * 2
        adapter.placement_owner = PlacementOwner.CONTROLLER
        adapter._placement_ledger = ledger
        adapter._placement_model_id = model_id
        planned.append(adapter._resolve_planned_placement(0)[3])

    # Both sit behind the rollout region, at their own instance offset.
    assert [item.reserved_offset for item in planned] == [4, 6]
    assert {item.group_id for item in ledger.allocations()} == {
        "genrm/judge-a/replica-0",
        "genrm/judge-b/replica-0",
    }


def test_genrm_init_uses_own_router_without_dcs():
    """The judge's static weights never join DCS; the adapter says so.

    This used to be hardcoded in a ``GenRMEngine.init`` override, which meant
    the judge could not start through the common engine path.
    """
    from relax.distributed.ray.genrm import GenRMEngineAdapter

    adapter = object.__new__(GenRMEngineAdapter)
    adapter.router_ip = "genrm-router"
    adapter.router_port = 3200
    addr = {"host": "judge.test", "port": 16000, "nccl_port": 16001, "dist_init_addr": "judge.test:16002"}

    result = adapter._build_engine_init_kwargs(0, addr)

    assert result == dict(
        addr,
        router_ip="genrm-router",
        router_port=3200,
        skip_dcs_registration=True,
        skip_router_registration=False,
    )
    assert "skip_dcs_registration" not in addr


def test_genrm_without_a_router_skips_router_registration():
    """debug_train_only leaves the judge with no router to register at."""
    from relax.distributed.ray.genrm import GenRMEngineAdapter

    adapter = object.__new__(GenRMEngineAdapter)
    adapter.router_ip = ""
    adapter.router_port = 0

    result = adapter._build_engine_init_kwargs(0, {})

    assert result["skip_router_registration"] is True
