# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import threading
from types import SimpleNamespace

import pytest

from relax.distributed.ray.inference_manager import InferenceManager, ModelHandle
from relax.engine.inference.config import ModelConfig
from relax.engine.inference.types import (
    LifecycleState,
    ModelSnapshot,
    ReplicaSnapshot,
    Role,
    RouteMode,
    WeightSource,
)


class FakePool:
    """An engine pool which reports READY while onloaded."""

    def __init__(self, name: str, *, version: str | None = None) -> None:
        policy = version is not None
        self.model_spec = ModelConfig(
            name,
            weight_source=WeightSource.DCS if policy else WeightSource.STATIC,
            route_mode=RouteMode.SGLANG_ROUTER if policy else RouteMode.DIRECT,
        )
        self.onloaded = True
        self.version = version
        self.calls: list[str] = []

    def observe(self) -> ModelSnapshot:
        state = LifecycleState.READY if self.onloaded else LifecycleState.SLEEPING
        replica = ReplicaSnapshot(f"{self.model_spec.name}/replica-0", state, "http://engine-0", self.version)
        return ModelSnapshot(
            self.model_spec.name,
            (replica,),
            "http://router" if self.model_spec.needs_router else None,
            state,
            admission=self.onloaded,
            required_weight_version=self.version,
        )

    def onload(self, tags=None) -> None:
        self.calls.append("onload")
        self.onloaded = True

    def offload(self) -> None:
        self.calls.append("offload")
        self.onloaded = False

    def shutdown(self, planner) -> None:
        self.calls.append("shutdown")

    def get_urls(self) -> list[str]:
        return ["http://engine-0"]


def _manager(**roles: dict[str, FakePool]) -> InferenceManager:
    manager = InferenceManager()
    for role, pools in roles.items():
        manager.register(role, pools)
    return manager


def test_manager_direct_eligible_only_for_ready_direct_replicas() -> None:
    manager = _manager(genrm={"judge": FakePool("judge")}, rollout={"policy": FakePool("policy", version="1")})

    genrm = manager.snapshot(Role.GENRM)
    assert genrm.routing.default_model == "judge"
    assert genrm.models[0].replicas[0].direct_eligible is True
    assert manager.snapshot(Role.ROLLOUT).models[0].replicas[0].direct_eligible is False

    manager.offload(Role.GENRM, "judge")
    replica = manager.snapshot(Role.GENRM).models[0].replicas[0]
    assert replica.state == LifecycleState.SLEEPING
    assert replica.direct_eligible is False


def test_manager_rejects_ready_policy_with_stale_weights() -> None:
    manager = _manager(rollout={"policy": FakePool("policy", version="1")})
    stale = ModelSnapshot(
        "policy",
        (ReplicaSnapshot("policy/replica-0", LifecycleState.READY, "http://engine-0", "0"),),
        "http://router",
        LifecycleState.READY,
        admission=True,
        required_weight_version="1",
    )
    with pytest.raises(ValueError, match="weight version"):
        manager.publish(Role.ROLLOUT, stale)


def test_manager_topology_revision_bumps_only_on_topology_change() -> None:
    manager = _manager(genrm={"judge": FakePool("judge")})
    revision = manager.snapshot(Role.GENRM).topology_revision
    manager.publish(Role.GENRM, manager.snapshot(Role.GENRM).models[0])
    assert manager.snapshot(Role.GENRM).topology_revision == revision
    manager.offload(Role.GENRM, "judge")
    assert manager.snapshot(Role.GENRM).topology_revision == revision
    moved = manager.snapshot(Role.GENRM).models[0]
    replica = ReplicaSnapshot("judge/replica-0", LifecycleState.SLEEPING, "http://engine-1")
    manager.publish(Role.GENRM, ModelSnapshot("judge", (replica,), None, LifecycleState.SLEEPING))
    assert moved.replicas[0].base_url == "http://engine-0"
    assert manager.snapshot(Role.GENRM).topology_revision == revision + 1


def test_manager_drain_closes_admission_and_waits_for_requests() -> None:
    manager = _manager(genrm={"judge": FakePool("judge")})
    assert manager.admit_request(Role.GENRM, "judge", "req-1") == "req-1"

    with pytest.raises(TimeoutError):
        manager.drain([Role.GENRM], timeout=0.01)
    with pytest.raises(RuntimeError, match="not ready"):
        manager.admit_request(Role.GENRM, "judge")

    manager.complete_request("req-1")
    manager.drain([Role.GENRM], timeout=0.01)


def test_manager_switch_offloads_outgoing_before_onloading_incoming() -> None:
    order: list[str] = []
    judge, teacher = FakePool("judge"), FakePool("teacher")
    teacher.onloaded = False
    manager = _manager(genrm={"judge": judge}, teacher={"teacher": teacher})
    judge.offload = lambda: (order.append("judge"), setattr(judge, "onloaded", False))
    teacher.onload = lambda tags=None: (order.append("teacher"), setattr(teacher, "onloaded", True))

    manager.switch([Role.GENRM], [Role.TEACHER], timeout=0.01)

    assert order == ["judge", "teacher"]
    assert manager.snapshot(Role.GENRM).models[0].state == LifecycleState.SLEEPING
    assert manager.snapshot(Role.TEACHER).models[0].state == LifecycleState.READY


def test_manager_shutdown_role_unregisters_the_role() -> None:
    pool = FakePool("judge")
    manager = _manager(genrm={"judge": pool})
    manager.shutdown_role(Role.GENRM)
    assert pool.calls == ["shutdown"]
    assert manager.roles() == ()
    manager.register(Role.GENRM, {"judge": FakePool("judge")})


def test_manager_call_only_exposes_lifecycle_and_queries() -> None:
    manager = _manager(genrm={"judge": FakePool("judge")})
    assert manager.call(Role.GENRM, "judge", "get_urls") == ["http://engine-0"]
    with pytest.raises(ValueError, match="Unsupported"):
        manager.call(Role.GENRM, "judge", "release_memory_occupation")


def test_model_handle_forwards_calls_through_manager() -> None:
    calls = []
    manager = SimpleNamespace(call=SimpleNamespace(remote=lambda *args, **kwargs: calls.append((args, kwargs))))
    ModelHandle(manager, Role.TEACHER, "default").onload.remote(tags=["kv_cache"])
    assert calls == [((Role.TEACHER, "default", "onload"), {"tags": ["kv_cache"]})]


def _shared(**roles: FakePool) -> tuple[InferenceManager, dict[str, FakePool]]:
    """Rollout, GenRM and Teacher on the same GPUs, each in its own phase."""
    from relax.engine.inference.placement import PlacementGroupView, PlacementOwner, PlacementRequest

    pools = {"rollout": FakePool("policy", version="1"), "genrm": FakePool("judge"), "teacher": FakePool("teacher")}
    pools.update(roles)
    manager = _manager(**{role: {pool.model_spec.name: pool} for role, pool in pools.items()})
    view = PlacementGroupView((0, 1), (0, 1), PlacementOwner.CONTROLLER, identity="shared")
    phases = {"rollout": "inference", "genrm": "genrm", "teacher": "teacher"}
    for role, pool in pools.items():
        request = PlacementRequest(f"{role}/{pool.model_spec.name}/group-0", "regular", 2, 1, 2, phases[role], 0)
        manager.plan_placement([request], view)
    manager.switch([Role.ROLLOUT, Role.GENRM, Role.TEACHER], [], timeout=0.01)
    return manager, pools


def _start(target, *args) -> threading.Thread:
    thread = threading.Thread(target=target, args=args)
    thread.start()
    thread.join(0.1)
    return thread


def test_manager_switch_keeps_scorers_mutually_exclusive() -> None:
    manager, pools = _shared()
    judge, teacher = pools["genrm"], pools["teacher"]
    both_ready = []

    def onload(pool, tags=None):
        pool.onloaded = True
        both_ready.append(judge.onloaded and teacher.onloaded)

    judge.onload = lambda tags=None: onload(judge)
    teacher.onload = lambda tags=None: onload(teacher)

    manager.switch([Role.ROLLOUT], [Role.GENRM], timeout=0.01)
    assert manager.snapshot(Role.TEACHER).phase == "genrm"
    waiter = _start(manager.switch, [Role.ROLLOUT], [Role.TEACHER], 5.0)
    # The teacher waits while GenRM holds the shared GPUs.
    assert waiter.is_alive() and not teacher.onloaded
    manager.switch([Role.GENRM], [], timeout=0.01)
    waiter.join(5.0)

    assert not waiter.is_alive() and teacher.onloaded and not judge.onloaded
    assert both_ready == [False, False]
    assert manager.snapshot(Role.GENRM).phase == "teacher"


def test_manager_switch_times_out_while_another_scorer_holds_the_gpus() -> None:
    manager, _ = _shared()
    manager.switch([], [Role.GENRM], timeout=0.01)
    with pytest.raises(TimeoutError, match="genrm"):
        manager.switch([], [Role.TEACHER], timeout=0.01)
    manager.switch([Role.GENRM], [Role.TEACHER], timeout=0.01)
    assert manager.snapshot(Role.TEACHER).models[0].state == LifecycleState.READY


def test_manager_switch_rejects_activating_two_conflicting_roles() -> None:
    manager, pools = _shared()
    for roles in ([Role.ROLLOUT, Role.GENRM], [Role.ROLLOUT, Role.TEACHER], [Role.GENRM, Role.TEACHER]):
        with pytest.raises(ValueError, match="cannot be active together"):
            manager.switch([], roles, timeout=0.01)
    assert not any(pool.onloaded for pool in pools.values())


def test_manager_switch_reactivates_the_resident_scorer() -> None:
    manager, pools = _shared()
    manager.switch([], [Role.GENRM], timeout=0.01)
    manager.switch([], [Role.GENRM], timeout=0.01)
    assert pools["genrm"].calls == ["offload", "onload", "onload"]


def test_manager_switch_releases_the_gpus_when_a_scorer_fails_to_load() -> None:
    manager, pools = _shared()
    pools["genrm"].onload = lambda tags=None: (_ for _ in ()).throw(RuntimeError("OOM"))

    with pytest.raises(RuntimeError, match="OOM"):
        manager.switch([], [Role.GENRM], timeout=0.01)

    manager.switch([], [Role.TEACHER], timeout=0.01)
    assert pools["teacher"].onloaded


def test_manager_snapshot_reports_the_generation_phase_without_a_scorer() -> None:
    manager = _manager(genrm={"judge": FakePool("judge")})
    assert manager.snapshot(Role.GENRM).phase == "inference"
    assert manager.snapshot(Role.GENRM).to_dict()["phase"] == "inference"


def test_manager_topology_revision_bumps_when_a_replica_restarts_in_place() -> None:
    pool = FakePool("judge")
    pool.recover = lambda: None
    manager = _manager(genrm={"judge": pool})
    revision = manager.snapshot(Role.GENRM).topology_revision

    manager.recover(Role.GENRM, "judge")

    snapshot = manager.snapshot(Role.GENRM)
    assert snapshot.models[0].replicas[0].base_url == "http://engine-0"
    assert snapshot.topology_revision == revision + 1
    # Waking a sleeping replica is not a restart.
    manager.offload(Role.GENRM, "judge")
    manager.onload(Role.GENRM, "judge")
    assert manager.snapshot(Role.GENRM).topology_revision == revision + 1


def test_manager_direct_onload_blocks_a_conflicting_switch() -> None:
    manager, pools = _shared()
    manager.onload(Role.GENRM, "judge")

    with pytest.raises(TimeoutError, match="genrm"):
        manager.switch([], [Role.TEACHER], timeout=0.01)
    assert not pools["teacher"].onloaded


def test_manager_direct_offload_releases_the_gpus() -> None:
    manager, pools = _shared()
    manager.switch([], [Role.GENRM], timeout=0.01)
    waiter = _start(manager.switch, [], [Role.TEACHER], 5.0)
    assert waiter.is_alive()

    # Training offloads the scorer directly, not through a switch.
    manager.offload(Role.GENRM, "judge")
    waiter.join(5.0)

    assert not waiter.is_alive() and pools["teacher"].onloaded
    assert manager.snapshot(Role.GENRM).phase == "teacher"


def test_manager_recover_waits_for_a_conflicting_role() -> None:
    manager, pools = _shared()
    recovered = []
    pools["teacher"].recover = lambda: recovered.append(True)
    manager.switch([], [Role.GENRM], timeout=0.01)

    waiter = _start(manager.recover, Role.TEACHER, "teacher")
    assert waiter.is_alive() and not recovered
    manager.switch([Role.GENRM], [], timeout=0.01)
    waiter.join(5.0)

    assert not waiter.is_alive() and recovered


def test_manager_onload_outside_a_switch_waits_for_the_scorer() -> None:
    manager, pools = _shared()
    manager.switch([], [Role.GENRM], timeout=0.01)
    # The role that holds the GPUs may still reload itself.
    manager.onload(Role.GENRM, "judge")

    # A weight sync waits for the release.
    waiter = _start(manager.onload, Role.ROLLOUT, "policy")
    assert waiter.is_alive() and not pools["rollout"].onloaded

    manager.switch([Role.GENRM], [], timeout=0.01)
    waiter.join(5.0)
    assert not waiter.is_alive() and pools["rollout"].onloaded
    assert manager.snapshot(Role.ROLLOUT).phase == "inference"


def test_manager_split_roles_load_without_waiting() -> None:
    judge, policy = FakePool("judge"), FakePool("policy", version="1")
    manager = _manager(genrm={"judge": judge}, rollout={"policy": policy})
    manager.offload(Role.ROLLOUT, "policy")
    manager.onload(Role.GENRM, "judge")
    manager.onload(Role.ROLLOUT, "policy")
    assert judge.onloaded and policy.onloaded


def test_manager_failed_offload_keeps_the_gpus_held() -> None:
    manager, pools = _shared()
    manager.switch([], [Role.GENRM], timeout=0.01)
    pools["genrm"].offload = lambda: (_ for _ in ()).throw(RuntimeError("release failed"))

    with pytest.raises(RuntimeError, match="release failed"):
        manager.offload(Role.GENRM, "judge")

    # DEAD, but the memory may still be held.
    assert manager.snapshot(Role.GENRM).models[0].state == LifecycleState.DEAD
    with pytest.raises(TimeoutError, match="genrm"):
        manager.switch([], [Role.TEACHER], timeout=0.01)
    assert not pools["teacher"].onloaded


def test_manager_shutdown_releases_the_gpus_only_once_it_completes() -> None:
    manager, pools = _shared()
    manager.switch([], [Role.GENRM], timeout=0.01)
    stopping, stopped = threading.Event(), threading.Event()

    def shutdown(planner) -> None:
        stopping.set()
        stopped.wait(5.0)

    pools["genrm"].shutdown = shutdown
    closer = _start(manager.shutdown, Role.GENRM, "judge")
    assert stopping.wait(5.0)
    waiter = _start(manager.switch, [], [Role.TEACHER], 5.0)
    assert waiter.is_alive() and not pools["teacher"].onloaded

    stopped.set()
    closer.join(5.0)
    waiter.join(5.0)
    assert not waiter.is_alive() and pools["teacher"].onloaded


def test_manager_failed_shutdown_keeps_the_gpus_held() -> None:
    manager, pools = _shared()
    manager.switch([], [Role.GENRM], timeout=0.01)
    pools["genrm"].shutdown = lambda planner: (_ for _ in ()).throw(RuntimeError("kill failed"))

    with pytest.raises(RuntimeError, match="kill failed"):
        manager.shutdown(Role.GENRM, "judge")
    with pytest.raises(TimeoutError, match="genrm"):
        manager.switch([], [Role.TEACHER], timeout=0.01)


def test_manager_weights_only_onload_holds_the_gpus() -> None:
    manager, pools = _shared()
    rollout = pools["rollout"]

    def onload(tags=None) -> None:
        # Like the engine pool, only a KV-cache onload reports the model loaded.
        rollout.onloaded = tags is None

    rollout.onload = onload
    manager.onload(Role.ROLLOUT, "policy", ["weights"])

    assert manager.snapshot(Role.ROLLOUT).models[0].state == LifecycleState.SLEEPING
    with pytest.raises(TimeoutError, match="rollout"):
        manager.switch([], [Role.TEACHER], timeout=0.01)
    manager.switch([Role.ROLLOUT], [Role.TEACHER], timeout=0.01)
    assert pools["teacher"].onloaded


def test_manager_failed_role_shutdown_keeps_the_gpus_held() -> None:
    manager, pools = _shared()
    manager.switch([], [Role.GENRM], timeout=0.01)
    stop = pools["genrm"].shutdown
    pools["genrm"].shutdown = lambda planner: (_ for _ in ()).throw(RuntimeError("kill failed"))

    with pytest.raises(RuntimeError, match="genrm"):
        manager.shutdown_role(Role.GENRM)
    assert "genrm" in manager.roles()
    with pytest.raises(TimeoutError, match="genrm"):
        manager.switch([], [Role.TEACHER], timeout=0.01)

    pools["genrm"].shutdown = stop
    manager.shutdown_role(Role.GENRM)
    assert "genrm" not in manager.roles()
    manager.switch([], [Role.TEACHER], timeout=0.01)
    assert pools["teacher"].onloaded


def test_manager_role_shutdown_retry_stops_only_the_failed_models() -> None:
    judge, backup = FakePool("judge"), FakePool("backup")
    manager = _manager(genrm={"judge": judge, "backup": backup})
    backup.shutdown = lambda planner: (_ for _ in ()).throw(RuntimeError("kill failed"))

    with pytest.raises(RuntimeError):
        manager.shutdown_role(Role.GENRM)
    assert manager.model_ids(Role.GENRM) == ("backup",)
    assert judge.calls == ["shutdown"]
