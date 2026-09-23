# Copyright (c) 2026 Relax Authors. All Rights Reserved.

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
