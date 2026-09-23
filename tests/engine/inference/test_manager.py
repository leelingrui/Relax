# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from dataclasses import replace

import pytest

from relax.engine.inference.capabilities import WeightSource
from relax.engine.inference.config import EngineGroupConfig, ModelConfig, SglangConfig
from relax.engine.inference.manager import InferenceManager, PreparationEvidence
from relax.engine.inference.specs import (
    EngineGroupSpec,
    ModelSpec,
    ReplicaSpec,
    RoleSpec,
    replicas_from_slots,
)
from relax.engine.inference.types import LifecycleState, ModelSnapshot, ReplicaSnapshot, Role, RoutingSpec


def test_manager_registration_is_idempotent_and_does_not_grant_admission():
    manager = InferenceManager(Role.TEACHER)
    spec = ModelSpec("teacher", "checkpoint", weight_source=WeightSource.CHECKPOINT)
    assert manager.register_model(spec, operation_id="register") == spec
    assert manager.register_model(spec, operation_id="register") == spec
    snapshot = manager.snapshot()
    assert len(snapshot.models) == 1
    assert snapshot.models[0].replicas == ()
    assert snapshot.models[0].state is None
    assert not snapshot.models[0].admission
    with pytest.raises(ValueError, match="Operation conflict"):
        manager.register_model(replace(spec, model_path="other"), operation_id="register")
    with pytest.raises(ValueError, match="configuration conflict"):
        manager.register_model(replace(spec, model_path="other"), operation_id="other")


@pytest.mark.parametrize("fails", [False, True])
def test_manager_shutdown_is_terminal_after_topology_callback(fails):
    from unittest.mock import MagicMock

    manager = InferenceManager(Role.ROLLOUT)
    manager.register_model(
        ModelSpec("model", "checkpoint", weight_source=WeightSource.POLICY), operation_id="register"
    )
    pool = MagicMock()
    manager.bind_pool("model", pool)

    def shutdown():
        manager.invalidate_model("model", state=LifecycleState.STARTING)
        if fails:
            raise RuntimeError("shutdown failed")

    pool.shutdown.side_effect = shutdown
    if fails:
        with pytest.raises(RuntimeError, match="shutdown failed"):
            manager.shutdown("model")
    else:
        manager.shutdown("model")
    assert manager.snapshot().models[0].state == LifecycleState.DEAD
    assert not manager.snapshot().models[0].admission


def test_manager_routes_validate_before_atomic_publication_and_seal_registration():
    manager = InferenceManager(Role.TEACHER)
    manager.register_model(
        ModelSpec("teacher", "checkpoint", weight_source=WeightSource.CHECKPOINT), operation_id="register"
    )
    before = manager.snapshot()
    with pytest.raises(ValueError, match="unregistered"):
        manager.configure_routes(RoutingSpec(default_model="missing"), operation_id="routes")
    assert manager.snapshot() is before
    routing = RoutingSpec(default_model="teacher")
    manager.configure_routes(routing, operation_id="routes")
    manager.configure_routes(routing, operation_id="routes")
    with pytest.raises(ValueError, match="sealed"):
        manager.register_model(
            ModelSpec("late", "checkpoint", weight_source=WeightSource.CHECKPOINT), operation_id="late"
        )
    assert manager.snapshot().routing == routing


@pytest.mark.parametrize("source", [WeightSource.POLICY, WeightSource.CHECKPOINT])
def test_manager_ready_requires_router_health_and_policy_weight_evidence(source):
    manager = InferenceManager(Role.ROLLOUT)
    manager.register_model(ModelSpec("model", "checkpoint", weight_source=source), operation_id="register")
    replica = ReplicaSnapshot("model/replica-0", LifecycleState.READY, "http://engine", "v1")
    model = ModelSnapshot("model", (replica,), state=LifecycleState.READY, admission=True)
    before = manager.snapshot()
    with pytest.raises(ValueError, match="Router"):
        manager.publish_model(model)
    assert manager.snapshot() is before
    model = replace(model, router_url="http://router")
    if source == WeightSource.POLICY:
        with pytest.raises(ValueError, match="weight version"):
            manager.publish_model(model)
        model = replace(model, required_weight_version="v1")
    manager.publish_model(model)
    assert manager.snapshot().models == (model,)
    assert before.models[0].state is None
    with pytest.raises(ValueError, match="Only READY"):
        manager.publish_model(replace(model, state=LifecycleState.SLEEPING))


def test_manager_topology_revision_ignores_lifecycle_state_changes():
    manager = InferenceManager(Role.TEACHER)
    manager.register_model(
        ModelSpec("model", "checkpoint", weight_source=WeightSource.CHECKPOINT), operation_id="register"
    )
    replica = ReplicaSnapshot("model/replica-0", LifecycleState.READY, "http://engine")
    model = ModelSnapshot("model", (replica,), router_url="http://router", state=LifecycleState.READY, admission=True)
    manager.publish_model(model)
    revision = manager.snapshot().topology_revision

    manager.invalidate_model("model", state=LifecycleState.SLEEPING)

    assert manager.snapshot().topology_revision == revision


def test_manager_topology_revision_changes_when_replica_endpoint_changes():
    manager = InferenceManager(Role.TEACHER)
    manager.register_model(
        ModelSpec("model", "checkpoint", weight_source=WeightSource.CHECKPOINT), operation_id="register"
    )
    model = ModelSnapshot(
        "model",
        (ReplicaSnapshot("model/replica-0", LifecycleState.READY, "http://engine"),),
        router_url="http://router",
        state=LifecycleState.READY,
        admission=True,
    )
    manager.publish_model(model)
    revision = manager.snapshot().topology_revision

    manager.publish_model(replace(model, replicas=(replace(model.replicas[0], base_url="http://engine-new"),)))

    assert manager.snapshot().topology_revision == revision + 1


def test_static_model_rejects_policy_weight_preparation():
    manager = InferenceManager(Role.TEACHER)
    manager.register_model(
        ModelSpec("teacher", "checkpoint", weight_source=WeightSource.CHECKPOINT), operation_id="register"
    )

    with pytest.raises(ValueError, match="Static or external weights"):
        manager.begin_preparation("teacher", required_weight_version="step-1")


def test_specs_map_stable_logical_replicas_to_all_node_actors():
    replicas = replicas_from_slots("model/group-0", 4, 2)
    assert replicas[0].node_ranks == (0, 1)
    assert replicas[1].node_ranks == (2, 3)
    assert replicas[1].replica_id == "model/group-0/replica-2"
    for slots, nodes in ((3, 2), (4, 0), (-1, 1)):
        with pytest.raises(ValueError):
            replicas_from_slots("model", slots, nodes)
    with pytest.raises(ValueError, match="share"):
        EngineGroupSpec("group", (ReplicaSpec("a", (0,)), ReplicaSpec("b", (0,))))
    with pytest.raises(ValueError, match="unregistered"):
        RoleSpec(Role.TEACHER, (), RoutingSpec(default_model="missing"))


def test_multinode_manager_shutdown_includes_followers(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from relax.distributed.ray import multi_engine_manager as module

    manager = module.MultiEngineManager(
        SimpleNamespace(),
        num_slots=4,
        nodes_per_engine=2,
        engine_actor_cls=object,
        skip_init=True,
        inference_manager=InferenceManager(Role.GENRM),
    )
    engines = [MagicMock() for _ in range(4)]
    manager.all_engines = list(engines)
    monkeypatch.setattr(module.ray, "get", lambda value, **kwargs: value)
    kill = MagicMock()
    monkeypatch.setattr(module.ray, "kill", kill)
    manager.shutdown()
    for engine in engines:
        engine.shutdown.remote.assert_called_once()
    assert kill.call_count == 4
    assert manager.all_engines == [None] * 4


def test_model_config_resolved_preserves_overrides_without_mutation():
    from types import SimpleNamespace

    config = ModelConfig("policy", engine_groups=[EngineGroupConfig("regular", 16)])
    args = SimpleNamespace(
        rollout_num_gpus_per_engine=8, num_gpus_per_node=4, sglang_hf_checkpoint="inference", hf_checkpoint="train"
    )
    spec = config.resolved(args)
    assert ModelSpec is ModelConfig
    assert type(spec) is ModelConfig
    assert spec.model_path == "inference"
    assert spec.engine_groups[0].replicas[1].node_ranks == (2, 3)
    assert dict(spec.engine_groups[0].overrides)["model_path"] == "inference"
    assert config.engine_groups[0].num_gpus_per_engine is None
    assert config.engine_groups[0].overrides == {}


def test_model_config_yaml_preserves_schema_and_group_overrides(tmp_path):
    from types import SimpleNamespace

    path = tmp_path / "engines.yaml"
    path.write_text(
        "sglang:\n"
        "  - name: teacher\n"
        "    weight_source: checkpoint\n"
        "    allow_defer: true\n"
        "    direct_eligible: true\n"
        "    num_gpus_per_engine: 4\n"
        "    engine_groups:\n"
        "      - worker_type: regular\n"
        "        num_gpus: 8\n"
        "        num_gpus_per_engine: 8\n"
        "        overrides: {model_path: custom}\n"
        "      - worker_type: placeholder\n"
        "        num_gpus: 1\n"
    )
    config = SglangConfig.from_yaml(str(path))
    args = SimpleNamespace(
        rollout_num_gpus_per_engine=2, num_gpus_per_node=4, sglang_hf_checkpoint=None, hf_checkpoint="train"
    )
    model = config.models[0].resolved(args)
    # Model-level unknown keys retain the old loader's ignore behavior.
    assert model.weight_source is WeightSource.POLICY
    assert not model.allow_defer and not model.direct_eligible
    assert model.total_num_gpus == 9
    regular, placeholder = model.engine_groups
    assert regular.num_gpus_per_engine == 8
    assert regular.overrides["model_path"] == "custom"
    assert regular.topology.replicas[0].node_ranks == (0, 1)
    assert placeholder.num_gpus_per_engine == 4
    assert placeholder.topology.replicas == ()
    assert model.resolved(args) == model


def test_model_config_registers_programmatic_capabilities():
    model = ModelConfig(
        "teacher", "checkpoint", weight_source=WeightSource.CHECKPOINT, allow_defer=True, direct_eligible=True
    )
    manager = InferenceManager(Role.TEACHER)
    assert manager.register_model(model, operation_id="register") == model
    snapshot = manager.snapshot().models[0]
    assert snapshot.allow_defer and snapshot.direct_eligible


@pytest.mark.parametrize("total, per_engine, per_node", [(3, 2, 4), (12, 6, 4), (4, 0, 4), (4, 4, 0)])
def test_model_config_rejects_incomplete_topology(total, per_engine, per_node):
    from types import SimpleNamespace

    config = ModelConfig("model", engine_groups=[EngineGroupConfig("regular", total, per_engine)])
    args = SimpleNamespace(
        rollout_num_gpus_per_engine=2, num_gpus_per_node=per_node, sglang_hf_checkpoint=None, hf_checkpoint="train"
    )
    with pytest.raises(ValueError):
        config.resolved(args)
    assert config.engine_groups[0].topology is None
    assert config.engine_groups[0].overrides == {}


def test_model_config_pool_preserves_topology_and_overrides():
    from relax.engine.inference.specs import model_spec_from_pool

    pool = model_spec_from_pool(
        "teacher",
        "checkpoint",
        weight_source=WeightSource.CHECKPOINT,
        num_slots=4,
        nodes_per_engine=2,
        num_gpus_per_engine=8,
        overrides=(("model_path", "custom"),),
    )
    assert isinstance(pool, ModelConfig)
    assert pool.total_num_gpus == 16
    assert pool.engine_groups[0].replicas[1].node_ranks == (2, 3)
    assert pool.engine_groups[0].overrides == {"model_path": "custom"}
    assert replace(pool, allow_defer=True).allow_defer


def test_preparation_requires_all_evidence_and_rejects_late_completion():
    manager = InferenceManager(Role.ROLLOUT)
    manager.register_model(
        ModelSpec("model", "checkpoint", weight_source=WeightSource.POLICY), operation_id="register"
    )
    old = manager.begin_preparation("model", required_weight_version="v1")
    token = manager.begin_preparation("model", required_weight_version="v2")
    model = ModelSnapshot(
        "model",
        (ReplicaSnapshot("replica", LifecycleState.READY, "http://engine", "v2"),),
        router_url="http://router",
        state=LifecycleState.READY,
        admission=True,
        required_weight_version="v2",
    )
    complete = PreparationEvidence(True, True, True, True)
    with pytest.raises(ValueError, match="Stale"):
        manager.complete_preparation(old, model, evidence=complete)
    with pytest.raises(ValueError, match="active preparation"):
        manager.publish_model(model)
    for field in ("initialized", "health_checked", "router_updated", "weights_synced"):
        with pytest.raises(ValueError, match="Preparation lacks"):
            manager.complete_preparation(token, model, evidence=replace(complete, **{field: False}))
        assert not manager.snapshot().models[0].admission
    manager.complete_preparation(token, model, evidence=complete)
    assert manager.snapshot().models[0].admission
    token = manager.begin_preparation("model", required_weight_version="v2")
    manager.invalidate_model("model", state=LifecycleState.SLEEPING)
    with pytest.raises(ValueError, match="Stale"):
        manager.complete_preparation(token, model, evidence=complete)
    assert manager.snapshot().models[0].state == LifecycleState.SLEEPING


def test_static_preparation_does_not_require_policy_sync_and_failed_publish_is_retryable():
    manager = InferenceManager(Role.TEACHER)
    manager.register_model(
        ModelSpec("model", "checkpoint", weight_source=WeightSource.CHECKPOINT), operation_id="register"
    )
    token = manager.begin_preparation("model")
    evidence = PreparationEvidence(True, True, True)
    model = ModelSnapshot("model", state=LifecycleState.READY, admission=True)
    with pytest.raises(ValueError, match="Router"):
        manager.complete_preparation(token, model, evidence=evidence)
    assert not manager.snapshot().models[0].admission
    model = replace(
        model,
        router_url="http://router",
        replicas=(ReplicaSnapshot("replica", LifecycleState.READY, "http://engine"),),
    )
    manager.complete_preparation(token, model, evidence=evidence)
    assert manager.snapshot().models[0].admission


def test_observation_cannot_replace_requested_policy_version():
    manager = InferenceManager(Role.ROLLOUT)
    manager.register_model(
        ModelSpec("model", "checkpoint", weight_source=WeightSource.POLICY), operation_id="register"
    )
    manager.begin_preparation("model", required_weight_version="v2")
    expected = manager.snapshot().models[0]
    stale = ModelSnapshot(
        "model",
        (ReplicaSnapshot("replica", LifecycleState.READY, "http://engine", "v1"),),
        router_url="http://router",
        state=LifecycleState.READY,
        admission=True,
        required_weight_version="v1",
    )
    with pytest.raises(ValueError, match="requested policy"):
        manager.commit_observation(expected, stale, evidence=PreparationEvidence(True, True, True, True))
    assert manager.snapshot().models[0] is expected


def test_observation_during_offload_cannot_reopen_admission():
    from unittest.mock import MagicMock

    manager = InferenceManager(Role.TEACHER)
    manager.register_model(
        ModelSpec("model", "checkpoint", weight_source=WeightSource.CHECKPOINT), operation_id="register"
    )
    pool = MagicMock()
    pool.is_onloaded.return_value = True
    manager.bind_pool("model", pool)

    def release(*args, **kwargs):
        expected = manager.snapshot().models[0]
        ready = replace(expected, state=LifecycleState.READY, admission=True)
        assert not manager.commit_observation(expected, ready, evidence=PreparationEvidence(True, True, True))
        return []

    pool.fanout.side_effect = release
    manager.offload("model")
    assert manager.snapshot().models[0].state == LifecycleState.SLEEPING


def test_manager_pool_lifecycle_delegates_and_closes_admission_before_rpc():
    from unittest.mock import MagicMock

    manager = InferenceManager(Role.TEACHER)
    manager.register_model(
        ModelSpec("model", "checkpoint", weight_source=WeightSource.CHECKPOINT), operation_id="register"
    )
    pool = MagicMock()
    pool.is_onloaded.return_value = True
    pool.fanout.return_value = [2]
    pool.recover.return_value = {2}
    manager.bind_pool("model", pool)

    def release(method, **kwargs):
        assert manager.snapshot().models[0].state == LifecycleState.DRAINING
        assert not manager.snapshot().models[0].admission
        assert method == "release_memory_occupation"
        return [2]

    pool.fanout.side_effect = release
    manager.offload("model")
    pool.retire.assert_called_once_with([2])
    pool.set_onloaded.assert_called_with(False)
    assert manager.snapshot().models[0].state == LifecycleState.SLEEPING
    pool.is_onloaded.return_value = False
    pool.fanout.side_effect = None
    pool.fanout.return_value = []
    manager.onload("model", tags=["weights"])
    pool.fanout.assert_called_with("resume_memory_occupation", skip_ranks={2}, tags=["weights"])
    pool.set_onloaded.assert_called_with(True)
    assert not manager.snapshot().models[0].admission
    manager.shutdown("model")
    pool.shutdown.assert_called_once()
    assert manager.snapshot().models[0].state == LifecycleState.DEAD


def test_manager_pool_failures_never_publish_ready():
    from unittest.mock import MagicMock

    manager = InferenceManager(Role.GENRM)
    manager.register_model(
        ModelSpec("model", "checkpoint", weight_source=WeightSource.CHECKPOINT), operation_id="register"
    )
    pool = MagicMock()
    pool.recover.side_effect = RuntimeError("rebuild failed")
    manager.bind_pool("model", pool)
    with pytest.raises(RuntimeError, match="rebuild failed"):
        manager.onload("model")
    assert manager.snapshot().models[0].state == LifecycleState.DEAD
    assert not manager.snapshot().models[0].admission
    with pytest.raises(ValueError, match="already bound"):
        manager.bind_pool("model", MagicMock())


def test_model_observation_cannot_overwrite_concurrent_offload():
    manager = InferenceManager(Role.TEACHER)
    manager.register_model(
        ModelSpec("model", "checkpoint", weight_source=WeightSource.CHECKPOINT), operation_id="register"
    )
    before_rpc = manager.snapshot().models[0]
    model = replace(
        before_rpc,
        router_url="http://router",
        state=LifecycleState.READY,
        admission=True,
        replicas=(ReplicaSnapshot("replica", LifecycleState.READY, "http://engine"),),
    )
    manager.invalidate_model("model", state=LifecycleState.SLEEPING)
    assert not manager.commit_observation(before_rpc, model, evidence=PreparationEvidence(True, True, True))
    assert manager.snapshot().models[0].state == LifecycleState.SLEEPING


def test_task_manager_shares_epoch_and_placement_owner_across_roles() -> None:
    manager = InferenceManager()
    rollout = manager.for_role(Role.ROLLOUT)
    teacher = manager.for_role(Role.TEACHER)
    genrm = manager.for_role(Role.GENRM)

    assert rollout.manager_epoch == teacher.manager_epoch == genrm.manager_epoch
    assert set(manager.snapshots()) == {Role.ROLLOUT, Role.TEACHER, Role.GENRM}


def test_task_manager_request_permit_is_shared_across_role_views() -> None:
    manager = InferenceManager()
    teacher = manager.for_role(Role.TEACHER)
    teacher.register_model(
        ModelSpec("teacher", "checkpoint", weight_source=WeightSource.CHECKPOINT), operation_id="register"
    )
    model = ModelSnapshot(
        "teacher",
        (ReplicaSnapshot("teacher/replica-0", LifecycleState.READY, "http://engine"),),
        router_url="http://router",
        state=LifecycleState.READY,
        admission=True,
    )
    teacher.publish_model(model)

    permit = manager.admit_request("teacher", request_id="request-1", role=Role.TEACHER, target="http://router")
    assert permit.manager_epoch == manager.manager_epoch
    assert manager.get_request("request-1") == permit
    manager.complete_request(permit)
    assert manager.get_request("request-1") is None


def test_task_manager_operation_snapshot_uses_shared_epoch() -> None:
    manager = InferenceManager()
    manager.register_model(
        ModelSpec("model", "checkpoint", weight_source=WeightSource.CHECKPOINT),
        operation_id="register",
        role=Role.ROLLOUT,
    )
    operation = manager.get_operation("register", role=Role.ROLLOUT)
    assert operation.owner_epoch == manager.manager_epoch
    assert operation.status == "completed"
    assert operation.kind == "register"


def test_request_permit_rejects_foreign_identity() -> None:
    manager = InferenceManager(Role.TEACHER)
    manager.register_model(
        ModelSpec("teacher", "checkpoint", weight_source=WeightSource.CHECKPOINT), operation_id="register"
    )
    model = ModelSnapshot(
        "teacher",
        (ReplicaSnapshot("teacher/replica-0", LifecycleState.READY, "http://engine"),),
        router_url="http://router",
        state=LifecycleState.READY,
        admission=True,
    )
    manager.publish_model(model)
    permit = manager.admit_request("teacher", request_id="request-1", target="http://router")
    with pytest.raises(ValueError, match="does not belong"):
        manager.complete_request(replace(permit, target="http://other"))
    assert manager.get_request("request-1") == permit


def test_discovery_capabilities_are_immutable_after_registration() -> None:
    manager = InferenceManager(Role.TEACHER)
    manager.register_model(
        ModelSpec("teacher", "checkpoint", weight_source=WeightSource.CHECKPOINT, allow_defer=True),
        operation_id="register",
    )
    with pytest.raises(ValueError, match="allow_defer"):
        manager.get_discovery_snapshot(model_id="teacher", allow_defer=False)
