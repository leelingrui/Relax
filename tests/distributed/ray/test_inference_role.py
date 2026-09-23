# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU-only tests for the task owner's role pools, dispatch and shutdown."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier, Event
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from relax.distributed.ray import inference_role as module
from relax.engine.inference.capabilities import WeightSource
from relax.engine.inference.manager import InferenceManager, PreparationEvidence
from relax.engine.inference.placement import PlacementGroupView, PlacementOwner, PlacementRequest
from relax.engine.inference.specs import ModelSpec
from relax.engine.inference.types import LifecycleState, ReplicaSnapshot, Role


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
            assert {model.model_id for model in snapshot.models} <= set(state.instances)
            assert dict(snapshot.routing.route_key_to_model) == {
                key: key for key, _ in snapshot.routing.route_key_to_model
            }
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


def _create(role: Role = Role.TEACHER, configs: dict | None = None) -> module.TaskInferenceManager:
    owner = module.TaskInferenceManager()
    owner.create_role(SimpleNamespace(), role, _configs() if configs is None else configs)
    return owner


def _manager(owner: module.TaskInferenceManager, role: Role = Role.TEACHER) -> InferenceManager:
    return owner._control_manager.for_role(role)


def _publish_ready(owner: module.TaskInferenceManager, role: Role, model_id: str) -> None:
    manager = _manager(owner, role)
    current = manager.snapshot((model_id,)).models[0]
    model = replace(
        current,
        state=LifecycleState.READY,
        admission=True,
        router_url="http://router:1",
        replicas=(ReplicaSnapshot(f"{model_id}/replica-0", LifecycleState.READY, "http://engine:1"),),
    )
    token = manager.begin_preparation(model_id)
    manager.complete_preparation(token, model, evidence=PreparationEvidence(True, True, True))


@pytest.mark.parametrize("role", [Role.GENRM, Role.TEACHER])
def test_owner_create_role_registers_all_models_before_initialization(pools: SimpleNamespace, role: Role) -> None:
    configs = _configs()
    owner = module.TaskInferenceManager()
    assert owner.create_role(SimpleNamespace(), role, configs) == ("a", "b")
    assert owner.ready(role=role) is True
    assert pools.events == [("construct", "a"), ("construct", "b"), ("initialize", "a"), ("initialize", "b")]
    assert pools.instances["a"].manager.role is pools.instances["b"].manager.role is role
    assert pools.instances["a"].manager.manager_epoch == owner._control_manager.manager_epoch
    assert pools.instances["b"].value == "b"
    # Every pool records its slices in the owner's one ledger.
    assert pools.instances["b"].kwargs == {"bundle_offset": 1, "placement_manager_handle": owner._placement_planner}
    assert configs == _configs()
    snapshot = owner.snapshot(role=role)
    assert snapshot.manager_epoch == owner.manager_epoch
    assert snapshot.routing.default_model is None
    assert all(not model.admission for model in snapshot.models)
    assert [call.args for call in pools.factory.call_args_list] == [(role, "a"), (role, "b")]
    assert owner.role_models(role) == ("a", "b")
    assert owner.registered_roles() == (role.value,)
    assert owner.get_operation(f"routes:{role.value}:startup").kind == "routes"


def test_owner_create_role_is_idempotent(pools: SimpleNamespace) -> None:
    owner = _create()
    assert owner.create_role(SimpleNamespace(), Role.TEACHER, _configs()) == ("a", "b")
    assert pools.factory.call_count == 2


def test_owner_single_model_has_default_route(pools: SimpleNamespace) -> None:
    owner = _create(configs={"a": _configs()["a"]})
    assert owner.snapshot(role=Role.TEACHER).routing.default_model == "a"


def test_owner_roles_share_one_task_manager(pools: SimpleNamespace) -> None:
    owner = module.TaskInferenceManager()
    owner.create_role(SimpleNamespace(), Role.TEACHER, {"teacher": _configs()["a"]})
    owner.create_role(SimpleNamespace(), Role.GENRM, {"genrm": _configs()["b"]})

    assert pools.instances["teacher"].manager.manager_epoch == pools.instances["genrm"].manager.manager_epoch
    assert owner.snapshot(role=Role.TEACHER).models[0].model_id == "teacher"
    assert owner.snapshot(role=Role.GENRM).models[0].model_id == "genrm"
    assert owner.snapshot(role=Role.TEACHER).manager_epoch == owner.snapshot(role=Role.GENRM).manager_epoch


@pytest.mark.parametrize("failing_model", ["a", "b"])
def test_owner_initialization_failure_cleans_even_uninitialized_pools(
    pools: SimpleNamespace, failing_model: str
) -> None:
    pools.fail_init = failing_model
    pools.fail_shutdown = "b"
    owner = module.TaskInferenceManager()
    with pytest.raises(RuntimeError, match="initialize failed"):
        owner.create_role(SimpleNamespace(), Role.GENRM, _configs())
    assert pools.events[-3:] == [("shutdown", "b"), ("shutdown", "a"), ("routers", "shutdown")]
    pools.router_cleanup.assert_called_once_with()
    assert owner.registered_roles() == ()


def test_owner_constructor_failure_cleans_previously_constructed_pools(pools: SimpleNamespace) -> None:
    pools.fail_construct = "b"
    with pytest.raises(RuntimeError, match="constructor failed"):
        _create()
    assert pools.events == [("construct", "a"), ("shutdown", "a"), ("routers", "shutdown")]


def test_owner_constructor_error_survives_router_cleanup_failure(pools: SimpleNamespace) -> None:
    pools.fail_construct = "a"
    pools.router_cleanup.side_effect = RuntimeError("router cleanup failed")
    with pytest.raises(RuntimeError, match="constructor failed"):
        _create()
    pools.router_cleanup.assert_called_once_with()


@pytest.mark.parametrize(
    "configs",
    [
        {},
        {"a": {"args": (), "kwargs": {"defer_init": False}}},
        {"a": {"args": (), "kwargs": {"placement_manager_handle": object()}}},
        {"a": {"args": "a"}},
        {"a": {"args": (), "unknown": 1}},
    ],
)
def test_owner_rejects_invalid_config_before_constructing(pools: SimpleNamespace, configs: dict) -> None:
    with pytest.raises(ValueError):
        _create(configs=configs)
    assert pools.events == []


def test_owner_rejects_rollout_pool_before_construction(pools: SimpleNamespace) -> None:
    with pytest.raises(ValueError, match="Unsupported inference pool role"):
        _create(Role.ROLLOUT)
    assert pools.events == []


@pytest.mark.parametrize("method", sorted(module._POOL_METHODS))
def test_owner_call_preserves_arguments_and_return_value(pools: SimpleNamespace, method: str) -> None:
    owner = _create(Role.GENRM)
    result = ([object()], object(), 2)
    implementation = MagicMock(return_value=result)
    setattr(pools.instances["b"], method, implementation)
    assert owner.call(Role.GENRM, "b", method, 7, model_id="alias", tags=["weights"]) is result
    implementation.assert_called_once_with(7, model_id="alias", tags=["weights"])


def test_owner_call_rejects_private_methods_unknown_models_and_roles(pools: SimpleNamespace) -> None:
    owner = _create()
    for method in ("initialize", "_init_engines", "__dict__", "snapshot"):
        with pytest.raises(ValueError, match="Unsupported pool method"):
            owner.call(Role.TEACHER, "a", method)
    with pytest.raises(KeyError, match="Unknown model"):
        owner.call(Role.TEACHER, "missing", "health_check")
    with pytest.raises(RuntimeError, match="not registered"):
        owner.call(Role.GENRM, "a", "health_check")
    with pytest.raises(RuntimeError, match="not registered"):
        owner.snapshot(role=Role.GENRM)
    assert owner.ready(role=Role.GENRM) is False
    owner.shutdown_role(Role.TEACHER)
    assert owner.ready(role=Role.TEACHER) is False
    with pytest.raises(RuntimeError, match="not registered"):
        owner.call(Role.TEACHER, "a", "recover")


def test_owner_lifecycle_accepts_only_lifecycle_methods(pools: SimpleNamespace) -> None:
    owner = _create()
    pools.instances["a"].onload = MagicMock(return_value="onloaded")
    assert owner.lifecycle(Role.TEACHER, "a", "onload") == "onloaded"
    with pytest.raises(ValueError, match="Unsupported lifecycle method"):
        owner.lifecycle(Role.TEACHER, "a", "get_urls")


def test_owner_shutdown_attempts_all_pools_on_error(pools: SimpleNamespace) -> None:
    owner = _create()
    pools.fail_shutdown = "b"
    with pytest.raises(RuntimeError, match="one or more"):
        owner.shutdown_role(Role.TEACHER)
    assert pools.events[-3:] == [("shutdown", "b"), ("shutdown", "a"), ("routers", "shutdown")]
    assert owner.ready(role=Role.TEACHER) is False


def test_owner_shutdown_all_closes_every_role(pools: SimpleNamespace) -> None:
    owner = module.TaskInferenceManager()
    owner.create_role(SimpleNamespace(), Role.TEACHER, {"teacher": _configs()["a"]})
    owner.create_role(SimpleNamespace(), Role.GENRM, {"genrm": _configs()["b"]})
    owner.shutdown_all()
    assert ("shutdown", "teacher") in pools.events and ("shutdown", "genrm") in pools.events
    assert owner.registered_roles() == ()


def test_owner_model_shutdown_keeps_other_model_routers(pools: SimpleNamespace) -> None:
    owner = _create()
    owner.call(Role.TEACHER, "a", "shutdown")
    pools.router_cleanup.assert_not_called()
    assert owner.ready(role=Role.TEACHER) is True
    assert pools.events[-1] == ("shutdown", "a")
    owner.shutdown_role(Role.TEACHER)
    pools.router_cleanup.assert_called_once_with()
    assert pools.events.count(("shutdown", "a")) == 1


def test_owner_last_model_shutdown_stops_routers_once(pools: SimpleNamespace) -> None:
    owner = _create()
    owner.call(Role.TEACHER, "a", "shutdown")
    owner.call(Role.TEACHER, "a", "shutdown")
    pools.router_cleanup.assert_not_called()
    owner.call(Role.TEACHER, "b", "shutdown")
    assert set(_manager(owner)._closed_models) == {"a", "b"}
    assert owner.ready(role=Role.TEACHER) is False
    owner.call(Role.TEACHER, "b", "shutdown")
    with pytest.raises(RuntimeError, match="shut down"):
        owner.call(Role.TEACHER, "a", "onload")
    owner.shutdown_role(Role.TEACHER)
    pools.router_cleanup.assert_called_once_with()
    assert pools.events.count(("shutdown", "a")) == 1
    assert pools.events.count(("shutdown", "b")) == 1


def test_owner_failed_model_shutdown_blocks_revival_but_allows_retry(pools: SimpleNamespace) -> None:
    owner = _create()
    pools.fail_shutdown = "a"
    with pytest.raises(RuntimeError, match="shutdown failed"):
        owner.call(Role.TEACHER, "a", "shutdown")
    assert "a" not in _manager(owner)._closed_models
    with pytest.raises(RuntimeError, match="closing"):
        owner.call(Role.TEACHER, "a", "recover")
    owner.call(Role.TEACHER, "b", "shutdown")
    pools.router_cleanup.assert_not_called()
    pools.fail_shutdown = None
    owner.call(Role.TEACHER, "a", "shutdown")
    pools.router_cleanup.assert_called_once_with()


def test_owner_last_model_shutdown_retries_failed_router_cleanup(pools: SimpleNamespace) -> None:
    owner = _create(configs={"a": _configs()["a"]})
    pools.router_cleanup.side_effect = RuntimeError("router cleanup failed")
    with pytest.raises(RuntimeError, match="router cleanup failed"):
        owner.call(Role.TEACHER, "a", "shutdown")
    assert owner.ready(role=Role.TEACHER) is False
    pools.router_cleanup.side_effect = None
    owner.call(Role.TEACHER, "a", "shutdown")
    assert pools.router_cleanup.call_count == 2
    assert pools.events.count(("shutdown", "a")) == 1


def test_owner_same_pool_calls_serialize_without_blocking_snapshot_or_other_pool(pools: SimpleNamespace) -> None:
    owner = _create()
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
        return owner.call(Role.TEACHER, "a", "offload")

    pools.instances["a"].onload = onload
    pools.instances["a"].offload = offload
    pools.instances["b"].health_check = lambda: True
    with ThreadPoolExecutor(max_workers=4) as executor:
        first = executor.submit(owner.call, Role.TEACHER, "a", "onload")
        try:
            assert entered.wait(2)
            second = executor.submit(queued_call)
            assert queued.wait(2)
            snapshot = executor.submit(owner.snapshot, role=Role.TEACHER).result(timeout=2)
            assert [model.model_id for model in snapshot.models] == ["a", "b"]
            assert executor.submit(owner.call, Role.TEACHER, "b", "health_check").result(timeout=2) is True
            # The owner waits for the model's operation lock instead of
            # rejecting: the queued call runs only once the first one ends.
            assert not second_entered.wait(0.05)
        finally:
            release.set()
        assert first.result(timeout=2) == "onloaded"
        assert second.result(timeout=2) == "offloaded"


def test_unified_service_manager_call_wait_rejects_non_pool_methods(pools: SimpleNamespace) -> None:
    owner = _create(configs={"a": _configs()["a"]})
    host = module.UnifiedServiceManager.__new__(module.UnifiedServiceManager)
    host.inference_manager = _manager(owner)
    pools.instances["a"].health_check = lambda: True
    assert host.call_wait("a", "health_check") is True
    with pytest.raises(ValueError, match="Unsupported pool method"):
        host.call_wait("a", "initialize")


def test_owner_shutdown_waits_for_calls_and_rejects_new_operations(
    pools: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = _create()
    entered, release, cleanup_entered = Event(), Event(), Event()
    cleanup = owner._control_manager._close_pool

    def onload() -> None:
        entered.set()
        assert release.wait(5)
        pools.events.append(("onload", "finished"))

    def shutdown_cleanup(*args: Any, **kwargs: Any) -> Any:
        cleanup_entered.set()
        return cleanup(*args, **kwargs)

    pools.instances["a"].onload = onload
    monkeypatch.setattr(owner._control_manager, "_close_pool", shutdown_cleanup)
    with ThreadPoolExecutor(max_workers=4) as executor:
        active = executor.submit(owner.call, Role.TEACHER, "a", "onload")
        try:
            assert entered.wait(2)
            closing = executor.submit(owner.shutdown_role, Role.TEACHER)
            assert cleanup_entered.wait(2)
            assert owner.ready(role=Role.TEACHER) is False
            assert executor.submit(owner.snapshot, role=Role.TEACHER).result(timeout=2).manager_epoch
            with pytest.raises(RuntimeError, match="shut down"):
                executor.submit(owner.call, Role.TEACHER, "b", "recover").result(timeout=2)
            pools.router_cleanup.assert_not_called()
            assert ("shutdown", "a") not in pools.events
        finally:
            release.set()
        active.result(timeout=2)
        closing.result(timeout=2)
    assert pools.events.index(("onload", "finished")) < pools.events.index(("shutdown", "a"))
    assert pools.events[-1] == ("routers", "shutdown")


def test_owner_concurrent_model_shutdown_cleans_routers_once(pools: SimpleNamespace) -> None:
    owner = _create()
    barrier = Barrier(2, timeout=5)

    def shutdown(model_id: str) -> str:
        barrier.wait()
        pools.events.append(("shutdown", model_id))
        return model_id

    pools.instances["a"].shutdown = lambda: shutdown("a")
    pools.instances["b"].shutdown = lambda: shutdown("b")
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(owner.call, Role.TEACHER, "a", "shutdown")
        second = executor.submit(owner.call, Role.TEACHER, "b", "shutdown")
        assert first.result(timeout=6) == "a"
        assert second.result(timeout=6) == "b"
    assert owner.ready(role=Role.TEACHER) is False
    assert pools.events[-1] == ("routers", "shutdown")
    pools.router_cleanup.assert_called_once_with()
    assert owner.call(Role.TEACHER, "a", "shutdown") == "a"
    owner.shutdown_role(Role.TEACHER)
    pools.router_cleanup.assert_called_once_with()


def test_owner_concurrent_role_and_model_shutdown_do_not_double_close(
    pools: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = _create(configs={"a": _configs()["a"]})
    entered, release, cleanup_entered = Event(), Event(), Event()
    cleanup = owner._control_manager._close_pool

    def shutdown_pool() -> None:
        entered.set()
        assert release.wait(5)
        pools.events.append(("shutdown", "a"))

    def shutdown_cleanup(*args: Any, **kwargs: Any) -> Any:
        cleanup_entered.set()
        return cleanup(*args, **kwargs)

    pools.instances["a"].shutdown = shutdown_pool
    monkeypatch.setattr(owner._control_manager, "_close_pool", shutdown_cleanup)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(owner.call, Role.TEACHER, "a", "shutdown")
        try:
            assert entered.wait(2)
            second = executor.submit(owner.shutdown_role, Role.TEACHER)
            assert cleanup_entered.wait(2)
            pools.router_cleanup.assert_not_called()
        finally:
            release.set()
        first.result(timeout=2)
        second.result(timeout=2)
    assert pools.events.count(("shutdown", "a")) == 1
    pools.router_cleanup.assert_called_once_with()


def test_owner_owns_request_permits_under_the_task_epoch(pools: SimpleNamespace) -> None:
    owner = _create(configs={"model": _configs()["a"]})
    _publish_ready(owner, Role.TEACHER, "model")

    permit = owner.admit_request("model", request_id="request", role=Role.TEACHER, target="http://router")

    assert owner.snapshot(role=Role.TEACHER).manager_epoch == owner.manager_epoch
    assert permit.manager_epoch == owner.manager_epoch
    assert owner.get_request("request") == permit
    owner.complete_request(permit)
    assert owner.get_request("request") is None


def test_owner_refuses_a_request_for_a_model_that_is_not_ready(pools: SimpleNamespace) -> None:
    owner = _create(configs={"model": _configs()["a"]})
    with pytest.raises(RuntimeError, match="not ready"):
        owner.admit_request("model", request_id="request", role=Role.TEACHER)
    assert owner.get_request("request") is None


# ---------------------------------------------------------------------------
# The task owner holds the one placement ledger for every role.
# ---------------------------------------------------------------------------
class _FakePlacementGroup:
    def __init__(self, value: str) -> None:
        self.id = SimpleNamespace(hex=lambda value=value: value)


def _placement_view(owner: PlacementOwner = PlacementOwner.CONTROLLER, size: int = 8, identity: str = "shared"):
    return PlacementGroupView(tuple(range(size)), tuple(range(size)), owner, identity=_FakePlacementGroup(identity))


def _placement_request(group_id: str, *, num_gpus: int, phase: str, bundle_offset: int | None = None):
    return PlacementRequest(
        group_id=group_id,
        worker_type="regular",
        num_gpus=num_gpus,
        num_gpus_per_engine=2,
        num_gpus_per_node=4,
        phase=phase,
        bundle_offset=bundle_offset,
    )


def test_task_manager_shares_one_placement_ledger_across_roles() -> None:
    owner = module.TaskInferenceManager()
    view = _placement_view()

    owner.plan_placement((_placement_request("rollout/group-0", num_gpus=4, phase="inference"),), view)
    (genrm,) = owner.plan_placement((_placement_request("genrm-0", num_gpus=4, phase="genrm"),), view)

    # Every role's slice lives in the same ledger, keyed by the group identity
    # rather than by a per-process object.
    assert {item.group_id for item in owner.allocations(view)} == {"rollout/group-0", "genrm-0"}
    assert genrm.reserved_offset == 0


def test_task_manager_rejects_co_resident_roles_in_one_phase() -> None:
    owner = module.TaskInferenceManager()
    view = _placement_view()
    owner.plan_placement(
        (_placement_request("rollout/group-0", num_gpus=4, phase="inference", bundle_offset=0),), view
    )

    with pytest.raises(ValueError, match="overlap"):
        owner.plan_placement((_placement_request("teacher-0", num_gpus=4, phase="inference", bundle_offset=2),), view)


def test_task_manager_reports_phase_exclusions_for_a_deferred_plan() -> None:
    owner = module.TaskInferenceManager()
    view = _placement_view()
    owner.plan_placement(
        (_placement_request("rollout/group-0", num_gpus=4, phase="inference", bundle_offset=0),), view
    )
    owner.plan_placement((_placement_request("teacher-0", num_gpus=4, phase="teacher_score", bundle_offset=0),), view)

    (contention,) = owner.contended_phases(view)

    assert contention.phases == ("inference", "teacher_score")


def test_task_manager_release_reports_ownership_of_the_group() -> None:
    owner = module.TaskInferenceManager()
    borrowed = _placement_view(PlacementOwner.CONTROLLER, identity="borrowed")
    created = _placement_view(PlacementOwner.MANAGER, identity="created")
    (from_borrowed,) = owner.plan_placement((_placement_request("genrm-0", num_gpus=4, phase="genrm"),), borrowed)
    (from_created,) = owner.plan_placement(
        (_placement_request("scale-out/replica-0", num_gpus=4, phase="inference"),), created
    )

    assert owner.release_placement(from_borrowed).remove_placement_group is False
    assert owner.release_placement(from_created).remove_placement_group is True
    assert owner.allocations() == ()


def test_task_manager_dry_run_validates_without_reserving() -> None:
    owner = module.TaskInferenceManager()
    view = _placement_view()

    owner.plan_placement(
        (_placement_request("genrm/a", num_gpus=4, phase="genrm", bundle_offset=0),), view, dry_run=True
    )

    assert owner.allocations(view) == ()
