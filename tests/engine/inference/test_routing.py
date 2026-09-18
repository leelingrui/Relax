# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from dataclasses import replace

import pytest

from relax.engine.inference.routing import RoutingError, resolve_model, select_target
from relax.engine.inference.types import (
    LifecycleState,
    ModelSnapshot,
    ReplicaSnapshot,
    Role,
    RoleSnapshot,
    RoutingSpec,
)


def _replica(name: str = "head") -> ReplicaSnapshot:
    return ReplicaSnapshot(
        engine_id=name,
        base_url=f"http://{name}.test",
        state=LifecycleState.READY,
        weight_version="v2",
    )


def _model(**kwargs) -> ModelSnapshot:
    return replace(
        ModelSnapshot(
            "student",
            (_replica(),),
            router_url="http://router.test",
            state=LifecycleState.READY,
            admission=True,
            direct_eligible=True,
        ),
        **kwargs,
    )


def _snapshot() -> RoleSnapshot:
    return RoleSnapshot(
        role=Role.ROLLOUT,
        manager_epoch="epoch-a",
        phase="rollout",
        models=(_model(), ModelSnapshot("teacher")),
        routing=RoutingSpec(default_model="student", route_key_to_model=(("score", "teacher"),)),
    )


@pytest.mark.parametrize(
    "kwargs,expected",
    [({}, "student"), ({"route_key": "score"}, "teacher"), ({"model": "student", "route_key": "score"}, "student")],
)
def test_routing_selection_precedence(kwargs: dict, expected: str) -> None:
    assert resolve_model(_snapshot(), **kwargs).model_id == expected


@pytest.mark.parametrize(
    "kwargs,code",
    [
        ({"model": "missing"}, "unknown_model"),
        ({"model": ""}, "unknown_model"),
        ({"route_key": "missing"}, "unknown_route"),
    ],
)
def test_routing_invalid_explicit_selection_never_falls_back(kwargs: dict, code: str) -> None:
    with pytest.raises(RoutingError) as error:
        resolve_model(_snapshot(), **kwargs)
    assert error.value.code == code
    assert error.value.status_code == 400


def test_routing_requires_configured_default_even_with_one_model() -> None:
    with pytest.raises(RoutingError, match="No model"):
        resolve_model(replace(_snapshot(), models=(_model(),), routing=RoutingSpec()))


@pytest.mark.parametrize("state", [None, LifecycleState.STARTING, LifecycleState.SLEEPING])
def test_routing_unknown_or_unready_model_is_unavailable(state: LifecycleState | None) -> None:
    with pytest.raises(RoutingError) as error:
        select_target(_model(state=state))
    assert error.value.status_code == 503


def test_routing_draining_model_closes_admission_even_with_ready_replica() -> None:
    with pytest.raises(RoutingError):
        select_target(_model(admission=False))


def test_allow_defer_is_model_capability_and_does_not_change_router_selection() -> None:
    target = select_target(_model(allow_defer=True))
    assert target.base_url == "http://router.test"


def test_routing_does_not_select_replica_even_when_replicas_are_ready() -> None:
    model = _model(replicas=(_replica("a"), _replica("b")))
    target = select_target(model, cursor=1)
    assert target.base_url == "http://router.test"


def test_routing_always_uses_router_and_never_replica_address() -> None:
    target = select_target(_model())
    assert target.base_url == "http://router.test"


def test_routing_missing_router_never_falls_back_to_replica() -> None:
    with pytest.raises(RoutingError) as error:
        select_target(_model(router_url=None))
    assert error.value.status_code == 503


def test_discovery_serializes_role_phase() -> None:
    assert _snapshot().to_dict()["phase"] == "rollout"


def test_discovery_rejects_ambiguous_model_and_route_identity() -> None:
    with pytest.raises(ValueError, match="Duplicate model"):
        replace(_snapshot(), models=(_model(), _model()))
    with pytest.raises(ValueError, match="Duplicate route"):
        RoutingSpec(route_key_to_model=(("score", "a"), ("score", "b")))
