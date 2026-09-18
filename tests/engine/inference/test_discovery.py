# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from unittest.mock import Mock

import httpx

from relax.engine.inference.client import InferenceDiscoveryClient
from relax.engine.inference.discovery import (
    DiscoveryState,
    role_snapshot_from_dict,
    snapshot_from_engine_urls,
    snapshot_from_legacy_engines,
)
from relax.engine.inference.types import LifecycleState, Role


def test_legacy_adapter_does_not_turn_active_actor_into_ready() -> None:
    snapshot = snapshot_from_legacy_engines(
        {
            "models": {
                "student": {
                    "router_ip": "router",
                    "router_port": 3000,
                    "engine_groups": [{"engines": [{"rank": 4, "status": "active", "url": "http://worker"}]}],
                }
            }
        },
        role=Role.ROLLOUT,
        manager_epoch="epoch-a",
        default_model="student",
    )
    model = snapshot.models[0]
    assert model.state is None
    assert model.admission is False
    assert model.router_url == "http://router:3000"
    assert model.replicas[0].engine_id == "student/replica-4"


def test_legacy_json_supports_status_filter() -> None:
    snapshot = snapshot_from_legacy_engines(
        {
            "models": {
                "student": {
                    "engine_groups": [{"engines": [{"rank": 0, "status": "active"}, {"rank": 1, "status": "dead"}]}]
                }
            }
        },
        role=Role.ROLLOUT,
        manager_epoch="epoch-a",
    )
    active = snapshot.to_legacy_dict(status_filter="active")
    assert active["total_engines"] == 1
    assert "worker_type" not in active["models"]["student"]["engine_groups"][0]


def test_engine_url_adapter_and_state_publisher() -> None:
    snapshot = snapshot_from_engine_urls(
        ["http://teacher-0/generate"],
        role=Role.TEACHER,
        model_id="teacher",
        manager_epoch="epoch-a",
        state=LifecycleState.READY,
        admission=True,
    )
    state = DiscoveryState(snapshot)
    published = state.update(phase="scoring")
    assert published.phase == "scoring"
    assert state.get().manager_epoch == "epoch-a"


def test_v2_round_trip() -> None:
    snapshot = snapshot_from_engine_urls(
        ["http://teacher-0"],
        role=Role.TEACHER,
        model_id="teacher",
        manager_epoch="epoch-a",
        topology_revision=7,
        state=LifecycleState.READY,
        admission=True,
    )
    restored = role_snapshot_from_dict(snapshot.to_dict())
    assert restored == snapshot
    assert restored.topology_revision == 7


def test_discovery_client_fetches_v2_snapshot() -> None:
    response = httpx.Response(
        200,
        json=snapshot_from_engine_urls(
            ["http://router"],
            role=Role.GENRM,
            model_id="judge",
            manager_epoch="epoch-a",
            router_url="http://router",
            state=LifecycleState.READY,
            admission=True,
        ).to_dict(),
        request=httpx.Request("GET", "http://service/genrm/engines"),
    )
    transport = Mock()
    transport.get.return_value = response
    with InferenceDiscoveryClient("http://service", client=transport) as client:
        snapshot = client.get_snapshot("genrm")
        target = client.select_target(client.resolve_model(snapshot, model="judge"))
    transport.get.assert_called_once_with("http://service/genrm/engines", params={"schema_version": 2})
    assert target.base_url == "http://router"


def test_role_snapshot_defaults_topology_revision_for_legacy_payload() -> None:
    snapshot = role_snapshot_from_dict(
        snapshot_from_engine_urls(
            ["http://teacher-0"],
            role=Role.TEACHER,
            model_id="teacher",
            manager_epoch="epoch-a",
        ).to_dict()
    )
    payload = snapshot.to_dict()
    payload.pop("topology_revision")

    assert role_snapshot_from_dict(payload).topology_revision == 0
