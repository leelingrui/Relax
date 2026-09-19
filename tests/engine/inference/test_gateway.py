# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json

import pytest

from relax.components.inference_gateway import InferenceGateway
from relax.engine.inference.discovery import snapshot_from_engine_urls
from relax.engine.inference.types import LifecycleState, Role


def _snapshot(*, role: Role = Role.ROLLOUT, state: LifecycleState = LifecycleState.READY, admission: bool = True):
    return snapshot_from_engine_urls(
        ["http://engine/generate"],
        role=role,
        model_id="model-a",
        manager_epoch="epoch-a",
        router_url="http://router",
        state=state,
        admission=admission,
    )


@pytest.mark.asyncio
async def test_one_gateway_class_serves_all_roles() -> None:
    gateways = [InferenceGateway(role, snapshot_provider=lambda role=role: _snapshot(role=role)) for role in Role]
    try:
        assert {gateway.role for gateway in gateways} == set(Role)
        model_lists = [await gateway.models() for gateway in gateways]
        assert all(models["data"][0]["id"] == "model-a" for models in model_lists)
    finally:
        for gateway in gateways:
            await gateway.close()


@pytest.mark.asyncio
async def test_gateway_reuses_discovery_routing_and_rejects_unavailable_models() -> None:
    ready = InferenceGateway(Role.ROLLOUT, snapshot_provider=lambda: _snapshot())
    sleeping = InferenceGateway(
        Role.ROLLOUT,
        snapshot_provider=lambda: _snapshot(state=LifecycleState.SLEEPING, admission=False),
    )
    try:
        assert await ready._target({"model": "model-a"}) == "http://router"
        with pytest.raises(Exception) as error:
            await sleeping._target({"model": "model-a"})
        assert error.value.status_code == 503
        assert "Retry-After" in (error.value.headers or {})
    finally:
        await ready.close()
        await sleeping.close()


@pytest.mark.asyncio
async def test_gateway_never_uses_replica_url_when_router_is_missing() -> None:
    snapshot = _snapshot()
    model = snapshot.models[0]
    snapshot = type(snapshot)(
        role=snapshot.role,
        manager_epoch=snapshot.manager_epoch,
        models=(
            type(model)(
                model_id=model.model_id,
                replicas=model.replicas,
                router_url=None,
                state=model.state,
                admission=model.admission,
                direct_eligible=True,
            ),
        ),
        routing=snapshot.routing,
    )
    gateway = InferenceGateway(Role.ROLLOUT, snapshot_provider=lambda: snapshot)
    try:
        with pytest.raises(Exception) as error:
            await gateway._target({"model": "model-a"})
        assert error.value.status_code == 503
    finally:
        await gateway.close()


@pytest.mark.asyncio
async def test_gateway_preserves_legacy_engines_and_genrm_response_shape() -> None:
    gateway = InferenceGateway(Role.GENRM, snapshot_provider=lambda: _snapshot(role=Role.GENRM))
    try:
        legacy = await gateway.engines()
        v2 = await gateway.engines(schema_version=2)
        assert legacy["models"]["model-a"]["total_engines"] == 1
        assert v2["role"] == "genrm"
        assert json.loads((await gateway.health()).body)["status"] == "healthy"
    finally:
        await gateway.close()
