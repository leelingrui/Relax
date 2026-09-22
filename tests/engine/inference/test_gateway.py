# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from starlette.requests import Request

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


def _request(payload: dict) -> Request:
    body = json.dumps(payload).encode()

    async def receive():
        return {"type": "http.request", "body": body}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/generate",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
        },
        receive,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True])
async def test_gateway_genrm_router_adapts_only_messages(native: bool) -> None:
    adapt = AsyncMock(return_value={"input_ids": [11, 12], "sampling_params": {"temperature": 0.3}})
    gateway = InferenceGateway(
        Role.GENRM,
        snapshot_provider=lambda: _snapshot(role=Role.GENRM),
        genrm_backend_handle=SimpleNamespace(prepare_generate_payload=SimpleNamespace(remote=adapt)),
    )
    sent = []

    def upstream(request):
        sent.append(json.loads(request.content))
        assert request.url == "http://router/generate"
        assert int(request.headers["content-length"]) == len(request.content)
        return httpx.Response(200, json={"text": " score \n", "meta_info": {"id": "result"}})

    await gateway._client.aclose()
    gateway._client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    content = {"text": "judge"} if native else {"messages": [{"role": "user", "content": "judge"}]}
    payload = {**content, "model": "model-a", "route_key": "model-a", "sampling_params": {"temperature": 0.3}}
    try:
        response = await gateway.proxy(_request(payload), "generate")
        assert response.status_code == 200
        assert int(response.headers["content-length"]) == len(response.body)
        assert "route_key" not in sent[0] and "model" not in sent[0]
        if native:
            adapt.assert_not_awaited()
            assert sent[0] == {**content, "sampling_params": {"temperature": 0.3}}
            assert json.loads(response.body) == {"text": " score \n", "meta_info": {"id": "result"}}
        else:
            adapt.assert_awaited_once_with("model-a", content["messages"], {"temperature": 0.3})
            assert sent[0] == adapt.return_value
            assert json.loads(response.body) == {"response": "score"}
    finally:
        await gateway.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["input_ids", "text"])
async def test_gateway_genrm_rejects_mixed_protocols(field: str) -> None:
    gateway = InferenceGateway(Role.GENRM)
    try:
        response = await gateway.proxy(_request({"messages": [], field: None}), "generate")
        assert response.status_code == 400
    finally:
        await gateway.close()


@pytest.mark.asyncio
async def test_gateway_genrm_rejects_missing_router_without_backend_fallback() -> None:
    snapshot = _snapshot(role=Role.GENRM)
    snapshot = replace(snapshot, models=(replace(snapshot.models[0], router_url=None),))
    adapt = AsyncMock()
    gateway = InferenceGateway(
        Role.GENRM,
        snapshot_provider=lambda: snapshot,
        upstream_url="http://backend",
        genrm_backend_handle=SimpleNamespace(prepare_generate_payload=SimpleNamespace(remote=adapt)),
    )
    messages = [{"role": "user", "content": "judge"}]

    def upstream(request):
        raise AssertionError("Gateway must not bypass the Router")

    await gateway._client.aclose()
    gateway._client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    try:
        response = await gateway.proxy(_request({"messages": messages, "model": "model-a"}), "generate")
        assert response.status_code == 503
        adapt.assert_not_awaited()
    finally:
        await gateway.close()


@pytest.mark.asyncio
async def test_gateway_reads_one_task_manager_snapshot() -> None:
    snapshot = _snapshot(role=Role.TEACHER)
    manager = SimpleNamespace(snapshot=SimpleNamespace(remote=lambda **kwargs: snapshot))
    gateway = InferenceGateway(Role.TEACHER, manager_handle=manager)
    try:
        first = await gateway._snapshot()
        assert first is snapshot
    finally:
        await gateway.close()


@pytest.mark.asyncio
async def test_gateway_teacher_native_strips_routing_metadata() -> None:
    gateway = InferenceGateway(Role.TEACHER, snapshot_provider=lambda: _snapshot(role=Role.TEACHER))

    def upstream(request):
        assert json.loads(request.content) == {"input_ids": [1, 2], "return_logprob": True}
        return httpx.Response(200, json={"text": "", "meta_info": {"input_token_logprobs": []}})

    await gateway._client.aclose()
    gateway._client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    try:
        response = await gateway.proxy(
            _request({"model": "model-a", "route_key": "model-a", "input_ids": [1, 2], "return_logprob": True}),
            "generate",
        )
        assert response.status_code == 200
        assert "meta_info" in json.loads(response.body)
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


@pytest.mark.asyncio
async def test_gateway_health_reports_manager_unavailable() -> None:
    gateway = InferenceGateway(Role.TEACHER, snapshot_provider=lambda: (_ for _ in ()).throw(RuntimeError("down")))
    try:
        response = await gateway.health()
        assert response.status_code == 503
        assert json.loads(response.body)["status"] == "unavailable"
    finally:
        await gateway.close()
