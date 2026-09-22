# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""A cancelled request must be aborted upstream, not declared finished.

The gateway is the only component that knows which engine received a request, so
it owns sending the abort. It is also the component most likely to be wrong about
completion: a 502, a broken stream or a client disconnect say nothing about
whether the engine is still generating, and treating them as completion is how a
deactivation ends up releasing memory under a running request.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from starlette.requests import Request

from relax.components.inference_gateway import InferenceGateway
from relax.engine.inference.discovery import snapshot_from_engine_urls
from relax.engine.inference.manager import RequestPermit
from relax.engine.inference.types import LifecycleState, Role


PERMIT = RequestPermit("request-1", "epoch-a", Role.GENRM, "model-a", "http://router")


def _snapshot():
    return snapshot_from_engine_urls(
        ["http://engine/generate"],
        role=Role.GENRM,
        model_id="model-a",
        manager_epoch="epoch-a",
        router_url="http://router",
        state=LifecycleState.READY,
        admission=True,
    )


def _request(payload: dict) -> Request:
    body = json.dumps({"model": "model-a", **payload}).encode()

    async def receive():
        return {"type": "http.request", "body": body}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/generate",
            "query_string": b"",
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                # Pin the permit identity so the abort target is predictable.
                (b"x-relax-request-id", b"request-1"),
            ],
        },
        receive,
    )


class FakeManager:
    """Records the admission bookkeeping the gateway drives."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.admit_request = SimpleNamespace(remote=AsyncMock(side_effect=self._admit))
        self.complete_request = SimpleNamespace(remote=AsyncMock(side_effect=self._complete))
        self.cancel_request = SimpleNamespace(remote=AsyncMock(side_effect=self._cancel))
        self.snapshot = SimpleNamespace(remote=AsyncMock(side_effect=self._snapshot))

    async def _snapshot(self, *, role):
        return _snapshot()

    async def _admit(self, **kwargs):
        self.calls.append(("admit", kwargs["request_id"]))
        return PERMIT

    async def _complete(self, *, permit):
        self.calls.append(("complete", permit.request_id))

    async def _cancel(self, *, permit, dispatched=True):
        self.calls.append(("cancel", permit.request_id, dispatched))


def build_gateway(upstream) -> tuple[InferenceGateway, FakeManager, list]:
    manager = FakeManager()
    gateway = InferenceGateway(Role.GENRM, manager_handle=manager)
    seen: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return upstream(request)

    # Replace the client entirely and ignore the environment: an ambient
    # http_proxy would otherwise be mounted ahead of the mock transport.
    gateway._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)
    return gateway, manager, seen


def aborted_rids(seen: list) -> list[str]:
    return [
        json.loads(request.content)["rid"]
        for request in seen
        if request.url.path == "/abort_request" and "rid" in json.loads(request.content)
    ]


@pytest.mark.asyncio
async def test_a_completed_request_is_reported_complete_and_carries_an_addressable_rid() -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": "ok", "meta_info": {}})

    gateway, manager, seen = build_gateway(upstream)
    try:
        response = await gateway.proxy(_request({"text": "hi"}), "generate")
        assert response.status_code == 200
    finally:
        await gateway.close()
    # The permit identity is handed to the engine, so a cancel has a target.
    assert json.loads(seen[0].content)["rid"] == "request-1"
    assert manager.calls == [("admit", "request-1"), ("complete", "request-1")]


@pytest.mark.asyncio
async def test_a_caller_supplied_rid_is_not_overwritten() -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": "ok", "meta_info": {}})

    gateway, _manager, seen = build_gateway(upstream)
    try:
        await gateway.proxy(_request({"text": "hi", "rid": "caller-rid"}), "generate")
    finally:
        await gateway.close()
    assert json.loads(seen[0].content)["rid"] == "caller-rid"


@pytest.mark.asyncio
async def test_a_read_failure_aborts_upstream_and_does_not_complete() -> None:
    """A 502 from a broken read is not evidence that the engine stopped."""

    def upstream(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/workers":
            return httpx.Response(200, json={"workers": [{"url": "http://worker-1:30000", "is_healthy": True}]})
        if request.url.path == "/abort_request":
            return httpx.Response(200, json={})
        raise httpx.ReadError("upstream went away")

    gateway, manager, seen = build_gateway(upstream)
    try:
        response = await gateway.proxy(_request({"text": "hi"}), "generate")
        assert response.status_code == 502
    finally:
        await gateway.close()
    assert aborted_rids(seen) == ["request-1"]
    assert manager.calls == [("admit", "request-1"), ("cancel", "request-1", True)]


@pytest.mark.asyncio
async def test_the_router_list_workers_fallback_is_used_when_workers_is_absent() -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/workers":
            return httpx.Response(404, json={})
        if request.url.path == "/list_workers":
            return httpx.Response(200, json={"urls": ["http://worker-1:30000"]})
        if request.url.path == "/abort_request":
            return httpx.Response(200, json={})
        raise httpx.ReadError("upstream went away")

    gateway, manager, seen = build_gateway(upstream)
    try:
        await gateway.proxy(_request({"text": "hi"}), "generate")
    finally:
        await gateway.close()
    assert aborted_rids(seen) == ["request-1"]
    assert manager.calls == [("admit", "request-1"), ("cancel", "request-1", True)]


@pytest.mark.asyncio
async def test_a_connect_failure_clears_the_registration_without_an_abort() -> None:
    """Nothing reached an engine, so there is nothing to wait for or abort."""

    def upstream(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to engine")

    gateway, manager, seen = build_gateway(upstream)
    try:
        response = await gateway.proxy(_request({"text": "hi"}), "generate")
        assert response.status_code == 502
    finally:
        await gateway.close()
    assert aborted_rids(seen) == []
    assert manager.calls == [("admit", "request-1"), ("cancel", "request-1", False)]


@pytest.mark.asyncio
async def test_a_broken_stream_aborts_upstream_and_keeps_the_request_in_flight() -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/workers":
            return httpx.Response(200, json={"workers": [{"url": "http://worker-1:30000", "is_healthy": True}]})
        if request.url.path == "/abort_request":
            return httpx.Response(200, json={})

        async def chunks():
            yield b"data: one\n\n"
            raise httpx.ReadError("client went away")

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=chunks())

    gateway, manager, seen = build_gateway(upstream)
    try:
        response = await gateway.proxy(_request({"text": "hi", "stream": True}), "generate")
        with pytest.raises(httpx.ReadError):
            async for _chunk in response.body_iterator:
                pass
    finally:
        await gateway.close()
    assert aborted_rids(seen) == ["request-1"]
    assert manager.calls == [("admit", "request-1"), ("cancel", "request-1", True)]


@pytest.mark.asyncio
async def test_a_stream_that_ends_normally_is_completed_without_an_abort() -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        async def chunks():
            yield b"data: [DONE]\n\n"

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=chunks())

    gateway, manager, seen = build_gateway(upstream)
    try:
        response = await gateway.proxy(_request({"text": "hi", "stream": True}), "generate")
        async for _chunk in response.body_iterator:
            pass
    finally:
        await gateway.close()
    assert aborted_rids(seen) == []
    assert manager.calls == [("admit", "request-1"), ("complete", "request-1")]


@pytest.mark.asyncio
async def test_an_unreachable_worker_list_still_records_the_cancel() -> None:
    """A request nobody could abort is exactly the one that must keep
    blocking."""

    def upstream(request: httpx.Request) -> httpx.Response:
        if request.url.path in ("/workers", "/list_workers"):
            raise httpx.ConnectError("router is gone")
        raise httpx.ReadError("upstream went away")

    gateway, manager, seen = build_gateway(upstream)
    try:
        await gateway.proxy(_request({"text": "hi"}), "generate")
    finally:
        await gateway.close()
    assert aborted_rids(seen) == []
    assert manager.calls == [("admit", "request-1"), ("cancel", "request-1", True)]


@pytest.mark.asyncio
async def test_the_abort_is_broadcast_to_every_worker_behind_the_router() -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/workers":
            # The router reports dp-rank-suffixed identities; they are normalized
            # to reachable base URLs the same way the rollout path does it.
            return httpx.Response(
                200,
                json={
                    "workers": [
                        {"url": "http://worker-1:30000@0", "is_healthy": True},
                        {"url": "http://worker-1:30000@1", "is_healthy": True},
                        {"url": "http://worker-2:30000", "is_healthy": True},
                    ]
                },
            )
        if request.url.path == "/abort_request":
            return httpx.Response(200, json={})
        raise httpx.ReadError("upstream went away")

    gateway, _manager, seen = build_gateway(upstream)
    try:
        await gateway.proxy(_request({"text": "hi"}), "generate")
    finally:
        await gateway.close()
    targets = sorted(str(request.url) for request in seen if request.url.path == "/abort_request")
    assert targets == ["http://worker-1:30000/abort_request", "http://worker-2:30000/abort_request"]
    assert aborted_rids(seen) == ["request-1", "request-1"]
