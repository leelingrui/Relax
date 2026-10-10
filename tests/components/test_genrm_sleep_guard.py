# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""The GenRM backend ``/generate`` admits direct requests through the Manager.

That endpoint reaches the engines without the Gateway, so it records its own
requests: a model that is not READY is refused with 503, every admitted request
is completed, and a cancelled caller aborts the engine request first. The real
``GenRM`` methods are driven on an instance built without ``__init__``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest


try:
    from fastapi import HTTPException

    import relax.components.genrm as genrm_module
    from relax.engine.inference.types import Role

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="requires ray[serve] + relax deps")


class _Manager:
    """An inference manager handle admitting only the ``ready`` instances."""

    def __init__(self, ready) -> None:
        self.inflight: dict[str, str] = {}
        self.completed: list[str] = []

        async def admit_request(role, model_id, trace_id):
            assert role == Role.GENRM
            if model_id not in ready:
                raise RuntimeError(f"Model genrm/{model_id} is not ready for inference")
            permit = f"permit-{trace_id}"
            self.inflight[permit] = model_id
            return permit

        async def complete_request(permit):
            self.inflight.pop(permit, None)
            self.completed.append(permit)

        self.admit_request = SimpleNamespace(remote=admit_request)
        self.complete_request = SimpleNamespace(remote=complete_request)


class _Client:
    def __init__(self, generate) -> None:
        self.posts: list[tuple[str, dict]] = []
        self._generate = generate

    async def post(self, url, json, timeout=None):
        self.posts.append((url, json))
        if url.endswith("/abort_request"):
            return None
        return await self._generate(json)


def _replica(manager, generate=None):
    async def ok(payload):
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"text": " 1 "})

    cls = genrm_module.GenRM.func_or_class.__bases__[0]
    replica = object.__new__(cls)
    replica._logger_instance = None
    replica._inference_manager = manager
    replica.genrm_managers = {"__default__": object(), "judge": object()}
    replica._http_client = _Client(generate or ok)
    replica._pick_engine = lambda route_key: (route_key, 0, "192.0.2.1", 16001)

    async def prepare(key, messages, sampling_params=None):
        return {"input_ids": [1]}

    replica.prepare_generate_payload = prepare
    return replica


def _request(route_key="judge"):
    return genrm_module.GenerateRequest(messages=[{"role": "user", "content": "q"}], route_key=route_key)


@pytest.mark.asyncio
async def test_genrm_generate_refuses_unadmitted_instance_with_503() -> None:
    manager = _Manager(ready={"__default__"})
    replica = _replica(manager)

    with pytest.raises(HTTPException) as caught:
        await replica.generate(_request("judge"))

    assert caught.value.status_code == 503
    assert caught.value.headers == {"Retry-After": "1"}
    assert replica._http_client.posts == []


@pytest.mark.asyncio
async def test_genrm_generate_completes_admitted_request_with_its_rid() -> None:
    manager = _Manager(ready={"judge"})
    replica = _replica(manager)

    response = await replica.generate(_request("judge"))

    assert response.response == "1"
    ((url, payload),) = replica._http_client.posts
    assert url.endswith("/generate")
    assert manager.completed == [f"permit-{payload['rid']}"]
    assert manager.inflight == {}


@pytest.mark.asyncio
async def test_genrm_generate_aborts_engine_request_when_cancelled() -> None:
    manager = _Manager(ready={"judge"})
    started = asyncio.Event()

    async def hang(payload):
        started.set()
        await asyncio.Event().wait()

    replica = _replica(manager, hang)
    task = asyncio.create_task(replica.generate(_request("judge")))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    (_, generate), (abort_url, abort) = replica._http_client.posts
    assert abort_url.endswith("/abort_request") and abort == {"rid": generate["rid"]}
    assert manager.completed == [f"permit-{generate['rid']}"] and manager.inflight == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("retried", [True, False])
async def test_genrm_generate_aborts_a_request_whose_reply_was_lost(monkeypatch, retried: bool) -> None:
    from unittest.mock import MagicMock

    import httpx

    monkeypatch.setenv("GENRM_ENGINE_RETRY_ATTEMPTS", "2" if retried else "1")
    manager = _Manager(ready={"judge"})
    hosts = iter([("judge", 0, "192.0.2.1", 16001), ("judge", 1, "192.0.2.2", 16002)])
    calls = []

    async def lose_first_reply(payload):
        calls.append(payload["rid"])
        if len(calls) == 1:
            raise httpx.ReadTimeout("reply lost")
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"text": " 1 "})

    replica = _replica(manager, lose_first_reply)
    replica._pick_engine = lambda route_key: next(hosts)
    replica._engine_caches = {"judge": MagicMock()}

    if retried:
        assert (await replica.generate(_request("judge"))).response == "1"
    else:
        with pytest.raises(httpx.ReadTimeout):
            await replica.generate(_request("judge"))

    urls = [url for url, _ in replica._http_client.posts]
    # The first engine may still run the request: it is aborted before the
    # retry reaches the second engine, or before the caller gives up.
    assert urls[:2] == ["http://192.0.2.1:16001/generate", "http://192.0.2.1:16001/abort_request"]
    assert replica._http_client.posts[1][1] == {"rid": calls[0]}
    assert urls[2:] == (["http://192.0.2.2:16002/generate"] if retried else [])
    assert manager.inflight == {}
