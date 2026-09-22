# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Legacy GenRM HTTP contract; retry behavior is not a future replay
guarantee."""

from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from relax.engine.inference.types import LifecycleState, Role
from relax.utils.genrm_client import GenRMClient


@pytest.mark.parametrize("sampling", [None, {"temperature": 0.7}])
async def test_genrm_client_preserves_messages_and_response(sampling: dict | None) -> None:
    client = object.__new__(GenRMClient)
    client.service_url = "http://genrm.test/genrm"
    request = httpx.Request("POST", f"{client.service_url}/generate")
    client._async_client = AsyncMock()
    client._async_client.post.return_value = httpx.Response(200, json={"response": "judge"}, request=request)
    messages = [{"role": "user", "content": "score"}]

    assert await client.generate(messages, sampling) == "judge"
    payload = {"messages": messages}
    if sampling is not None:
        payload["sampling_params"] = sampling
    client._async_client.post.assert_awaited_once_with(str(request.url), json=payload)


@pytest.mark.parametrize("status,attempts", [(400, 1), (503, 3)])
async def test_genrm_client_legacy_http_retry_limit(monkeypatch, status: int, attempts: int) -> None:
    client = object.__new__(GenRMClient)
    client.service_url = "http://genrm.test/genrm"
    request = httpx.Request("POST", f"{client.service_url}/generate")
    client._async_client = AsyncMock()
    client._async_client.post.return_value = httpx.Response(status, request=request)
    sleep = AsyncMock()
    monkeypatch.setattr("relax.utils.genrm_client.asyncio.sleep", sleep)

    with pytest.raises(httpx.HTTPStatusError):
        await client.generate([])
    assert client._async_client.post.await_count == attempts
    assert sleep.await_count == attempts - 1


@pytest.mark.parametrize("status_filter", [None, "active", "dead"])
def test_genrm_client_reads_v2_discovery(status_filter: str | None) -> None:
    client = object.__new__(GenRMClient)
    client.service_url = "http://genrm.test/genrm"
    snapshot = {
        "schema_version": 2,
        "role": "genrm",
        "manager_epoch": "epoch-a",
        "phase": None,
        "routing": {"default_model": "judge", "route_key_to_model": {}, "config_version": 0},
        "models": {
            "judge": {
                "state": LifecycleState.READY.value,
                "admission": True,
                "allow_defer": True,
                "direct_eligible": False,
                "router_url": "http://router",
                "required_weight_version": None,
                "engines": [],
            }
        },
    }
    response = httpx.Response(
        200,
        json=snapshot,
        request=httpx.Request("GET", "http://genrm.test/genrm/engines"),
    )
    client._sync_client = Mock()
    client._sync_client.get.return_value = response
    result = client.get_discovery(status_filter=status_filter)
    assert result.role is Role.GENRM
    assert result.models[0].state is LifecycleState.READY
    expected_params = {"schema_version": 2}
    if status_filter is not None:
        expected_params["status_filter"] = status_filter
    client._sync_client.get.assert_called_once_with("http://genrm.test/genrm/engines", params=expected_params)
