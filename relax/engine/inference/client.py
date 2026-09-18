# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Small HTTP client for v2 inference discovery."""

from typing import Any

import httpx

from relax.engine.inference.discovery import role_snapshot_from_dict
from relax.engine.inference.routing import resolve_model, select_target
from relax.engine.inference.types import ModelSnapshot, RoleSnapshot, RouteTarget


class InferenceDiscoveryClient:
    """Fetch discovery snapshots; request admission remains server-side."""

    def __init__(self, base_url: str, *, timeout: float = 10.0, client: httpx.Client | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None

    def get_snapshot(
        self, role: str, *, schema_version: int = 2, status_filter: str | None = None
    ) -> RoleSnapshot:
        params: dict[str, Any] = {"schema_version": schema_version}
        if status_filter is not None:
            params["status_filter"] = status_filter
        response = self._client.get(f"{self.base_url}/{role}/engines", params=params)
        response.raise_for_status()
        return role_snapshot_from_dict(response.json())

    def resolve_model(
        self, snapshot: RoleSnapshot, *, model: str | None = None, route_key: str | None = None
    ) -> ModelSnapshot:
        return resolve_model(snapshot, model=model, route_key=route_key)

    def select_target(self, model: ModelSnapshot, *, cursor: int = 0) -> RouteTarget:
        return select_target(model, cursor=cursor)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "InferenceDiscoveryClient":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
