# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU HTTP gateway shared by rollout, GenRM, and Teacher services."""

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any
from uuid import uuid4

import httpx
import ray
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from ray import serve

from relax.engine.inference.discovery import role_snapshot_from_dict
from relax.engine.inference.routing import RoutingError, resolve_model, select_target
from relax.engine.inference.types import Role, RoleSnapshot
from relax.utils.logging_utils import get_logger


GatewaySnapshotProvider = Callable[[], RoleSnapshot | Mapping[str, Any] | Awaitable[RoleSnapshot | Mapping[str, Any]]]

_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
GATEWAY_REQUEST_HEADER = "x-relax-inference-gateway"


def _json_error(status_code: int, message: str, *, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
        headers={"Retry-After": "1"} if status_code == 503 else None,
    )


class InferenceGateway:
    """Role-neutral HTTP ingress which routes through a Manager snapshot.

    The gateway owns no GPU resources and keeps no availability state of its
    own. ``manager_handle`` or ``snapshot_provider`` is the only state source.
    """

    app = FastAPI()

    def __init__(
        self,
        role: str | Role,
        *,
        manager_handle: Any | None = None,
        snapshot_provider: GatewaySnapshotProvider | None = None,
        role_manager_handle: Any | None = None,
        upstream_url: str | None = None,
        # Kept only so old deployment construction fails at request time with
        # a clear manager error instead of failing while binding Serve.
        discovery_url: str | None = None,
        manager_handles: Mapping[str, Any] | None = None,
        genrm_backend_handle: Any | None = None,
        timeout: float = 1800.0,
    ) -> None:
        self.role = Role(role)
        self.manager_handle = manager_handle
        self.snapshot_provider = snapshot_provider
        if manager_handle is not None and role_manager_handle is not None:
            raise ValueError("Use manager_handle or role_manager_handle, not both")
        self.role_manager_handle = role_manager_handle
        self.upstream_url = upstream_url.rstrip("/") if upstream_url else None
        self.genrm_backend_handle = genrm_backend_handle
        if discovery_url is not None or manager_handles:
            self._logger = get_logger(__name__)
            self._logger.warning("Ignoring legacy discovery inputs; configure one task InferenceManager handle")
        self._client = httpx.AsyncClient(timeout=timeout, limits=httpx.Limits(max_connections=2048))
        self._logger = get_logger(__name__)

    async def _snapshot(self) -> RoleSnapshot:
        if self.manager_handle is not None:
            value = self.manager_handle.snapshot.remote(role=self.role)
            if inspect.isawaitable(value):
                value = await value
            elif not isinstance(value, RoleSnapshot):
                value = await asyncio.to_thread(ray.get, value)
            return value
        if self.role_manager_handle is not None:
            return await asyncio.to_thread(ray.get, self.role_manager_handle.get_role_snapshot.remote())
        if self.snapshot_provider is not None:
            value = self.snapshot_provider()
            if inspect.isawaitable(value):
                value = await value
            if isinstance(value, RoleSnapshot):
                return value
            return role_snapshot_from_dict(value)
        raise HTTPException(status_code=503, detail="Inference manager is not configured")

    async def _target(self, payload: Mapping[str, Any] | None = None) -> str:
        target, _ = await self._resolve_target(payload)
        return target

    async def _resolve_target(self, payload: Mapping[str, Any] | None = None) -> tuple[str, str]:
        snapshot = await self._snapshot()
        payload = payload or {}
        try:
            model = resolve_model(
                snapshot,
                model=payload.get("model"),
                route_key=payload.get("route_key"),
            )
            try:
                return select_target(model).base_url.rstrip("/"), model.model_id
            except RoutingError:
                raise
        except RoutingError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail={"code": exc.code, "message": str(exc)},
                headers={"Retry-After": "1"} if exc.status_code == 503 else None,
            ) from exc

    async def engines(self, schema_version: int | None = None, status_filter: str | None = None) -> Any:
        snapshot = await self._snapshot()
        if schema_version == 2:
            return snapshot.to_dict(status_filter)
        return snapshot.to_legacy_dict(status_filter)

    async def health(self) -> JSONResponse:
        """Report gateway liveness separately from model readiness."""
        try:
            snapshot = await self._snapshot()
        except Exception as exc:
            self._logger.warning("Inference discovery unavailable: %s", exc)
            return JSONResponse(
                {"status": "unavailable", "service": self.role.value, "model_status": "unknown"}, status_code=503
            )
        ready = any(model.admission and model.state and model.state.value == "ready" for model in snapshot.models)
        return JSONResponse(
            {
                "status": "healthy",
                "service": self.role.value,
                "model_status": "ready" if ready else "unavailable",
                "manager_epoch": snapshot.manager_epoch,
            }
        )

    async def models(self) -> dict[str, Any]:
        snapshot = await self._snapshot()
        return {
            "object": "list",
            "data": [
                {"id": model.model_id, "object": "model", "created": 0, "owned_by": "relax"}
                for model in snapshot.models
            ],
        }

    async def proxy(self, request: Request, path: str) -> Response:
        body = await request.body()
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError as exc:
            return _json_error(400, f"Invalid JSON request body: {exc}", code="invalid_request")
        if not isinstance(payload, dict):
            return _json_error(400, "Request body must be a JSON object", code="invalid_request")

        messages_request = self.role == Role.GENRM and path == "generate" and "messages" in payload
        if messages_request and ("input_ids" in payload or "text" in payload):
            return _json_error(400, "messages cannot be combined with input_ids or text", code="invalid_request")
        if messages_request and (
            not isinstance(payload["messages"], list)
            or (payload.get("sampling_params") is not None and not isinstance(payload["sampling_params"], dict))
        ):
            return _json_error(400, "Invalid messages or sampling_params", code="invalid_request")

        try:
            target, model_id = await self._resolve_target(payload)
        except HTTPException as exc:
            return _json_error(
                exc.status_code, str(exc.detail), code="unavailable" if exc.status_code == 503 else "routing_error"
            )

        permit = await self._admit(request, model_id, target)

        try:
            payload.pop("route_key", None)
            if path == "generate":
                payload.pop("model", None)
            wrap_response = False
            if messages_request:
                if self.genrm_backend_handle is None:
                    raise HTTPException(status_code=503, detail="GenRM adapter is not configured")
                payload = await self.genrm_backend_handle.prepare_generate_payload.remote(
                    model_id, payload["messages"], payload.get("sampling_params")
                )
                wrap_response = True
            response = await self._forward(
                request, path, target, json.dumps(payload).encode(), wrap_response=wrap_response, permit=permit
            )
            if permit is not None and not isinstance(response, StreamingResponse):
                await self._complete(permit)
            return response
        except Exception:
            if permit is not None:
                await self._cancel(permit)
            raise

    async def _manager_call(self, method: str, **kwargs: Any) -> Any:
        if self.manager_handle is None:
            return None
        remote_method = getattr(self.manager_handle, method, None)
        if remote_method is None:
            return None
        value = remote_method.remote(**kwargs)
        if inspect.isawaitable(value):
            return await value
        return await asyncio.to_thread(ray.get, value)

    async def _admit(self, request: Request, model_id: str, target: str) -> Any:
        if self.manager_handle is None:
            return None
        request_id = request.headers.get("x-relax-request-id") or uuid4().hex
        try:
            return await self._manager_call(
                "admit_request",
                model_id=model_id,
                request_id=request_id,
                role=self.role,
                target=target,
            )
        except Exception as exc:
            self._logger.warning("Inference request admission failed: %s", exc)
            raise HTTPException(status_code=503, detail="Inference request is not admitted") from exc

    async def _complete(self, permit: Any) -> None:
        await self._manager_call("complete_request", permit=permit)

    async def _cancel(self, permit: Any) -> None:
        await self._manager_call("cancel_request", permit=permit)

    async def proxy_backend(self, request: Request, path: str) -> Response:
        """Forward role control-plane endpoints to the internal service."""
        if self.upstream_url is None:
            return _json_error(404, "Gateway backend is not configured", code="backend_unavailable")
        return await self._forward(request, path, self.upstream_url, await request.body())

    async def _forward(
        self,
        request: Request,
        path: str,
        target: str,
        body: bytes,
        *,
        wrap_response: bool = False,
        permit: Any = None,
    ) -> Response:
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in _HOP_BY_HOP_HEADERS and key.lower() not in {"host", "content-length"}
        }
        headers[GATEWAY_REQUEST_HEADER] = "1"
        upstream_url = f"{target}/{path.lstrip('/')}"
        if request.url.query:
            upstream_url = f"{upstream_url}?{request.url.query}"
        upstream = self._client.build_request(request.method, upstream_url, content=body, headers=headers)
        response: httpx.Response | None = None
        try:
            response = await self._client.send(upstream, stream=True)
            if response.headers.get("content-type", "").startswith("text/event-stream"):
                stream_response = self._stream_response(response, permit)
                permit = None
                response = None
                return stream_response
            content = await response.aread()
            response_headers = {
                key: value for key, value in response.headers.items() if key.lower() not in _HOP_BY_HOP_HEADERS
            }
            if wrap_response and response.is_success:
                data = json.loads(content)
                data = {"response": data.get("text", "").strip()}
                response_headers.pop("content-length", None)
                response_headers.pop("content-encoding", None)
                return JSONResponse(data, status_code=response.status_code, headers=response_headers)
            return Response(
                content=content, status_code=response.status_code, headers=response_headers, media_type=None
            )
        except httpx.RequestError as exc:
            return _json_error(502, f"Failed to connect to inference router: {exc}", code="upstream_unavailable")
        finally:
            if response is not None and not response.is_closed:
                await response.aclose()

    def _stream_response(self, response: httpx.Response, permit: Any = None) -> StreamingResponse:
        async def body():
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            except BaseException:
                if permit is not None:
                    await self._cancel(permit)
                raise
            else:
                if permit is not None:
                    await self._complete(permit)
            finally:
                await response.aclose()

        headers = {key: value for key, value in response.headers.items() if key.lower() not in _HOP_BY_HOP_HEADERS}
        return StreamingResponse(
            body(), status_code=response.status_code, headers=headers, media_type="text/event-stream"
        )

    async def close(self) -> None:
        await self._client.aclose()


gateway_app = FastAPI()


@gateway_app.get("/engines")
async def _engines(request: Request, schema_version: int | None = None, status_filter: str | None = None):
    return await request.app.state.gateway.engines(schema_version, status_filter)


@gateway_app.get("/health")
async def _health(request: Request):
    return await request.app.state.gateway.health()


@gateway_app.get("/v1/models")
async def _models(request: Request):
    return await request.app.state.gateway.models()


@gateway_app.api_route("/generate", methods=["POST"])
async def _generate(request: Request):
    return await request.app.state.gateway.proxy(request, "generate")


@gateway_app.api_route("/v1/chat/completions", methods=["POST"])
async def _chat(request: Request):
    return await request.app.state.gateway.proxy(request, "v1/chat/completions")


@gateway_app.api_route("/chat/completions", methods=["POST"])
async def _chat_legacy(request: Request):
    return await request.app.state.gateway.proxy(request, "v1/chat/completions")


@gateway_app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def _backend(request: Request, path: str):
    return await request.app.state.gateway.proxy_backend(request, path)


@serve.deployment(ray_actor_options={"num_gpus": 0}, max_ongoing_requests=128)
@serve.ingress(gateway_app)
class InferenceGatewayDeployment(InferenceGateway):
    """Ray Serve deployment form of :class:`InferenceGateway`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        gateway_app.state.gateway = self


__all__ = ["InferenceGateway", "InferenceGatewayDeployment", "gateway_app"]
