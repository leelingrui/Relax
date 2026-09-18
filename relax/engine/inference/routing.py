# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Pure selection shared by discovery clients and Manager admission.

Gateway callers must perform selection under Manager admission, not dispatch
from a previously observed snapshot. No function here sends or retries
requests.
"""

from relax.engine.inference.types import (
    LifecycleState,
    ModelSnapshot,
    RoleSnapshot,
    RouteTarget,
)


class RoutingError(ValueError):
    def __init__(self, code: str, message: str, *, status_code: int, model_id: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.model_id = model_id


def resolve_model(snapshot: RoleSnapshot, *, model: str | None = None, route_key: str | None = None) -> ModelSnapshot:
    """Resolve explicit model, explicit route key, then configured default."""
    if model is not None:
        selected = model
    elif route_key is not None:
        selected = dict(snapshot.routing.route_key_to_model).get(route_key)
        if selected is None:
            raise RoutingError("unknown_route", f"No model configured for route key {route_key!r}", status_code=400)
    else:
        selected = snapshot.routing.default_model
    if selected is None:
        raise RoutingError("model_required", "No model or default model selected", status_code=400)
    for candidate in snapshot.models:
        if candidate.model_id == selected:
            return candidate
    raise RoutingError("unknown_model", f"Unknown model {selected!r}", status_code=400, model_id=selected)


def select_target(model: ModelSnapshot, *, cursor: int = 0) -> RouteTarget:
    """Select the model's Router endpoint.

    Internal callers always use the Router. Replica addresses remain discovery
    observations for external integrations and diagnostics, not request
    targets.
    """
    if cursor < 0:
        raise RoutingError("invalid_cursor", "Round-robin cursor must be non-negative", status_code=400)
    if not model.admission or model.state != LifecycleState.READY:
        raise RoutingError("unavailable", "Model is not accepting requests", status_code=503, model_id=model.model_id)
    if not model.router_url:
        raise RoutingError("unavailable", "Model router is unavailable", status_code=503, model_id=model.model_id)
    return RouteTarget(model_id=model.model_id, base_url=model.router_url)
