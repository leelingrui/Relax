# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Adapters and storage for the common inference discovery contract."""

from dataclasses import replace
from threading import RLock
from typing import Any, Mapping
from uuid import uuid4

from relax.engine.inference.types import (
    LifecycleState,
    ModelSnapshot,
    ReplicaSnapshot,
    Role,
    RoleSnapshot,
    RoutingSpec,
)


def new_manager_epoch() -> str:
    """Create an opaque identifier for one Manager/control-plane lifetime."""
    return uuid4().hex


class DiscoveryState:
    """Thread-safe publisher for the latest complete role snapshot."""

    def __init__(self, snapshot: RoleSnapshot) -> None:
        self._snapshot = snapshot
        self._lock = RLock()

    def get(self) -> RoleSnapshot:
        with self._lock:
            return self._snapshot

    def publish(self, snapshot: RoleSnapshot) -> RoleSnapshot:
        if snapshot.role != self._snapshot.role:
            raise ValueError("Discovery snapshots must keep the same role")
        with self._lock:
            self._snapshot = snapshot
            return snapshot

    def update(self, **changes: Any) -> RoleSnapshot:
        """Publish one immutable snapshot while retaining unchanged fields."""
        return self.publish(replace(self.get(), **changes))


def snapshot_from_legacy_engines(
    payload: Mapping[str, Any],
    *,
    role: Role,
    manager_epoch: str,
    topology_revision: int = 0,
    phase: str | None = None,
    default_model: str | None = None,
    route_key_to_model: tuple[tuple[str, str], ...] = (),
    model_states: Mapping[str, LifecycleState | None] | None = None,
    admission: Mapping[str, bool] | None = None,
    allow_defer: Mapping[str, bool] | None = None,
    direct_eligible: Mapping[str, bool] | None = None,
    required_weight_versions: Mapping[str, str | None] | None = None,
) -> RoleSnapshot:
    """Adapt the legacy ``/engines`` response into a v2 snapshot.

    Legacy ``active`` only proves that an actor slot exists.  It therefore
    becomes a replica with ``state=None`` and never implies model readiness.
    """
    model_states = model_states or {}
    admission = admission or {}
    allow_defer = allow_defer or {}
    direct_eligible = direct_eligible or {}
    required_weight_versions = required_weight_versions or {}
    models = []
    for model_id, model_payload in dict(payload.get("models", {})).items():
        replicas = []
        for group in model_payload.get("engine_groups", ()):
            for engine in group.get("engines", ()):
                rank = engine.get("rank")
                engine_id = f"{model_id}/replica-{rank}" if rank is not None else f"{model_id}/replica-{len(replicas)}"
                replicas.append(
                    ReplicaSnapshot(
                        engine_id=engine_id,
                        base_url=engine.get("url"),
                        state=LifecycleState.DEAD if engine.get("status") == "dead" else None,
                        weight_version=engine.get("weight_version"),
                    )
                )
        router_ip = model_payload.get("router_ip")
        router_port = model_payload.get("router_port")
        router_url = f"http://{router_ip}:{router_port}" if router_ip and router_port else None
        models.append(
            ModelSnapshot(
                model_id=model_id,
                replicas=tuple(replicas),
                router_url=router_url,
                state=model_states.get(model_id),
                admission=admission.get(model_id, False),
                allow_defer=allow_defer.get(model_id, False),
                direct_eligible=direct_eligible.get(model_id, False),
                required_weight_version=required_weight_versions.get(model_id),
            )
        )
    return RoleSnapshot(
        role=role,
        manager_epoch=manager_epoch,
        topology_revision=topology_revision,
        phase=phase,
        models=tuple(models),
        routing=RoutingSpec(default_model=default_model, route_key_to_model=route_key_to_model),
    )


def snapshot_from_engine_urls(
    urls: list[str] | tuple[str, ...],
    *,
    role: Role,
    model_id: str,
    manager_epoch: str,
    topology_revision: int = 0,
    router_url: str | None = None,
    state: LifecycleState | None = None,
    admission: bool = False,
    allow_defer: bool = False,
    direct_eligible: bool = False,
    required_weight_version: str | None = None,
    phase: str | None = None,
) -> RoleSnapshot:
    """Adapt GenRM/Teacher URL lists that have no legacy group structure."""
    model = ModelSnapshot(
        model_id=model_id,
        replicas=tuple(
            ReplicaSnapshot(engine_id=f"{model_id}/replica-{rank}", base_url=url, state=None)
            for rank, url in enumerate(urls)
        ),
        router_url=router_url,
        state=state,
        admission=admission,
        allow_defer=allow_defer,
        direct_eligible=direct_eligible,
        required_weight_version=required_weight_version,
    )
    return RoleSnapshot(
        role=role,
        manager_epoch=manager_epoch,
        topology_revision=topology_revision,
        phase=phase,
        models=(model,),
    )


def role_snapshot_from_dict(payload: Mapping[str, Any]) -> RoleSnapshot:
    """Parse a v2 JSON discovery response without trusting missing fields."""
    models = []
    for model_id, model_payload in dict(payload.get("models", {})).items():
        replicas = tuple(
            ReplicaSnapshot(
                engine_id=replica["engine_id"],
                base_url=replica.get("base_url"),
                state=LifecycleState(replica["state"]) if replica.get("state") else None,
                weight_version=replica.get("weight_version"),
            )
            for replica in model_payload.get("engines", ())
        )
        models.append(
            ModelSnapshot(
                model_id=model_id,
                replicas=replicas,
                router_url=model_payload.get("router_url"),
                state=LifecycleState(model_payload["state"]) if model_payload.get("state") else None,
                admission=bool(model_payload.get("admission", False)),
                allow_defer=bool(model_payload.get("allow_defer", False)),
                direct_eligible=bool(model_payload.get("direct_eligible", False)),
                required_weight_version=model_payload.get("required_weight_version"),
            )
        )
    routing_payload = payload.get("routing", {})
    return RoleSnapshot(
        role=Role(payload["role"]),
        manager_epoch=payload["manager_epoch"],
        topology_revision=int(payload.get("topology_revision", 0)),
        phase=payload.get("phase"),
        models=tuple(models),
        routing=RoutingSpec(
            default_model=routing_payload.get("default_model"),
            route_key_to_model=tuple(routing_payload.get("route_key_to_model", {}).items()),
            config_version=int(routing_payload.get("config_version", 0)),
        ),
    )
