# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Internal definitions; placement and process creation remain separate."""

from dataclasses import dataclass
from typing import Any

from relax.engine.inference.capabilities import WeightSource
from relax.engine.inference.config import EngineGroupConfig, ModelConfig
from relax.engine.inference.types import Role, RoutingSpec


@dataclass(frozen=True)
class ReplicaSpec:
    replica_id: str
    node_ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.replica_id or not self.node_ranks:
            raise ValueError("Replica identity and node ranks are required")
        if min(self.node_ranks) < 0 or len(set(self.node_ranks)) != len(self.node_ranks):
            raise ValueError("Node ranks must be unique and non-negative")


def replicas_from_slots(
    prefix: str, num_slots: int, nodes_per_engine: int, *, first_slot: int = 0
) -> tuple[ReplicaSpec, ...]:
    """Name the replicas of one engine group.

    ``first_slot`` offsets the identity, not the node ranks: a group that
    starts at engine slot N names its replicas from N while its ranks stay
    local to the group. That keeps an identity stable for the life of the
    replica even when another group is added or removed around it.
    """
    if num_slots < 0 or nodes_per_engine < 1 or num_slots % nodes_per_engine:
        raise ValueError("Engine slots must contain complete logical replicas")
    return tuple(
        ReplicaSpec(f"{prefix}/replica-{first_slot + head}", tuple(range(head, head + nodes_per_engine)))
        for head in range(0, num_slots, nodes_per_engine)
    )


@dataclass(frozen=True)
class EngineGroupSpec:
    """Runtime topology only; deployment parameters live in
    EngineGroupConfig."""

    group_id: str
    replicas: tuple[ReplicaSpec, ...]

    def __post_init__(self) -> None:
        if not self.group_id:
            raise ValueError("Engine group identity is required")
        ranks = [rank for replica in self.replicas for rank in replica.node_ranks]
        if len(set(ranks)) != len(ranks):
            raise ValueError("Replicas cannot share node actor ranks within an engine group")


ModelSpec = ModelConfig


@dataclass(frozen=True)
class RoleSpec:
    role: Role
    models: tuple[ModelConfig, ...]
    routing: RoutingSpec = RoutingSpec()

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", Role(self.role))
        names = [model.model_id for model in self.models]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate model identities")
        validate_routes(self.routing, set(names))


def validate_routes(routing: RoutingSpec, model_names: set[str]) -> None:
    targets = {model for _, model in routing.route_key_to_model}
    if routing.default_model is not None:
        targets.add(routing.default_model)
    if not targets <= model_names:
        raise ValueError(f"Routing references unregistered models: {sorted(targets - model_names)}")


def model_spec_from_pool(
    model_id: str,
    model_path: str,
    *,
    weight_source: WeightSource,
    num_slots: int,
    nodes_per_engine: int,
    num_gpus_per_engine: int = 1,
    overrides: tuple[tuple[str, Any], ...] = (),
    allow_defer: bool = False,
    direct_eligible: bool = False,
) -> ModelConfig:
    """Describe a single homogeneous pool owned by a legacy manager."""
    replicas = replicas_from_slots(model_id, num_slots, nodes_per_engine)
    topology = EngineGroupSpec("group-0", replicas)
    group = EngineGroupConfig(
        worker_type="regular",
        num_gpus=max(1, len(replicas)) * num_gpus_per_engine,
        num_gpus_per_engine=num_gpus_per_engine,
        overrides=dict(overrides),
        topology=topology,
    )
    return ModelConfig(
        model_id,
        model_path,
        engine_groups=[group],
        weight_source=weight_source,
        allow_defer=allow_defer,
        direct_eligible=direct_eligible,
    )
