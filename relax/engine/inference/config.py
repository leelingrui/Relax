# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Shared deployment configuration, independent of Ray and process creation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from relax.engine.inference.capabilities import WeightSource


if TYPE_CHECKING:
    from relax.engine.inference.specs import EngineGroupSpec


@dataclass
class EngineGroupConfig:
    worker_type: str
    num_gpus: int
    num_gpus_per_engine: int | None = None
    overrides: dict[str, Any] = field(default_factory=dict)
    topology: EngineGroupSpec | None = None

    def __post_init__(self) -> None:
        valid_types = {"regular", "prefill", "decode", "placeholder"}
        assert self.worker_type in valid_types, (
            f"Invalid worker_type '{self.worker_type}', must be one of {valid_types}"
        )
        assert self.num_gpus > 0, f"num_gpus must be > 0, got {self.num_gpus}"

    @property
    def replicas(self) -> tuple:
        """Compatibility view for consumers of the former ModelSpec."""
        return self.topology.replicas if self.topology is not None else ()


@dataclass
class ModelConfig:
    name: str
    model_path: str | None = None
    num_gpus_per_engine: int | None = None
    engine_groups: list[EngineGroupConfig] = field(default_factory=list)
    weight_source: WeightSource = WeightSource.POLICY
    allow_defer: bool = False
    direct_eligible: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Model identity is required")
        self.weight_source = WeightSource(self.weight_source)
        topologies = [group.topology for group in self.engine_groups if group.topology is not None]
        groups = [group.group_id for group in topologies]
        replicas = [replica.replica_id for group in topologies for replica in group.replicas]
        if len(set(groups)) != len(groups) or len(set(replicas)) != len(replicas):
            raise ValueError("Group and replica identities must be unique within a model")

    @property
    def model_id(self) -> str:
        return self.name

    def resolve(self, args: Any) -> None:
        """Resolve launch defaults in place (legacy API)."""
        default_gpus = self.num_gpus_per_engine or args.rollout_num_gpus_per_engine
        self.model_path = self.model_path or args.sglang_hf_checkpoint or args.hf_checkpoint
        if not self.model_path:
            raise ValueError("Model checkpoint path is required")
        for group in self.engine_groups:
            if group.num_gpus_per_engine is None:
                group.num_gpus_per_engine = default_gpus
            group.overrides.setdefault("model_path", self.model_path)

    def resolved(self, args: Any) -> ModelConfig:
        """Resolve once, then attach topology without a second configuration
        representation."""
        from relax.engine.inference.specs import EngineGroupSpec, replicas_from_slots

        model = deepcopy(self)
        model.resolve(args)
        if args.num_gpus_per_node < 1:
            raise ValueError("GPUs per node must be positive")
        for index, group in enumerate(model.engine_groups):
            group_id = f"{model.name}/group-{index}"
            gpus = group.num_gpus_per_engine
            placeholder = group.worker_type == "placeholder"
            if gpus < 1 or (not placeholder and group.num_gpus % gpus):
                raise ValueError("Engine group GPUs must contain complete replicas")
            if not placeholder and gpus > args.num_gpus_per_node and gpus % args.num_gpus_per_node:
                raise ValueError("Multi-node engines must occupy complete nodes")
            nodes = max(1, gpus // args.num_gpus_per_node)
            replicas = () if placeholder else replicas_from_slots(group_id, group.num_gpus // gpus * nodes, nodes)
            group.topology = EngineGroupSpec(group_id, replicas)
        return model

    @property
    def has_pd_disaggregation(self) -> bool:
        return any(group.worker_type in ("prefill", "decode") for group in self.engine_groups)

    @property
    def total_num_gpus(self) -> int:
        return sum(group.num_gpus for group in self.engine_groups)


@dataclass
class SglangConfig:
    models: list[ModelConfig]

    @staticmethod
    def from_yaml(path: str) -> SglangConfig:
        import yaml

        with open(path) as stream:
            data = yaml.safe_load(stream)
        assert "sglang" in data, "sglang config must have a 'sglang' key"
        return SglangConfig(
            models=[
                ModelConfig(
                    name=model["name"],
                    model_path=model.get("model_path"),
                    num_gpus_per_engine=model.get("num_gpus_per_engine"),
                    engine_groups=[EngineGroupConfig(**g) for g in model.get("engine_groups", [])],
                )
                for model in data["sglang"]
            ]
        )

    @staticmethod
    def from_prefill_num_servers(args: Any) -> SglangConfig:
        prefill_gpus = args.prefill_num_servers * args.rollout_num_gpus_per_engine
        decode_gpus = args.rollout_num_gpus - prefill_gpus
        assert decode_gpus > 0, f"No decode GPUs: total {args.rollout_num_gpus}, prefill {prefill_gpus}"
        return SglangConfig(
            [
                ModelConfig(
                    "default",
                    engine_groups=[
                        EngineGroupConfig("prefill", prefill_gpus),
                        EngineGroupConfig("decode", decode_gpus),
                    ],
                )
            ]
        )

    @property
    def has_pd_disaggregation(self) -> bool:
        return any(model.has_pd_disaggregation for model in self.models)

    @property
    def total_num_gpus(self) -> int:
        return sum(model.total_num_gpus for model in self.models)
