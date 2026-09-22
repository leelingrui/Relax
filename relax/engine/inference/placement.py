# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Compatibility types for legacy placement adapters.

Placement ownership is not global. New control-plane code must carry its
planner instance explicitly; this module only preserves the old adapter API
while the engine backends are migrated.
"""

from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import ClassVar, Sequence


class PlacementOwner(str, Enum):
    MANAGER = "manager"
    CONTROLLER = "controller"
    EXTERNAL = "external"


@dataclass(frozen=True)
class PlacementGroupView:
    bundle_indices: tuple[int, ...]
    gpu_ids: tuple[int, ...]
    owner: PlacementOwner
    identity: object = field(default_factory=object, repr=False, compare=False)


@dataclass(frozen=True)
class PlacementRequest:
    group_id: str
    worker_type: str
    num_gpus: int
    num_gpus_per_engine: int
    num_gpus_per_node: int
    phase: str = "inference"
    bundle_offset: int | None = None
    active: bool = True


@dataclass(frozen=True)
class PlacementSlice:
    group_id: str
    worker_type: str
    phase: str
    owner: PlacementOwner
    reserved_offset: int
    reserved_size: int
    referenced_offsets: tuple[int, ...]
    bundle_indices: tuple[int, ...]
    gpu_ids: tuple[int, ...]
    placement_group_key: int = 0


class PlacementPlanner:
    _lock: ClassVar[Lock] = Lock()
    _allocations: ClassVar[dict[int, dict[str, PlacementSlice]]] = {}

    def plan(
        self, requests: Sequence[PlacementRequest], placement_group: PlacementGroupView
    ) -> tuple[PlacementSlice, ...]:
        key = id(placement_group.identity)
        with self._lock:
            allocations = self._allocations.setdefault(key, {})
            result = []
            for request in requests:
                if request.group_id in allocations:
                    result.append(allocations[request.group_id])
                    continue
                if request.num_gpus <= 0 or request.num_gpus > len(placement_group.bundle_indices):
                    raise ValueError(f"Invalid placement size for {request.group_id}")
                offset = request.bundle_offset
                if offset is None:
                    offset = max(
                        (item.reserved_offset + item.reserved_size for item in allocations.values()), default=0
                    )
                end = offset + request.num_gpus
                if end > len(placement_group.bundle_indices):
                    raise ValueError(f"Placement group is too small for {request.group_id}")
                for item in allocations.values():
                    if (
                        item.phase == request.phase
                        and offset < item.reserved_offset + item.reserved_size
                        and item.reserved_offset < end
                    ):
                        raise ValueError(
                            f"Placement overlap in phase {request.phase!r}: {request.group_id} overlaps {item.group_id}"
                        )
                local = max(1, min(request.num_gpus_per_engine, request.num_gpus_per_node))
                referenced = tuple(offset + index * local for index in range(request.num_gpus // local))
                item = PlacementSlice(
                    request.group_id,
                    request.worker_type,
                    request.phase,
                    placement_group.owner,
                    offset,
                    request.num_gpus,
                    referenced,
                    tuple(placement_group.bundle_indices[index] for index in referenced),
                    tuple(placement_group.gpu_ids[index] for index in referenced),
                    key,
                )
                allocations[request.group_id] = item
                result.append(item)
            return tuple(result)

    @classmethod
    def cancel(cls, placement: PlacementSlice | PlacementGroupView, group_id: str | None = None) -> None:
        key = placement.placement_group_key if isinstance(placement, PlacementSlice) else id(placement.identity)
        with cls._lock:
            allocations = cls._allocations.get(key)
            if allocations is None:
                return
            if isinstance(placement, PlacementSlice):
                allocations.pop(placement.group_id, None)
            elif group_id is None:
                allocations.clear()
            else:
                allocations.pop(group_id, None)
            if not allocations:
                cls._allocations.pop(key, None)

    @classmethod
    def allocations(cls, placement_group: PlacementGroupView) -> tuple[PlacementSlice, ...]:
        with cls._lock:
            return tuple(cls._allocations.get(id(placement_group.identity), {}).values())
