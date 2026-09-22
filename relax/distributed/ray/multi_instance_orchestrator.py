# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Launch N named managers (GenRM judges, OPD teachers, ...) sharing one
placement group, each carved out at a distinct, non-overlapping GPU offset.

This is the domain-agnostic core of what MOPD's
``_start_managed_multi_teacher`` did for OPD teachers: split a GPU budget
across N keyed instances and launch one manager actor per instance at a
``bundle_offset`` within a shared placement group. Manager constructor
signatures differ (``TeacherManager`` takes ``num_replicas``/
``gpus_per_replica``/``bundle_offset`` directly; ``GenRMManager`` derives them
from its args namespace), so the actual ``.remote(...)`` call is left to the
caller-supplied ``spawn_manager`` hook -- this function only owns the GPU-
budget bookkeeping (equal or unequal per-instance splits, non-overlapping
offsets) and the offload-on-start behavior shared by every manager type.
"""

from __future__ import annotations

from typing import Any, Callable

import ray

from relax.distributed.ray.placement_ledger import plan_placement
from relax.engine.inference.placement import (
    PlacementGroupView,
    PlacementOwner,
    PlacementPlanner,
    PlacementRequest,
)
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def start_multi_instance_managers(
    *,
    args: Any,
    instance_specs: dict[str, dict],
    build_manager_args: Callable[[Any, str, dict], Any],
    spawn_manager: Callable[[str, Any, int, dict], Any],
    region_offset: int = 0,
    placement_group: tuple | None = None,
    placement_owner: PlacementOwner = PlacementOwner.CONTROLLER,
    worker_type: str = "regular",
    num_gpus_per_node: int | None = None,
    phase: str = "inference",
    placement_ledger: Any | None = None,
) -> dict[str, Any]:
    """Launch one manager per entry in ``instance_specs``, each at a non-
    overlapping GPU offset within whatever shared placement group
    ``spawn_manager`` closes over.

    Args:
        args: Base argument namespace.
        instance_specs: ``{key: spec}``. Each ``spec`` must contain
            ``num_gpus`` (int); any other keys are opaque to this function and
            forwarded verbatim to ``build_manager_args``/``spawn_manager``.
            ``num_gpus`` need not be equal across instances -- the offset
            passed to ``spawn_manager`` is the prefix sum of GPUs used by
            earlier instances (in dict order), not ``idx * num_gpus``, so
            heterogeneously-sized instances never overlap.
        build_manager_args: ``(args, key, spec) -> per_instance_args``, used
            to inject instance-specific overrides (e.g. a distinct model path)
            into a copy of ``args`` before it's passed to ``spawn_manager``.
        spawn_manager: ``(key, per_instance_args, bundle_offset, spec) -> manager_handle``.
            Owns the manager class's actual constructor signature and its
            shared placement group -- e.g. for ``TeacherManager`` this calls
            ``TeacherManager.options(...).remote(per_instance_args,
            num_replicas, gpus_per_replica, pg=shared_pg, shared_pg=True,
            bundle_offset=bundle_offset)``; for ``GenRMManager`` it sets
            ``per_instance_args._genrm_bundle_offset = bundle_offset`` and
            calls ``GenRMManager.options(...).remote(per_instance_args, shared_pg)``.
        region_offset: GPU offset within the shared placement group where the
            first instance's region starts (e.g. ``rollout_num_gpus``, to skip
            a rollout region that precedes all instances).
        placement_group: Optional reordered ``(pg, bundle_indices, gpu_ids)``
            view. When supplied, all instance slices are validated before any
            manager is spawned.

    Returns:
        ``{key: manager_handle}`` in ``instance_specs`` iteration order.
    """
    offsets: dict[str, int] = {}
    managers: dict[str, Any] = {}
    bundle_offset = region_offset
    for key, spec in instance_specs.items():
        offsets[key] = bundle_offset
        bundle_offset += spec["num_gpus"]

    if placement_group is not None:
        if num_gpus_per_node is None or num_gpus_per_node < 1:
            raise ValueError("num_gpus_per_node is required when validating shared placement")
        pg_view = PlacementGroupView(
            tuple(placement_group[1]), tuple(placement_group[2]), placement_owner, identity=placement_group[0]
        )
        # Pre-flight only: the role adapters record the authoritative slices.
        plan_placement(
            placement_ledger or PlacementPlanner(),
            tuple(
                PlacementRequest(
                    group_id=f"{worker_type}/{key}",
                    worker_type=worker_type,
                    num_gpus=spec["num_gpus"],
                    num_gpus_per_engine=spec.get("num_gpus_per_engine", spec["num_gpus"]),
                    num_gpus_per_node=num_gpus_per_node,
                    phase=phase,
                    bundle_offset=offset,
                )
                for key, spec in instance_specs.items()
                for offset in (offsets[key],)
            ),
            pg_view,
            dry_run=True,
        )

    for key, spec in instance_specs.items():
        per_instance_args = build_manager_args(args, key, spec)
        manager = spawn_manager(key, per_instance_args, offsets[key], spec)
        managers[key] = manager
        logger.info(f"Launched instance '{key}': bundle_offset={offsets[key]}, num_gpus={spec['num_gpus']}")

    if getattr(args, "offload_rollout", False):
        ray.get([m.offload.remote() for m in managers.values()])

    return managers
