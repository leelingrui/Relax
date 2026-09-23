# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Access to the one placement ledger a control domain owns.

Every role reaches the ledger through these helpers, so no role keeps its own
allocation bookkeeping. ``ledger`` is the task-level owner's ledger: its actor
handle from outside the owner process, or the owner's own
:class:`PlacementPlanner` for the engine pools created inside it. It is never
implicit shared state.
"""

from typing import Any, Sequence

import ray

from relax.engine.inference.placement import (
    PlacementGroupView,
    PlacementPlanner,
    PlacementRelease,
    PlacementRequest,
    PlacementSlice,
)


def plan_placement(
    ledger: Any,
    requests: Sequence[PlacementRequest],
    placement_group: PlacementGroupView,
    *,
    dry_run: bool = False,
) -> tuple[PlacementSlice, ...]:
    """Resolve and record a layout, or only validate it when ``dry_run``."""
    if isinstance(ledger, PlacementPlanner):
        return ledger.plan(requests, placement_group, dry_run=dry_run)
    return ray.get(ledger.plan_placement.remote(requests, placement_group, dry_run=dry_run))


def release_placement(
    ledger: Any, placement: PlacementSlice | PlacementGroupView, group_id: str | None = None
) -> PlacementRelease:
    """Release ledger entries and report whether the group may be removed."""
    if isinstance(ledger, PlacementPlanner):
        return ledger.release(placement, group_id=group_id)
    return ray.get(ledger.release_placement.remote(placement, group_id=group_id))
