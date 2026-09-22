# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Shared inference contracts, independent of Ray and GPU backends."""

from relax.engine.inference.client import InferenceDiscoveryClient
from relax.engine.inference.discovery import (
    DiscoveryState,
    new_manager_epoch,
    role_snapshot_from_dict,
    snapshot_from_engine_urls,
    snapshot_from_legacy_engines,
)
from relax.engine.inference.manager import OperationSnapshot, RequestPermit
from relax.engine.inference.routing import resolve_model, select_target
from relax.engine.inference.types import Role, RoleSnapshot


__all__ = [
    "DiscoveryState",
    "InferenceDiscoveryClient",
    "new_manager_epoch",
    "role_snapshot_from_dict",
    "snapshot_from_engine_urls",
    "snapshot_from_legacy_engines",
    "Role",
    "RoleSnapshot",
    "RequestPermit",
    "OperationSnapshot",
    "resolve_model",
    "select_target",
]
