# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""MOPD teachers are created on the task owner with prefix-sum offsets."""

import json
from argparse import Namespace

from conftest import FakeOwnerHandle


def _install_fake_gateway(monkeypatch):
    from relax.utils.opd import opd_utils

    monkeypatch.setattr(opd_utils, "_deploy_teacher_gateway", lambda owner: "http://gateway/teacher")


def _base_args(**overrides):
    defaults = dict(
        use_opd=True,
        opd_type="sglang",
        colocate=True,
        hybrid=False,
        debug_train_only=False,
        offload_rollout=False,
        enable_affinity=True,
        rollout_num_gpus=8,
        teacher_num_gpus_per_engine=None,
        opd_teacher_key=None,
        resource={"actor": [1, 16], "rollout": [1, 8], "teacher": [1, 8]},
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def test_multi_teacher_bundle_offsets_are_prefix_sums_not_index_times_size(monkeypatch):
    """Two teachers with equal GPU shares (the only shape MOPD currently
    supports, since it enforces an even split) must land at non-overlapping,
    monotonically increasing bundle offsets starting after the rollout
    region."""
    import ray

    from relax.core.service import create_placement_group as _real_create_pg  # noqa: F401
    from relax.engine.inference.types import ModelRef, Role
    from relax.utils.opd import opd_utils

    _install_fake_gateway(monkeypatch)
    monkeypatch.setattr(opd_utils, "is_managed_opd_teacher_colocate", lambda args: True)
    full_pg = ("pg", list(range(16)), list(range(16)))
    monkeypatch.setattr(
        "relax.core.service.create_placement_group",
        lambda **kwargs: full_pg,
    )
    owner = FakeOwnerHandle(urls={"math": ["http://math"], "code": ["http://code/generate"]})
    monkeypatch.setattr(ray, "get", lambda ref, **kwargs: ref)

    args = _base_args()
    routes_json = json.dumps({"math": "/ckpt/math", "code": "/ckpt/code"})

    shared_pg, models = opd_utils._start_managed_multi_teacher(args, routes_json, inference_manager_handle=owner)

    assert shared_pg == full_pg
    assert models == (ModelRef(Role.TEACHER, "math"), ModelRef(Role.TEACHER, "code"))

    # The whole layout is validated on the owner's ledger before any teacher
    # starts, and reserves nothing.
    ((plan_args, plan_kwargs),) = owner.named("plan_placement")
    assert plan_kwargs == {"dry_run": True}
    assert [request.bundle_offset for request in plan_args[0]] == [8, 12]

    ((create_args, _),) = owner.named("create_role")
    assert create_args[1] == "teacher"
    ctor = {model_id: config["kwargs"] for model_id, config in create_args[2].items()}
    assert create_args[2]["math"]["args"][0].teacher_hf_checkpoint == "/ckpt/math"
    # The adapter adds rollout_num_gpus itself, so these offsets are relative
    # to the teacher region: math at 0, code at 0+4=4.
    assert ctor["math"]["bundle_offset"] == 0
    assert ctor["code"]["bundle_offset"] == 4
    assert ctor["math"]["num_replicas"] == 1
    assert ctor["math"]["gpus_per_replica"] == 4
    assert ctor["math"]["shared_pg"] is True
    assert ctor["math"]["pg"] == full_pg

    assert args.opd_teacher_routes_map == {
        "math": ["http://math/generate"],
        "code": ["http://code/generate"],
    }


def test_multi_teacher_requires_colocate(monkeypatch):
    import pytest

    from relax.utils.opd import opd_utils

    monkeypatch.setattr(opd_utils, "is_managed_opd_teacher_colocate", lambda args: False)
    args = _base_args()
    routes_json = json.dumps({"math": "/ckpt/math"})

    with pytest.raises(ValueError, match="requires colocate mode"):
        opd_utils._start_managed_multi_teacher(args, routes_json, inference_manager_handle=FakeOwnerHandle())


def test_multi_teacher_rejects_uneven_gpu_split(monkeypatch):
    import pytest

    from relax.utils.opd import opd_utils

    monkeypatch.setattr(opd_utils, "is_managed_opd_teacher_colocate", lambda args: True)
    args = _base_args(resource={"actor": [1, 16], "rollout": [1, 8], "teacher": [1, 7]})
    routes_json = json.dumps({"math": "/ckpt/math", "code": "/ckpt/code"})

    with pytest.raises(ValueError, match="evenly divisible"):
        opd_utils._start_managed_multi_teacher(args, routes_json, inference_manager_handle=FakeOwnerHandle())
