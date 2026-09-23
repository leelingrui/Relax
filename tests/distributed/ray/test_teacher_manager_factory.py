# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

from conftest import FakeOwnerHandle


def test_create_managed_opd_teacher_offloads_shared_pg_teacher(monkeypatch):
    import ray

    from relax.engine.inference.types import ModelRef, Role
    from relax.utils.opd.opd_utils import create_managed_opd_teacher

    owner = FakeOwnerHandle(urls={"default": ["http://teacher/generate"]})
    monkeypatch.setattr(ray, "get", lambda ref, **kwargs: ref)
    args = SimpleNamespace(offload_rollout=True, enable_affinity=True)
    pg = ("pg", list(range(8)), list(range(8)))

    models, urls = create_managed_opd_teacher(
        args, num_replicas=1, gpus_per_replica=4, inference_manager_handle=owner, pg=pg, shared_pg=True
    )

    assert models == (ModelRef(Role.TEACHER, "default"),)
    assert urls == ["http://teacher/generate"]
    assert [name for name, _, _ in owner.calls] == ["create_role", "call", "lifecycle"]
    ((create_args, _),) = owner.named("create_role")
    assert create_args[1] == "teacher"
    assert create_args[2]["default"]["args"] == (args,)
    assert create_args[2]["default"]["kwargs"] == {
        "num_replicas": 1,
        "gpus_per_replica": 4,
        "pg": pg,
        "shared_pg": True,
    }
    assert owner.named("lifecycle") == [((Role.TEACHER, "default", "offload"), {})]


def test_create_managed_opd_teacher_keeps_a_dedicated_teacher_loaded(monkeypatch):
    import ray

    from relax.utils.opd.opd_utils import create_managed_opd_teacher

    owner = FakeOwnerHandle(urls={"default": ["http://teacher/generate"]})
    monkeypatch.setattr(ray, "get", lambda ref, **kwargs: ref)
    args = SimpleNamespace(offload_rollout=True, enable_affinity=True)

    create_managed_opd_teacher(args, num_replicas=2, gpus_per_replica=4, inference_manager_handle=owner)

    assert owner.named("lifecycle") == []
