# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import sys
from types import ModuleType, SimpleNamespace


def _install_fake_teacher_manager(monkeypatch, captured):
    teacher_manager_module = ModuleType("relax.distributed.ray.inference_role")

    class _RemoteMethod:
        def __init__(self, name):
            self.name = name

        def remote(self):
            captured["calls"].append(self.name)
            return f"{self.name}-ref"

    class _TeacherManagerHandle:
        get_urls = _RemoteMethod("get_urls")
        offload = _RemoteMethod("offload")

    def create_role_managers(args, role, pool_configs, runtime_env=None):
        assert role == "teacher"
        captured["runtime_env"] = runtime_env
        captured["pool_configs"] = pool_configs
        captured["handle"] = _TeacherManagerHandle()
        return {"default": captured["handle"]}

    teacher_manager_module.create_role_managers = create_role_managers
    monkeypatch.setitem(sys.modules, "relax.distributed.ray.inference_role", teacher_manager_module)


def test_create_managed_opd_teacher_manager_offloads_shared_pg_teacher(monkeypatch, tmp_path):
    import ray

    from relax.utils.opd.opd_utils import create_managed_opd_teacher_manager

    captured = {"calls": []}
    _install_fake_teacher_manager(monkeypatch, captured)
    monkeypatch.setattr(
        ray,
        "get",
        lambda ref: ["http://teacher/generate"] if ref == "get_urls-ref" else None,
    )

    autoscaler_config = tmp_path / "autoscaler.yaml"
    autoscaler_config.write_text("enabled: true\n")
    args = SimpleNamespace(
        offload_rollout=True,
        autoscaler_config=str(autoscaler_config),
        enable_affinity=True,
    )
    manager, urls = create_managed_opd_teacher_manager(
        args,
        num_replicas=1,
        gpus_per_replica=4,
        pg=("pg", list(range(8)), list(range(8))),
        shared_pg=True,
        runtime_env={"env_vars": {"A": "B"}},
    )

    assert manager is captured["handle"]
    assert urls == ["http://teacher/generate"]
    assert captured["calls"] == ["get_urls", "offload"]
    assert captured["runtime_env"] == {"env_vars": {"A": "B"}}
    assert captured["pool_configs"]["default"]["args"] == (args,)
    assert captured["pool_configs"]["default"]["kwargs"] == {
        "num_replicas": 1,
        "gpus_per_replica": 4,
        "pg": ("pg", list(range(8)), list(range(8))),
        "shared_pg": True,
    }
