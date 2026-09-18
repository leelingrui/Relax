# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest

from relax.utils.http_utils import router_worker_base_url


@pytest.fixture()
def sglang_engine_module(monkeypatch):
    ray = ModuleType("ray")
    ray.get_runtime_context = lambda: SimpleNamespace()
    monkeypatch.setitem(sys.modules, "ray", ray)

    sglang_router = ModuleType("sglang_router")
    sglang_router.__version__ = "0.3.2"
    monkeypatch.setitem(sys.modules, "sglang_router", sglang_router)

    sglang = ModuleType("sglang")
    sglang_srt = ModuleType("sglang.srt")
    server_args = ModuleType("sglang.srt.server_args")
    server_args.ServerArgs = object
    sglang_utils = ModuleType("sglang.srt.utils")
    sglang_utils.kill_process_tree = lambda _pid: None
    monkeypatch.setitem(sys.modules, "sglang", sglang)
    monkeypatch.setitem(sys.modules, "sglang.srt", sglang_srt)
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", server_args)
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", sglang_utils)

    checkpoint_client = ModuleType("relax.distributed.checkpoint_service.client.engine")
    checkpoint_client.create_client = lambda **_kwargs: None
    monkeypatch.setitem(sys.modules, "relax.distributed.checkpoint_service.client.engine", checkpoint_client)

    ray_actor = ModuleType("relax.distributed.ray.ray_actor")
    ray_actor.RayActor = object
    monkeypatch.setitem(sys.modules, "relax.distributed.ray.ray_actor", ray_actor)

    device = ModuleType("relax.utils.device")
    device.get_visible_devices_env_var = lambda: "CUDA_VISIBLE_DEVICES"
    monkeypatch.setitem(sys.modules, "relax.utils.device", device)

    async_utils = ModuleType("relax.utils.async_utils")
    async_utils.run = lambda value: value
    monkeypatch.setitem(sys.modules, "relax.utils.async_utils", async_utils)

    env = ModuleType("relax.utils.env")
    env.Envs = SimpleNamespace(
        RELAX_SCALE_OUT_MAX_REASON_ITEMS=3,
        RELAX_SCALE_OUT_MAX_REASON_ITEM_LEN=120,
        RELAX_SCALE_OUT_MAX_REASON_TOTAL_LEN=512,
    )
    monkeypatch.setitem(sys.modules, "relax.utils.env", env)

    http_utils = ModuleType("relax.utils.http_utils")
    http_utils.get_host_info = lambda: ("worker", "127.0.0.1")
    http_utils.router_worker_base_url = router_worker_base_url
    monkeypatch.setitem(sys.modules, "relax.utils.http_utils", http_utils)

    logging_utils = ModuleType("relax.utils.logging_utils")
    logging_utils.get_logger = logging.getLogger
    monkeypatch.setitem(sys.modules, "relax.utils.logging_utils", logging_utils)

    megatron_peft_utils = ModuleType("relax.utils.megatron_peft_utils")
    megatron_peft_utils.convert_megatron_to_sglang_target_modules = lambda value: value
    megatron_peft_utils.is_lora_enabled = lambda _args: False
    monkeypatch.setitem(sys.modules, "relax.utils.megatron_peft_utils", megatron_peft_utils)

    # Force a fresh import so the module binds to the stubbed dependencies above,
    # but restore the original module object on teardown. Leaving the key popped
    # corrupts sys.modules for any later test that patches this module: their
    # patch() re-imports a *new* module object distinct from the one already
    # bound in other test files' top-level imports, so the patch silently misses.
    original_module = sys.modules.pop("relax.backends.sglang.sglang_engine", None)
    module = importlib.import_module("relax.backends.sglang.sglang_engine")
    yield module
    if original_module is not None:
        sys.modules["relax.backends.sglang.sglang_engine"] = original_module
    else:
        sys.modules.pop("relax.backends.sglang.sglang_engine", None)


class _Response:
    def __init__(self, status_code: int, payload: dict | None = None, headers: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"status {self.status_code}")

    def json(self):
        return self._payload


class _RouterRequests:
    def __init__(
        self,
        delete_status: int = 202,
        worker_lists: list[list[dict]] | None = None,
        worker_id: str | int | None = "physical-worker-id",
        location: str | None = None,
        body_location: str | None = None,
    ):
        self.delete_status = delete_status
        self.worker_lists = worker_lists
        self.worker_id = worker_id
        self.location = location
        self.body_location = body_location
        self.posts: list[str] = []
        self.deletes: list[str] = []
        self.gets: list[str] = []

    def post(self, url, json=None, timeout=None):
        self.posts.append(url)
        headers = {"Location": self.location} if self.location is not None else {}
        payload = {"worker_id": self.worker_id}
        if self.body_location is not None:
            payload["location"] = self.body_location
        return _Response(202, payload, headers)

    def delete(self, url, timeout=None):
        self.deletes.append(url)
        return _Response(self.delete_status)

    def get(self, url, timeout=None):
        if self.worker_lists is None:
            raise AssertionError("registration-level worker_id should avoid listing DP rank workers")
        self.gets.append(url)
        workers = self.worker_lists[0] if len(self.worker_lists) == 1 else self.worker_lists.pop(0)
        return _Response(200, {"workers": workers})


class _Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def _make_engine(sglang_engine_module):
    engine = sglang_engine_module.SGLangEngine.__new__(sglang_engine_module.SGLangEngine)
    engine.args = SimpleNamespace(use_slime_router=False)
    engine.node_rank = 0
    engine.worker_type = "regular"
    engine.router_ip = "router"
    engine.router_port = 30000
    engine.server_host = "worker"
    engine.server_port = 8000
    engine._router_worker_id = None
    engine._router_unregister_submitted = False
    return engine


def test_missing_load_format_choices_fails_closed(sglang_engine_module):
    with pytest.raises(RuntimeError, match="cannot report whether runai_streamer is supported"):
        sglang_engine_module._preferred_s3_stream_load_format()


def test_unregister_uses_registration_worker_id_once(monkeypatch, sglang_engine_module):
    requests = _RouterRequests()
    monkeypatch.setattr(sglang_engine_module, "requests", requests)
    engine = _make_engine(sglang_engine_module)

    assert engine.register_to_router()
    assert engine._router_worker_id == "physical-worker-id"
    assert engine.unregister_from_router()
    assert engine.unregister_from_router()

    assert requests.posts == ["http://router:30000/workers"]
    assert requests.deletes == ["http://router:30000/workers/physical-worker-id"]


def test_registration_normalizes_numeric_worker_id(monkeypatch, sglang_engine_module):
    requests = _RouterRequests(worker_id=42)
    monkeypatch.setattr(sglang_engine_module, "requests", requests)
    engine = _make_engine(sglang_engine_module)

    assert engine.register_to_router()
    assert engine._router_worker_id == "42"
    assert engine.unregister_from_router()

    assert requests.deletes == ["http://router:30000/workers/42"]


@pytest.mark.parametrize(
    ("worker_id", "location", "body_location", "expected_worker_id"),
    [
        (None, "http://router:30000/workers/header-worker-id", None, "header-worker-id"),
        ("", None, "/workers/body-worker-id", "body-worker-id"),
    ],
)
def test_registration_falls_back_to_location_worker_id(
    monkeypatch,
    sglang_engine_module,
    worker_id,
    location,
    body_location,
    expected_worker_id,
):
    requests = _RouterRequests(worker_id=worker_id, location=location, body_location=body_location)
    monkeypatch.setattr(sglang_engine_module, "requests", requests)
    engine = _make_engine(sglang_engine_module)

    assert engine.register_to_router()
    assert engine._router_worker_id == expected_worker_id
    assert engine.unregister_from_router()

    assert requests.deletes == [f"http://router:30000/workers/{expected_worker_id}"]
    assert requests.gets == []


def test_unregister_falls_back_to_exact_non_dp_worker(monkeypatch, sglang_engine_module):
    requests = _RouterRequests(worker_lists=[[{"id": "listed-worker-id", "url": "http://worker:8000"}]])
    monkeypatch.setattr(sglang_engine_module, "requests", requests)
    engine = _make_engine(sglang_engine_module)

    assert engine.unregister_from_router()

    assert requests.deletes == ["http://router:30000/workers/listed-worker-id"]


def test_unregister_does_not_delete_dp_rank_worker_as_fallback(monkeypatch, sglang_engine_module):
    requests = _RouterRequests(worker_lists=[[{"id": "rank-worker-id", "url": "http://worker:8000@0"}]])
    monkeypatch.setattr(sglang_engine_module, "requests", requests)
    engine = _make_engine(sglang_engine_module)

    assert not engine.unregister_from_router()

    assert requests.deletes == []


def test_unregister_rejects_ambiguous_exact_and_dp_rank_workers(monkeypatch, sglang_engine_module):
    requests = _RouterRequests(
        worker_lists=[
            [
                {"id": "exact-worker-id", "url": "http://worker:8000"},
                {"id": "rank-worker-id", "url": "http://worker:8000@0"},
            ]
        ]
    )
    monkeypatch.setattr(sglang_engine_module, "requests", requests)
    engine = _make_engine(sglang_engine_module)

    assert not engine.unregister_from_router()

    assert requests.deletes == []


def test_unregister_can_retry_after_worker_is_not_listed_yet(monkeypatch, sglang_engine_module):
    requests = _RouterRequests(
        worker_lists=[
            [],
            [{"id": "listed-worker-id", "url": "http://worker:8000"}],
        ]
    )
    monkeypatch.setattr(sglang_engine_module, "requests", requests)
    engine = _make_engine(sglang_engine_module)

    assert not engine.unregister_from_router()
    assert engine._router_unregister_submitted is False
    assert engine.unregister_from_router()

    assert requests.deletes == ["http://router:30000/workers/listed-worker-id"]


def test_unregister_treats_missing_registration_as_complete(monkeypatch, sglang_engine_module):
    requests = _RouterRequests(delete_status=404)
    monkeypatch.setattr(sglang_engine_module, "requests", requests)
    engine = _make_engine(sglang_engine_module)
    engine._router_worker_id = "physical-worker-id"

    assert engine.unregister_from_router()
    assert requests.deletes == ["http://router:30000/workers/physical-worker-id"]


def test_unregister_waits_for_all_dp_rank_workers_to_leave(monkeypatch, sglang_engine_module):
    requests = _RouterRequests(
        worker_lists=[
            [{"url": "http://worker:8000@0"}, {"url": "http://worker:8000@1"}],
            [],
        ]
    )
    monkeypatch.setattr(sglang_engine_module, "requests", requests)
    monkeypatch.setattr(sglang_engine_module, "time", _Clock())
    engine = _make_engine(sglang_engine_module)
    engine._router_worker_id = "physical-worker-id"

    assert engine.unregister_from_router(wait_for_removal=True, timeout=5.0)
    assert requests.deletes == ["http://router:30000/workers/physical-worker-id"]
    assert requests.gets == [
        "http://router:30000/workers",
        "http://router:30000/workers",
    ]


def test_unregister_timeout_allows_a_later_retry(monkeypatch, sglang_engine_module):
    requests = _RouterRequests(worker_lists=[[{"url": "http://worker:8000@0"}]])
    monkeypatch.setattr(sglang_engine_module, "requests", requests)
    monkeypatch.setattr(sglang_engine_module, "time", _Clock())
    engine = _make_engine(sglang_engine_module)
    engine._router_worker_id = "physical-worker-id"

    assert not engine.unregister_from_router(wait_for_removal=True, timeout=1.0)
    assert engine._router_unregister_submitted is False


@pytest.mark.parametrize(
    "node_rank,fully_async,override", [(0, True, 4), (0, True, None), (1, True, 4), (0, False, 4)]
)
def test_engine_dcs_registration_eligibility_and_gpu_metadata(
    monkeypatch, sglang_engine_module, node_rank, fully_async, override
):
    from unittest.mock import MagicMock

    engine = _make_engine(sglang_engine_module)
    engine.node_rank = node_rank
    engine.rank = 2
    engine.num_gpus_per_engine = override
    engine.args = SimpleNamespace(
        fully_async=fully_async, rollout_num_gpus_per_engine=2, coordinator_url="http://dcs.test"
    )
    engine.checkpoint_engine_client = None
    create_client = MagicMock(return_value=object())
    monkeypatch.setattr(sglang_engine_module, "create_client", create_client)

    engine.register_dcs()

    if node_rank == 0 and fully_async:
        create_client.assert_called_once_with(
            args=engine.args,
            coordinator_url="http://dcs.test",
            role="rollout",
            ip="worker",
            port=8000,
            rank=2,
            metadata={"num_gpus_per_engine": override or 2},
        )
        assert engine.checkpoint_engine_client is create_client.return_value
    else:
        create_client.assert_not_called()
        assert engine.checkpoint_engine_client is None


@pytest.mark.parametrize("skip_dcs", [False, True])
def test_engine_startup_precedes_dcs_registration(monkeypatch, sglang_engine_module, skip_dcs):
    engine = _make_engine(sglang_engine_module)
    engine.args = SimpleNamespace(sglang_router_ip="router", sglang_router_port=30000, rollout_external=False)
    engine.rank = 0
    engine.base_gpu_id = 0
    engine.sglang_overrides = {}
    engine.num_gpus_per_engine = 2
    events = []
    monkeypatch.setattr(
        sglang_engine_module,
        "_compute_server_args",
        lambda *a, **kw: ({"node_rank": 0, "host": "worker", "port": 8000}, []),
    )
    engine._init_normal = lambda args: events.append("started")
    engine.register_dcs = lambda: events.append("dcs")

    engine.init("worker:8001", 8000, 8002, skip_router_registration=True, skip_dcs_registration=skip_dcs)

    assert events == (["started"] if skip_dcs else ["started", "dcs"])
    assert engine._skip_router_registration is True


@pytest.mark.parametrize("worker_type", ["regular", "prefill", "decode"])
def test_router_registration_preserves_pd_bootstrap_payload(monkeypatch, sglang_engine_module, worker_type):
    from unittest.mock import MagicMock

    engine = _make_engine(sglang_engine_module)
    engine.worker_type = worker_type
    post = MagicMock(return_value=_Response(200, {"worker_id": "worker-id"}))
    monkeypatch.setattr(sglang_engine_module.requests, "post", post)

    assert engine.register_to_router(bootstrap_port=9000)

    payload = {"url": "http://worker:8000", "worker_type": worker_type}
    if worker_type == "prefill":
        payload["bootstrap_port"] = 9000
    post.assert_called_once_with("http://router:30000/workers", json=payload, timeout=30)


@pytest.mark.parametrize("external", [False, True])
def test_engine_shutdown_respects_external_ownership(monkeypatch, sglang_engine_module, external):
    engine = _make_engine(sglang_engine_module)
    engine.args.rollout_external = external
    events = []
    engine.unregister_from_router = lambda: events.append("router")
    engine.unregister_dcs = lambda: events.append("dcs")
    engine.process = SimpleNamespace(pid=42)
    monkeypatch.setattr(sglang_engine_module, "kill_process_tree", lambda pid: events.append(("kill", pid)))
    try:
        engine.shutdown()
        assert events == ([] if external else ["router", ("kill", 42)])
    finally:
        del engine.process  # Do not invoke the destructor's process cleanup in this unit test.
