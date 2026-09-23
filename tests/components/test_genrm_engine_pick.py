# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Engine selection in the GenRM Serve replica.

The replica caches a model's engine list, read from the task inference owner,
and round-robins over it. Single-engine recovery makes that list *mutable at
runtime*: the pool compacts it
when an engine is retired and hands back a rebuilt engine on a fresh port, so
every assumption the cache makes has to survive the list changing underneath it.

Three ways this has gone wrong, one test class each:
  - the ``itertools.cycle`` outliving the list it was built for (IndexError one
    request after a recovery),
  - the replica addressing an engine's pre-rebuild port forever,
  - an empty list being cached as if it were valid, which strands the replica
    permanently even after recovery succeeds.

These drive the real ``GenRM`` methods -- the class is unwrapped from its Serve
deployment and instantiated without ``__init__`` so no cluster is needed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


try:
    import relax.components.genrm as genrm_module

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="requires ray[serve] + relax deps")

COOLDOWN_S = 5.0

# RFC 5737 TEST-NET-1, reserved for documentation and examples. Deliberately not
# 10.x/192.168.x: the gitleaks pre-commit hook rejects private-range literals so
# real cluster addresses cannot be committed by accident.
LIVE_SIX = [(f"192.0.2.{i}", 16001) for i in range(6)]
LIVE_FIVE = LIVE_SIX[:5]  # one engine retired; the manager compacts the list
REBUILT = ("192.0.2.5", 16007)  # rank 5 back after recovery, on a fresh port
REBUILT_ON_NEW_PORT = LIVE_FIVE + [REBUILT]


class _Replica:
    """A bare ``GenRM`` driving its real engine-picking methods.

    ``ray.get`` and ``time`` are stubbed in the namespace the methods actually
    resolve names from, so the clock is deterministic and no Ray runtime is
    touched. Two distinct namespaces are involved, and both must be patched:

    - ``GenRM._pick_engine``/``_resolve_instance_key`` resolve names from a
      globals dict FastAPI's class-based-view rewriting builds for route
      handlers -- a *copy* of ``relax.components.genrm.__dict__``, not the
      module dict itself (``@serve.ingress`` rebuilds the class against this
      copy, so patching only the module would silently miss and let the real
      ``ray.get`` auto-init a cluster).
    - ``_EngineCacheState.needs_refresh``/``refresh``/``invalidate`` are a
      plain module-level class, never touched by that rewriting, so their
      ``time`` still resolves from the *original* module dict. Patching only
      the FastAPI copy leaves this class reading the real wall clock, and
      every cooldown/refresh assertion silently uses live time instead of the
      test's fake clock.
    """

    def __init__(self, monkeypatch, responses, start_time=100.0):
        cls = genrm_module.GenRM.func_or_class
        method_globals = cls._pick_engine.__globals__
        assert method_globals.get("ray") is not None, "engine picking no longer resolves 'ray' from its globals"
        module_globals = genrm_module.__dict__
        assert method_globals is not module_globals, (
            "FastAPI route rewriting no longer copies GenRM's globals -- "
            "the module_globals patch below may now be redundant, not wrong"
        )

        self._replica = object.__new__(cls)
        self._replica._engine_caches = {"__default__": genrm_module._EngineCacheState()}
        # Only the owner's ``call.remote(GENRM, key, "get_engine_hosts_ports")``
        # is ever reached; the token it returns is resolved by the patched
        # ``ray.get`` below, mirroring how a real ObjectRef is consumed.
        self._replica.model_ids = ("__default__",)
        self._replica._owner = SimpleNamespace(call=SimpleNamespace(remote=self._engine_list_call))

        self.now = start_time
        self.fetches = 0
        # Successive manager replies; the last one repeats once exhausted.
        self._responses = list(responses)

        def fake_ray_get(token):
            assert token is _TOKEN, "only the engine-list call should reach ray.get"
            reply = self._responses[min(self.fetches, len(self._responses) - 1)]
            self.fetches += 1
            return reply

        fake_time = SimpleNamespace(monotonic=lambda: self.now)
        monkeypatch.setitem(method_globals, "ray", SimpleNamespace(get=fake_ray_get))
        monkeypatch.setitem(method_globals, "time", fake_time)
        monkeypatch.setitem(module_globals, "time", fake_time)
        monkeypatch.setenv("GENRM_ENGINE_CACHE_REFRESH_COOLDOWN_S", str(COOLDOWN_S))

    @staticmethod
    def _engine_list_call(role, model_id, method):
        assert (role, model_id, method) == (genrm_module.Role.GENRM, "__default__", "get_engine_hosts_ports")
        return _TOKEN

    def pick(self):
        _key, idx, host, port = self._replica._pick_engine(None)
        return idx, host, port

    def invalidate(self):
        self._replica._engine_caches["__default__"].invalidate()

    def advance(self, seconds):
        self.now += seconds


_TOKEN = object()


@pytest.fixture
def replica(monkeypatch):
    def _make(responses, start_time=100.0):
        return _Replica(monkeypatch, responses, start_time)

    return _make


class TestRoundRobin:
    def test_covers_every_engine_without_refetching(self, replica):
        r = replica([LIVE_SIX])
        assert {r.pick()[0] for _ in range(12)} == set(range(6))
        assert r.fetches == 1, "a valid cache must not be re-read"

    def test_shrunk_list_does_not_go_out_of_range(self, replica):
        """The manager compacts over the dead engine, so a cycle built for the
        6-element list would hand back index 5 into a 5-element one."""
        r = replica([LIVE_SIX, LIVE_FIVE])
        for _ in range(6):
            r.pick()
        r.advance(COOLDOWN_S + 1)
        r.invalidate()
        for _ in range(20):
            idx, host, port = r.pick()  # must not IndexError
            assert 0 <= idx < len(LIVE_FIVE)
            assert (host, port) in LIVE_FIVE

    def test_refresh_picks_up_the_rebuilt_port(self, replica):
        """A rebuilt engine comes back on a different port; the replica must
        stop addressing the old one."""
        r = replica([LIVE_FIVE, REBUILT_ON_NEW_PORT])
        assert REBUILT not in {r.pick()[1:] for _ in range(10)}

        r.advance(COOLDOWN_S + 1)
        r.invalidate()
        assert REBUILT in {r.pick()[1:] for _ in range(12)}


class TestInvalidationThrottle:
    def test_failure_burst_collapses_into_one_refetch(self, replica):
        """One dead engine fails many in-flight requests at once.

        The refresh is a blocking ray.get on the replica's event loop, so a
        burst must not cost one round-trip per failure.
        """
        r = replica([LIVE_SIX, LIVE_FIVE])
        r.pick()
        assert r.fetches == 1

        for _ in range(50):  # burst, no time passing
            r.invalidate()
            r.pick()
        assert r.fetches == 1, "cooldown must suppress repeat refreshes"

        r.advance(COOLDOWN_S + 1)
        r.invalidate()
        r.pick()
        assert r.fetches == 2, "a later failure must still be able to refresh"


class TestEmptyEngineList:
    """The manager reports ``[]`` for the whole window between offload()
    retiring every dead engine and the next onload() rebuilding them."""

    def test_no_engines_raises(self, replica):
        r = replica([[]])
        with pytest.raises(RuntimeError, match="No genRM engines available"):
            r.pick()

    def test_empty_list_is_not_cached_forever(self, replica):
        """Regression: caching ``[]`` stranded the replica permanently.

        ``_pick_engine`` raises before any HTTP request is made, so the retry
        loop in ``_call_engine`` -- the only caller of
        ``_invalidate_engine_cache`` -- is never reached. Nothing would ever
        re-read the list, so GenRM stayed dead for the rest of training even
        after recovery rebuilt every engine.
        """
        r = replica([[], LIVE_SIX])
        with pytest.raises(RuntimeError, match="No genRM engines available"):
            r.pick()

        # Engines are back. No invalidate() call here on purpose: nothing in the
        # request path can issue one from this state.
        r.advance(COOLDOWN_S + 1)
        assert r.pick()[1:] in LIVE_SIX
        assert r.fetches == 2

    def test_empty_list_refetch_is_throttled(self, replica):
        """Every in-flight request lands on the empty branch, so re-reading has
        to obey the same cooldown an invalidation does."""
        r = replica([[]])
        for _ in range(50):
            with pytest.raises(RuntimeError):
                r.pick()
        assert r.fetches == 1, "burst while empty must not fan out one ray.get each"

        r.advance(COOLDOWN_S + 1)
        with pytest.raises(RuntimeError):
            r.pick()
        assert r.fetches == 2, "but it must keep retrying so recovery is noticed"


class TestInstanceLookup:
    def test_selects_instance_by_route_key(self):
        cls = genrm_module.GenRM.func_or_class
        replica = object.__new__(cls)
        replica.model_ids = ("quality", "safety")

        assert replica._resolve_instance_key("quality") == "quality"
        assert replica._resolve_instance_key("safety") == "safety"

    def test_omitted_route_key_requires_exactly_one_instance(self):
        cls = genrm_module.GenRM.func_or_class
        replica = object.__new__(cls)
        replica.model_ids = ("quality",)

        assert replica._resolve_instance_key(None) == "quality"

        replica.model_ids = ("quality", "safety")
        with pytest.raises(RuntimeError, match="route_key"):
            replica._resolve_instance_key(None)


async def test_genrm_endpoint_preserves_route_and_strips_response():
    from unittest.mock import AsyncMock

    cls = genrm_module.GenRM.func_or_class
    replica = object.__new__(cls)
    replica._call_engine = AsyncMock(return_value={"text": "  score\n"})
    request = genrm_module.GenerateRequest(
        route_key="quality", messages=[{"role": "user", "content": "judge"}], sampling_params={"temperature": 0.1}
    )

    result = await replica.generate(request)

    assert result.response == "score"
    replica._call_engine.assert_awaited_once_with("quality", request.messages, {"temperature": 0.1})


@pytest.mark.asyncio
async def test_genrm_prepare_payload_uses_instance_tokenizer_and_defaults():
    from unittest.mock import Mock

    cls = genrm_module.GenRM.func_or_class
    replica = object.__new__(cls)
    replica.model_ids = ("quality", "safety")
    replica.instance_specs = {
        "quality": {"sampling_config": {}},
        "safety": {
            "sampling_config": {
                "temperature": 0.4,
                "top_p": 0.8,
                "top_k": 20,
                "max_response_len": 512,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        },
    }
    replica.tokenizers = {
        "quality": SimpleNamespace(apply_chat_template=Mock(return_value=[1])),
        "safety": SimpleNamespace(apply_chat_template=Mock(return_value={"input_ids": [2, 3]})),
    }
    messages = [genrm_module.Message(role="user", content="judge")]
    payload = await replica.prepare_generate_payload("safety", messages, {"temperature": 0.1})
    assert payload == {
        "input_ids": [2, 3],
        "sampling_params": {"temperature": 0.1, "top_p": 0.8, "top_k": 20, "max_new_tokens": 512},
    }
    replica.tokenizers["safety"].apply_chat_template.assert_called_once_with(
        [{"role": "user", "content": "judge"}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    replica.tokenizers["quality"].apply_chat_template.assert_not_called()
    assert replica.instance_specs["safety"]["sampling_config"]["temperature"] == 0.4
    payload = await replica.prepare_generate_payload("quality", messages)
    assert payload["sampling_params"] == {"temperature": 0.2, "top_p": 1.0, "top_k": -1, "max_new_tokens": 1024}


@pytest.mark.asyncio
async def test_genrm_lifecycle_and_health_go_through_the_task_owner(monkeypatch):
    """Every instance is a model of the task owner; the service holds no per-
    instance actor to call."""
    cls = genrm_module.GenRM.func_or_class
    method_globals = cls.onload.__globals__
    monkeypatch.setitem(method_globals, "ray", SimpleNamespace(get=lambda refs: refs))
    calls = []

    def lifecycle(role, model_id, method):
        calls.append((role, model_id, method))
        return model_id != "safety" if method == "health_check" else None

    replica = object.__new__(cls)
    replica._logger_instance = SimpleNamespace(info=lambda *a, **k: None, error=lambda *a, **k: None)
    replica.model_ids = ("quality", "safety")
    replica._owner = SimpleNamespace(lifecycle=SimpleNamespace(remote=lifecycle))

    replica.onload()
    replica.offload()
    health = await replica.health()

    role = genrm_module.Role.GENRM
    assert calls[:4] == [
        (role, "quality", "onload"),
        (role, "safety", "onload"),
        (role, "quality", "offload"),
        (role, "safety", "offload"),
    ]
    assert calls[4:] == [(role, "quality", "health_check"), (role, "safety", "health_check")]
    assert health["status"] == "unhealthy"
    assert health["instances"] == {"quality": {"status": "healthy"}, "safety": {"status": "unhealthy"}}
