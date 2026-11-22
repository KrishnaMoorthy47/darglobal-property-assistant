"""Gateway tests: failover, circuit breaker, daily cap, mid-stream failure."""
import asyncio
from types import SimpleNamespace

import httpx
import openai
import pytest

from app.llm import gateway as gw_mod
from app.llm.gateway import AllProvidersUnavailable, Endpoint, Gateway, StreamResult


def _rate_limit_error():
    req = httpx.Request("POST", "https://x")
    resp = httpx.Response(429, request=req, headers={"retry-after": "120"})
    return openai.RateLimitError("rate limited", response=resp, body=None)


class FakeStream:
    def __init__(self, tokens, fail_after=None):
        self.tokens, self.fail_after = tokens, fail_after

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for i, t in enumerate(self.tokens):
            if self.fail_after is not None and i == self.fail_after:
                raise httpx.ReadTimeout("boom")
            yield SimpleNamespace(usage=None, choices=[SimpleNamespace(delta=SimpleNamespace(content=t))])


def fake_client(behaviour):
    async def create(**kwargs):
        if isinstance(behaviour, Exception):
            raise behaviour
        return behaviour

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    monkeypatch.setattr(gw_mod, "_record", lambda *a, **k: None)
    monkeypatch.setattr(gw_mod, "usage_today", lambda ep: 0)
    monkeypatch.setattr(gw_mod, "key_usage_today", lambda p: 0)
    g = Gateway.__new__(Gateway)
    g.key_blocked_until = {}
    from app.config import get_settings
    g.cfg = get_settings().llm
    g.clients = {}
    g.chains = {"generation": [Endpoint("a", "m1", 50, 10), Endpoint("b", "m2", 50, 10)]}
    return g


async def collect(g, res):
    return "".join([t async for t in g.stream("generation", [{"role": "user", "content": "hi"}], res)])


def test_failover_on_rate_limit(gateway):
    gateway.clients = {"a": fake_client(_rate_limit_error()), "b": fake_client(FakeStream(["Hel", "lo"]))}
    res = StreamResult()
    out = asyncio.run(collect(gateway, res))
    assert out == "Hello"
    assert res.provider == "b"
    assert res.attempts[0]["error"] == "rate_limited"
    # 429 should open a's circuit for the Retry-After time
    assert gateway.chains["generation"][0].is_open()


def test_open_circuit_is_skipped(gateway):
    gateway.chains["generation"][0].trip(60, "test")
    gateway.clients = {"a": fake_client(RuntimeError("must not be called")), "b": fake_client(FakeStream(["ok"]))}
    res = StreamResult()
    assert asyncio.run(collect(gateway, res)) == "ok"
    assert res.attempts[0] == {"endpoint": "a:m1", "skipped": "circuit_open"}


def test_daily_budget_guard(gateway, monkeypatch):
    monkeypatch.setattr(gw_mod, "usage_today", lambda ep: 40 if ep.provider == "a" else 0)  # 40 >= 50*0.8
    gateway.clients = {"a": fake_client(RuntimeError("must not be called")), "b": fake_client(FakeStream(["ok"]))}
    res = StreamResult()
    assert asyncio.run(collect(gateway, res)) == "ok"
    assert res.attempts[0]["skipped"] == "daily_budget_reached"


def test_all_providers_down_raises(gateway):
    gateway.clients = {"a": fake_client(_rate_limit_error()), "b": fake_client(_rate_limit_error())}
    with pytest.raises(AllProvidersUnavailable) as e:
        asyncio.run(collect(gateway, StreamResult()))
    assert len(e.value.attempts) == 2


def test_no_splicing_after_first_token(gateway):
    """A stream that fails mid-answer shouldn't be continued by another model."""
    gateway.clients = {"a": fake_client(FakeStream(["Part", "ial", "x"], fail_after=2)),
                       "b": fake_client(FakeStream(["SHOULD NOT APPEAR"]))}
    out = asyncio.run(collect(gateway, StreamResult()))
    assert out.startswith("Partial")
    assert "SHOULD NOT APPEAR" not in out
    assert "interrupted" in out


def test_token_bucket_limits_rpm():
    e = Endpoint("a", "m", 100, 3)
    assert [e.take_token() for _ in range(4)] == [True, True, True, False]


def _per_day_error():
    req = httpx.Request("POST", "https://x")
    resp = httpx.Response(429, request=req)
    return openai.RateLimitError("Rate limit exceeded: free-models-per-day", response=resp, body=None)


def test_key_daily_limit_blocks_every_model_on_that_key(gateway):
    gateway.chains["generation"] = [Endpoint("a", "m1", 50, 10), Endpoint("a", "m2", 50, 10), Endpoint("b", "m3", 50, 10)]
    gateway.clients = {"a": fake_client(_per_day_error()), "b": fake_client(FakeStream(["ok"]))}
    res = StreamResult()
    assert asyncio.run(collect(gateway, res)) == "ok"
    assert res.attempts[1] == {"endpoint": "a:m2", "skipped": "key_daily_limit"}


def test_key_cap_counts_all_models(gateway, monkeypatch):
    monkeypatch.setattr(gateway.cfg, "key_daily_caps", {"a": 10})
    monkeypatch.setattr(gw_mod, "key_usage_today", lambda p: 10 if p == "a" else 0)
    gateway.clients = {"a": fake_client(RuntimeError("must not be called")), "b": fake_client(FakeStream(["ok"]))}
    res = StreamResult()
    assert asyncio.run(collect(gateway, res)) == "ok"
    assert res.attempts[0]["skipped"] == "key_daily_budget_reached"


def _fake_embed_client(behaviour):
    async def create(**kwargs):
        if isinstance(behaviour, Exception):
            raise behaviour
        data = [SimpleNamespace(index=i, embedding=[3.0, 4.0]) for i in range(len(kwargs["input"]))]
        return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=5))
    return SimpleNamespace(embeddings=SimpleNamespace(create=create))


def test_embed_fails_over_and_normalises(gateway):
    gateway.chains["embedding"] = [Endpoint("a", "e", 50, 10), Endpoint("b", "e", 50, 10)]
    gateway.clients = {"a": _fake_embed_client(_rate_limit_error()), "b": _fake_embed_client(None)}
    v, res = asyncio.run(gateway.embed(["x", "y"]))
    assert v.shape == (2, 2) and abs(float(v[0] @ v[0]) - 1) < 1e-6
    assert res.provider == "b"


def test_store_falls_back_to_local_embeddings(monkeypatch):
    import numpy as np

    from app.rag.store import get_store
    s = get_store()

    class Down:
        async def embed(self, texts):
            raise AllProvidersUnavailable([])

    monkeypatch.setattr("app.llm.gateway.get_gateway", lambda: Down())
    monkeypatch.setitem(s.emb, "remote", {"chunk": np.zeros((1, 2)), "prop": np.zeros((1, 2))})
    s._remote_cache.clear()
    qv = asyncio.run(s.embed_query("villa in riyadh"))
    assert qv.space == "local"
