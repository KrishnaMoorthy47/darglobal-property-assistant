"""LLM calls with failover across OpenRouter and Groq models.

Each model gets a per-minute token bucket, a daily request cap and a circuit breaker (429 respects
Retry-After, auth errors back off longer). Streams only fail over before the first token is sent.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
import openai
from openai import AsyncOpenAI

from app.config import get_settings
from app.db import conn, write
from app.resources import get_prompts
from app.storage import get_sink

log = logging.getLogger("gateway")

class InputRejected(Exception):
    """The provider refused the input itself (e.g. content screening); other keys would refuse it too."""


class AllProvidersUnavailable(Exception):
    def __init__(self, attempts: list[dict]):
        super().__init__("all LLM providers unavailable")
        self.attempts = attempts


@dataclass
class Endpoint:
    provider: str
    model: str
    daily_cap: int
    rpm: int
    tokens: float = 0.0
    last_refill: float = field(default_factory=time.monotonic)
    open_until: float = 0.0
    consecutive_failures: int = 0

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    def __post_init__(self):
        self.tokens = float(self.rpm)

    def take_token(self) -> bool:
        now = time.monotonic()
        self.tokens = min(self.rpm, self.tokens + (now - self.last_refill) * self.rpm / 60.0)
        self.last_refill = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

    def is_open(self) -> bool:
        return time.monotonic() < self.open_until

    def trip(self, seconds: float, reason: str) -> None:
        self.open_until = time.monotonic() + seconds
        log.warning("circuit OPEN for %s (%.0fs): %s", self.name, seconds, reason)


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def usage_today(ep: Endpoint) -> int:
    row = conn().execute(
        "SELECT requests FROM provider_usage WHERE day=? AND provider=? AND model=?",
        (_today(), ep.provider, ep.model),
    ).fetchone()
    return row["requests"] if row else 0


def key_usage_today(provider: str) -> int:
    """All of today's requests made with one API key, across models."""
    row = conn().execute("SELECT SUM(requests) AS n FROM provider_usage WHERE day=? AND provider=?",
                         (_today(), provider)).fetchone()
    return (row["n"] if row else 0) or 0


def _record(ep: Endpoint, ok: bool, rate_limited: bool = False, pt: int = 0, ct: int = 0) -> None:
    with write() as c:
        c.execute(
            """INSERT INTO provider_usage(day, provider, model, requests, failures, rate_limited, prompt_tokens, completion_tokens)
               VALUES (?,?,?,1,?,?,?,?)
               ON CONFLICT(day, provider, model) DO UPDATE SET
                 requests=requests+1, failures=failures+excluded.failures,
                 rate_limited=rate_limited+excluded.rate_limited,
                 prompt_tokens=prompt_tokens+excluded.prompt_tokens,
                 completion_tokens=completion_tokens+excluded.completion_tokens""",
            (_today(), ep.provider, ep.model, 0 if ok else 1, 1 if rate_limited else 0, pt, ct),
        )
        row = c.execute("SELECT * FROM provider_usage WHERE day=? AND provider=? AND model=?",
                        (_today(), ep.provider, ep.model)).fetchone()
    # so the daily cap survives restarts
    get_sink().upsert("provider_usage", dict(row), on_conflict="day,provider,model")


async def hydrate_usage_from_sink() -> int:
    """Load today's request counts from Supabase into SQLite at startup."""
    rows = await get_sink().select("provider_usage", {"select": "*", "day": f"eq.{_today()}"})
    if not rows:
        return 0
    cols = ("day", "provider", "model", "requests", "failures", "rate_limited", "prompt_tokens", "completion_tokens")
    with write() as c:
        for r in rows:
            c.execute(
                f"""INSERT INTO provider_usage({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})
                    ON CONFLICT(day, provider, model) DO UPDATE SET
                      requests=max(requests, excluded.requests), failures=max(failures, excluded.failures),
                      rate_limited=max(rate_limited, excluded.rate_limited),
                      prompt_tokens=max(prompt_tokens, excluded.prompt_tokens),
                      completion_tokens=max(completion_tokens, excluded.completion_tokens)""",
                tuple(r.get(k) or 0 if k not in ("day", "provider", "model") else r[k] for k in cols),
            )
    return len(rows)


def _model_extras(e: Endpoint) -> dict:
    """Extra request params for a model: every matching entry of llm.request_extras, in order."""
    out: dict = {}
    for fragment, extras in get_settings().llm.request_extras.items():
        if fragment in e.model:
            out |= extras
    return out


def build_chain(entries) -> list[Endpoint]:
    return [Endpoint(c.provider, c.model, c.daily_cap, c.rpm) for c in entries]


@dataclass
class StreamResult:
    """Filled in during the stream."""
    provider: str = ""
    model: str = ""
    attempts: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_ms: float | None = None


class Gateway:
    def __init__(self) -> None:
        s = get_settings()
        self.cfg = s.llm
        keys = {"openrouter": s.openrouter_api_key, "openrouter_2": s.openrouter_api_key_2,
                "openrouter_3": s.openrouter_api_key_3, "groq": s.groq_api_key}
        self.clients: dict[str, AsyncOpenAI] = {}
        for p, key in keys.items():
            if key:
                headers = {"HTTP-Referer": s.public_url, "X-Title": "Property Intelligence Assistant"} if p.startswith("openrouter") else None
                self.clients[p] = AsyncOpenAI(api_key=key, base_url=self.cfg.base_urls[p], timeout=self.cfg.timeout_s,
                                              max_retries=0, default_headers=headers)
        self.chains = {
            "generation": [e for e in build_chain(self.cfg.generation_chain) if e.provider in self.clients],
            "router": [e for e in build_chain(self.cfg.router_chain) if e.provider in self.clients],
            "embedding": [e for e in build_chain(self.cfg.embedding_chain) if e.provider in self.clients],
        }
        self.key_blocked_until: dict[str, float] = {}  # a key that hit its daily limit, for every model

    def status(self) -> list[dict]:
        out = []
        for chain, eps in self.chains.items():
            for e in eps:
                used = usage_today(e)
                out.append({
                    "chain": chain, "endpoint": e.name, "provider": e.provider, "model": e.model,
                    "daily_cap": e.daily_cap, "used_today": used,
                    "safe_limit": int(e.daily_cap * self.cfg.quota_safety_ratio),
                    "remaining_safe": max(0, int(e.daily_cap * self.cfg.quota_safety_ratio) - used),
                    "rpm_limit": e.rpm, "circuit_open": e.is_open(),
                    "open_for_s": max(0, round(e.open_until - time.monotonic())),
                })
        return out

    def key_status(self) -> list[dict]:
        out = []
        for p in self.clients:
            cap = self.cfg.key_daily_caps.get(p)
            out.append({"key": p, "used_today": key_usage_today(p), "daily_cap": cap,
                        "blocked_for_s": max(0, round(self.key_blocked_until.get(p, 0) - time.monotonic()))})
        return out

    def available(self, chain: str) -> bool:
        return any(self._usable(e) is None for e in self.chains[chain])

    def _usable(self, e: Endpoint) -> str | None:
        """Return a skip reason, or None if the endpoint may be called now."""
        if e.is_open():
            return "circuit_open"
        if self.key_blocked_until.get(e.provider, 0) > time.monotonic():
            return "key_daily_limit"
        cap = self.cfg.key_daily_caps.get(e.provider)
        if cap and key_usage_today(e.provider) >= cap:
            return "key_daily_budget_reached"
        if usage_today(e) >= int(e.daily_cap * self.cfg.quota_safety_ratio):
            return "daily_budget_reached"
        return None

    def _handle_error(self, e: Endpoint, err: Exception) -> tuple[str, bool]:
        cb = self.cfg.circuit_breaker
        e.consecutive_failures += 1
        if isinstance(err, openai.RateLimitError):
            retry_after = cb.default_retry_after_s
            try:
                retry_after = float(err.response.headers.get("retry-after", cb.default_retry_after_s))
            except (AttributeError, TypeError, ValueError):
                pass
            if "per-day" in str(err):  # OpenRouter free-models-per-day: the whole key is done for a while
                self.key_blocked_until[e.provider] = time.monotonic() + cb.rate_limit_max_s
                log.warning("key %s hit its daily limit; skipping it for %ss", e.provider, cb.rate_limit_max_s)
            e.trip(min(max(retry_after, cb.rate_limit_min_s), cb.rate_limit_max_s), "429 rate limited")
            return "rate_limited", True
        if isinstance(err, (openai.AuthenticationError, openai.PermissionDeniedError)) or getattr(err, "status_code", None) == 402:
            e.trip(cb.auth_error_s, f"auth/credits error {getattr(err, 'status_code', '')}")
            return "auth_or_credits", False
        if isinstance(err, openai.NotFoundError):
            e.trip(cb.model_not_found_s, "model not found (free model may have been retired)")
            return "model_not_found", False
        if e.consecutive_failures >= cb.transient_failures_to_trip:
            e.trip(cb.transient_cooldown_s, f"{type(err).__name__}")
        return type(err).__name__, False

    async def complete_json(self, chain: str, messages: list[dict]) -> tuple[str, StreamResult]:
        """Non-streaming call used by the router."""
        res = StreamResult()
        for e in self.chains[chain]:
            reason = self._usable(e)
            if reason or not e.take_token():
                res.attempts.append({"endpoint": e.name, "skipped": reason or "local_rpm_limit"})
                continue
            t0 = time.perf_counter()
            try:
                params = {"model": e.model, "messages": messages, "temperature": self.cfg.router.temperature,
                          "max_tokens": self.cfg.router.max_tokens}
                r = await self.clients[e.provider].chat.completions.create(**params | _model_extras(e))
                pt = getattr(r.usage, "prompt_tokens", 0) or 0
                ct = getattr(r.usage, "completion_tokens", 0) or 0
                _record(e, True, pt=pt, ct=ct)
                e.consecutive_failures = 0
                res.provider, res.model, res.prompt_tokens, res.completion_tokens = e.provider, e.model, pt, ct
                res.attempts.append({"endpoint": e.name, "ok": True, "ms": round((time.perf_counter() - t0) * 1000)})
                return r.choices[0].message.content or "", res
            except Exception as err:  # noqa: BLE001
                kind, rl = self._handle_error(e, err)
                _record(e, False, rate_limited=rl)
                res.attempts.append({"endpoint": e.name, "error": kind})
                log.warning("router call failed on %s: %s", e.name, err)
        raise AllProvidersUnavailable(res.attempts)

    async def embed(self, texts: list[str], chain: str = "embedding") -> tuple[np.ndarray, StreamResult]:
        """Embeddings from the first usable endpoint. Rows are L2-normalised."""
        res = StreamResult()
        for e in self.chains[chain]:
            reason = self._usable(e)
            if reason or not e.take_token():
                res.attempts.append({"endpoint": e.name, "skipped": reason or "local_rpm_limit"})
                continue
            t0 = time.perf_counter()
            try:
                r = await self.clients[e.provider].embeddings.create(model=e.model, input=texts)
                v = np.array([d.embedding for d in sorted(r.data, key=lambda d: d.index)], dtype=np.float32)
                if v.shape[0] != len(texts):
                    raise ValueError(f"got {v.shape[0]} embeddings for {len(texts)} texts")
                v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
                pt = getattr(r.usage, "prompt_tokens", 0) or 0
                _record(e, True, pt=pt)
                e.consecutive_failures = 0
                res.provider, res.model, res.prompt_tokens = e.provider, e.model, pt
                res.attempts.append({"endpoint": e.name, "ok": True, "ms": round((time.perf_counter() - t0) * 1000)})
                return v, res
            except openai.UnprocessableEntityError as err:
                _record(e, False)
                raise InputRejected(str(err)) from err
            except Exception as err:  # noqa: BLE001
                kind, rl = self._handle_error(e, err)
                _record(e, False, rate_limited=rl)
                res.attempts.append({"endpoint": e.name, "error": kind})
                log.warning("embedding call failed on %s: %s", e.name, err)
        raise AllProvidersUnavailable(res.attempts)

    async def stream(self, chain: str, messages: list[dict], res: StreamResult) -> AsyncIterator[str]:
        """Stream tokens; switch endpoints only before the first token."""
        gen = self.cfg.generation
        for e in self.chains[chain]:
            reason = self._usable(e)
            if reason or not e.take_token():
                res.attempts.append({"endpoint": e.name, "skipped": reason or "local_rpm_limit"})
                continue
            t0 = time.perf_counter()
            started = False
            try:
                params = {"model": e.model, "messages": messages, "temperature": gen.temperature,
                          "max_tokens": gen.max_tokens, "stream": True, "stream_options": {"include_usage": True}}
                stream = await self.clients[e.provider].chat.completions.create(**params | _model_extras(e))
                async for chunk in stream:
                    if chunk.usage:
                        res.prompt_tokens = chunk.usage.prompt_tokens or 0
                        res.completion_tokens = chunk.usage.completion_tokens or 0
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta.content
                    if delta:
                        if not started:
                            started = True
                            res.ttft_ms = (time.perf_counter() - t0) * 1000
                            res.provider, res.model = e.provider, e.model
                        yield delta
                if not started:
                    raise RuntimeError("empty completion")
                _record(e, True, pt=res.prompt_tokens, ct=res.completion_tokens)
                e.consecutive_failures = 0
                res.attempts.append({"endpoint": e.name, "ok": True, "ms": round((time.perf_counter() - t0) * 1000)})
                return
            except asyncio.CancelledError:
                raise
            except Exception as err:  # noqa: BLE001
                kind, rl = self._handle_error(e, err)
                _record(e, False, rate_limited=rl)
                res.attempts.append({"endpoint": e.name, "error": kind})
                log.warning("stream failed on %s: %s", e.name, err)
                if started:
                    # don't continue with a different model mid-answer
                    yield get_prompts().interrupted_notice
                    return
        raise AllProvidersUnavailable(res.attempts)


_gateway: Gateway | None = None


def get_gateway() -> Gateway:
    global _gateway
    if _gateway is None:
        _gateway = Gateway()
    return _gateway
