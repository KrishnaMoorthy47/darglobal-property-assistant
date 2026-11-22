"""Chat traces and the metrics shown on /admin."""
from __future__ import annotations

import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass, field

from app.db import conn, write
from app.storage import get_sink, jsonable

log = logging.getLogger("telemetry")


@dataclass
class Trace:
    id: str
    session_id: str = ""
    ip_hash: str = ""
    question: str = ""
    intent: str = ""
    source_filter: str = ""
    filters: dict = field(default_factory=dict)
    router_mode: str = ""
    retrieved: list = field(default_factory=list)
    provider: str = ""
    model: str = ""
    attempts: list = field(default_factory=list)
    cache_hit: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    router_ms: float = 0
    retrieval_ms: float = 0
    ttft_ms: float | None = None
    llm_ms: float = 0
    total_ms: float = 0
    answer: str = ""
    error: str = ""
    prompt_version: str = ""
    ts: float = field(default_factory=time.time)

    def save(self) -> None:
        d = asdict(self)
        for k in ("filters", "retrieved", "attempts"):
            d[k] = json.dumps(d[k], ensure_ascii=False)
        cols = ", ".join(d)
        with write() as c:
            c.execute(f"INSERT OR REPLACE INTO traces ({cols}) VALUES ({', '.join('?' * len(d))})", tuple(d.values()))
        get_sink().insert("chat_logs", jsonable(d))
        log.info(json.dumps({"event": "chat_trace", "id": self.id, "session": self.session_id, "intent": self.intent,
                             "provider": self.provider, "model": self.model, "cache": self.cache_hit,
                             "total_ms": round(self.total_ms), "ttft_ms": round(self.ttft_ms or 0),
                             "fallbacks": max(0, len(self.attempts) - 1), "error": self.error or None}))


def set_feedback(trace_id: str, value: int, note: str = "") -> bool:
    with write() as c:
        cur = c.execute("UPDATE traces SET feedback=?, feedback_note=? WHERE id=?", (value, note[:500], trace_id))
        found = cur.rowcount > 0
    # not in SQLite after a restart, so try Supabase
    get_sink().update("chat_logs", {"id": trace_id}, {"feedback": value, "feedback_note": note[:500]})
    return found or get_sink().enabled


def _pct(vals: list[float], p: float) -> float | None:
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    return round(vals[min(len(vals) - 1, round(p / 100 * (len(vals) - 1)))])


def _normalise(rows: list[dict]) -> list[dict]:
    for d in rows:
        for k in ("filters", "retrieved", "attempts"):
            if isinstance(d.get(k), str):
                d[k] = json.loads(d[k] or "[]")
            d[k] = d.get(k) or ([] if k != "filters" else {})
    return rows


async def _rows_since(since: float, limit: int = 5000) -> tuple[list[dict], str]:
    """Chat rows newer than `since`: Supabase if configured (durable history), else local SQLite."""
    rows = await get_sink().select("chat_logs", {"select": "*", "ts": f"gte.{since}", "order": "ts.asc", "limit": str(limit)})
    if rows is not None:
        return _normalise(rows), "supabase"
    local = [dict(r) for r in conn().execute("SELECT * FROM traces WHERE ts>=? ORDER BY ts LIMIT ?", (since, limit))]
    return _normalise(local), "sqlite"


async def metrics(hours: int = 24) -> dict:
    rows, backend = await _rows_since(time.time() - hours * 3600)
    n = len(rows)
    llm_rows = [r for r in rows if r["provider"] and not r["cache_hit"]]
    by_hour: dict[str, int] = {}
    for r in rows:
        h = time.strftime("%Y-%m-%d %H:00", time.gmtime(r["ts"]))
        by_hour[h] = by_hour.get(h, 0) + 1
    count = lambda key: dict(sorted(
        ((v, sum(1 for r in rows if (r[key] or "none") == v)) for v in {r[key] or "none" for r in rows}),
        key=lambda kv: -kv[1]))
    fb = [r["feedback"] for r in rows if r["feedback"] is not None]
    fallbacks = sum(1 for r in llm_rows if len(r["attempts"]) > 1)
    top_scores = [r["retrieved"][0].get("score") or 0 for r in rows if r["retrieved"]]
    return {
        "window_hours": hours,
        "storage": backend,
        "requests": n,
        "sessions": len({r["session_id"] for r in rows}),
        "errors": sum(1 for r in rows if r["error"]),
        "cache_hit_rate": round(sum(r["cache_hit"] or 0 for r in rows) / n, 3) if n else 0,
        "fallback_rate": round(fallbacks / len(llm_rows), 3) if llm_rows else 0,
        "router_llm_share": round(sum(1 for r in rows if r["router_mode"] == "llm") / n, 3) if n else 0,
        "latency_ms": {
            stage: {"p50": _pct([r[col] for r in rows], 50), "p95": _pct([r[col] for r in rows], 95)}
            for stage, col in [("total", "total_ms"), ("ttft", "ttft_ms"), ("router", "router_ms"),
                               ("retrieval", "retrieval_ms"), ("llm", "llm_ms")]
        },
        "tokens": {"prompt": sum(r["prompt_tokens"] or 0 for r in rows),
                   "completion": sum(r["completion_tokens"] or 0 for r in rows)},
        "by_hour": by_hour,
        "intents": count("intent"),
        "models": count("model"),
        "feedback": {"up": fb.count(1), "down": fb.count(-1),
                     "satisfaction": round(fb.count(1) / len(fb), 3) if fb else None},
        "avg_retrieved_top_score": round(statistics.mean(top_scores), 4) if top_scores else None,
    }


async def recent(limit: int = 50) -> list[dict]:
    rows = await get_sink().select("chat_logs", {"select": "*", "order": "ts.desc", "limit": str(limit)})
    if rows is None:
        rows = [dict(r) for r in conn().execute("SELECT * FROM traces ORDER BY ts DESC LIMIT ?", (limit,))]
    for d in _normalise(rows):
        d.pop("ip_hash", None)
    return rows
