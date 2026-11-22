"""Answer cache, matched on the normalised question or by embedding similarity.

Follow-up questions aren't cached since their answers depend on the history.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from typing import TYPE_CHECKING

import numpy as np

from app.config import get_settings
from app.db import conn, write
from app.resources import get_prompts

if TYPE_CHECKING:
    from app.rag.store import QVec


def _norm(q: str) -> str:
    return re.sub(r"[^\w؀-ۿ ]+", "", q.lower()).strip()


def _scope(source: str | None) -> str:
    # include the prompt version so a prompt change doesn't keep serving old answers
    return f"{source or 'any'}@{get_prompts().version}"


def _key(q: str, source: str | None) -> str:
    return hashlib.sha1(f"{_scope(source)}|{_norm(q)}".encode()).hexdigest()


def lookup(question: str, source: str | None, qvec: QVec) -> dict | None:
    s = get_settings()
    min_ts = time.time() - s.cache.ttl_hours * 3600
    c = conn()
    row = c.execute("SELECT * FROM response_cache WHERE key=? AND ts>=?", (_key(question, source), min_ts)).fetchone()
    kind = "exact"
    if row is None:
        kind = "semantic"
        best, best_sim = None, 0.0
        for r in c.execute("SELECT * FROM response_cache WHERE source_filter=? AND ts>=? ORDER BY ts DESC LIMIT ?",
                           (_scope(source), min_ts, s.cache.scan_limit)):
            emb = np.frombuffer(r["embedding"], dtype=np.float32)
            if emb.shape != qvec.v.shape:  # stored with the other embedding model
                continue
            sim = float(emb @ qvec.v)
            if sim > best_sim:
                best, best_sim = r, sim
        if best is None or best_sim < s.cache.semantic_threshold:
            return None
        row = best
    with write() as w:
        w.execute("UPDATE response_cache SET hits=hits+1 WHERE key=?", (row["key"],))
    return {"answer": row["answer"], "payload": json.loads(row["payload"]), "provider": row["provider"],
            "model": row["model"], "kind": kind}


def store(question: str, source: str | None, qvec: QVec, answer: str, payload: dict, provider: str, model: str) -> None:
    with write() as w:
        w.execute(
            "INSERT OR REPLACE INTO response_cache(key, ts, source_filter, question, embedding, answer, payload, provider, model, hits) "
            "VALUES (?,?,?,?,?,?,?,?,?,0)",
            (_key(question, source), time.time(), _scope(source), question, qvec.v.astype(np.float32).tobytes(),
             answer, json.dumps(payload, ensure_ascii=False), provider, model),
        )
