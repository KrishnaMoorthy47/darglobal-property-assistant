"""Writable SQLite for traces, the response cache and quota counters. Resets when the container restarts."""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager

from app.config import get_settings

_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    id TEXT PRIMARY KEY, ts REAL, session_id TEXT, ip_hash TEXT,
    question TEXT, intent TEXT, source_filter TEXT, filters TEXT, router_mode TEXT,
    retrieved TEXT, provider TEXT, model TEXT, attempts TEXT, cache_hit INTEGER DEFAULT 0,
    prompt_tokens INTEGER, completion_tokens INTEGER,
    router_ms REAL, retrieval_ms REAL, ttft_ms REAL, llm_ms REAL, total_ms REAL,
    answer TEXT, error TEXT, feedback INTEGER, feedback_note TEXT, prompt_version TEXT
);
CREATE INDEX IF NOT EXISTS traces_ts ON traces(ts);
CREATE TABLE IF NOT EXISTS response_cache (
    key TEXT PRIMARY KEY, ts REAL, source_filter TEXT, question TEXT, embedding BLOB,
    answer TEXT, payload TEXT, provider TEXT, model TEXT, hits INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS provider_usage (
    day TEXT, provider TEXT, model TEXT, requests INTEGER DEFAULT 0, failures INTEGER DEFAULT 0,
    rate_limited INTEGER DEFAULT 0, prompt_tokens INTEGER DEFAULT 0, completion_tokens INTEGER DEFAULT 0,
    PRIMARY KEY (day, provider, model)
);
"""

_conn: sqlite3.Connection | None = None


def conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        path = get_settings().runtime_db
        path.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.executescript(SCHEMA)
        cols = {r[1] for r in _conn.execute("PRAGMA table_info(traces)")}
        if "prompt_version" not in cols:  # older DBs lack this column
            _conn.execute("ALTER TABLE traces ADD COLUMN prompt_version TEXT")
    return _conn


@contextmanager
def write():
    """One writer at a time; reads don't need the lock."""
    with _lock:
        yield conn()
