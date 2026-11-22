"""Supabase log storage over its REST API.

Writes are queued and sent by a background task with retries, so chat requests don't wait on them.
When the queue is full the oldest entries are dropped. Without SUPABASE_URL / SUPABASE_SECRET_KEY
everything here is a no-op and only local SQLite is used.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

from app.config import get_settings

log = logging.getLogger("storage")


class SupabaseSink:
    def __init__(self) -> None:
        s = get_settings()
        self.url = s.supabase_url.rstrip("/")
        key = s.supabase_secret_key
        self.enabled = bool(self.url and key)
        self.headers = {"apikey": key, "Content-Type": "application/json"}
        if key.startswith("eyJ"):  # old service_role JWTs also need Authorization
            self.headers["Authorization"] = f"Bearer {key}"
        self.queue: asyncio.Queue | None = None
        self.client: httpx.AsyncClient | None = None
        self.worker: asyncio.Task | None = None
        self.stats = {"written": 0, "failed": 0, "dropped": 0}

    async def start(self) -> None:
        if not self.enabled:
            log.info("Supabase sink disabled (no SUPABASE_URL / SUPABASE_SECRET_KEY); using local SQLite only")
            return
        self.client = httpx.AsyncClient(base_url=f"{self.url}/rest/v1", headers=self.headers,
                                        timeout=get_settings().storage.timeout_s)
        self.queue = asyncio.Queue(maxsize=get_settings().storage.queue_max)
        self.worker = asyncio.create_task(self._drain())
        log.info("Supabase sink enabled: %s", self.url)

    async def stop(self) -> None:
        if not self.enabled or self.queue is None:
            return
        try:
            await asyncio.wait_for(self.queue.join(), timeout=5)
        except TimeoutError:
            log.warning("Supabase flush timed out with %d pending", self.queue.qsize())
        if self.worker:
            self.worker.cancel()
        if self.client:
            await self.client.aclose()

    def _enqueue(self, op: dict) -> None:
        if not self.enabled or self.queue is None:
            return
        if self.queue.full():
            self.queue.get_nowait()
            self.queue.task_done()
            self.stats["dropped"] += 1
        self.queue.put_nowait(op)

    def insert(self, table: str, row: dict) -> None:
        self._enqueue({"method": "POST", "path": f"/{table}", "json": row, "prefer": "return=minimal"})

    def upsert(self, table: str, row: dict, on_conflict: str) -> None:
        self._enqueue({"method": "POST", "path": f"/{table}?on_conflict={on_conflict}", "json": row,
                       "prefer": "resolution=merge-duplicates,return=minimal"})

    def update(self, table: str, match: dict, data: dict) -> None:
        q = "&".join(f"{k}=eq.{v}" for k, v in match.items())
        self._enqueue({"method": "PATCH", "path": f"/{table}?{q}", "json": data, "prefer": "return=minimal"})

    async def _drain(self) -> None:
        assert self.queue is not None and self.client is not None
        while True:
            op = await self.queue.get()
            try:
                for attempt in range(get_settings().storage.write_retries):
                    try:
                        r = await self.client.request(op["method"], op["path"], json=op["json"],
                                                      headers={"Prefer": op["prefer"]})
                        if r.status_code < 300:
                            self.stats["written"] += 1
                            break
                        if r.status_code < 500 and r.status_code != 429:
                            log.warning("Supabase rejected %s %s: %s %s", op["method"], op["path"], r.status_code, r.text[:300])
                            self.stats["failed"] += 1
                            break
                    except httpx.HTTPError as e:
                        log.warning("Supabase write error (attempt %d): %s", attempt + 1, e)
                    await asyncio.sleep(2 ** attempt)
                else:
                    self.stats["failed"] += 1
            finally:
                self.queue.task_done()

    async def select(self, table: str, params: dict[str, Any]) -> list[dict] | None:
        """Rows, or None if Supabase is off or unreachable."""
        if not self.enabled or self.client is None:
            return None
        try:
            r = await self.client.get(f"/{table}", params=params)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            log.warning("Supabase read failed for %s: %s", table, e)
            return None


def jsonable(row: dict) -> dict:
    """SQLite stores JSON columns as text; Supabase wants real JSON for jsonb columns."""
    out = dict(row)
    for k in ("filters", "retrieved", "attempts"):
        if isinstance(out.get(k), str):
            try:
                out[k] = json.loads(out[k])
            except json.JSONDecodeError:
                pass
    return out


_sink: SupabaseSink | None = None


def get_sink() -> SupabaseSink:
    global _sink
    if _sink is None:
        _sink = SupabaseSink()
    return _sink
