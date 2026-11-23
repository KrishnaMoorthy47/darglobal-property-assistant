"""Supabase sink tests against a fake PostgREST server."""
import asyncio
import json

import httpx

from app.storage import SupabaseSink, jsonable


def make_sink(handler) -> SupabaseSink:
    sink = SupabaseSink.__new__(SupabaseSink)
    sink.url, sink.enabled = "https://x.supabase.co", True
    sink.headers = {"apikey": "test"}
    sink.stats = {"written": 0, "failed": 0, "dropped": 0}
    sink.client = httpx.AsyncClient(base_url="https://x.supabase.co/rest/v1", transport=httpx.MockTransport(handler))
    return sink


async def run_ops(sink: SupabaseSink, ops) -> None:
    sink.queue = asyncio.Queue()
    sink.worker = asyncio.create_task(sink._drain())
    ops(sink)
    await asyncio.wait_for(sink.queue.join(), timeout=10)
    sink.worker.cancel()


def test_insert_upsert_update_requests():
    seen = []

    def handler(req: httpx.Request):
        seen.append((req.method, str(req.url), req.headers.get("prefer"), json.loads(req.content or b"null")))
        return httpx.Response(201)

    sink = make_sink(handler)
    asyncio.run(run_ops(sink, lambda s: (
        s.insert("chat_logs", {"id": "t1", "question": "hi", "answer": "hello", "session_id": "s1"}),
        s.upsert("provider_usage", {"day": "2000-01-01", "provider": "groq", "model": "m", "requests": 3}, "day,provider,model"),
        s.update("chat_logs", {"id": "t1"}, {"feedback": 1}),
    )))
    assert [m for m, *_ in seen] == ["POST", "POST", "PATCH"]
    assert seen[0][1].endswith("/rest/v1/chat_logs") and seen[0][3]["question"] == "hi"
    assert "on_conflict=day%2Cprovider%2Cmodel" in seen[1][1] or "on_conflict=day,provider,model" in seen[1][1]
    assert "merge-duplicates" in seen[1][2]
    assert seen[2][1].endswith("chat_logs?id=eq.t1") and seen[2][3] == {"feedback": 1}
    assert sink.stats["written"] == 3


def test_retries_transient_errors(monkeypatch):
    calls = {"n": 0}

    async def no_sleep(_):
        return None

    monkeypatch.setattr("app.storage.asyncio.sleep", no_sleep)

    def handler(req):
        calls["n"] += 1
        return httpx.Response(503 if calls["n"] < 3 else 201)

    sink = make_sink(handler)
    asyncio.run(run_ops(sink, lambda s: s.insert("api_logs", {"ts": 1})))
    assert calls["n"] == 3 and sink.stats["written"] == 1


def test_rejected_write_is_not_retried():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(400, json={"message": "bad column"})

    sink = make_sink(handler)
    asyncio.run(run_ops(sink, lambda s: s.insert("api_logs", {"nope": 1})))
    assert calls["n"] == 1 and sink.stats["failed"] == 1


def test_select_and_fallback():
    ok = make_sink(lambda req: httpx.Response(200, json=[{"id": "t1"}]))
    assert asyncio.run(ok.select("chat_logs", {"select": "*"})) == [{"id": "t1"}]
    down = make_sink(lambda req: httpx.Response(500))
    assert asyncio.run(down.select("chat_logs", {"select": "*"})) is None


def test_disabled_sink_is_noop():
    sink = SupabaseSink.__new__(SupabaseSink)
    sink.enabled, sink.queue = False, None
    sink.insert("chat_logs", {"id": "x"})  # must not raise
    assert asyncio.run(sink.select("chat_logs", {})) is None


def test_jsonable_parses_json_columns():
    row = jsonable({"filters": '{"city": "Riyadh"}', "retrieved": "[]", "question": "q"})
    assert row["filters"] == {"city": "Riyadh"} and row["retrieved"] == [] and row["question"] == "q"
