"""API tests with no LLM keys, so the rule router and search-only answers are used."""
import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module):
    for key in ("OPENROUTER_API_KEY", "OPENROUTER_API_KEY_2", "OPENROUTER_API_KEY_3", "GROQ_API_KEY"):
        monkeypatch_module.setenv(key, "")
    monkeypatch_module.setenv("ADMIN_TOKEN", "test-admin")
    monkeypatch_module.setenv("SUPABASE_URL", "")         # don't write to the real logs
    monkeypatch_module.setenv("SUPABASE_SECRET_KEY", "")
    monkeypatch_module.setenv("PATHS__RUNTIME_DB", str(tmp_path_factory.mktemp("rt") / "rt.db"))
    from app import config, db, storage
    from app.llm import gateway
    config.get_settings.cache_clear()
    db._conn, gateway._gateway, storage._sink = None, None, None
    from app import main
    main.s = config.get_settings()
    with TestClient(main.app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def events(resp):
    return [json.loads(line[5:]) for line in resp.text.split("\n\n") if line.startswith("data:")]


def test_health(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["properties"] > 0


def test_chat_stream_degrades_gracefully(client):
    r = client.post("/api/chat", json={"message": "villas for sale in Riyadh", "session_id": "t1"})
    evs = events(r)
    types = [e["type"] for e in evs]
    assert types[0] == "meta" and "token" in types and types[-1] == "done"
    meta = evs[0]
    assert meta["cards"], "property cards expected"
    assert evs[-1]["provider"] == "retrieval-only"


def test_greeting_costs_no_llm(client):
    evs = events(client.post("/api/chat", json={"message": "hi"}))
    assert evs[-1]["provider"] == "rules"


def test_input_validation(client):
    assert client.post("/api/chat", json={"message": "x" * 5000}).status_code == 413
    assert client.post("/api/chat", json={"message": ""}).status_code == 422


def test_feedback_and_admin(client):
    evs = events(client.post("/api/chat", json={"message": "Tell me about DarGlobal"}))
    tid = evs[-1]["trace_id"]
    assert client.post("/api/feedback", json={"trace_id": tid, "value": 1}).status_code == 200
    assert client.get("/api/admin/metrics").status_code == 401
    m = client.get("/api/admin/metrics", headers={"X-Admin-Token": "test-admin"}).json()
    assert m["metrics"]["requests"] >= 1 and m["metrics"]["feedback"]["up"] >= 1
    traces = client.get("/api/admin/traces", headers={"X-Admin-Token": "test-admin"}).json()
    assert traces and "ip_hash" not in traces[0]


def test_contact_details_are_never_shared(client):
    evs = events(client.post("/api/chat", json={"message": "What is the agent's phone number for listing 5879365?"}))
    text = "".join(e.get("text", "") for e in evs if e["type"] == "token")
    assert evs[-1]["provider"] == "rules"
    assert "don't share personal contact details" in text
