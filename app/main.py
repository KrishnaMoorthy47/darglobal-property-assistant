"""FastAPI entrypoint."""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import ROOT, get_settings
from app.db import conn
from app.llm.gateway import get_gateway, hydrate_usage_from_sink
from app.rag.chat import run_chat
from app.rag.store import get_store
from app.storage import get_sink
from app.telemetry import metrics, recent, set_feedback

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")
STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    store = get_store()
    store.embed("warm up")  # load the model now, not on the first request
    conn()
    sink = get_sink()
    await sink.start()
    restored = await hydrate_usage_from_sink()
    gw = get_gateway()
    log.info("ready: generation=%s router=%s durable_logs=%s restored_usage_rows=%d",
             [e.name for e in gw.chains["generation"]], [e.name for e in gw.chains["router"]], sink.enabled, restored)
    yield
    await sink.stop()


app = FastAPI(title="Property Intelligence Assistant", version="1.0.0", lifespan=lifespan,
              docs_url="/api/docs", redoc_url=None)
s = get_settings()
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in s.ops.allowed_origins.split(",")],
                   allow_methods=["GET", "POST"], allow_headers=["*"])


@app.middleware("http")
async def security_headers(request: Request, call_next):
    t0 = time.perf_counter()
    resp = await call_next(request)
    if request.url.path.startswith("/api/"):
        get_sink().insert("api_logs", {
            "ts": time.time(), "session_id": request.headers.get("x-session-id", "")[:64] or None,
            "method": request.method, "path": request.url.path, "status": resp.status_code,
            "duration_ms": round((time.perf_counter() - t0) * 1000, 1),
            "ip_hash": hashlib.sha256(f"{client_ip(request)}|pia".encode()).hexdigest()[:16],
            "user_agent": request.headers.get("user-agent", "")[:200],
        })
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    resp.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    return resp


_minute: dict[str, deque] = defaultdict(deque)
_day: dict[str, deque] = defaultdict(deque)


def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "unknown")


def rate_limit(request: Request) -> str:
    ip = client_ip(request)
    now = time.time()
    for q, window, limit in ((_minute[ip], 60, s.guardrails.user_rpm), (_day[ip], 86400, s.guardrails.user_daily)):
        while q and q[0] < now - window:
            q.popleft()
        if len(q) >= limit:
            raise HTTPException(429, "You're sending messages too quickly. Please wait a moment and try again.")
    _minute[ip].append(now)
    _day[ip].append(now)
    return hashlib.sha256(f"{ip}|pia".encode()).hexdigest()[:16]


def require_admin(x_admin_token: str = Header(default="")) -> None:
    if not s.admin_token or not hmac.compare_digest(x_admin_token, s.admin_token):
        raise HTTPException(401, "invalid admin token")


class Msg(BaseModel):
    role: str
    content: str = Field(max_length=4000)


class ChatIn(BaseModel):
    message: str = Field(min_length=1)
    history: list[Msg] = Field(default_factory=list, max_length=40)
    session_id: str = Field(default="anon", max_length=64)
    source: str | None = Field(default=None, pattern="^(darglobal|wasalt|any)?$")


@app.post("/api/chat")
async def chat(body: ChatIn, ip_hash: str = Depends(rate_limit)):
    msg = body.message.strip()
    if len(msg) > s.guardrails.max_input_chars:
        raise HTTPException(413, f"Message too long (max {s.guardrails.max_input_chars} characters).")
    source = None if body.source in (None, "", "any") else body.source

    async def events():
        async for ev in run_chat(msg, [m.model_dump() for m in body.history], body.session_id, source, ip_hash):
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


class FeedbackIn(BaseModel):
    trace_id: str = Field(max_length=32)
    value: int = Field(ge=-1, le=1)
    note: str = Field(default="", max_length=500)


@app.post("/api/feedback")
async def feedback(body: FeedbackIn):
    if not set_feedback(body.trace_id, body.value, body.note):
        raise HTTPException(404, "trace not found")
    return {"ok": True}


@app.get("/api/info")
async def info():
    m = get_store().manifest
    return {"manifest": m, "llm_available": get_gateway().available("generation")}


@app.get("/healthz")
async def healthz():
    store = get_store()
    return {"status": "ok", "chunks": len(store.chunk_ids), "properties": len(store.prop_ids),
            "llm_endpoints": len(get_gateway().chains["generation"])}


@app.get("/api/admin/metrics", dependencies=[Depends(require_admin)])
async def admin_metrics(hours: int = 24):
    eval_path = ROOT / "evals" / "report.json"
    return {
        "metrics": await metrics(min(max(hours, 1), 24 * 30)),
        "durable_logs": {"enabled": get_sink().enabled, **get_sink().stats},
        "providers": get_gateway().status(),
        "keys": get_gateway().key_status(),
        "embeddings_in_use": (f"OpenRouter {get_settings().embedder.remote.model}, local MiniLM as backup"
                              if get_store().remote_ready else "local MiniLM only (no OpenRouter index built)"),
        "manifest": get_store().manifest,
        "eval": json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else None,
        "cache": dict(conn().execute("SELECT count(*) entries, COALESCE(sum(hits),0) hits FROM response_cache").fetchone()),
    }


@app.get("/api/admin/traces", dependencies=[Depends(require_admin)])
async def admin_traces(limit: int = 50):
    return await recent(min(max(limit, 1), 200))


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/admin", include_in_schema=False)
async def admin_page():
    return FileResponse(STATIC / "admin.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)
