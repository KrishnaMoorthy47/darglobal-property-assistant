"""Handles one chat turn: cache, route, retrieve, generate, log.

Yields events for the SSE endpoint:
  {"type": "meta",  "route": {...}, "cards": [...], "sources": [...]}
  {"type": "token", "text": "..."}
  {"type": "done",  "trace_id": "...", "provider": "...", "model": "...", "cached": bool}
"""
from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import AsyncIterator

from app.config import get_settings
from app.llm import cache
from app.llm.gateway import AllProvidersUnavailable, StreamResult, get_gateway
from app.rag.prompts import build_messages
from app.rag.router import Route, route
from app.rag.store import Filters, get_store
from app.resources import get_prompts, get_reference, render
from app.telemetry import Trace

log = logging.getLogger("chat")


def _price_label(p: dict) -> str | None:
    if not p.get("price"):
        return None
    v = p["price"]
    s = f"{v / 1e6:.2f}M".replace(".00M", "M") if v >= 1e6 else f"{v:,.0f}"
    freq = p.get("rent_frequency")
    per = {"yearly": " / year", "monthly": " / month", "unverified": " (period unclear)"}.get(freq or "", "")
    return f"{p.get('currency') or ''} {s}{per}".strip()


def to_card(p: dict) -> dict:
    beds = None
    if p.get("bedrooms"):
        b, bm = int(p["bedrooms"]), int(p["bedrooms_max"] or 0)
        beds = f"{b}-{bm}" if bm and bm != b else f"{b}"
    return {
        "id": p["id"], "source": p["source"], "name": p["name"], "url": p["url"], "image": p.get("image"),
        "price": _price_label(p), "purpose": p.get("purpose"), "type": p.get("property_type"),
        "location": ", ".join(x for x in [p.get("district"), p.get("city"), p.get("country")] if x),
        "beds": beds, "area_sqm": p.get("area_sqm"), "status": p.get("status"), "brand": p.get("brand"),
    }


def retrieve(r: Route, question: str, ui_source: str | None, qvec) -> dict:
    """Returns properties, text chunks and stats for the given intent."""
    c, store = get_settings().retrieval, get_store()
    f = r.to_filters(ui_source)
    q = r.search_query or question
    props, chunks, stats, relaxed, total = [], [], None, [], 0
    # "which DarGlobal projects are by Trump / in Oman": answer from the full list, not the top few
    list_mode = (r.intent in ("property_search", "knowledge") and not r.project_names and f.source != "wasalt"
                 and (f.brand or (f.source == "darglobal" and (f.city or f.country or f.property_type))))
    if list_mode:
        res = store.search_properties(Filters(**{**f.__dict__, "source": "darglobal"}), q, qvec, k=c.list_property_k)
        seen_names, props = set(), []
        for p in res["items"]:
            if p["name"].lower() not in seen_names:
                seen_names.add(p["name"].lower()); props.append(p)
        chunks = store.search_chunks(q, qvec, source="darglobal", k=c.chunks_with_property_search)
        return {"properties": props, "chunks": chunks, "stats": None, "relaxed": res["relaxed"],
                "total_matches": len(props), "filters": f.active(),
                "full_list": not res["relaxed"] and res["total_matches"] <= c.list_property_k}

    if r.intent in ("property_search", "market_stats"):
        res = store.search_properties(f, q, qvec, k=c.top_k_properties)
        props, stats, relaxed, total = res["items"], res["stats"], res["relaxed"], res["total_matches"]
        chunks = store.search_chunks(q, qvec, source=f.source, k=c.chunks_with_property_search)
    elif r.intent == "compare":
        # named projects are DarGlobal's; skip them when the user limited the search to Wasalt
        props = store.find_projects(r.project_names) if r.project_names and f.source != "wasalt" else []
        if len(props) < 2:
            props += store.search_properties(f, q, qvec, k=c.compare_property_k)["items"]
        chunks = store.search_chunks(q, qvec, source=f.source, k=c.top_k_chunks + 2)
    else:
        named = store.find_projects(r.project_names) if r.project_names and f.source != "wasalt" else []
        if named:
            # a named project: its own pages plus general company info (FAQ, payments), not other projects
            chunks = store.search_chunks(q, qvec, urls=[p["url"] for p in named], k=c.top_k_chunks)
            chunks += store.search_chunks(q, qvec, source="darglobal", doc_types=["company"], k=2)
        else:
            chunks = store.search_chunks(q, qvec, source=f.source, k=c.top_k_chunks)
        props = named or ([] if f.source == "wasalt" else store.search_properties(
            f if f.source else type(f)(source="darglobal"), q, qvec, k=c.knowledge_property_k)["items"])
    seen, uniq = set(), []
    for p in props:
        if p["id"] not in seen:
            seen.add(p["id"]); uniq.append(p)
    return {"properties": uniq[:c.max_properties_in_context], "chunks": chunks, "stats": stats, "relaxed": relaxed,
            "total_matches": total, "filters": f.active()}


def contact_reply(question: str, ui_source: str | None) -> str:
    pr, ref, store = get_prompts(), get_reference(), get_store()
    dg = {"dg_phone": ref.darglobal_contact.phone, "dg_website": ref.darglobal_contact.website}
    q = question.lower()
    project = next((p for p in store.find_projects([question]) if p["source"] == "darglobal"
                    and p["name"].lower() in q), None)
    if project or ui_source == "darglobal" or re.search(r"\bdar ?global\b", q):
        link = render(pr.contact_link, url=project["url"]).replace("Listing page", "Project page") if project else ""
        return render(pr.darglobal_contact_reply, link=link, **dg)
    m = re.search(r"\b(\d{6,8})\b", question)  # Wasalt listing id
    row = store.db.execute("SELECT url FROM properties WHERE id=?", (f"ws-{m.group(1)}",)).fetchone() if m else None
    return render(pr.contact_reply, link=render(pr.contact_link, url=row[0]) if row else "", **dg)


def retrieval_only_answer(ctx: dict) -> str:
    """Answer built from search results alone, for when no LLM is available."""
    lines = [get_prompts().retrieval_only_header, ""]
    for p in ctx["properties"][:get_settings().retrieval.max_cards]:
        c = to_card(p)
        lines.append(f"- **[{c['name']}]({c['url']})** - {c['location']}" + (f" - {c['price']}" if c["price"] else ""))
    for st in ctx["stats"] or []:
        per = " per year" if st["purpose"] == "rent" else ""
        what = f"{st['type'].capitalize()}s for {st['purpose']}" if st.get("type") else f"For {st['purpose']}"
        lines.append(f"\n{what} ({st['count']} listings): median {st['median']:,.0f} {st['currency']}{per} "
                     f"(range {st['min']:,.0f}-{st['max']:,.0f}).")
    for c in ctx["chunks"][:get_settings().retrieval.chunks_with_property_search]:
        lines.append(f"- [{c['title']}]({c['url']}): {c['text'].split(': ', 1)[-1][:220]}...")
    return "\n".join(lines)


async def run_chat(question: str, history: list[dict], session_id: str, ui_source: str | None,
                   ip_hash: str, include_context: bool = False) -> AsyncIterator[dict]:
    s = get_settings()
    t0 = time.perf_counter()
    tr = Trace(id=uuid.uuid4().hex[:16], session_id=session_id, ip_hash=ip_hash, question=question,
               source_filter=ui_source or "any", prompt_version=get_prompts().version)
    history = history[-s.guardrails.max_history_turns * 2:]
    store = get_store()
    try:
        qvec = store.embed(question)  # local model: cache matching only, costs no OpenRouter request

        if not history:
            hit = cache.lookup(question, ui_source, qvec)
            if hit:
                tr.cache_hit, tr.provider, tr.model, tr.intent = 1, hit["provider"], hit["model"], "cached"
                tr.answer = hit["answer"]
                yield {"type": "meta", **hit["payload"], "cached": hit["kind"]}
                yield {"type": "token", "text": hit["answer"]}
                tr.total_ms = (time.perf_counter() - t0) * 1000
                yield {"type": "done", "trace_id": tr.id, "provider": hit["provider"], "model": hit["model"], "cached": True}
                return

        # owners' and agents' numbers are never given out; DarGlobal's own sales contact is
        if get_reference().compiled("contact_request").search(question):
            text = contact_reply(question, ui_source)
            tr.intent, tr.router_mode, tr.answer = "contact_request", "rules", text
            yield {"type": "meta", "route": {"intent": "contact_request"}, "cards": [], "sources": []}
            yield {"type": "token", "text": text}
            tr.total_ms = (time.perf_counter() - t0) * 1000
            yield {"type": "done", "trace_id": tr.id, "provider": "rules", "model": "none", "cached": False}
            return

        tr0 = time.perf_counter()
        r = await route(question, history)
        tr.router_ms = (time.perf_counter() - tr0) * 1000
        tr.intent, tr.router_mode = r.intent, r.mode

        if r.intent in ("greeting", "out_of_scope"):
            text = get_prompts().greeting_reply if r.intent == "greeting" else get_prompts().out_of_scope_reply
            tr.answer = text
            yield {"type": "meta", "route": r.model_dump(), "cards": [], "sources": []}
            yield {"type": "token", "text": text}
            tr.total_ms = (time.perf_counter() - t0) * 1000
            yield {"type": "done", "trace_id": tr.id, "provider": "rules", "model": "none", "cached": False}
            return

        rt0 = time.perf_counter()
        sq = r.search_query or question
        svec = await store.embed_query(sq)
        ctx = retrieve(r, question, ui_source, svec)
        log.info("retrieval used the %s embedding index", svec.space)
        tr.retrieval_ms = (time.perf_counter() - rt0) * 1000
        tr.filters = ctx["filters"]
        tr.retrieved = ([{"type": "property", "id": p["id"], "title": p["name"], "score": p.get("score")} for p in ctx["properties"]]
                        + [{"type": "chunk", "id": c["id"], "title": c["title"], "score": c["score"], "url": c["url"]} for c in ctx["chunks"]])

        messages, sources = build_messages(question, history, r, ctx, store.manifest)
        cards = [to_card(p) for p in ctx["properties"][:s.retrieval.max_cards]]
        payload = {"route": r.model_dump(), "cards": cards, "sources": sources,
                   "stats": ctx["stats"], "relaxed": ctx["relaxed"], "total_matches": ctx["total_matches"]}
        # include_context is used by evals/run.py
        yield {"type": "meta", **payload, **({"context": messages[-1]["content"]} if include_context else {})}

        res = StreamResult()
        lt0 = time.perf_counter()
        parts: list[str] = []
        try:
            async for tok in get_gateway().stream("generation", messages, res):
                parts.append(tok)
                yield {"type": "token", "text": tok}
        except AllProvidersUnavailable as e:
            res.attempts = e.attempts
            fallback = retrieval_only_answer(ctx)
            parts = [fallback]
            res.provider, res.model = "retrieval-only", "none"
            yield {"type": "token", "text": fallback}
        tr.llm_ms = (time.perf_counter() - lt0) * 1000
        tr.ttft_ms = res.ttft_ms
        tr.provider, tr.model, tr.attempts = res.provider, res.model, res.attempts
        tr.prompt_tokens, tr.completion_tokens = res.prompt_tokens, res.completion_tokens
        tr.answer = "".join(parts)

        if not history and res.provider not in ("", "retrieval-only") and "interrupted" not in tr.answer:
            cache.store(question, ui_source, qvec, tr.answer, payload, res.provider, res.model)
        tr.total_ms = (time.perf_counter() - t0) * 1000
        yield {"type": "done", "trace_id": tr.id, "provider": res.provider, "model": res.model, "cached": False}
    except Exception as e:
        log.exception("chat failed")
        tr.error = f"{type(e).__name__}: {e}"[:500]
        yield {"type": "error", "message": "Something went wrong on our side. Please try again."}
    finally:
        if not tr.total_ms:
            tr.total_ms = (time.perf_counter() - t0) * 1000
        try:
            tr.save()
        except Exception:
            log.exception("trace save failed")
