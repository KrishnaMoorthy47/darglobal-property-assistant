"""Builds the chat messages. The wording is in config/prompts.yaml; context blocks are numbered for [n] citations."""
from __future__ import annotations

from app.config import get_settings
from app.rag.router import Route
from app.resources import get_prompts, get_reference, render


def _fmt_prop(p: dict) -> str:
    fields = [
        ("source", p["source"]), ("name", p["name"]), ("type", p.get("property_type")), ("purpose", p.get("purpose")),
        ("location", ", ".join(x for x in [p.get("district"), p.get("city"), p.get("country")] if x)),
        ("price", f"{p['price']:,.0f} {p.get('currency') or ''}" + {"yearly": " per year", "monthly": " per month", "unverified": " (rent period unclear at source)"}.get(p.get("rent_frequency") or "", "")
         if p.get("price") else "not published"),
        ("bedrooms", (f"{p['bedrooms']:.0f}" + (f"-{p['bedrooms_max']:.0f}" if p.get("bedrooms_max") and p["bedrooms_max"] != p["bedrooms"] else ""))
         if p.get("bedrooms") else None),
        ("area_sqm", p.get("area_sqm")), ("status", p.get("status")), ("brand", p.get("brand")),
        ("handover", p.get("handover")), ("payment_plan", p.get("payment_plan")),
        ("details", (p.get("description") or "")[:get_settings().retrieval.property_details_chars]), ("url", p["url"]),
    ]
    return "; ".join(f"{k}: {v}" for k, v in fields if v not in (None, ""))


def build_messages(question: str, history: list[dict], r: Route, ctx: dict, manifest: dict) -> tuple[list[dict], list[dict]]:
    blocks, sources = [], []
    n = 0
    for p in ctx["properties"]:
        n += 1
        blocks.append(f"[{n}] PROPERTY - {_fmt_prop(p)}")
        sources.append({"n": n, "title": p["name"], "url": p["url"], "source": p["source"], "kind": "property",
                        "snippet": _fmt_prop(p)[:500]})
    for c in ctx["chunks"]:
        n += 1
        blocks.append(f"[{n}] {c['source'].upper()} {c['doc_type']} \"{c['title']}\" ({c['url']}): {c['text']}")
        sources.append({"n": n, "title": c["title"], "url": c["url"], "source": c["source"], "kind": c["doc_type"],
                        "snippet": c["text"][:500]})

    pr = get_prompts()
    cn = pr.context_notes
    notes = []
    for st in ctx.get("stats") or []:
        label = cn.stats_label_sale if st["purpose"] == "sale" else cn.stats_label_rent
        if st.get("type"):
            label = f"{st['type']}s, {label}"
        notes.append(render(cn.stats, label=label, count=st["count"], median=f"{st['median']:,.0f}", currency=st["currency"],
                            min=f"{st['min']:,.0f}", max=f"{st['max']:,.0f}"))
    purposes = {p.get("purpose") for p in ctx.get("properties") or [] if p.get("source") == "wasalt"}
    if len(purposes) > 1 and "purpose" not in (ctx.get("filters") or {}):
        notes.append(cn.both_purposes)
    if ctx.get("full_list"):
        notes.append(render(cn.full_list, count=len(ctx["properties"])))
    if ctx.get("relaxed"):
        notes.append(render(cn.relaxed, relaxed=", ".join(ctx["relaxed"])))
    if ctx.get("filters"):
        notes.append(render(cn.filters, filters=ctx["filters"]))

    sw = manifest.get("scraped_window", {})
    dg = get_reference().darglobal_contact
    note = render(pr.darglobal_contact_note, dg_phone=dg.phone, dg_website=dg.website)
    system = render(pr.answer_system, language="Arabic" if r.language == "ar" else "English", darglobal_contact_note=note,
                    dg_date=sw.get("darglobal", "recently"), ws_date=sw.get("wasalt", "recently"))
    context = "CONTEXT:\n" + ("\n\n".join(blocks) if blocks else cn.empty_context)
    if notes:
        context += "\n\nNOTES:\n" + "\n".join(notes)

    messages = [{"role": "system", "content": system}]
    for m in history:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            messages.append({"role": m["role"], "content": m["content"][:get_settings().guardrails.history_message_chars]})
    messages.append({"role": "user", "content": f"{context}\n\nQUESTION: {question}"})
    return messages, sources
