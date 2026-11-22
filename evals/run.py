"""Evals. Writes evals/report.json and evals/report.md.

  python -m evals.run                 # retrieval + rule router (no API calls, runs in CI)
  python -m evals.run --llm-router    # also the LLM router
  python -m evals.run --answers       # full answers graded by an LLM judge
  python -m evals.run --remote-embeddings   # retrieval with the OpenRouter embedding index too
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from app.config import get_settings
from app.llm.gateway import AllProvidersUnavailable
from app.rag.router import route, rule_route
from app.rag.store import QVec, get_store

HERE = Path(__file__).parent


def load_cases() -> list[dict]:
    return [json.loads(line) for line in (HERE / "golden.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


async def eval_retrieval(cases: list[dict], remote: bool = False) -> dict:
    """remote=True uses the OpenRouter embedding index (all questions in one API call)."""
    store = get_store()
    if remote:
        if not store.remote_ready:
            raise SystemExit("no OpenRouter embedding index; build it with: python -m pipeline.build --remote-embed")
        from app.llm.gateway import get_gateway
        prefix = get_settings().embedder.remote.query_prefix
        vecs, _ = await get_gateway().embed([prefix + c["question"].strip().lower() for c in cases])
    hits, rr, rows = 0, [], []
    for i, c in enumerate(cases):
        qv = QVec(vecs[i], "remote") if remote else store.embed(c["question"])
        res = store.search_chunks(c["question"], qv, source=c.get("source"), k=5)
        rank = next((i + 1 for i, h in enumerate(res) if any(e.lower() in h["url"].lower() for e in c["expect_url"])), None)
        hits += rank is not None
        rr.append(1 / rank if rank else 0)
        rows.append({"id": c["id"], "question": c["question"], "rank": rank, "top": [h["url"] for h in res[:3]]})
    return {"hit_at_5": round(hits / len(cases), 3), "mrr": round(statistics.mean(rr), 3), "rows": rows}


FILTER_KEYS = ["city", "property_type", "purpose", "min_bedrooms", "max_price"]


def _router_score(r, exp: dict) -> tuple[bool, float]:
    intent_ok = r.intent == exp["intent"]
    if exp.get("source"):
        intent_ok = intent_ok and r.source == exp["source"]
    keys = [k for k in FILTER_KEYS if k in exp]
    if not keys:
        return intent_ok, 1.0
    ok = 0
    for k in keys:
        got = getattr(r.filters, k)
        want = exp[k]
        if isinstance(want, str):
            ok += bool(got) and want.lower() in str(got).lower()
        else:
            ok += got is not None and abs(float(got) - float(want)) < 1e-6
    return intent_ok, ok / len(keys)


async def eval_router(cases: list[dict], llm: bool) -> dict:
    out = {}
    for mode in (["rules", "llm"] if llm else ["rules"]):
        intents, filt, rows = [], [], []
        for c in cases:
            r = rule_route(c["question"]) if mode == "rules" else await route(c["question"], [])
            i_ok, f_score = _router_score(r, c["expect"])
            intents.append(i_ok); filt.append(f_score)
            rows.append({"id": c["id"], "intent": r.intent, "ok": i_ok, "filter_score": round(f_score, 2), "mode": r.mode})
            if mode == "llm":
                await asyncio.sleep(2.5)  # rate limit
        out[mode] = {"intent_accuracy": round(sum(intents) / len(cases), 3),
                     "filter_accuracy": round(statistics.mean(filt), 3), "rows": rows}
    return out


async def eval_answers(cases: list[dict]) -> dict:
    from openai import AsyncOpenAI

    from app.config import get_settings
    from app.rag.chat import run_chat
    from app.resources import get_prompts, render

    s = get_settings()
    jc = s.evals.judge
    key = {"groq": s.groq_api_key, "openrouter": s.openrouter_api_key}[jc.provider]
    judge = AsyncOpenAI(api_key=key, base_url=s.llm.base_urls[jc.provider], max_retries=2)
    rows = []
    for c in cases:
        t0 = time.perf_counter()
        answer, meta, done = "", {}, {}
        async for ev in run_chat(c["question"], [], "eval", None, "eval", include_context=True):
            if ev["type"] == "meta":
                meta = ev
            elif ev["type"] == "token":
                answer += ev["text"]
            elif ev["type"] == "done":
                done = ev
        latency = (time.perf_counter() - t0) * 1000
        ctx = (meta.get("context") or "")[:12000]
        if not meta.get("sources"):
            ctx = "(no sources: deterministic reply)"
        try:
            r = await judge.chat.completions.create(
                model=jc.model, reasoning_effort=jc.reasoning_effort, temperature=0, max_tokens=jc.max_tokens,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": render(get_prompts().judge, q=c["question"], ref=c["reference"],
                                                             ctx=ctx, ans=answer[:3000])}])
            verdict = json.loads(r.choices[0].message.content)
        except Exception as e:  # noqa: BLE001
            verdict = {"correct": None, "faithful": None, "refused_properly": None, "reason": f"judge error: {e}"}
        rows.append({"id": c["id"], "question": c["question"], "model": done.get("model"), "cached": done.get("cached"),
                     "latency_ms": round(latency), "answer": answer[:600], **verdict})
        await asyncio.sleep(s.evals.pause_s)

    def rate(key, subset=None):
        vals = [r[key] for r in (subset or rows) if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 3) if vals else None

    refusal_rows = [r for r, c in zip(rows, cases) if c.get("must_refuse")]
    return {"correctness": rate("correct"), "faithfulness": rate("faithful"),
            "refusal_accuracy": rate("refused_properly", refusal_rows),
            "p50_latency_ms": round(statistics.median(r["latency_ms"] for r in rows)), "rows": rows}


async def main(llm_router: bool, answers: bool, remote_embeddings: bool = False) -> None:
    cases = load_cases()
    by_type = {t: [c for c in cases if c["type"] == t] for t in ("retrieval", "router", "answer")}
    report = {"run_at": datetime.now(UTC).isoformat(timespec="seconds"), "cases": len(cases),
              "mode": "+".join(["retrieval", "router-rules"] + (["router-llm"] if llm_router else []) + (["answers"] if answers else []))}
    prev = json.loads((HERE / "report.json").read_text(encoding="utf-8")) if (HERE / "report.json").exists() else {}

    report["retrieval"] = await eval_retrieval(by_type["retrieval"])
    if remote_embeddings:
        try:
            report["retrieval_remote"] = await eval_retrieval(by_type["retrieval"], remote=True)
        except AllProvidersUnavailable:
            print("OpenRouter embeddings unavailable (daily limit?); keeping the previous OpenRouter retrieval score")
    if "retrieval_remote" not in report and prev.get("retrieval_remote"):
        report["retrieval_remote"] = prev["retrieval_remote"]
    report["router"] = await eval_router(by_type["router"], llm_router)
    if not llm_router and "llm" in prev.get("router", {}):
        report["router"]["llm"] = prev["router"]["llm"]
    if answers:
        report["answers"] = await eval_answers(by_type["answer"])
    elif prev.get("answers"):
        report["answers"] = prev["answers"]

    summary = {
        "retrieval_hit_at_5": report["retrieval"]["hit_at_5"],
        "retrieval_mrr": report["retrieval"]["mrr"],
        **({"retrieval_openrouter_hit_at_5": report["retrieval_remote"]["hit_at_5"],
            "retrieval_openrouter_mrr": report["retrieval_remote"]["mrr"]} if report.get("retrieval_remote") else {}),
        "router_rules_intent_accuracy": report["router"]["rules"]["intent_accuracy"],
        "router_rules_filter_accuracy": report["router"]["rules"]["filter_accuracy"],
    }
    if "llm" in report["router"]:
        summary["router_llm_intent_accuracy"] = report["router"]["llm"]["intent_accuracy"]
        summary["router_llm_filter_accuracy"] = report["router"]["llm"]["filter_accuracy"]
    if report.get("answers"):
        a = report["answers"]
        summary.update({"answer_correctness": a["correctness"], "answer_faithfulness": a["faithfulness"],
                        "refusal_accuracy": a["refusal_accuracy"], "answer_p50_latency_ms": a["p50_latency_ms"]})
    report["summary"] = summary
    (HERE / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# Evaluation report", f"Run: {report['run_at']} · mode: {report['mode']}", "", "| Metric | Value |", "|---|---|"]
    md += [f"| {k.replace('_', ' ')} | {v} |" for k, v in summary.items()]
    md += ["", "## Retrieval misses"] + [f"- `{r['id']}` {r['question']} -> top: {r['top'][:2]}"
                                         for r in report["retrieval"]["rows"] if not r["rank"]] or ["- none"]
    (HERE / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-router", action="store_true")
    ap.add_argument("--answers", action="store_true")
    ap.add_argument("--remote-embeddings", action="store_true", help="also score the OpenRouter embedding index")
    a = ap.parse_args()
    asyncio.run(main(a.llm_router, a.answers, a.remote_embeddings))
