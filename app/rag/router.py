"""Works out the intent and filters of a question, using an LLM with a rule-based fallback."""
from __future__ import annotations

import json
import logging
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from app.config import get_settings
from app.llm.gateway import AllProvidersUnavailable, get_gateway
from app.rag.store import Filters, get_store
from app.resources import get_prompts, get_reference, render

log = logging.getLogger("router")

Intent = Literal["property_search", "knowledge", "compare", "market_stats", "greeting", "out_of_scope"]
INTENTS = set(Intent.__args__)


class RouterFilters(BaseModel):
    city: str | None = None
    country: str | None = None
    property_type: str | None = None
    purpose: Literal["sale", "rent"] | None = None
    min_bedrooms: float | None = None
    max_bedrooms: float | None = None
    min_price: float | None = None
    max_price: float | None = None
    currency: str | None = None
    brand: str | None = None

    @field_validator("*", mode="before")
    @classmethod
    def _nullish(cls, v):
        if v in ("", "null", "none", "any", "None", []):
            return None
        if isinstance(v, list):  # several values ("Dubai or Riyadh"); the store splits them again
            v = ", ".join(str(x) for x in v if x not in (None, ""))
        return v.lower() if isinstance(v, str) and v.lower() in ("sale", "rent") else v


class Route(BaseModel):
    intent: Intent = "knowledge"
    source: Literal["darglobal", "wasalt", "any"] = "any"
    filters: RouterFilters = Field(default_factory=RouterFilters)
    project_names: list[str] = Field(default_factory=list)
    search_query: str = ""
    language: Literal["en", "ar"] = "en"
    mode: str = "llm"  # llm | rules

    @model_validator(mode="before")
    @classmethod
    def _tolerate_llm_shapes(cls, d):
        """Fix up common LLM output quirks (null instead of {}, slightly wrong labels)."""
        if not isinstance(d, dict):
            return d
        d = dict(d)
        d["filters"] = d.get("filters") or {}
        d["project_names"] = [p for p in (d.get("project_names") or []) if isinstance(p, str)]
        d["search_query"] = d.get("search_query") or ""
        if d.get("intent") not in INTENTS:
            d["intent"] = "knowledge"
        if d.get("source") not in ("darglobal", "wasalt", "any"):
            d["source"] = "any"
        if d.get("language") not in ("en", "ar"):
            d["language"] = "en"
        if isinstance(d["filters"], dict) and str(d["filters"].get("purpose", "")).lower() not in ("sale", "rent"):
            d["filters"] = {**d["filters"], "purpose": None}
        return d

    def to_filters(self, ui_source: str | None) -> Filters:
        src = ui_source if ui_source in ("darglobal", "wasalt") else (self.source if self.source != "any" else None)
        return Filters(source=src, **self.filters.model_dump())


def _money(num: str, unit: str | None) -> float:
    return float(num.replace(",", "")) * get_reference().rule_router.number_units.get((unit or "").lower(), 1)


def is_greeting(q: str) -> bool:
    return bool(get_reference().compiled("greeting").match(q))


def is_injection(q: str) -> bool:
    return bool(get_reference().compiled("injection").search(q))


def rule_route(q: str) -> Route:
    """Keyword/regex parser. Also the baseline in evals."""
    store, rr = get_store(), get_reference().rule_router
    ql = q.lower()
    is_ar = bool(re.search(r"[؀-ۿ]", q))
    r = Route(mode="rules", search_query=q, language="ar" if is_ar else "en")
    if is_greeting(q):
        r.intent = "greeting"
        return r
    if is_injection(q):
        r.intent = "out_of_scope"
        return r

    f = r.filters
    for city in store.cities:
        if re.search(rf"\b{re.escape(city.lower())}\b", ql):
            f.city = city
            break
    for country, keys in rr.countries.items():
        if any(re.search(rf"\b{re.escape(k)}\b", ql) for k in keys):
            f.country = country
    for t, keys in rr.property_types.items():
        if any(k in ql for k in keys):
            f.property_type = t
            break
    if re.search(rr.rent_pattern, ql):
        f.purpose = "rent"
    elif re.search(rr.sale_pattern, ql):
        f.purpose = "sale"
    if m := re.search(rr.bedrooms_pattern, ql):
        f.min_bedrooms = float(m.group(1))
    units = "|".join(sorted((re.escape(u) for u in rr.number_units), key=len, reverse=True))
    num = rf"(\d+(?:[.,]\d+)?)\s*({units})?"
    cur_words = r"(?:aed|sar|usd|\$|£|€)?"
    if m := re.search(rf"{rr.max_price_words}\s*{cur_words}\s*{num}", ql):
        f.max_price = _money(m.group(1), m.group(2))
    if m := re.search(rf"{rr.min_price_words}\s*{cur_words}\s*{num}", ql):
        f.min_price = _money(m.group(1), m.group(2))
    for cur, keys in rr.currencies.items():
        if any(k in ql for k in keys):
            f.currency = cur
    for b in store.brands:
        if b.lower() in ql:
            f.brand = b
    r.project_names = [n for n in store.project_names if len(n) > 4 and n.lower().split(",")[0] in ql][:3]

    if "darglobal" in ql or "dar global" in ql or f.brand or r.project_names:
        r.source = "darglobal"
    elif "wasalt" in ql or (f.city and f.city in rr.wasalt_default_cities):
        r.source = "wasalt"

    if re.search(rr.compare_pattern, ql):
        r.intent = "compare"
    elif re.search(rr.stats_pattern, ql):
        r.intent = "market_stats"
    elif any([f.min_bedrooms, f.max_price, f.min_price, f.property_type, f.purpose]) or re.search(rr.search_pattern, ql):
        r.intent = "property_search"
    else:
        r.intent = "knowledge"
    return r


async def route(question: str, history: list[dict]) -> Route:
    if is_greeting(question):
        return Route(intent="greeting", mode="rules", search_query=question)
    gw = get_gateway()
    if not gw.available("router"):
        return rule_route(question)
    store, rc = get_store(), get_settings().llm.router
    hist = "\n".join(f"{m['role']}: {m['content'][:rc.history_chars]}" for m in history[-rc.history_turns:])
    system = render(get_prompts().router_system, cities=", ".join(store.cities[:rc.max_cities_in_prompt]))
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": (f"Chat history:\n{hist}\n\n" if hist else "") + f"Question: {question}"},
    ]
    try:
        raw, _ = await gw.complete_json("router", messages)
        raw = raw[raw.find("{"): raw.rfind("}") + 1]
        r = Route.model_validate(json.loads(raw))
        r.search_query = r.search_query or question
        if is_injection(question):
            r.intent = "out_of_scope"
        return r
    except (AllProvidersUnavailable, json.JSONDecodeError, ValidationError, ValueError) as e:
        log.warning("router fallback to rules: %s", type(e).__name__)
        return rule_route(question)  # LLM failed or returned bad JSON
