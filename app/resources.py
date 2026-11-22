"""Loads config/prompts.yaml and config/reference.yaml."""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache

import yaml
from pydantic import BaseModel

from app.config import CONFIG_DIR


class _Strict(BaseModel):
    model_config = {"extra": "forbid"}


class ContextNotes(_Strict):
    stats: str
    stats_label_sale: str
    stats_label_rent: str
    relaxed: str
    filters: str
    empty_context: str
    both_purposes: str
    full_list: str


class Prompts(_Strict):
    version: str
    answer_system: str
    context_notes: ContextNotes
    router_system: str
    greeting_reply: str
    contact_reply: str
    contact_link: str
    darglobal_contact_reply: str
    darglobal_contact_note: str
    out_of_scope_reply: str
    retrieval_only_header: str
    interrupted_notice: str
    enrichment_system: str
    judge: str


class Patterns(_Strict):
    greeting: str
    injection: str
    contact_request: str
    phone: str
    email: str


class RuleRouter(_Strict):
    countries: dict[str, list[str]]
    property_types: dict[str, list[str]]
    currencies: dict[str, list[str]]
    rent_pattern: str
    sale_pattern: str
    bedrooms_pattern: str
    max_price_words: str
    min_price_words: str
    number_units: dict[str, float]
    compare_pattern: str
    stats_pattern: str
    search_pattern: str
    wasalt_default_cities: list[str]


class PipelineRef(_Strict):
    brands: list[str]
    developer_names_not_brands: list[str]
    country_aliases: dict[str, str]
    city_country: dict[str, str]
    city_display_overrides: dict[str, str]
    city_normalize: dict[str, str]
    boilerplate: list[str]


class CompanyContact(_Strict):
    phone: str
    website: str


class Reference(_Strict):
    darglobal_contact: CompanyContact
    currency_usd_rates: dict[str, float]
    default_currency: dict[str, str]
    patterns: Patterns
    stopwords: list[str]
    rule_router: RuleRouter
    pipeline: PipelineRef

    def compiled(self, name: str) -> re.Pattern:
        return _compile(getattr(self.patterns, name))


@lru_cache(maxsize=64)
def _compile(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


def _load(name: str) -> dict:
    with (CONFIG_DIR / name).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache
def get_prompts() -> Prompts:
    # the version is a fingerprint of the file: it changes whenever the wording does, which clears
    # cached answers and is logged with each chat
    raw = (CONFIG_DIR / "prompts.yaml").read_bytes()
    return Prompts.model_validate({**_load("prompts.yaml"), "version": hashlib.sha1(raw).hexdigest()[:10]})


@lru_cache
def get_reference() -> Reference:
    ref = Reference.model_validate(_load("reference.yaml"))
    for name in ("greeting", "injection", "contact_request", "phone", "email"):
        ref.compiled(name)  # surface bad regexes at startup
    return ref


def redact_contacts(text: str | None) -> str:
    """Remove phone numbers and e-mail addresses from scraped text."""
    ref = get_reference()
    text = ref.compiled("phone").sub("[number removed]", text or "")
    return ref.compiled("email").sub("[email removed]", text)


def render(template: str, **values) -> str:
    """Replace {name} placeholders. Plain str.replace so JSON braces in prompts are left alone."""
    for k, v in values.items():
        template = template.replace("{" + k + "}", str(v))
    return template
