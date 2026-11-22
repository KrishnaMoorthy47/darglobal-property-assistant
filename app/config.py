"""Settings from config/settings.yaml, overridable by env vars (LLM__GENERATION__MAX_TOKENS=900).

API keys only come from the environment / .env. There are no defaults in code, so a missing key in the
YAML is a startup error.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict, YamlConfigSettingsSource

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"


class _Strict(BaseModel):
    model_config = {"extra": "forbid"}  # catch typos in YAML keys


class ChainEntry(_Strict):
    provider: Literal["openrouter", "openrouter_2", "openrouter_3", "groq"]
    model: str
    daily_cap: int = Field(gt=0)
    rpm: int = Field(gt=0)


class GenerationCfg(_Strict):
    temperature: float
    max_tokens: int


class RouterCfg(_Strict):
    temperature: float
    max_tokens: int
    history_turns: int
    history_chars: int
    max_cities_in_prompt: int


class CircuitBreakerCfg(_Strict):
    default_retry_after_s: float
    rate_limit_min_s: float
    rate_limit_max_s: float
    auth_error_s: float
    model_not_found_s: float
    transient_failures_to_trip: int
    transient_cooldown_s: float


class LlmCfg(_Strict):
    generation_chain: list[ChainEntry]
    router_chain: list[ChainEntry]
    embedding_chain: list[ChainEntry]
    key_daily_caps: dict[str, int]
    base_urls: dict[str, str]
    quota_safety_ratio: float = Field(gt=0, le=1)
    timeout_s: float
    generation: GenerationCfg
    router: RouterCfg
    request_extras: dict[str, dict[str, Any]]
    circuit_breaker: CircuitBreakerCfg


class RetrievalCfg(_Strict):
    top_k_chunks: int
    top_k_properties: int
    candidate_pool: int
    rrf_k: int
    max_chunks_per_page: int
    bm25_title_weight: float
    bm25_text_weight: float
    property_fts_bonus: float
    property_fts_limit: int
    property_candidate_limit: int
    relax_order: list[str]
    stats_min_count: int
    chunks_with_property_search: int
    compare_property_k: int
    knowledge_property_k: int
    max_properties_in_context: int
    list_property_k: int
    max_cards: int
    property_details_chars: int
    max_fts_terms: int


class DataQualityCfg(_Strict):
    min_plausible_annual_rent_sar: float


class CacheCfg(_Strict):
    semantic_threshold: float
    ttl_hours: float
    scan_limit: int


class GuardrailsCfg(_Strict):
    max_input_chars: int
    max_history_turns: int
    history_message_chars: int
    user_rpm: int
    user_daily: int


class StorageCfg(_Strict):
    queue_max: int
    write_retries: int
    timeout_s: float


class RemoteEmbedCfg(_Strict):
    enabled: bool
    model: str
    query_prefix: str
    doc_prefix: str
    batch_size: int
    pause_s: float


class EmbedderCfg(_Strict):
    remote: RemoteEmbedCfg
    hf_repo: str
    display_name: str
    model_dir: Path
    max_tokens: int
    max_piece_len: int
    threads: int
    batch_size: int

    @property
    def path(self) -> Path:
        return self.model_dir if self.model_dir.is_absolute() else ROOT / self.model_dir


class EnrichmentCfg(_Strict):
    provider: Literal["openrouter", "groq"]
    model: str
    max_tokens: int
    max_input_chars: int
    pause_s: float


class PipelineCfg(_Strict):
    chunk_size: int
    chunk_overlap: int
    min_page_chars: int
    enrichment: EnrichmentCfg


class JudgeCfg(_Strict):
    provider: Literal["openrouter", "groq"]
    model: str
    reasoning_effort: str
    max_tokens: int


class EvalsCfg(_Strict):
    judge: JudgeCfg
    pause_s: float


class PathsCfg(_Strict):
    data_dir: Path
    runtime_db: Path


class OpsCfg(_Strict):
    app_url: str
    allowed_origins: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore",
        env_nested_delimiter="__", yaml_file=CONFIG_DIR / "settings.yaml", yaml_file_encoding="utf-8",
    )

    # from env only; empty turns the feature off
    openrouter_api_key: str = ""
    openrouter_api_key_2: str = ""   # optional second OpenRouter account
    openrouter_api_key_3: str = ""   # optional, for embeddings + question understanding
    groq_api_key: str = ""
    admin_token: str = ""
    supabase_url: str = ""
    supabase_secret_key: str = ""    # secret key, not the anon key
    app_url: str = ""                # overrides ops.app_url


    llm: LlmCfg
    retrieval: RetrievalCfg
    data_quality: DataQualityCfg
    cache: CacheCfg
    guardrails: GuardrailsCfg
    storage: StorageCfg
    embedder: EmbedderCfg
    pipeline: PipelineCfg
    evals: EvalsCfg
    paths: PathsCfg
    ops: OpsCfg

    @model_validator(mode="after")
    def _embedding_chain_matches_index(self):
        # query vectors must come from the model the index was built with
        wrong = {e.model for e in self.llm.embedding_chain} - {self.embedder.remote.model}
        if wrong:
            raise ValueError(f"llm.embedding_chain models {wrong} differ from embedder.remote.model")
        return self

    @classmethod
    def settings_customise_sources(cls, settings_cls: type[BaseSettings], init_settings: PydanticBaseSettingsSource,
                                   env_settings: PydanticBaseSettingsSource, dotenv_settings: PydanticBaseSettingsSource,
                                   file_secret_settings: PydanticBaseSettingsSource):
        # env vars win over .env, which wins over the YAML
        return init_settings, env_settings, dotenv_settings, YamlConfigSettingsSource(settings_cls)

    @property
    def data_dir(self) -> Path:
        return self.paths.data_dir if self.paths.data_dir.is_absolute() else ROOT / self.paths.data_dir

    @property
    def runtime_db(self) -> Path:
        return self.paths.runtime_db if self.paths.runtime_db.is_absolute() else ROOT / self.paths.runtime_db

    @property
    def public_url(self) -> str:
        return self.app_url or self.ops.app_url


@lru_cache
def get_settings() -> Settings:
    return Settings()
