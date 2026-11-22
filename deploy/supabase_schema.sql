-- Log tables for Supabase. Run in the SQL editor; safe to re-run.
-- RLS is on with no policies, so only the secret key can read or write.

-- one row per chat turn
create table if not exists public.chat_logs (
    id               text primary key,
    created_at       timestamptz not null default now(),
    ts               double precision not null,          -- unix seconds
    session_id       text,
    ip_hash          text,                               -- hashed
    question         text,
    answer           text,
    intent           text,
    source_filter    text,
    filters          jsonb,
    router_mode      text,                               -- llm | rules
    retrieved        jsonb,                              -- ids/titles/scores of retrieved listings & pages
    provider         text,
    model            text,
    attempts         jsonb,                              -- failover attempts
    cache_hit        integer default 0,
    prompt_tokens    integer,
    completion_tokens integer,
    router_ms        real,
    retrieval_ms     real,
    ttft_ms          real,
    llm_ms           real,
    total_ms         real,
    error            text,
    feedback         integer,                            -- 1 = up, -1 = down
    prompt_version   text,
    feedback_note    text
);
create index if not exists chat_logs_ts_idx on public.chat_logs (ts desc);
create index if not exists chat_logs_session_idx on public.chat_logs (session_id);

-- one row per API call
create table if not exists public.api_logs (
    id           bigint generated always as identity primary key,
    created_at   timestamptz not null default now(),
    ts           double precision not null,
    session_id   text,
    method       text,
    path         text,
    status       integer,
    duration_ms  real,
    ip_hash      text,
    user_agent   text
);
create index if not exists api_logs_ts_idx on public.api_logs (ts desc);

-- daily LLM request counts per model
create table if not exists public.provider_usage (
    day               text not null,
    provider          text not null,
    model             text not null,
    requests          integer not null default 0,
    failures          integer not null default 0,
    rate_limited      integer not null default 0,
    prompt_tokens     integer not null default 0,
    completion_tokens integer not null default 0,
    updated_at        timestamptz not null default now(),
    primary key (day, provider, model)
);

alter table public.chat_logs      enable row level security;
alter table public.api_logs       enable row level security;
alter table public.provider_usage enable row level security;

-- added later; harmless on a fresh install
alter table public.chat_logs add column if not exists prompt_version text;
