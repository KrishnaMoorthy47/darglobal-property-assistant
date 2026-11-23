"""Search over the built knowledge base (data/app.db + embeddings)."""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import statistics
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from app.config import get_settings
from app.resources import get_reference

log = logging.getLogger("store")

REMOTE_FILES = {"chunk": "chunk_emb_remote.npy", "prop": "prop_emb_remote.npy"}
LOCAL_FILES = {"chunk": "chunk_emb.npy", "prop": "prop_emb.npy"}


@dataclass(frozen=True)
class QVec:
    """A query embedding plus which index it belongs to ("remote" = OpenRouter model, "local" = MiniLM)."""
    v: np.ndarray
    space: str


def values(v: str | None) -> list[str]:
    """Split a filter value like "Dubai, Saudi Arabia" or "apartment | villa" into its parts."""
    return [p.strip() for p in re.split(r",|\||\bor\b|/", v or "") if p.strip()]


def fts_query(text: str) -> str:
    """Build an FTS5 OR-query; every term is quoted so user input can't inject FTS syntax."""
    stop = _stopwords()
    toks = [t for t in re.findall(r"[\w؀-ۿ]+", text.lower()) if t not in stop and len(t) > 1]
    return " OR ".join(f'"{t}"' for t in toks[:get_settings().retrieval.max_fts_terms])


@lru_cache(maxsize=1)
def _stopwords() -> frozenset[str]:
    return frozenset(get_reference().stopwords)


@dataclass
class Filters:
    source: str | None = None          # darglobal | wasalt
    city: str | None = None
    country: str | None = None
    property_type: str | None = None
    purpose: str | None = None         # sale | rent
    min_bedrooms: float | None = None
    max_bedrooms: float | None = None
    min_price: float | None = None
    max_price: float | None = None
    currency: str | None = None        # of the user's price
    brand: str | None = None

    def active(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v not in (None, "")}


class Store:
    def __init__(self) -> None:
        s = get_settings()
        self.cfg = s.retrieval
        db_path = s.data_dir / "app.db"
        self.db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.chunk_ids = [r[0] for r in self.db.execute("SELECT rowid FROM chunks ORDER BY rowid")]
        self.prop_ids = [r[0] for r in self.db.execute("SELECT rowid FROM properties ORDER BY rowid")]
        mf = s.data_dir / "manifest.json"
        self.manifest = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else {}
        self.emb = {"local": {k: np.load(s.data_dir / f) for k, f in LOCAL_FILES.items()}}
        rcfg = s.embedder.remote
        built_with = (self.manifest.get("embeddings") or {}).get("remote")
        if rcfg.enabled and built_with == rcfg.model and all((s.data_dir / f).exists() for f in REMOTE_FILES.values()):
            self.emb["remote"] = {k: np.load(s.data_dir / f) for k, f in REMOTE_FILES.items()}
        elif rcfg.enabled:
            log.warning("no OpenRouter embedding index for %s (built with %s); using the local model only",
                        rcfg.model, built_with)
        self._remote_cache: dict[str, np.ndarray] = {}
        self.chunk_pos = {rid: i for i, rid in enumerate(self.chunk_ids)}
        self.prop_pos = {rid: i for i, rid in enumerate(self.prop_ids)}
        self.cities = sorted({r[0] for r in self.db.execute("SELECT DISTINCT city FROM properties WHERE city IS NOT NULL")},
                             key=len, reverse=True)
        self.countries = sorted({r[0] for r in self.db.execute("SELECT DISTINCT country FROM properties WHERE country IS NOT NULL")})
        self.types = sorted({r[0] for r in self.db.execute("SELECT DISTINCT property_type FROM properties WHERE source='wasalt' AND property_type IS NOT NULL")})
        self.brands = sorted({r[0] for r in self.db.execute("SELECT DISTINCT brand FROM properties WHERE brand IS NOT NULL")})
        self.project_names = [r[0] for r in self.db.execute("SELECT name FROM properties WHERE source='darglobal'")]
        self._model = None
        log.info("store loaded: %d chunks, %d properties", len(self.chunk_ids), len(self.prop_ids))

    @property
    def model(self):
        if self._model is None:
            from app.rag.embedder import Embedder
            self._model = Embedder(get_settings().embedder)
        return self._model

    @lru_cache(maxsize=2048)  # noqa: B019 - Store is a process-wide singleton
    def _embed_cached(self, text: str) -> bytes:
        return self.model.embed([text])[0].tobytes()

    def embed(self, text: str) -> QVec:
        """Local (MiniLM) embedding."""
        return QVec(np.frombuffer(self._embed_cached(text.strip().lower()[:1000]), dtype=np.float32), "local")

    @property
    def remote_ready(self) -> bool:
        return "remote" in self.emb

    async def embed_query(self, text: str) -> QVec:
        """OpenRouter embedding if its index is loaded and a key is available, otherwise the local model."""
        if not self.remote_ready:
            return self.embed(text)
        key = text.strip().lower()[:1000]
        if key in self._remote_cache:
            return QVec(self._remote_cache[key], "remote")
        from app.llm.gateway import get_gateway
        try:
            v, _ = await get_gateway().embed([get_settings().embedder.remote.query_prefix + key])
        except Exception as e:  # noqa: BLE001
            log.warning("OpenRouter embedding unavailable (%s); using the local model", e)
            return self.embed(text)
        if len(self._remote_cache) > 2000:
            self._remote_cache.clear()
        self._remote_cache[key] = v[0]
        return QVec(v[0], "remote")

    def search_chunks(self, query: str, qvec: QVec, source: str | None = None, k: int | None = None,
                      urls: list[str] | None = None, doc_types: list[str] | None = None) -> list[dict]:
        """Hybrid search over page text. urls / doc_types restrict it to certain pages."""
        c = self.cfg
        k = k or c.top_k_chunks
        pool = c.candidate_pool
        where, params = ("AND c.source = ?", [source]) if source else ("", [])
        for col, vals in (("url", urls), ("doc_type", doc_types)):
            if vals:
                where += f" AND c.{col} IN ({','.join('?' * len(vals))})"
                params += list(vals)
        bm25_ranked: list[int] = []
        q = fts_query(query)
        if q:
            rows = self.db.execute(
                f"""SELECT c.rowid FROM chunks_fts f JOIN chunks c ON c.rowid = f.rowid
                    WHERE chunks_fts MATCH ? {where} ORDER BY bm25(chunks_fts, ?, ?) LIMIT ?""",
                [q, *params, c.bm25_title_weight, c.bm25_text_weight, pool]).fetchall()
            bm25_ranked = [r[0] for r in rows]

        sims = self.emb[qvec.space]["chunk"] @ qvec.v
        if source or urls or doc_types:
            keep = np.array([(not source or s == source) and (not urls or u in urls) and (not doc_types or d in doc_types)
                             for s, u, d in self._chunk_meta()])
            sims = np.where(keep, sims, -1.0)
        top = np.argsort(-sims)[:pool]
        vec_ranked = [self.chunk_ids[i] for i in top]

        # RRF, since BM25 and cosine scores aren't on the same scale
        scores: dict[int, float] = {}
        for ranked in (bm25_ranked, vec_ranked):
            for rank, rid in enumerate(ranked):
                scores[rid] = scores.get(rid, 0) + 1.0 / (c.rrf_k + rank)
        best = sorted(scores, key=scores.get, reverse=True)

        out, per_url = [], {}
        if urls:  # vector ranking covers every chunk; keep only the requested pages
            best = [rid for rid in best if sims[self.chunk_pos[rid]] > -1.0]
        for rid in best:
            row = dict(self.db.execute("SELECT * FROM chunks WHERE rowid=?", (rid,)).fetchone())
            per_url[row["url"]] = per_url.get(row["url"], 0) + 1
            if per_url[row["url"]] > (k if urls else c.max_chunks_per_page):
                continue
            row["score"] = round(scores[rid], 4)
            row["cosine"] = round(float(sims[self.chunk_pos[rid]]), 3)
            out.append(row)
            if len(out) >= k:
                break
        return out

    @lru_cache(maxsize=1)  # noqa: B019 - singleton
    def _chunk_meta(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(tuple(r) for r in self.db.execute("SELECT source, url, doc_type FROM chunks ORDER BY rowid"))

    def _where(self, f: Filters) -> tuple[str, list]:
        clauses, params = [], []
        if f.source:
            clauses.append("source = ?"); params.append(f.source)
        # cities and countries are OR'ed ("Dubai or Saudi Arabia"), except a country that just
        # contains one of the cities ("Riyadh, Saudi Arabia" means Riyadh)
        aliases = get_reference().pipeline.country_aliases
        cities = values(f.city)
        countries = [aliases.get(c.lower(), c) for c in values(f.country)]  # "UAE" -> "United Arab Emirates"
        countries = [c for c in countries if not any(self._city_in_country(ct, c) for ct in cities)]
        loc = []
        for ct in cities:
            loc.append("(lower(city) LIKE ? OR lower(district) LIKE ?)"); params += [f"%{ct.lower()}%"] * 2
        for c in countries:
            loc.append("lower(country) LIKE ?"); params.append(f"%{c.lower()}%")
        if loc:
            clauses.append("(" + " OR ".join(loc) + ")")
        types = [t.lower().rstrip("s") for t in values(f.property_type)]
        if types:
            clauses.append("(" + " OR ".join(["(lower(property_type) LIKE ? OR lower(name) LIKE ?)"] * len(types)) + ")")
            for t in types:
                params += [f"%{t}%"] * 2
        if f.purpose:
            clauses.append("purpose = ?"); params.append(f.purpose)
        brands = values(f.brand)
        if brands:
            clauses.append("(" + " OR ".join(["lower(brand) LIKE ?"] * len(brands)) + ")")
            params += [f"%{b.lower()}%" for b in brands]
        if f.min_bedrooms is not None:
            clauses.append("COALESCE(bedrooms_max, bedrooms) >= ?"); params.append(f.min_bedrooms)
        if f.max_bedrooms is not None:
            clauses.append("bedrooms <= ?"); params.append(f.max_bedrooms)
        # compare in USD
        ref = get_reference()
        cur = (f.currency or ref.default_currency.get(f.source or "wasalt", "SAR")).upper()
        rates = ref.currency_usd_rates
        rate = rates.get(cur, rates[ref.default_currency["wasalt"]])
        if f.max_price is not None:
            clauses.append("price_usd <= ?"); params.append(f.max_price * rate)
        if f.min_price is not None:
            clauses.append("price_usd >= ?"); params.append(f.min_price * rate)
        return (" AND ".join(clauses) or "1=1"), params

    def search_properties(self, f: Filters, query: str, qvec: QVec, k: int | None = None) -> dict:
        """SQL filters, then rerank. Drops filters one by one if nothing matches."""
        c = self.cfg
        k = k or c.top_k_properties
        relaxed: list[str] = []
        f = Filters(**f.__dict__)
        rows = self._candidates(f)
        for field in c.relax_order:
            if rows or getattr(f, field) is None:
                continue
            relaxed.append(f"{field}={getattr(f, field)}")
            setattr(f, field, None)
            rows = self._candidates(f)

        fts_ids = set()
        q = fts_query(query)
        if q and rows:
            fts_ids = {r[0] for r in self.db.execute(
                "SELECT rowid FROM properties_fts WHERE properties_fts MATCH ? LIMIT ?", (q, c.property_fts_limit)).fetchall()}
        for r in rows:
            r["_score"] = float(self.emb[qvec.space]["prop"][self.prop_pos[r["rowid"]]] @ qvec.v) + (c.property_fts_bonus if r["rowid"] in fts_ids else 0)
        rows.sort(key=lambda r: r["_score"], reverse=True)

        types = [t.lower().rstrip("s") for t in values(f.property_type)]
        stats = self._stats(rows, types if len(types) > 1 else [])
        top = self._mix(rows, k, by_purpose=not f.purpose, types=types if len(types) > 1 else [])
        for r in top:
            r["score"] = round(r.pop("_score"), 3)
        return {"items": top, "total_matches": len(rows), "stats": stats, "relaxed": relaxed,
                "applied_filters": f.active()}

    def _candidates(self, f: Filters) -> list[dict]:
        where, params = self._where(f)
        return [dict(r) for r in self.db.execute(f"SELECT rowid, * FROM properties WHERE {where} LIMIT ?", [*params, self.cfg.property_candidate_limit])]

    @lru_cache(maxsize=256)  # noqa: B019 - singleton
    def _city_in_country(self, city: str, country: str) -> bool:
        return self.db.execute("SELECT 1 FROM properties WHERE lower(city) LIKE ? AND lower(country) LIKE ? LIMIT 1",
                               (f"%{city.lower()}%", f"%{country.lower()}%")).fetchone() is not None

    @staticmethod
    def _group(r: dict, by_purpose: bool, types: list[str]) -> tuple:
        ptype = next((t for t in types if t in f"{r.get('property_type')} {r.get('name')}".lower()), "")
        return (r.get("purpose") if by_purpose else "", ptype)

    @classmethod
    def _mix(cls, rows: list[dict], k: int, by_purpose: bool, types: list[str]) -> list[dict]:
        """Top k, taking turns between groups (sale/rent, requested types) so one group doesn't fill every slot."""
        groups: dict[tuple, list[dict]] = {}
        for r in rows:  # rows are already sorted by score
            groups.setdefault(cls._group(r, by_purpose, types), []).append(r)
        if len(groups) < 2:
            return rows[:k]
        out, queues = [], list(groups.values())
        while len(out) < k and any(queues):
            for q in queues:
                if q and len(out) < k:
                    out.append(q.pop(0))
        return out

    @staticmethod
    def _stats(rows: list[dict], types: list[str] | None = None) -> list[dict]:
        """Price stats, kept separate for sale and rent (and per type when several were asked for)."""
        out = []
        for ptype in types or [None]:
            subset = [r for r in rows if ptype is None or ptype in f"{r.get('property_type')} {r.get('name')}".lower()]
            for purpose in ("sale", "rent"):
                priced = [r for r in subset if r.get("purpose") == purpose and r.get("price") and r.get("currency")
                          and r.get("rent_frequency") not in ("unverified", "monthly")]
                if len(priced) < get_settings().retrieval.stats_min_count:
                    continue
                cur = statistics.mode(r["currency"] for r in priced)
                prices = sorted(r["price"] for r in priced if r["currency"] == cur)
                out.append({"purpose": purpose, "type": ptype, "count": len(prices), "currency": cur,
                            "median": statistics.median(prices), "min": prices[0], "max": prices[-1],
                            "period": "per year" if purpose == "rent" else None})
        return out

    def find_projects(self, names: list[str], limit_each: int = 1) -> list[dict]:
        out = []
        for n in names:
            q = fts_query(n)
            if not q:
                continue
            rows = self.db.execute(
                """SELECT p.rowid, p.* FROM properties_fts f JOIN properties p ON p.rowid=f.rowid
                   WHERE properties_fts MATCH ? AND p.source='darglobal' ORDER BY bm25(properties_fts, 5.0) LIMIT ?""",
                (q, limit_each)).fetchall()
            out += [dict(r) for r in rows]
        return out

_store: Store | None = None


def get_store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store
