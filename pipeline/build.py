"""Builds data/app.db, the embedding files and manifest.json from the raw scrapes.

The .npy rows follow the rowid order of the chunks and properties tables.

Usage: python -m pipeline.build [--enrich] [--remote-embed]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sqlite3
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from app.config import get_settings
from app.resources import get_prompts, get_reference, redact_contacts
from pipeline.schemas import Chunk, Property

log = logging.getLogger("build")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT_DB = ROOT / "data" / "app.db"
ENRICH_CACHE = RAW / "darglobal" / "enriched.json"

REF = get_reference().pipeline
RATES = get_reference().currency_usd_rates
CFG = get_settings()


def read_jsonl(path: Path) -> list[dict]:
    """Read JSONL, skipping a half-written last line."""
    out = []
    # split("\n") not splitlines(): scraped text has U+2028 chars it would split on
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            log.warning("skipping malformed line in %s", path.name)
    return out


def clean_text(t: str) -> str:
    for pat in REF.boilerplate:
        t = re.sub(pat, " ", t, flags=re.DOTALL | re.IGNORECASE)
    t = re.sub(r"(?<=[a-z0-9.])(?=[A-Z][a-z])", "\n", t)  # "GalleryDa Vinci" -> "Gallery" / "Da Vinci"
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n", t)
    return t.strip()


def chunk_text(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    """Split on sentences into overlapping windows of about 250 tokens."""
    size = size or CFG.pipeline.chunk_size
    overlap = overlap or CFG.pipeline.chunk_overlap
    sents = re.split(r"(?<=[.!?])\s+|\n", text)
    chunks, cur = [], ""
    for s in sents:
        s = s.strip()
        if not s:
            continue
        if len(cur) + len(s) + 1 > size and cur:
            chunks.append(cur.strip())
            cur = cur[-overlap:] + " " + s
        else:
            cur = f"{cur} {s}"
    if cur.strip():
        chunks.append(cur.strip())
    return [c for c in chunks if len(c) > 80]


def page_name(row: dict) -> str:
    name = (row.get("h1") or "").strip() or row["title"].split("|")[0].strip()
    return name.title() if name.isupper() else name


def find_location(text: str, name: str) -> tuple[str | None, str | None]:
    head = text[:600].lower()
    for city, country in REF.city_country.items():
        if re.search(rf"\b{re.escape(city)}\b", head):
            return REF.city_display_overrides.get(city, city.title()), country
    for alias, country in REF.country_aliases.items():
        if re.search(rf"\b{re.escape(alias)}\b", head):
            return None, country
    return None, None


def norm_city(c: str | None) -> str | None:
    if not c:
        return c
    c = c.split(" - ")[0].strip()
    return REF.city_normalize.get(c.lower(), c)


def to_usd(price: float | None, cur: str | None) -> float | None:
    if price is None or not cur:
        return None
    rate = RATES.get(cur.upper())
    return round(price * rate, 2) if rate else None


def regex_project_facts(text: str) -> dict:
    facts = {}
    m = re.search(r"Property Type\s*(.{2,60}?)\s*Status\s*(.{2,30}?)\s*Unit Type\s*(.{2,60}?)(?:\s*Area|\s*Size|\n|$)", text)
    if m:
        facts["property_type"], facts["status"], facts["unit_types"] = (g.strip() for g in m.groups())
    beds = [int(b) for b in re.findall(r"(\d)\s*(?:,|&|and|to|-)?\s*(?:\d\s*(?:,|&|and|-)\s*)*\d?\s*-?\s*bed", text[:3000], re.IGNORECASE)]
    if beds:
        facts["bedrooms_min"], facts["bedrooms_max"] = min(beds), max(beds)
    return facts


def llm_enrich(rows: list[dict]) -> dict[str, dict]:
    """Extract project facts with an LLM. Results are cached in enriched.json."""
    cache = json.loads(ENRICH_CACHE.read_text(encoding="utf-8")) if ENRICH_CACHE.exists() else {}
    todo = [r for r in rows if r["url"] not in cache]
    if not todo:
        return cache
    import time

    from openai import OpenAI

    from app.config import get_settings

    s = get_settings()
    ec = s.pipeline.enrichment
    key = {"groq": s.groq_api_key, "openrouter": s.openrouter_api_key}[ec.provider]
    if not key:
        log.warning("%s API key not set - skipping LLM enrichment", ec.provider)
        return cache
    client = OpenAI(api_key=key, base_url=s.llm.base_urls[ec.provider], max_retries=2)
    prompt = get_prompts().enrichment_system
    for i, r in enumerate(todo, 1):
        try:
            resp = client.chat.completions.create(
                model=ec.model, temperature=0, max_tokens=ec.max_tokens,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": prompt},
                          {"role": "user", "content": f"URL: {r['url']}\nTITLE: {r['title']}\n\n{clean_text(r['text'])[:ec.max_input_chars]}"}],
            )
            cache[r["url"]] = json.loads(resp.choices[0].message.content)
            log.info("enriched %d/%d %s", i, len(todo), r["url"])
        except Exception as e:  # noqa: BLE001
            log.warning("enrich failed %s: %s", r["url"], e)
        ENRICH_CACHE.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
        time.sleep(ec.pause_s)
    return cache


def darglobal(enrich: bool) -> tuple[list[Property], list[Chunk]]:
    path = RAW / "darglobal" / "pages.jsonl"
    rows = {r["url"]: r for r in read_jsonl(path)}  # last row per URL wins
    rows = [r for r in rows.values() if len(clean_text(r["text"])) > CFG.pipeline.min_page_chars]
    log.info("darglobal pages usable: %d", len(rows))

    projects = [r for r in rows if r["category"] == "project"]
    enriched = llm_enrich(projects) if enrich else (
        json.loads(ENRICH_CACHE.read_text(encoding="utf-8")) if ENRICH_CACHE.exists() else {})

    props, chunks = [], []
    for r in rows:
        text = clean_text(r["text"])
        name = page_name(r)
        city, country = find_location(text, name)
        if r["category"] == "project":
            f = regex_project_facts(text)
            e = enriched.get(r["url"], {})
            e = {k: v for k, v in e.items() if v not in (None, "", [], "null")}
            ptypes = e.get("property_types") or ([f["property_type"]] if f.get("property_type") else [])
            if str(e.get("brand_partner", "")).lower().replace(" ", "") in REF.developer_names_not_brands:
                e.pop("brand_partner")
            # if the LLM read the page, trust its "no brand"; scanning the text picks up nearby amenities
            # (e.g. "home to Trump International Golf Club" on DarGlobal Mansions)
            scan = name if r["url"] in enriched else name + text[:800]
            brand = e.get("brand_partner") or next((b for b in REF.brands if b.lower() in scan.lower()), None)
            price = e.get("starting_price") if isinstance(e.get("starting_price"), (int, float)) else None
            cur = e.get("currency") if price else None
            desc = r.get("description") or text[:400]
            highlights = e.get("highlights") or []
            props.append(Property(
                id="dg-" + hashlib.sha1(r["url"].encode()).hexdigest()[:10], source="darglobal", url=r["url"],
                name=e.get("name") or name, purpose="sale",
                property_type=", ".join(map(str, ptypes))[:80] or None,
                city=e.get("city") or city, country=REF.country_aliases.get(str(e.get("country", "")).lower(), e.get("country")) or country,
                price=price, currency=cur, price_usd=to_usd(price, cur),
                bedrooms=e.get("bedrooms_min") or f.get("bedrooms_min"),
                bedrooms_max=e.get("bedrooms_max") or f.get("bedrooms_max"),
                area_sqm=round(e["area_min_sqft"] * 0.0929, 1) if isinstance(e.get("area_min_sqft"), (int, float)) else None,
                status=e.get("status") or f.get("status"), brand=brand, handover=e.get("handover"),
                payment_plan=e.get("payment_plan"),
                image=r.get("og_image") or (r.get("images") or [None])[0],
                description=(desc + (" Highlights: " + "; ".join(highlights) if highlights else ""))[:900],
            ))
        prefix = f"[DarGlobal {r['category']}] {name}" + (f" ({city or ''}{', ' if city and country else ''}{country or ''})" if city or country else "")
        for i, c in enumerate(chunk_text(text)):
            chunks.append(Chunk(id=f"dgc-{hashlib.sha1(r['url'].encode()).hexdigest()[:8]}-{i}", source="darglobal",
                                doc_type=r["category"], title=name, url=r["url"], text=f"{prefix}: {c}",
                                image=r.get("og_image")))
    return props, chunks


def wasalt() -> tuple[list[Property], list[Chunk]]:
    path = RAW / "wasalt" / "listings.jsonl"
    seen, props = set(), []
    for r in read_jsonl(path):
        if r["listing_id"] in seen or not r.get("price"):
            continue
        seen.add(r["listing_id"])
        a = r.get("attributes") or {}
        land = r.get("land_area_sqm") or next((float(v) for k, v in a.items() if "land" in k.lower() or "plot" in k.lower()
                                               if str(v).replace(".", "", 1).isdigit()), None)
        area = r.get("built_up_area_sqm") or land
        freq = r.get("rent_frequency")
        # some "yearly" rents are really monthly; keep the listing but leave it out of stats
        if r["purpose"] == "rent" and freq == "yearly" and r["price"] < CFG.data_quality.min_plausible_annual_rent_sar:
            freq = "unverified"
        rent_note = f" per {freq.replace('ly', '')}" if freq and freq != "unverified" else (" (rent period unclear at source)" if freq else "")
        props.append(Property(
            id=f"ws-{r['listing_id']}", source="wasalt", url=r["url"], name=redact_contacts(r["title"]), purpose=r["purpose"],
            property_type=r.get("property_type"), city=norm_city(r.get("city")), district=r.get("district"),
            country=r.get("country") or "Saudi Arabia", price=r["price"], currency=r.get("currency") or "SAR",
            price_usd=to_usd(r["price"], r.get("currency") or "SAR"), rent_frequency=freq,
            bedrooms=r.get("bedrooms"), bathrooms=r.get("bathrooms"), area_sqm=area,
            status="listed", image=r.get("image"), furnishing=r.get("furnishing"),
            description=f"{r['title']} for {r['purpose']} in {r.get('district') or ''}, {r.get('city') or ''}. "
                        f"Price {r['price']:,.0f} {r.get('currency', 'SAR')}{rent_note}. {redact_contacts(r.get('description'))}"[:900],
            updated_at=r.get("updated_at"),
        ))
    log.info("wasalt listings usable: %d", len(props))

    # per city/type price summaries for market questions
    chunks = []
    groups: dict[tuple, list[float]] = {}
    for p in props:
        if p.purpose == "rent" and p.rent_frequency != "yearly":
            continue
        groups.setdefault((p.city, p.property_type, p.purpose), []).append(p.price)
    for (city, ptype, purpose), prices in groups.items():
        if len(prices) < 5 or not city or not ptype:
            continue
        txt = (f"[Wasalt market snapshot] {ptype} for {purpose} in {city}, Saudi Arabia: {len(prices)} listings in our sample. "
               f"Median price {statistics.median(prices):,.0f} SAR, range {min(prices):,.0f} - {max(prices):,.0f} SAR"
               + (" (annual rent)." if purpose == "rent" else "."))
        chunks.append(Chunk(id=f"wsm-{hashlib.sha1(txt.encode()).hexdigest()[:10]}", source="wasalt", doc_type="market_stats",
                            title=f"{ptype} for {purpose} in {city} - market snapshot",
                            url="https://wasalt.sa/en", text=txt))
    return props, chunks


def embed(texts: list[str]) -> np.ndarray:
    from app.config import get_settings
    from app.rag.embedder import Embedder

    return Embedder(get_settings().embedder, threads=4).embed(texts)


def embed_remote(texts: list[str], fetch: bool) -> np.ndarray | None:
    """Embeddings from the OpenRouter model in embedder.remote, cached on disk per text.

    Without fetch, only the cache is used (no API calls); returns None if anything is missing.
    """
    import asyncio

    from app.llm.gateway import Endpoint, InputRejected, get_gateway

    rc = CFG.embedder.remote
    cache_file = RAW / "embed_cache" / (re.sub(r"\W+", "_", rc.model) + ".npz")
    cache: dict[str, np.ndarray] = {}
    if cache_file.exists():
        z = np.load(cache_file)
        cache = dict(zip(z["keys"].tolist(), z["vecs"], strict=True))
    def key(t: str) -> str:
        return hashlib.sha1((rc.doc_prefix + t).encode()).hexdigest()

    missing = list(dict.fromkeys(t for t in texts if key(t) not in cache))
    if missing and not fetch:
        log.warning("%d texts have no cached %s embedding; run with --remote-embed", len(missing), rc.model)
        return None

    def save() -> None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        ks = list(cache)
        np.savez(cache_file, keys=np.array(ks), vecs=np.stack([cache[k] for k in ks]))

    async def fetch_missing() -> bool:
        g = get_gateway()
        # the build may use each OpenRouter key's whole daily allowance, key 3 first
        g.chains["build"] = [Endpoint(p, rc.model, g.cfg.key_daily_caps.get(p, 48), 15)
                             for p in ("openrouter_3", "openrouter", "openrouter_2") if p in g.clients]
        log.info("embedding %d texts with %s in batches of %d", len(missing), rc.model, rc.batch_size)

        async def embed_into_cache(batch: list[str]) -> None:
            try:
                v, _ = await g.embed([rc.doc_prefix + t for t in batch], chain="build")
            except InputRejected:
                # the provider's content screening refused something in this batch: halve until it's found
                if len(batch) == 1:
                    rejected.append(batch[0])
                    return
                await asyncio.sleep(rc.pause_s)
                await embed_into_cache(batch[:len(batch) // 2])
                await asyncio.sleep(rc.pause_s)
                await embed_into_cache(batch[len(batch) // 2:])
                return
            for t, vec in zip(batch, v, strict=True):
                cache[key(t)] = vec

        for i in range(0, len(missing), rc.batch_size):
            try:
                await embed_into_cache(missing[i:i + rc.batch_size])
            except Exception as e:  # noqa: BLE001
                log.error("stopped after %d/%d texts: %s (progress is cached; rerun later)", i, len(missing), e)
                return False
            save()
            log.info("  %d/%d done", min(i + rc.batch_size, len(missing)), len(missing))
            await asyncio.sleep(rc.pause_s)
        return True

    # texts the provider refuses get a zero vector: never found by vector search, still found by keyword search
    rejected: list[str] = []

    # one event loop for all batches: the API client is tied to the loop it was first used in
    if missing and not asyncio.run(fetch_missing()):
        save()
        return None
    if rejected:
        dim = next(iter(cache.values())).shape[0]
        for t in rejected:
            cache[key(t)] = np.zeros(dim, dtype=np.float32)
        save()
        log.warning("%d texts were refused by the embedding provider's content screening: %s",
                    len(rejected), [t[:60] for t in rejected])
    return np.stack([cache[key(t)] for t in texts]).astype(np.float32)


def write_db(props: list[Property], chunks: list[Chunk]) -> None:
    tmp = OUT_DB.with_suffix(".db.tmp")
    tmp.unlink(missing_ok=True)
    db = sqlite3.connect(tmp)
    cols = list(Property.model_fields)
    db.execute(f"CREATE TABLE properties ({', '.join(c + (' TEXT PRIMARY KEY' if c == 'id' else '') for c in cols)})")
    db.executemany(f"INSERT INTO properties VALUES ({','.join('?' * len(cols))})",
                   [tuple(getattr(p, c) for c in cols) for p in props])
    db.execute("CREATE VIRTUAL TABLE properties_fts USING fts5(name, city, district, country, property_type, brand, description, "
               "content='properties', content_rowid='rowid', tokenize='unicode61 remove_diacritics 2')")
    db.execute("INSERT INTO properties_fts(rowid, name, city, district, country, property_type, brand, description) "
               "SELECT rowid, name, city, district, country, property_type, brand, description FROM properties")
    ccols = list(Chunk.model_fields)
    db.execute(f"CREATE TABLE chunks ({', '.join(c + (' TEXT PRIMARY KEY' if c == 'id' else '') for c in ccols)})")
    db.executemany(f"INSERT INTO chunks VALUES ({','.join('?' * len(ccols))})", [tuple(getattr(c, k) for k in ccols) for c in chunks])
    db.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(title, text, content='chunks', content_rowid='rowid', "
               "tokenize='unicode61 remove_diacritics 2')")
    db.execute("INSERT INTO chunks_fts(rowid, title, text) SELECT rowid, title, text FROM chunks")
    for idx in ["source", "city", "purpose", "property_type", "price_usd", "bedrooms"]:
        db.execute(f"CREATE INDEX idx_p_{idx} ON properties({idx})")
    db.commit()
    db.close()
    tmp.replace(OUT_DB)


def main(enrich: bool, remote_embed: bool = False) -> None:
    dg_props, dg_chunks = darglobal(enrich)
    ws_props, ws_chunks = wasalt()
    props, chunks = dg_props + ws_props, dg_chunks + ws_chunks
    leaks = [x.id for x in props if redact_contacts(x.description) != x.description]
    leaks += [x.id for x in chunks if redact_contacts(x.text) != x.text]
    if leaks:
        raise SystemExit(f"contact details found in {len(leaks)} records, e.g. {leaks[:3]}; refusing to build")
    write_db(props, chunks)

    chunk_texts = [c.text for c in chunks]
    prop_texts = [f"{p.name}. {p.property_type or ''} for {p.purpose} in {p.district or ''} {p.city or ''} {p.country or ''}. "
                  f"{p.brand or ''}. {p.description[:300]}" for p in props]
    log.info("embedding %d chunks + %d properties (local model) ...", len(chunks), len(props))
    np.save(ROOT / "data" / "chunk_emb.npy", embed(chunk_texts))
    np.save(ROOT / "data" / "prop_emb.npy", embed(prop_texts))

    # OpenRouter embeddings; without a complete set the app just uses the local index
    remote_model = None
    if CFG.embedder.remote.enabled:
        both = embed_remote(chunk_texts + prop_texts, fetch=remote_embed)
        if both is not None:
            np.save(ROOT / "data" / "chunk_emb_remote.npy", both[:len(chunk_texts)])
            np.save(ROOT / "data" / "prop_emb_remote.npy", both[len(chunk_texts):])
            remote_model = CFG.embedder.remote.model
    if remote_model is None:
        for f in ("chunk_emb_remote.npy", "prop_emb_remote.npy"):
            (ROOT / "data" / f).unlink(missing_ok=True)

    manifest = {
        "scraped_window": {
            "darglobal": _scrape_date(RAW / "darglobal" / "pages.jsonl"),
            "wasalt": _scrape_date(RAW / "wasalt" / "listings.jsonl"),
        },
        "counts": {
            "darglobal_projects": len(dg_props), "darglobal_chunks": len(dg_chunks),
            "darglobal_pages_by_type": dict(Counter(c.doc_type for c in dg_chunks if c.id.endswith("-0"))),
            "wasalt_listings": len(ws_props), "wasalt_market_snapshots": len(ws_chunks),
            "wasalt_by_city": dict(Counter(p.city for p in ws_props).most_common(12)),
            "wasalt_by_purpose": dict(Counter(p.purpose for p in ws_props)),
        },
        "embedding_model": remote_model or CFG.embedder.display_name,
        "embeddings": {"remote": remote_model, "local": CFG.embedder.display_name},
        "llm_enriched_projects": sum(1 for p in dg_props if p.brand or p.status),
    }
    (ROOT / "data" / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("manifest: %s", json.dumps(manifest["counts"], ensure_ascii=False))


def _scrape_date(path: Path) -> str | None:
    dates = [r["scraped_at"][:10] for r in read_jsonl(path) if r.get("scraped_at")]
    return max(dates) if dates else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--enrich", action="store_true", help="LLM-extract DarGlobal project facts")
    ap.add_argument("--remote-embed", action="store_true", help="call OpenRouter for missing embeddings (uses quota)")
    a = ap.parse_args()
    main(a.enrich, a.remote_embed)
