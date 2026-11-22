"""Wasalt scraper (wasalt.sa).

Follows robots.txt and llms.txt: listing pages only, about one request a second, and no
owner/agent contact data is kept. Listing data is read from the page's __NEXT_DATA__ JSON.

Output: data/raw/wasalt/listings.jsonl
Usage:  python -m scrapers.wasalt [--n 1200]
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import subprocess
from datetime import UTC, datetime

from selectolax.lexbor import LexborHTMLParser as HTMLParser
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.resources import redact_contacts
from scrapers.common import RAW_DIR, USER_AGENT, append_jsonl, load_done, polite_sleep

log = logging.getLogger("wasalt")
SITEMAP = "https://cdn.wasalt.sa/sitemap/product_sitemap_en_sa.xml.gz"
IMG_BASE = "https://imagedelivery.net/1DNKFJPRaeUdy_j8F7HT3w/production/properties/{id}/images/{name}/width=800,quality=60,format=auto"
OUT = RAW_DIR / "wasalt" / "listings.jsonl"
NEXT_RE = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.DOTALL)
STATUS_MARK = "\n__HTTP_STATUS__:"


class Retryable(Exception):
    pass


@retry(retry=retry_if_exception_type(Retryable),
       wait=wait_exponential(multiplier=2, min=4, max=60), stop=stop_after_attempt(4), reraise=True)
def fetch(url: str) -> tuple[int, str]:
    """GET with curl. httpx and headless Chromium get a Cloudflare challenge here; curl doesn't."""
    proc = subprocess.run(
        ["curl", "-sL", "--compressed", "-m", "40", "-A", USER_AGENT, "-H", "Accept-Language: en",
         "-w", STATUS_MARK + "%{http_code}", url],
        capture_output=True, timeout=60, check=False,
    )
    out = proc.stdout.decode("utf-8", errors="replace")
    body, _, code = out.rpartition(STATUS_MARK)
    status = int(code.strip() or 0)
    if proc.returncode != 0 or status in (0, 429, 500, 502, 503, 504):
        log.warning("status %s (curl rc=%s) on %s, backing off", status, proc.returncode, url)
        raise Retryable(str(status))
    return status, body


def sample_urls(n: int, rent_share: float, recent: int = 15000) -> list[str]:
    """Sample sale and rent listings, newest first."""
    _, xml = fetch(SITEMAP)
    entries = re.findall(r"<loc>([^<]+)</loc>\s*<lastmod>([^<]+)</lastmod>", xml)
    entries.sort(key=lambda e: e[1], reverse=True)
    urls = [u for u, _ in entries[:recent]]
    sale = [u for u in urls if "/property/sale/" in u]
    rent = [u for u in urls if "/property/rent/" in u]
    log.info("sitemap: %d entries; recent pool %d sale, %d rent", len(entries), len(sale), len(rent))
    rng = random.Random(42)
    n_rent = int(n * rent_share)
    return rng.sample(sale, min(n - n_rent, len(sale))) + rng.sample(rent, min(n_rent, len(rent)))


def _strip_html(s: str | None) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", HTMLParser(s).text(separator=" ")).strip()


def _num(v):
    try:
        return float(str(v).replace(",", "")) if v not in (None, "", "-") else None
    except ValueError:
        return None


def parse_listing(url: str, html: str) -> dict | None:
    m = NEXT_RE.search(html)
    if not m:
        return None
    pp = json.loads(m.group(1)).get("props", {}).get("pageProps", {})
    d = pp.get("propertyDetailsV3") or {}
    info = d.get("propertyInfo") or {}
    if not info:
        return None
    attrs = {a.get("key"): a for a in d.get("attributes") or []}

    purpose = info.get("propertyFor")
    price, freq = _num(info.get("salePrice")), None
    if purpose == "rent":
        price = _num(info.get("expectedRent"))
        rf = info.get("rentFreq") or {}
        if isinstance(rf, dict) and rf:
            default = next((v for v in rf.values() if v.get("default_freq")), next(iter(rf.values())))
            price, freq = _num(default.get("amount")) or price, default.get("freq")
    if price is None:
        price = _num(info.get("conversionPrice"))

    imgs = ((pp.get("galleryDetails") or {}).get("images") or {}).get("data") or []
    imgs = sorted(imgs, key=lambda x: x.get("sequence", 99))
    image = IMG_BASE.format(id=d.get("id"), name=imgs[0]["content"]) if imgs else None
    amen = pp.get("infoDetails") or {}

    def names(key):
        return [a.get("name") or a.get("title") for a in amen.get(key) or [] if isinstance(a, dict)]

    return {
        "url": url,
        "listing_id": d.get("id"),
        "title": info.get("title"),
        "purpose": purpose,                         # sale | rent
        "main_type": info.get("propertyMainType"),  # Residential | Commercial
        "property_type": info.get("propertySubType"),
        "city": info.get("city"),
        "district": info.get("district") or info.get("zone"),
        "territory": info.get("territory"),
        "address": info.get("address"),
        "country": info.get("country") or "Saudi Arabia",
        "price": price,
        "currency": info.get("currencyType") or "SAR",
        "rent_frequency": freq,
        "bedrooms": _num((attrs.get("noOfBedrooms") or {}).get("value")),
        "bathrooms": _num((attrs.get("noOfBathrooms") or {}).get("value")),
        "built_up_area_sqm": _num((attrs.get("builtUpArea") or {}).get("value")),
        "land_area_sqm": _num((attrs.get("landArea") or attrs.get("plotArea") or {}).get("value")),
        "attributes": {k: v.get("value") for k, v in attrs.items()},
        "furnishing": info.get("furnishingType"),
        "amenities": [x for x in names("amenities") if x],
        "description": redact_contacts(_strip_html(info.get("description")))[:2000],
        "is_verified": bool(d.get("isVerified")),
        "rega_licensed": bool(d.get("isRegaProp")),
        "project_id": info.get("projectId"),
        "image": image,
        "published_at": d.get("publishedAt"),
        "updated_at": d.get("updatedAt"),
        "scraped_at": datetime.now(UTC).isoformat(),
        # not stored: propertyOwner, leadContactInfo, regaVerifiedInfo, lat/lon
    }


def main(n: int, rent_share: float, delay: float) -> None:
    done = load_done(OUT)
    urls = [u for u in sample_urls(n, rent_share) if u not in done]
    log.info("to scrape: %d (already %d)", len(urls), len(done))
    ok = gone = fail = 0
    consecutive_blocks = 0
    for i, url in enumerate(urls, 1):
        try:
            status, body = fetch(url)
            row = parse_listing(url, body) if status == 200 else None
            if row:
                append_jsonl(OUT, [row])
                ok += 1
                consecutive_blocks = 0
            elif status in (404, 410):
                gone += 1  # removed listing
            else:
                fail += 1
                consecutive_blocks += status == 403
                log.info("no listing data (%s): %s", status, url)
        except Exception as e:  # noqa: BLE001
            fail += 1
            log.warning("failed %s: %s", url, e)
        if consecutive_blocks >= 5:
            log.error("5 consecutive 403s: the site is refusing us. Stopping (do not escalate).")
            break
        if i % 50 == 0:
            log.info("progress %d/%d ok=%d gone=%d fail=%d", i, len(urls), ok, gone, fail)
        polite_sleep(delay)
    log.info("done ok=%d gone=%d fail=%d -> %s", ok, gone, fail, OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--rent-share", type=float, default=0.3)
    ap.add_argument("--delay", type=float, default=1.0)
    a = ap.parse_args()
    main(a.n, a.rent_share, a.delay)
