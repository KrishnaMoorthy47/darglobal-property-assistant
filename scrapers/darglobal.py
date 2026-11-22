"""DarGlobal scraper. The site renders client-side, so it uses Playwright.
Pages that return a challenge are skipped.

Writes data/raw/darglobal/pages.jsonl
Usage: python -m scrapers.darglobal [--blog-limit 120] [--press-limit 40]
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import random
import re
from datetime import UTC, datetime

from playwright.async_api import async_playwright

from scrapers.common import RAW_DIR, append_jsonl, load_done

log = logging.getLogger("darglobal")
BASE = "https://www.darglobal.co.uk"
OUT = RAW_DIR / "darglobal" / "pages.jsonl"
HTML_DIR = RAW_DIR / "darglobal" / "html"

# pages with nothing useful to index
EXCLUDE = re.compile(
    r"/(thank-you|success|pay-online|landing-?page|bookYourUnit|get-in-touch|careers|internship"
    r"|become-a|terms|privacy|cookie|win-a-trip|campaigns|new-home-page|broker-contest|amour-sans"
    r"|dg1$|investor$|aston-martin-media)",
    re.IGNORECASE,
)


def categorize(path: str) -> str:
    first = path.strip("/").split("/")[0].lower()
    if first in {"blog", "press", "insights", "partners"}:
        return {"blog": "blog", "press": "press", "insights": "insight", "partners": "partner"}[first]
    if first == "projects":
        return "listing_index"
    if first in {"about", "faq", "why-invest", "hospitality", "commercial", "development-management",
                 "tokenization", "invest-crypto", "exclusive-membership", "dg-circle", "insignia",
                 "darglobal-insignia-card", "live-all-in", "luxury-golf-communities", "one-of-one"} and "/" not in path.strip("/"):
        return "company"
    return "project"


async def get_sitemap_urls(page) -> list[str]:
    await page.goto(f"{BASE}/sitemap.xml", wait_until="domcontentloaded", timeout=60000)
    await page.wait_for_timeout(2500)
    content = await page.content()
    paths = sorted(set(re.findall(r"darglobal\.co\.uk(/[^<\"\s&]*)", content)))
    return [BASE + p for p in paths if p not in ("/", "/sitemap.xml")]


def select_urls(urls: list[str], blog_limit: int, press_limit: int) -> list[str]:
    keep, blog, press = [], [], []
    for u in urls:
        path = u.replace(BASE, "")
        if EXCLUDE.search(path):
            continue
        cat = categorize(path)
        if cat == "blog" and path.strip("/") != "blog":
            blog.append(u)
        elif cat == "press" and path.strip("/") != "press":
            press.append(u)
        elif path.strip("/") in {"blog", "press", "insights", "partners"}:
            continue  # link lists only
        else:
            keep.append(u)
    # main pages first, then a fixed sample of articles
    random.Random(42).shuffle(blog)
    random.Random(42).shuffle(press)
    return keep + blog[:blog_limit] + press[:press_limit]


EXTRACT_JS = """
() => {
  const clone = document.body.cloneNode(true);
  clone.querySelectorAll('header, footer, nav, script, style, noscript, iframe, form, [class*="cookie"], [class*="modal"]').forEach(e => e.remove());
  const meta = (n) => (document.querySelector(`meta[name="${n}"], meta[property="${n}"]`) || {}).content || '';
  const imgs = [...document.querySelectorAll('img')].map(i => i.currentSrc || i.src)
      .filter(s => s && s.startsWith('http') && !s.includes('logo') && !s.endsWith('.svg')).slice(0, 4);
  return {
    title: document.title,
    h1: (document.querySelector('h1') || {}).innerText || '',
    description: meta('description') || meta('og:description'),
    og_image: meta('og:image'),
    images: imgs,
    text: clone.innerText,
  };
}
"""


async def scrape(blog_limit: int, press_limit: int, delay: float) -> None:
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    done = load_done(OUT)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1366, "height": 900}, locale="en-GB")
        page = await ctx.new_page()
        urls = select_urls(await get_sitemap_urls(page), blog_limit, press_limit)
        todo = [u for u in urls if u not in done]
        log.info("sitemap selection=%d, already done=%d, to scrape=%d", len(urls), len(done), len(todo))
        ok = fail = 0
        for i, url in enumerate(todo, 1):
            try:
                resp = await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                await page.wait_for_timeout(2500)
                # scroll so lazy sections (prices, payment plans) load
                for _ in range(4):
                    await page.mouse.wheel(0, 2500)
                    await page.wait_for_timeout(400)
                html = await page.content()
                if "_Incapsula_Resource" in html and len(html) < 5000:
                    log.warning("challenge page, skipping: %s", url)
                    fail += 1
                    continue
                data = await page.evaluate(EXTRACT_JS)
                text = re.sub(r"\n{3,}", "\n\n", data["text"]).strip()
                if len(text) < 200:
                    log.warning("too little text (%d chars): %s", len(text), url)
                h = hashlib.sha1(url.encode()).hexdigest()[:16]
                (HTML_DIR / f"{h}.html").write_text(html, encoding="utf-8")
                append_jsonl(OUT, [{
                    "url": url,
                    "category": categorize(url.replace(BASE, "")),
                    "status": resp.status if resp else None,
                    "title": data["title"],
                    "h1": data["h1"],
                    "description": data["description"],
                    "og_image": data["og_image"],
                    "images": data["images"],
                    "text": text,
                    "html_file": f"{h}.html",
                    "scraped_at": datetime.now(UTC).isoformat(),
                }])
                ok += 1
            except Exception as e:  # noqa: BLE001
                log.warning("failed %s: %s", url, e)
                fail += 1
            if i % 10 == 0:
                log.info("progress %d/%d ok=%d fail=%d", i, len(todo), ok, fail)
            await asyncio.sleep(delay + random.random())
        await browser.close()
    log.info("done ok=%d fail=%d -> %s", ok, fail, OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--blog-limit", type=int, default=120)
    ap.add_argument("--press-limit", type=int, default=40)
    ap.add_argument("--delay", type=float, default=1.5)
    a = ap.parse_args()
    asyncio.run(scrape(a.blog_limit, a.press_limit, a.delay))
