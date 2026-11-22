"""Helpers shared by the scrapers."""
from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

USER_AGENT = "PropertyAssistantResearchBot/1.0 (candidate technical assessment; low-rate; respects robots.txt)"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def polite_sleep(base: float, jitter: float = 0.6) -> None:
    """Sleep base to base+jitter seconds."""
    time.sleep(base + random.random() * jitter)


def load_done(path: Path, key: str = "url") -> set[str]:
    """Keys already in the file, so a rerun can resume."""
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)[key])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
