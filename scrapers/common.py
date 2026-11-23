"""Helpers shared by the scrapers."""
from __future__ import annotations

import json
import logging
import random
import re
import time
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

USER_AGENT = "PropertyAssistantResearchBot/1.0 (candidate technical assessment; low-rate; respects robots.txt)"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


class RobotsRules:
    """robots.txt following RFC 9309: the longest matching rule wins, and Allow wins a tie.

    (urllib.robotparser uses the first matching rule instead, so "Allow: /" followed by
    "Disallow: /search" would wrongly allow /search.)
    """

    def __init__(self, rules: list[tuple[bool, str]] | None = None, allow_all: bool = False):
        self.rules = rules or []
        self.allow_all = allow_all
        self.disallow_all = rules is None and not allow_all

    @classmethod
    def parse(cls, text: str, agent: str) -> RobotsRules:
        groups: list[tuple[list[str], list[tuple[bool, str]]]] = []
        new_group = True
        for line in text.splitlines():
            key, _, value = line.split("#", 1)[0].partition(":")
            key, value = key.strip().lower(), value.strip()
            if key == "user-agent":
                if new_group:
                    groups.append(([], []))
                    new_group = False
                groups[-1][0].append(value.lower())
            elif key in ("allow", "disallow") and groups:
                new_group = True
                if value:
                    groups[-1][1].append((key == "allow", value))
        token = agent.split("/")[0].lower()
        mine = [r for agents, rules in groups if token in agents for r in rules]
        anyone = [r for agents, rules in groups if "*" in agents for r in rules]
        return cls(mine if any(token in a for a, _ in groups) else anyone)

    def can_fetch(self, url: str) -> bool:
        if self.allow_all:
            return True
        if self.disallow_all:
            return False
        parts = urlsplit(url)
        path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        best = (-1, True)
        for allow, pattern in self.rules:
            regex = re.escape(pattern).replace(r"\*", ".*")
            regex = regex[:-2] + "$" if regex.endswith(r"\$") else regex
            if re.match(regex, path) and (len(pattern), allow) > best:
                best = (len(pattern), allow)
        return best[1]


def robots_rules(status: int, text: str, agent: str = "PropertyAssistantResearchBot") -> RobotsRules:
    """A missing robots.txt (4xx) allows everything; an unreachable one (5xx, timeout, or a 200 that isn't a
    robots file, e.g. a firewall challenge page) allows nothing."""
    if status == 200 and "user-agent" in text.lower():
        return RobotsRules.parse(text, agent)
    return RobotsRules(allow_all=400 <= status < 500)


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
