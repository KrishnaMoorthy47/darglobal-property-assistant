"""Index build: OpenRouter embeddings with batching, caching and refused inputs."""
import numpy as np

from app.llm import gateway as gw_mod
from pipeline import build


class FakeGateway:
    def __init__(self):
        self.calls = 0
        self.chains, self.clients = {}, {"openrouter_3": object()}
        self.cfg = type("Cfg", (), {"key_daily_caps": {}})()

    async def embed(self, texts, chain="embedding"):
        self.calls += 1
        if any("BAD" in t for t in texts):
            raise gw_mod.InputRejected("422 content screening")
        v = np.ones((len(texts), 4), dtype=np.float32) / 2
        return v, gw_mod.StreamResult(provider="openrouter_3")


def test_refused_text_gets_zero_vector_and_the_rest_are_embedded(tmp_path, monkeypatch):
    fake = FakeGateway()
    monkeypatch.setattr(build, "RAW", tmp_path)
    monkeypatch.setattr(gw_mod, "get_gateway", lambda: fake)
    monkeypatch.setattr(build.CFG.embedder.remote, "pause_s", 0)
    monkeypatch.setattr(build.CFG.embedder.remote, "batch_size", 4)
    texts = ["a", "b", "BAD", "c", "d", "e"]
    v = build.embed_remote(texts, fetch=True)
    assert v.shape == (6, 4)
    assert not v[2].any() and all(v[i].any() for i in (0, 1, 3, 4, 5))

    # second run uses the cache only
    calls = fake.calls
    assert build.embed_remote(texts, fetch=False) is not None and fake.calls == calls
