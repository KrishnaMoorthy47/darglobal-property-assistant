"""Multilingual MiniLM embedder (int8 ONNX) with a small Unigram tokenizer.

The HF tokenizers object alone used ~260 MB for this vocabulary, too much for a 512 MB host, so the
model is quantized at build time and tokenized with the Viterbi code below.

Build: python -m app.rag.embedder prepare [out_dir]
"""
from __future__ import annotations

import json
import math
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np

SPACE = "▁"            # SentencePiece word boundary
BOS, PAD, EOS, UNK = 0, 1, 2, 3


class UnigramTokenizer:
    def __init__(self, vocab_path: Path, max_tokens: int, max_piece_len: int):
        pieces, scores = json.loads(vocab_path.read_text(encoding="utf-8"))
        self.ids = {p: i for i, p in enumerate(pieces) if i > 3}
        self.scores = np.asarray(scores, dtype=np.float32)
        self.max_len = max_piece_len
        self.max_tokens = max_tokens
        self.unk_score = float(self.scores.min()) - 10.0

    @lru_cache(maxsize=50_000)  # noqa: B019 - one tokenizer per process
    def _word(self, w: str) -> tuple[int, ...]:
        """Viterbi split of one word into the most likely pieces."""
        n = len(w)
        best = [0.0] + [-math.inf] * n
        back: list[tuple[int, int]] = [(0, UNK)] * (n + 1)
        for end in range(1, n + 1):
            for start in range(max(0, end - self.max_len), end):
                if best[start] == -math.inf:
                    continue
                tid = self.ids.get(w[start:end])
                if tid is not None:
                    s = best[start] + float(self.scores[tid])
                elif end - start == 1:
                    s, tid = best[start] + self.unk_score, UNK
                else:
                    continue
                if s > best[end]:
                    best[end], back[end] = s, (start, tid)
        out, i = [], n
        while i > 0:
            start, tid = back[i]
            out.append(tid)
            i = start
        return tuple(reversed(out))

    def encode(self, text: str) -> list[int]:
        text = unicodedata.normalize("NFKC", text)
        ids = [BOS]
        for w in text.split():
            ids.extend(self._word(SPACE + w))
            if len(ids) >= self.max_tokens - 1:
                break
        return ids[: self.max_tokens - 1] + [EOS]


class Embedder:
    def __init__(self, cfg, threads: int | None = None):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.enable_cpu_mem_arena = False
        so.enable_mem_pattern = False
        so.intra_op_num_threads = threads or cfg.threads
        so.add_session_config_entry("session.disable_prepacking", "1")
        self.session = ort.InferenceSession(str(cfg.path / "model_int8.onnx"), so, providers=["CPUExecutionProvider"])
        self.tokenizer = UnigramTokenizer(cfg.path / "vocab.json", cfg.max_tokens, cfg.max_piece_len)
        self.batch_size = cfg.batch_size

    def embed(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        batch_size = batch_size or self.batch_size
        out = []
        for i in range(0, len(texts), batch_size):
            enc = [self.tokenizer.encode(t) for t in texts[i:i + batch_size]]
            width = max(len(e) for e in enc)
            ids = np.full((len(enc), width), PAD, dtype=np.int64)
            mask = np.zeros_like(ids)
            for r, e in enumerate(enc):
                ids[r, : len(e)] = e
                mask[r, : len(e)] = 1
            h = self.session.run(None, {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)})[0]
            v = (h * mask[..., None]).sum(1) / mask.sum(1, keepdims=True)
            out.append(v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9))
        return np.vstack(out).astype(np.float32)


def prepare(out_dir: Path, hf_repo: str) -> None:
    """Download the model, quantize it to int8 and write vocab.json."""
    import onnx
    from huggingface_hub import hf_hub_download
    from onnx import TensorProto, numpy_helper
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out_dir.mkdir(parents=True, exist_ok=True)
    src = hf_hub_download(hf_repo, "model_optimized.onnx")
    tok = json.loads(Path(hf_hub_download(hf_repo, "tokenizer.json")).read_text(encoding="utf-8"))

    m = onnx.load(src)
    g = m.graph
    for i, t in enumerate(g.initializer):
        if t.data_type == TensorProto.FLOAT16:
            g.initializer[i].CopyFrom(numpy_helper.from_array(numpy_helper.to_array(t).astype(np.float32), t.name))
    for node in g.node:
        for a in node.attribute:
            if node.op_type == "Cast" and a.name == "to" and a.i == TensorProto.FLOAT16:
                a.i = TensorProto.FLOAT
            if node.op_type == "Constant" and a.name == "value" and a.t.data_type == TensorProto.FLOAT16:
                a.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(a.t).astype(np.float32), a.t.name))
    for vi in [*g.input, *g.output, *g.value_info]:
        if vi.type.tensor_type.elem_type == TensorProto.FLOAT16:
            vi.type.tensor_type.elem_type = TensorProto.FLOAT
    fp32 = out_dir / "model_fp32.onnx"
    onnx.save(m, str(fp32))
    quantize_dynamic(str(fp32), str(out_dir / "model_int8.onnx"), weight_type=QuantType.QUInt8,
                     op_types_to_quantize=["MatMul", "Gather"])
    fp32.unlink()

    vocab = tok["model"]["vocab"]
    (out_dir / "vocab.json").write_text(json.dumps([[p for p, _ in vocab], [round(s, 5) for _, s in vocab]],
                                                   ensure_ascii=False), encoding="utf-8")
    print(f"prepared {out_dir}: model_int8.onnx {round((out_dir / 'model_int8.onnx').stat().st_size / 1e6)} MB")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "prepare":
        from app.config import get_settings
        cfg = get_settings().embedder
        prepare(Path(sys.argv[2]) if len(sys.argv) > 2 else cfg.path, cfg.hf_repo)
