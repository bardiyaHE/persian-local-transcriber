from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


class OnnxSentenceEncoder:
    """Small local sentence encoder with mean pooling and cosine normalization.

    The encoder never generates text.  Its only output is a 384-dimensional
    vector used to rank candidates that already exist in the six-ASR lattice.
    """

    def __init__(self, model_dir: Path, max_length: int = 128,
                 threads: int | None = None) -> None:
        self.model_dir = model_dir.resolve()
        self.model_path = self.model_dir / "onnx" / "model_quint8_avx2.onnx"
        self.tokenizer_path = self.model_dir / "tokenizer.json"
        for required in (self.model_path, self.tokenizer_path):
            if not required.is_file():
                raise FileNotFoundError(f"Semantic encoder asset is missing: {required}")

        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        self.tokenizer.enable_truncation(max_length=max_length)
        pad_id = self.tokenizer.token_to_id("<pad>")
        if pad_id is None:
            raise RuntimeError("Semantic encoder tokenizer has no <pad> token.")
        self.tokenizer.enable_padding(pad_id=pad_id, pad_token="<pad>")

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads is not None and threads > 0:
            options.intra_op_num_threads = int(threads)
        available = ort.get_available_providers()
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if "CUDAExecutionProvider" in available
                     else ["CPUExecutionProvider"])
        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=options, providers=providers)
        self.provider = self.session.get_providers()[0]
        self.dimension = 384
        self.max_length = max_length

    def encode(self, texts: Iterable[str], batch_size: int = 64) -> np.ndarray:
        values = [str(text).strip() for text in texts]
        if not values:
            return np.empty((0, self.dimension), dtype=np.float32)
        batches: list[np.ndarray] = []
        for offset in range(0, len(values), batch_size):
            encoded = self.tokenizer.encode_batch(values[offset:offset + batch_size])
            input_ids = np.asarray([row.ids for row in encoded], dtype=np.int64)
            attention_mask = np.asarray(
                [row.attention_mask for row in encoded], dtype=np.int64)
            token_type_ids = np.asarray([row.type_ids for row in encoded], dtype=np.int64)
            hidden = self.session.run(None, {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            })[0]
            mask = attention_mask[:, :, None].astype(np.float32)
            pooled = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1.0)
            pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9)
            batches.append(pooled.astype(np.float32, copy=False))
        return np.concatenate(batches, axis=0)

    @staticmethod
    def cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        if not len(left) or not len(right):
            return np.empty((len(left), len(right)), dtype=np.float32)
        return left @ right.T

    def describe(self) -> dict[str, object]:
        digest = hashlib.sha256()
        with self.model_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return {
            "repository": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "revision": "e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
            "license": "Apache-2.0",
            "language_support": "multilingual including fa",
            "architecture": "MiniLM sentence encoder; ONNX UINT8 AVX2",
            "embedding_dimension": self.dimension,
            "max_tokens": self.max_length,
            "provider": self.provider,
            "model_path": str(self.model_path),
            "model_bytes": self.model_path.stat().st_size,
            "sha256": digest.hexdigest(),
            "text_generation": False,
        }
