from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


REPOSITORY = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
FILES = (
    "README.md",
    "config.json",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "1_Pooling/config.json",
    "onnx/model_quint8_avx2.onnx",
)


def valid(target: Path) -> bool:
    model = target / "onnx" / "model_quint8_avx2.onnx"
    tokenizer = target / "tokenizer.json"
    return (model.is_file() and model.stat().st_size > 100_000_000
            and tokenizer.is_file() and tokenizer.stat().st_size > 1_000_000)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args()
    target = args.root.resolve() / "models" / "semantic-encoder-v1"
    if not valid(target):
        for filename in FILES:
            source = Path(hf_hub_download(
                repo_id=REPOSITORY,
                filename=filename,
                revision=REVISION,
                local_files_only=args.local_only,
            ))
            destination = target / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    if not valid(target):
        raise RuntimeError("Semantic ONNX encoder validation failed after download.")
    print(json.dumps({
        "repository": REPOSITORY,
        "revision": REVISION,
        "target": str(target),
        "model_bytes": (target / "onnx" / "model_quint8_avx2.onnx").stat().st_size,
        "local_only": args.local_only,
        "text_generation": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
