from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import sqlite3
from pathlib import Path


REQUIRED_PACKAGES = (
    "faster-whisper",
    "ctranslate2",
    "huggingface-hub",
    "numpy",
    "gradio",
)


def require_file(path: Path, minimum_bytes: int = 1) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"Required file is missing: {path}")
    size = path.stat().st_size
    if size < minimum_bytes:
        raise RuntimeError(f"Required file is incomplete: {path} ({size} bytes)")
    return size


def validate_model(root: Path, name: str) -> dict[str, int]:
    model_dir = root / "models" / name
    return {
        filename: require_file(model_dir / filename)
        for filename in ("model.bin", "config.json", "tokenizer.json")
    }


def validate_corpus(path: Path) -> dict[str, int]:
    require_file(path)
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        if metadata.get("schema_version") != "1":
            raise RuntimeError("Unsupported or incomplete n-gram database schema.")
        counts = {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("sentences", "unigrams", "bigrams", "trigrams")
        }
        if not all(counts.values()):
            raise RuntimeError("The n-gram database contains an empty required table.")
        return counts
    finally:
        connection.close()


def run(root: Path, profile: str) -> dict:
    result = {
        "status": "ok",
        "profile": profile,
        "packages": {name: importlib.metadata.version(name) for name in REQUIRED_PACKAGES},
        "binaries": {},
        "models": {},
    }
    for name, relative in {
        "ffmpeg": "runtime/ffmpeg/ffmpeg.exe",
        "ffprobe": "runtime/ffmpeg/ffprobe.exe",
        "deepfilternet": "runtime/deepfilternet/deep-filter.exe",
    }.items():
        result["binaries"][name] = require_file(root / relative)

    selected_models = (
        ("large-v3-turbo",)
        if profile == "lite" else ("medium", "large-v3-turbo", "large-v3")
    )
    for name in selected_models:
        result["models"][name] = validate_model(root, name)

    if profile == "full":
        qwen = root / "models/qwen3.5-35b-a3b-gguf/Qwen3.5-35B-A3B-UD-Q4_K_L.gguf"
        qwen_bytes = require_file(qwen, minimum_bytes=20_205_632_160)
        if qwen_bytes != 20_205_632_160:
            raise RuntimeError(f"Unexpected Qwen model size: {qwen_bytes}")
        runtime_name = "cuda" if shutil.which("nvidia-smi") else "cpu"
        llama_server = root / f"runtime/llama.cpp/{runtime_name}/llama-server.exe"
        encoder = root / "models/semantic-encoder-v1"
        lexicon = root / "offline-lexicon/combined-medical-drug-index.json"
        require_file(lexicon, minimum_bytes=10_000_000)
        lexicon_payload = json.loads(lexicon.read_text(encoding="utf-8"))
        if int(lexicon_payload.get("unique_terms") or 0) < 50_000:
            raise RuntimeError("Bundled public lexicon is incomplete.")
        for overlay in (
            "fa_modern_spoken_allowlist.txt",
            "fa_spoken_medical_concepts.txt",
            "fa_user_blocklist.txt",
            "LEXICON_SOURCES.md",
        ):
            require_file(root / "offline-lexicon" / overlay)
        result["full"] = {
            "qwen_bytes": qwen_bytes,
            "llama_server_bytes": require_file(llama_server),
            "semantic_encoder_bytes": require_file(
                encoder / "onnx/model_quint8_avx2.onnx", minimum_bytes=100_000_000
            ),
            "lexicon_terms": int(lexicon_payload["unique_terms"]),
            "corpus_rows": validate_corpus(root / "offline-corpus/domain-ngrams-v1.sqlite3"),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a local installation without user data.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--profile", choices=["lite", "full"], required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root.resolve(), args.profile), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
