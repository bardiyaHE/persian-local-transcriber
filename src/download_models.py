from pathlib import Path
import argparse
import json

from huggingface_hub import snapshot_download


MODELS = {
    "medium": "Systran/faster-whisper-medium",
    "large-v3-turbo": "dropbox-dash/faster-whisper-large-v3-turbo",
    "large-v3": "Systran/faster-whisper-large-v3",
}


def validate(path: Path) -> dict:
    required = ["model.bin", "config.json", "tokenizer.json"]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise RuntimeError(f"Model validation failed for {path}: missing {missing}")
    return {name: (path / name).stat().st_size for name in required}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args()
    models_root = args.root.resolve() / "models"
    models_root.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, repo in MODELS.items():
        target = models_root / name
        print(f"[{name}] validating/downloading {repo} -> {target}", flush=True)
        snapshot_download(
            repo_id=repo,
            local_dir=str(target),
            local_files_only=args.local_only,
            max_workers=4,
        )
        manifest[name] = {"repository": repo, "files": validate(target)}
    (models_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("All three CTranslate2 models validated.")


if __name__ == "__main__":
    main()
