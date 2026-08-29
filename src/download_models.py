from pathlib import Path
import argparse
import hashlib
import json

from huggingface_hub import snapshot_download


MODELS = {
    "medium": {
        "repository": "Systran/faster-whisper-medium",
        "revision": "08e178d48790749d25932bbc082711ddcfdfbc4f",
        "model_bytes": 1527906378,
        "model_sha256": "9b45e1009dcc4ab601eff815b61d80e60ce3fd8c74c1a14f4a282258286b51ae",
    },
    "large-v3-turbo": {
        "repository": "dropbox-dash/faster-whisper-large-v3-turbo",
        "revision": "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf",
        "model_bytes": 1617884929,
        "model_sha256": "e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da",
    },
    "large-v3": {
        "repository": "Systran/faster-whisper-large-v3",
        "revision": "edaa852ec7e145841d8ffdb056a99866b5f0a478",
        "model_bytes": 3087284237,
        "model_sha256": "69f74147e3334731bc3a76048724833325d2ec74642fb52620eda87352e3d4f1",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(path: Path, spec: dict) -> dict:
    required = ["model.bin", "config.json", "tokenizer.json"]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise RuntimeError(f"Model validation failed for {path}: missing {missing}")
    model = path / "model.bin"
    if model.stat().st_size != spec["model_bytes"]:
        raise RuntimeError(f"Unexpected model size for {path}: {model.stat().st_size}")
    actual_hash = sha256(model)
    if actual_hash != spec["model_sha256"]:
        raise RuntimeError(f"SHA-256 mismatch for {model}: {actual_hash}")
    return {name: (path / name).stat().st_size for name in required}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--profile", choices=["lite", "full"], default="full")
    args = parser.parse_args()
    models_root = args.root.resolve() / "models"
    models_root.mkdir(parents=True, exist_ok=True)
    selected = ["large-v3-turbo"] if args.profile == "lite" else list(MODELS)
    manifest = {"profile": args.profile, "models": {}}
    for name in selected:
        spec = MODELS[name]
        repo = spec["repository"]
        target = models_root / name
        print(f"[{name}] validating/downloading {repo} -> {target}", flush=True)
        snapshot_download(
            repo_id=repo,
            revision=spec["revision"],
            local_dir=str(target),
            local_files_only=args.local_only,
            max_workers=4,
        )
        manifest["models"][name] = {
            **spec,
            "files": validate(target, spec),
        }
    (models_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"CTranslate2 models validated for {args.profile} profile: {', '.join(selected)}")


if __name__ == "__main__":
    main()
