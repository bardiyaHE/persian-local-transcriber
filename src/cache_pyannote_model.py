"""Cache the gated pyannote Community-1 pipeline during setup."""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore", message="(?s).*torchcodec is not installed correctly.*", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module=r"pyannote\.audio\.core\.io")

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    from pyannote.audio import Pipeline


MODEL_ID = "pyannote/speaker-diarization-community-1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    pipeline = Pipeline.from_pretrained(
        MODEL_ID, token=token, cache_dir=args.cache_dir)
    if pipeline is None:
        raise RuntimeError(
            "Could not cache pyannote Community-1. Accept its Hugging Face terms and set HF_TOKEN.")
    print(json.dumps({
        "model": MODEL_ID,
        "cache_dir": str(args.cache_dir.resolve()),
        "audio_uploaded": False,
        "telemetry_disabled": True,
    }, indent=2))


if __name__ == "__main__":
    main()
