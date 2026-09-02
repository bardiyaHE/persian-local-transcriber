"""Offline pyannote Community-1 diarization for a pre-normalized PCM16 WAV."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
import wave
from pathlib import Path

warnings.filterwarnings(
    "ignore", message="(?s).*torchcodec is not installed correctly.*", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module=r"pyannote\.audio\.core\.io")

import numpy as np
import torch
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    from pyannote.audio import Pipeline


MODEL_ID = "pyannote/speaker-diarization-community-1"


def load_pcm16_waveform(path: Path) -> dict:
    with wave.open(str(path), "rb") as stream:
        channels = stream.getnchannels()
        sample_width = stream.getsampwidth()
        sample_rate = stream.getframerate()
        frames = stream.readframes(stream.getnframes())
    if sample_width != 2:
        raise ValueError("pyannote input must be PCM16 WAV")
    samples = np.frombuffer(frames, dtype="<i2").reshape(-1, channels)
    waveform = torch.from_numpy((samples.astype(np.float32) / 32768.0).T.copy())
    return {"waveform": waveform, "sample_rate": sample_rate}


def annotation_rows(annotation) -> list[dict]:
    return [
        {"start": round(segment.start, 3), "end": round(segment.end, 3),
         "speaker": speaker}
        for segment, _, speaker in annotation.itertracks(yield_label=True)
    ]


def write_rttm(rows: list[dict], path: Path, uri: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            duration = float(row["end"]) - float(row["start"])
            stream.write(
                f"SPEAKER {uri} 1 {float(row['start']):.3f} {duration:.3f} "
                f"<NA> <NA> {row['speaker']} <NA> <NA>\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--uri", default="audio")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    args.output.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for pyannote but this Torch runtime has no CUDA")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline = Pipeline.from_pretrained(
            MODEL_ID, token=token, cache_dir=args.cache_dir)
        if pipeline is None:
            raise RuntimeError(
                "pyannote Community-1 could not be loaded; accept its model terms and set HF_TOKEN"
            )
        pipeline.to(torch.device(args.device))
        audio = load_pcm16_waveform(args.audio)
        audio["uri"] = args.uri
        output = pipeline(audio)

    regular = annotation_rows(output.speaker_diarization)
    exclusive = annotation_rows(output.exclusive_speaker_diarization)
    write_rttm(regular, args.output / "regular.rttm", args.uri)
    write_rttm(exclusive, args.output / "exclusive.rttm", args.uri)
    payload = {
        "model": MODEL_ID,
        "device": args.device,
        "audio_uploaded": False,
        "telemetry_disabled": True,
        "regular_diarization": regular,
        "exclusive_diarization": exclusive,
        "detected_speakers": sorted({row["speaker"] for row in regular}),
        "detected_speaker_count": len({row["speaker"] for row in regular}),
        "processing_seconds": round(time.perf_counter() - started, 6),
        "warnings": [str(item.message) for item in caught],
    }
    (args.output / "output.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
