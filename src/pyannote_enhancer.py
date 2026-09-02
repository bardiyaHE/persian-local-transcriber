"""Create the ASR enhanced branch with Demucs and pyannote Community-1."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np


DEMUCS_MODEL = "htdemucs"
DEMUCS_REVISION = "cbc8a9b1a87023b7fd74e7b3412e6321c0eab003"
PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"


def run_checked(command: list[str], environment: dict[str, str] | None = None) -> dict:
    started = time.perf_counter()
    completed = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", env=environment)
    elapsed = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError(
            f"Enhancement command failed ({completed.returncode}): "
            f"{subprocess.list2cmdline(command)}\n{completed.stderr}"
        )
    return {
        "command": command,
        "seconds": round(elapsed, 6),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def read_pcm16(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as stream:
        if stream.getnchannels() != 1 or stream.getsampwidth() != 2:
            raise ValueError("Expected mono PCM16 WAV")
        rate = stream.getframerate()
        signal = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
    return signal.astype(np.float32) / 32768.0, rate


def write_pcm16(path: Path, signal: np.ndarray, rate: int) -> None:
    pcm = np.clip(signal * 32768.0, -32768, 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(pcm.tobytes())


def speaker_activity(rows: list[dict]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for row in rows:
        totals[str(row["speaker"])] += max(
            0.0, float(row["end"]) - float(row["start"]))
    return dict(totals)


def dominant_speaker(rows: list[dict]) -> tuple[str | None, dict[str, float]]:
    totals = speaker_activity(rows)
    if not totals:
        return None, totals
    return max(totals, key=lambda speaker: (totals[speaker], speaker)), totals


def gate_speaker(audio: Path, output: Path, rows: list[dict], speaker: str,
                 pad_ms: float = 60.0, fade_ms: float = 8.0) -> None:
    signal, rate = read_pcm16(audio)
    envelope = np.zeros(len(signal), dtype=np.float32)
    pad = round(rate * pad_ms / 1000.0)
    fade = round(rate * fade_ms / 1000.0)
    for row in rows:
        if str(row["speaker"]) != speaker:
            continue
        start = max(0, round(float(row["start"]) * rate) - pad)
        end = min(len(signal), round(float(row["end"]) * rate) + pad)
        if end <= start:
            continue
        current = np.ones(end - start, dtype=np.float32)
        current_fade = min(fade, (end - start) // 2)
        if current_fade:
            current[:current_fade] = np.linspace(
                0.0, 1.0, current_fade, endpoint=False, dtype=np.float32)
            current[-current_fade:] = np.linspace(
                1.0, 0.0, current_fade, endpoint=False, dtype=np.float32)
        envelope[start:end] = np.maximum(envelope[start:end], current)
    write_pcm16(output, signal * envelope, rate)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--demucs-python", type=Path, required=True)
    parser.add_argument("--pyannote-python", type=Path, required=True)
    parser.add_argument("--pyannote-runner", type=Path, required=True)
    parser.add_argument("--demucs-cache", type=Path, required=True)
    parser.add_argument("--pyannote-cache", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--segment", type=int, default=7)
    parser.add_argument("--overlap", type=float, default=0.1)
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    demucs_root = args.work_dir / "demucs"
    demucs_environment = os.environ.copy()
    demucs_environment["HF_HOME"] = str(args.demucs_cache.resolve())
    demucs_environment["HF_HUB_OFFLINE"] = "1"
    demucs_environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
    demucs_environment["DO_NOT_TRACK"] = "1"
    demucs = run_checked([
        str(args.demucs_python.resolve()), "-m", "demucs",
        "--two-stems", "vocals", "-n", DEMUCS_MODEL,
        "--device", args.device, "--shifts", "0",
        "--overlap", str(args.overlap), "--segment", str(args.segment),
        "--float32", "--out", str(demucs_root.resolve()),
        str(args.audio.resolve()),
    ], demucs_environment)
    vocals = [path for path in demucs_root.rglob("vocals.wav") if path.is_file()]
    removed = [path for path in demucs_root.rglob("no_vocals.wav") if path.is_file()]
    if len(vocals) != 1:
        raise RuntimeError(f"Demucs produced {len(vocals)} vocals stems instead of one")

    speech_mix = args.work_dir / "speech_without_background_music.wav"
    conversion = run_checked([
        str(args.ffmpeg.resolve()), "-hide_banner", "-y", "-i", str(vocals[0]),
        "-map", "0:a:0", "-vn", "-map_metadata", "-1",
        "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(speech_mix),
    ])

    diarization_root = args.work_dir / "pyannote"
    pyannote_environment = os.environ.copy()
    pyannote_environment["PYANNOTE_METRICS_ENABLED"] = "0"
    pyannote_environment["HF_HUB_OFFLINE"] = "1"
    pyannote_environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
    pyannote_environment["DO_NOT_TRACK"] = "1"
    pyannote = run_checked([
        str(args.pyannote_python.resolve()), str(args.pyannote_runner.resolve()),
        str(speech_mix.resolve()), "--output", str(diarization_root.resolve()),
        "--cache-dir", str(args.pyannote_cache.resolve()),
        "--device", args.device, "--uri", args.work_dir.parent.name,
    ], pyannote_environment)
    diarization = json.loads((diarization_root / "output.json").read_text(encoding="utf-8"))
    regular = list(diarization.get("regular_diarization") or [])
    exclusive = list(diarization.get("exclusive_diarization") or [])
    main_speaker, activity = dominant_speaker(regular)

    if main_speaker is not None and len(activity) > 1:
        gate_speaker(speech_mix, args.output, exclusive, main_speaker)
        selection = "dominant-speaker-gated"
    else:
        shutil.copy2(speech_mix, args.output)
        selection = "single-or-undetected-speaker-preserved"

    report = {
        "pipeline": "Demucs HTDemucs music separation + pyannote Community-1 diarization",
        "audio_uploaded": False,
        "input": str(args.audio.resolve()),
        "output": str(args.output.resolve()),
        "device": args.device,
        "demucs": {
            "model": DEMUCS_MODEL,
            "revision": DEMUCS_REVISION,
            "vocals_stem": str(vocals[0].resolve()),
            "removed_background_stem": str(removed[0].resolve()) if len(removed) == 1 else None,
            "run": demucs,
        },
        "pyannote": {
            "model": PYANNOTE_MODEL,
            "detected_speaker_count": len(activity),
            "speaker_activity_seconds": activity,
            "selected_main_speaker": main_speaker,
            "selection_policy": selection,
            "run": pyannote,
        },
        "conversion": conversion,
        "processing_seconds": round(
            float(demucs["seconds"]) + float(conversion["seconds"]) +
            float(pyannote["seconds"]), 6),
    }
    (args.work_dir / "enhancement-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
