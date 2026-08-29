from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import speech_recognition as sr

from consensus_v5 import protect_honorific_names


OUTPUT_RELATIVE = Path("final-delivery") / "12-google-recognition-fallback"
PLACEHOLDER_RE = re.compile(r"\[[^\]]*نامفهوم[^\]]*\]")
TOKEN_RE = re.compile(r"[\u0600-\u06ffA-Za-z0-9]+")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("ي", "ی").replace("ك", "ک")).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_metrics(text: str) -> dict[str, Any]:
    normalized = normalize_text(text)
    tokens = TOKEN_RE.findall(normalized)
    placeholders = PLACEHOLDER_RE.findall(normalized)
    letters = [character for character in normalized if character.isalpha()]
    persian_letters = sum("\u0600" <= character <= "\u06ff" for character in letters)
    return {
        "token_count": len(tokens),
        "placeholder_count": len(placeholders),
        "placeholder_ratio": round(len(placeholders) / max(1, len(tokens)), 4),
        "non_persian_letter_ratio": round(
            1.0 - (persian_letters / len(letters)), 4) if letters else 0.0,
    }


def fallback_reasons(local_text: str, duration_seconds: float) -> list[str]:
    metrics = text_metrics(local_text)
    reasons: list[str] = []
    if metrics["token_count"] == 0:
        reasons.append("local-transcript-empty")
    if (duration_seconds >= 8.0
            and metrics["token_count"] < max(3, round(duration_seconds * 0.08))):
        reasons.append("local-transcript-too-short-for-duration")
    if metrics["placeholder_count"] >= 4 and metrics["placeholder_ratio"] >= 0.16:
        reasons.append("local-placeholder-ratio-high")
    if (metrics["placeholder_count"] >= 3
            and metrics["non_persian_letter_ratio"] >= 0.08):
        reasons.append("local-language-anomaly-high")
    return reasons


def make_chunks(ffmpeg: Path, source: Path, target_dir: Path,
                chunk_seconds: float) -> list[Path]:
    pattern = target_dir / "chunk-%03d.wav"
    command = [
        str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source), "-map", "0:a:0", "-vn", "-map_metadata", "-1",
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        "-f", "segment", "-segment_time", f"{chunk_seconds:.3f}",
        "-reset_timestamps", "1", str(pattern),
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False)
    if completed.returncode:
        raise RuntimeError(f"FFmpeg chunking failed: {completed.stderr[-1000:]}")
    chunks = sorted(target_dir.glob("chunk-*.wav"))
    if not chunks:
        raise RuntimeError("FFmpeg did not create a Google Recognition audio chunk")
    return chunks


def recognize_chunk(recognizer: sr.Recognizer, chunk: Path,
                    language: str) -> tuple[str, float | None]:
    with sr.AudioFile(str(chunk)) as source:
        audio = recognizer.record(source)
    result = recognizer.recognize_google(
        audio, language=language, pfilter=0, with_confidence=True)
    if isinstance(result, tuple):
        text, confidence = result
        return normalize_text(str(text)), float(confidence)
    return normalize_text(str(result)), None


def recognize_audio(source: Path, ffmpeg: Path, language: str,
                    timeout_seconds: float, chunk_seconds: float) \
        -> tuple[str, list[dict[str, Any]]]:
    recognizer = sr.Recognizer()
    recognizer.operation_timeout = timeout_seconds
    rows: list[dict[str, Any]] = []
    text_parts: list[str] = []
    with tempfile.TemporaryDirectory(prefix="google-recognition-") as temp:
        chunks = make_chunks(ffmpeg, source, Path(temp), chunk_seconds)
        for index, chunk in enumerate(chunks):
            started = time.perf_counter()
            try:
                text, confidence = recognize_chunk(recognizer, chunk, language)
                if text:
                    text_parts.append(text)
                rows.append({
                    "index": index, "recognized": bool(text), "text": text,
                    "confidence": confidence,
                    "seconds": round(time.perf_counter() - started, 3),
                    "error": None,
                })
            except sr.UnknownValueError:
                text_parts.append("[نامفهوم]")
                rows.append({
                    "index": index, "recognized": False, "text": "",
                    "confidence": None,
                    "seconds": round(time.perf_counter() - started, 3),
                    "error": "unknown-value",
                })
            except (sr.RequestError, OSError, TimeoutError) as error:
                rows.append({
                    "index": index, "recognized": False, "text": "",
                    "confidence": None,
                    "seconds": round(time.perf_counter() - started, 3),
                    "error": f"{type(error).__name__}: {error}",
                })
    return normalize_text(" ".join(text_parts)), rows


def google_text_is_usable(text: str, chunks: list[dict[str, Any]]) -> bool:
    metrics = text_metrics(text)
    recognized = sum(bool(row.get("recognized")) for row in chunks)
    return bool(metrics["token_count"] >= 2 and recognized >= 1)


def load_cache(cache_path: Path, audio_sha256: str) -> dict[str, Any] | None:
    if not cache_path.is_file():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return payload if payload.get("audio_sha256") == audio_sha256 else None


def run(run_dir: Path, ffmpeg: Path, timeout_seconds: float = 12.0,
        chunk_seconds: float = 45.0, language: str = "fa-IR",
        force: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = run_dir / OUTPUT_RELATIVE
    output_dir.mkdir(parents=True, exist_ok=True)
    output_text = output_dir / "google-recognition.txt"
    output_json = output_dir / "google-recognition.json"
    local_path = run_dir / "final-delivery" / "10-local-qwen-reranker" / "final-v10.txt"
    run_summary_path = run_dir / "run-summary.json"
    if not local_path.is_file() or not run_summary_path.is_file():
        raise FileNotFoundError("V10 transcript or run summary is missing")
    if not ffmpeg.is_file():
        raise FileNotFoundError(f"FFmpeg is missing: {ffmpeg}")
    local_text = normalize_text(local_path.read_text(encoding="utf-8-sig"))
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    duration = float(
        ((run_summary.get("raw_metadata") or {}).get("format") or {}).get("duration") or 0.0)
    reasons = fallback_reasons(local_text, duration)
    source = run_dir / "normalized_mono_48k.wav"
    if not source.is_file():
        raise FileNotFoundError(f"Normalized audio is missing: {source}")
    audio_sha256 = sha256_file(source)
    cache_path = run_dir.parent.parent / "runtime" / "google-speech-recognition-cache" \
        / f"{audio_sha256}.json"
    requested = bool(force or reasons)
    cache_hit = False
    text = ""
    chunks: list[dict[str, Any]] = []
    error: str | None = None
    if requested:
        cached = load_cache(cache_path, audio_sha256)
        if cached:
            cache_hit = True
            text = normalize_text(str(cached.get("text") or ""))
            chunks = list(cached.get("chunks") or [])
        else:
            try:
                text, chunks = recognize_audio(
                    source, ffmpeg, language, timeout_seconds, chunk_seconds)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps({
                    "provider": "Google Speech Recognition (generic key)",
                    "audio_sha256": audio_sha256, "language": language,
                    "text": text, "chunks": chunks,
                }, ensure_ascii=False, indent=2), encoding="utf-8")
            except (OSError, RuntimeError, ValueError, TypeError) as exc:
                error = f"{type(exc).__name__}: {exc}"
    protected_text, protected_slots = protect_honorific_names(text)
    selected = requested and google_text_is_usable(protected_text, chunks)
    if selected:
        output_text.write_text(protected_text + "\n", encoding="utf-8")
    elif output_text.exists():
        output_text.unlink()
    payload = {
        "provider": "Google Speech Recognition (generic key)",
        "language": language,
        "requested": requested,
        "selected": selected,
        "selection_reasons": reasons + (["forced"] if force and not reasons else []),
        "fail_open": True,
        "cache_hit": cache_hit,
        "audio_sha256": audio_sha256,
        "source_audio": str(source),
        "local_transcript": str(local_path),
        "local_metrics": text_metrics(local_text),
        "google_metrics": text_metrics(protected_text),
        "google_transcript": str(output_text) if selected else None,
        "protected_name_slots": protected_slots,
        "chunk_count": len(chunks),
        "recognized_chunk_count": sum(bool(row.get("recognized")) for row in chunks),
        "chunks": chunks,
        "error": error,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "external_audio_sent": bool(requested and not cache_hit and chunks),
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    review_lines = [
        "# Google Recognition fallback", "",
        f"- درخواست شد: `{'بله' if requested else 'خیر'}`",
        f"- برای خروجی انتخاب شد: `{'بله' if selected else 'خیر'}`",
        f"- کش محلی: `{'بله' if cache_hit else 'خیر'}`",
        f"- دلیل‌ها: `{', '.join(reasons) if reasons else 'کیفیت محلی قابل قبول'}`",
    ]
    if error:
        review_lines.append(f"- خطای غیرمسدودکننده: `{error}`")
    (output_dir / "review-google-recognition.md").write_text(
        "\n".join(review_lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="Use free Google Speech Recognition only for poor local transcripts.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--chunk-seconds", type=float, default=45.0)
    parser.add_argument("--language", default="fa-IR")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = run(
        args.run_dir.resolve(), args.ffmpeg.resolve(), args.timeout,
        args.chunk_seconds, args.language, args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
