from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel


MODEL_ORDER = ["medium", "large-v3-turbo", "large-v3"]
SECONDARY_MODEL_ORDER = ["medium", "large-v3"]
SOURCE_ORDER = ["raw", "enhanced"]
SENSITIVE = {
    "قرص", "کپسول", "شربت", "آمپول", "دارو", "دوز", "واحد", "درصد",
    "میلیگرم", "میکروگرم", "گرم", "کیلوگرم", "میلیلیتر", "لیتر",
    "منفی", "نه", "نیست", "نبود", "ندارد", "نخورید", "قطع",
    "یک", "دو", "سه", "چهار", "پنج", "شش", "هفت", "هشت", "نه", "ده",
}
DRUG_CONTEXT = {"قرص", "کپسول", "شربت", "آمپول", "دارو", "دوز", "مصرف"}
DRUG_SUFFIXES = ("سین", "مایسین", "زول", "پرازول", "پام", "پرین", "کسین", "فلوکس")


def run(cmd: list[str], log, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    rendered = subprocess.list2cmdline(cmd)
    log.write(f"$ {rendered}\n")
    log.flush()
    result = subprocess.run(cmd, cwd=cwd, text=True, encoding="utf-8", errors="replace",
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write(result.stdout + "\n")
    log.flush()
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {rendered}\n{result.stdout}")
    return result


def probe(ffprobe: Path, audio: Path, log) -> dict[str, Any]:
    result = run([str(ffprobe), "-v", "error", "-show_entries",
                  "format=duration,size,bit_rate,format_name:stream=codec_name,sample_rate,channels,bit_rate",
                  "-select_streams", "a:0", "-of", "json", str(audio)], log)
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise ValueError(
            "FFprobe found no decodable audio stream. The filename extension is ignored, "
            "but the file content must contain audio supported by the bundled FFmpeg.")
    payload["content_detection"] = {
        "extension_ignored": True,
        "detected_audio_codec": streams[0].get("codec_name"),
        "detected_container": (payload.get("format") or {}).get("format_name"),
    }
    return payload


def stamp(seconds: float, vtt: bool = False) -> str:
    millis = max(0, round(seconds * 1000))
    h, rem = divmod(millis, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    sep = "." if vtt else ","
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("ي", "ی").replace("ك", "ک")
    return re.sub(r"[^\w\u0600-\u06ff]+", "", text).casefold()


def segment_dict(seg) -> dict[str, Any]:
    words = []
    for word in seg.words or []:
        words.append({"start": word.start, "end": word.end, "word": word.word,
                      "probability": word.probability})
    return {
        "id": seg.id, "seek": seg.seek, "start": seg.start, "end": seg.end,
        "text": seg.text, "tokens": list(seg.tokens), "temperature": seg.temperature,
        "avg_logprob": seg.avg_logprob, "compression_ratio": seg.compression_ratio,
        "no_speech_prob": seg.no_speech_prob, "words": words,
    }


def write_subtitles(segments: list[dict[str, Any]], base: Path) -> None:
    srt, vtt = [], ["WEBVTT", ""]
    for idx, seg in enumerate(segments, 1):
        text = seg["text"].strip()
        srt += [str(idx), f"{stamp(seg['start'])} --> {stamp(seg['end'])}", text, ""]
        vtt += [f"{stamp(seg['start'], True)} --> {stamp(seg['end'], True)}", text, ""]
    base.with_suffix(".srt").write_text("\n".join(srt), encoding="utf-8")
    base.with_suffix(".vtt").write_text("\n".join(vtt), encoding="utf-8")


def transcribe_one(model_path: Path, audio: Path, out_base: Path, model_name: str,
                   source: str, device: str, compute_type: str, threads: int, log,
                   loaded_model: WhisperModel | None = None,
                   model_load_seconds: float = 0.0) -> dict[str, Any]:
    warning = None
    owns_model = loaded_model is None
    model = loaded_model
    if model is None:
        started = time.perf_counter()
        try:
            model = WhisperModel(str(model_path), device=device, compute_type=compute_type,
                                 cpu_threads=threads, num_workers=1, local_files_only=True)
        except Exception as exc:
            raise RuntimeError(
                f"Failed loading {model_name} on {device}/{compute_type}; no silent CPU fallback. {exc}"
            ) from exc
        model_load_seconds = time.perf_counter() - started
    transcribe_started = time.perf_counter()
    try:
        iterator, info = model.transcribe(
            str(audio), language="fa", task="transcribe", beam_size=5, patience=1.0,
            temperature=[0.0, 0.2, 0.4], compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0, no_speech_threshold=0.6,
            condition_on_previous_text=False, prompt_reset_on_temperature=0.5,
            vad_filter=True, vad_parameters={"threshold": 0.5, "min_speech_duration_ms": 250,
                                             "min_silence_duration_ms": 500, "speech_pad_ms": 300},
            word_timestamps=True, hallucination_silence_threshold=2.0,
        )
        segments = [segment_dict(seg) for seg in iterator]
    except Exception as exc:
        raise RuntimeError(f"Transcription failed for {model_name}/{source}: {exc}") from exc
    transcribe_seconds = time.perf_counter() - transcribe_started
    text = "".join(seg["text"] for seg in segments).strip()
    payload = {
        "model": model_name, "source": source, "model_path": str(model_path),
        "audio": str(audio), "hardware": platform.processor(), "device": device,
        "compute_type": compute_type, "cpu_threads": threads,
        "detected_language": info.language,
        "detected_language_probability": info.language_probability,
        "duration": info.duration, "duration_after_vad": info.duration_after_vad,
        "load_seconds": model_load_seconds, "transcription_seconds": transcribe_seconds,
        "warning": warning, "error": None, "text": text, "segments": segments,
    }
    out_base.parent.mkdir(parents=True, exist_ok=True)
    out_base.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
    out_base.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_subtitles(segments, out_base)
    log.write(f"Completed {model_name}/{source}: load={model_load_seconds:.2f}s transcribe={transcribe_seconds:.2f}s\n")
    if owns_model:
        del model
        gc.collect()
    return payload


def empty_hypothesis(audio: Path, out_base: Path, model_name: str, source: str,
                     device: str, compute_type: str, threads: int,
                     reason: str) -> dict[str, Any]:
    """Write a valid empty secondary hypothesis when Turbo needs no review."""
    payload = {
        "model": model_name, "source": source, "model_path": None,
        "audio": str(audio), "hardware": platform.processor(), "device": device,
        "compute_type": compute_type, "cpu_threads": threads,
        "detected_language": "fa", "detected_language_probability": None,
        "duration": 0.0, "duration_after_vad": 0.0,
        "load_seconds": 0.0, "transcription_seconds": 0.0,
        "warning": reason, "error": None, "text": "", "segments": [],
        "selective_intervals": [], "selective_secondary_asr": True,
    }
    out_base.parent.mkdir(parents=True, exist_ok=True)
    out_base.with_suffix(".txt").write_text("\n", encoding="utf-8")
    out_base.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_subtitles([], out_base)
    return payload


def merge_intervals(intervals: list[dict[str, Any]], duration: float,
                    padding: float = 1.15, join_gap: float = 0.85) -> list[dict[str, Any]]:
    expanded = []
    for row in intervals:
        start = max(0.0, float(row["start"]) - padding)
        end = min(duration, float(row["end"]) + padding)
        if end <= start:
            continue
        expanded.append({"start": start, "end": end, "reasons": set(row["reasons"])})
    expanded.sort(key=lambda row: (row["start"], row["end"]))
    merged: list[dict[str, Any]] = []
    for row in expanded:
        if merged and row["start"] <= merged[-1]["end"] + join_gap:
            merged[-1]["end"] = max(merged[-1]["end"], row["end"])
            merged[-1]["reasons"].update(row["reasons"])
        else:
            merged.append(row)
    return [{"start": round(row["start"], 3), "end": round(row["end"], 3),
             "reasons": sorted(row["reasons"])} for row in merged]


def turbo_uncertainty_intervals(raw_payload: dict[str, Any],
                                enhanced_payload: dict[str, Any],
                                duration: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Find spans that deserve an independent ASR family.

    This is deliberately recall-oriented: it can spend more compute, but it
    must not silently skip a low-confidence or medically sensitive Turbo span.
    """
    raw_segments = list(raw_payload.get("segments") or [])
    raw_words = words_of(raw_payload)
    flagged: list[dict[str, Any]] = []
    segment_audit: list[dict[str, Any]] = []
    for segment in enhanced_payload.get("segments") or []:
        segment_reasons: set[str] = set()
        word_reasons_seen: set[str] = set()
        segment_words = list(segment.get("words") or [])
        probabilities = [float(row.get("probability") or 0.0) for row in segment_words]
        if float(segment.get("avg_logprob") or -2.0) < -0.72:
            segment_reasons.add("low-segment-logprob")
        if float(segment.get("compression_ratio") or 0.0) > 2.15:
            segment_reasons.add("high-compression-ratio")
        if float(segment.get("no_speech_prob") or 0.0) > 0.38:
            segment_reasons.add("possible-nonspeech")
        if probabilities and sum(probabilities) / len(probabilities) < 0.63:
            segment_reasons.add("low-mean-word-probability")

        overlapping_raw_segments = [row for row in raw_segments
                                    if min(float(row["end"]), float(segment["end"]))
                                    > max(float(row["start"]), float(segment["start"]))]
        raw_text = " ".join(str(row.get("text") or "") for row in overlapping_raw_segments)
        normalized_enhanced = " ".join(norm(str(segment.get("text") or "")).split())
        normalized_raw = " ".join(norm(raw_text).split())
        agreement = SequenceMatcher(
            None, normalized_enhanced, normalized_raw, autojunk=False).ratio()
        if normalized_raw and agreement < 0.62:
            segment_reasons.add("severe-raw-enhanced-disagreement")
        if not normalized_raw:
            segment_reasons.add("missing-raw-counterpart")

        for index, word in enumerate(segment_words):
            word_reasons: set[str] = set()
            token = norm(str(word.get("word") or ""))
            nearby = {norm(str(item.get("word") or ""))
                      for item in segment_words[max(0, index - 2):index + 3]}
            if float(word.get("probability") or 0.0) < 0.43:
                word_reasons.add("very-low-word-probability")
            if (re.search(r"\d|[۰-۹]", token) or token in SENSITIVE
                    or nearby & DRUG_CONTEXT or token.endswith(DRUG_SUFFIXES)):
                word_reasons.add("critical-medical-number-or-negation")
            best_raw = max(raw_words, key=lambda row: temporal_similarity(word, row), default=None)
            if best_raw and temporal_similarity(word, best_raw) >= 0.45:
                lexical = SequenceMatcher(
                    None, token, norm(str(best_raw.get("word") or "")),
                    autojunk=False).ratio()
                if lexical < 0.72:
                    word_reasons.add("word-level-turbo-disagreement")
            elif normalized_raw:
                word_reasons.add("missing-raw-word-counterpart")
            if word_reasons:
                word_reasons_seen.update(word_reasons)
                flagged.append({
                    "start": float(word["start"]), "end": float(word["end"]),
                    "reasons": sorted(word_reasons),
                })

        if segment_reasons:
            flagged.append({
                "start": float(segment["start"]), "end": float(segment["end"]),
                "reasons": sorted(segment_reasons),
            })
        reasons = segment_reasons | word_reasons_seen

        audit = {
            "start": round(float(segment["start"]), 3),
            "end": round(float(segment["end"]), 3),
            "text": str(segment.get("text") or "").strip(),
            "raw_enhanced_similarity": round(agreement, 6),
            "reasons": sorted(reasons),
            "needs_secondary_asr": bool(reasons),
        }
        segment_audit.append(audit)

    intervals = merge_intervals(flagged, duration)
    covered = sum(float(row["end"]) - float(row["start"]) for row in intervals)
    coverage = min(1.0, covered / max(duration, 0.001))
    return intervals, {
        "policy": "Turbo raw+enhanced first; secondary families only on uncertain spans",
        "audio_duration_seconds": round(duration, 3),
        "review_interval_count": len(intervals),
        "review_coverage_ratio": round(coverage, 6),
        "segment_audit": segment_audit,
    }


def extract_interval(ffmpeg: Path, audio: Path, target: Path,
                     start: float, end: float, log) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    run([str(ffmpeg), "-hide_banner", "-y", "-ss", f"{start:.3f}",
         "-t", f"{max(0.05, end - start):.3f}", "-i", str(audio),
         "-vn", "-map_metadata", "-1", "-ac", "1", "-ar", "48000",
         "-c:a", "pcm_s16le", str(target)], log)


def transcribe_selective(model_path: Path, audio: Path, out_base: Path,
                         model_name: str, source: str, device: str,
                         compute_type: str, threads: int, log,
                         loaded_model: WhisperModel,
                         model_load_seconds: float,
                         intervals: list[dict[str, Any]], ffmpeg: Path,
                         work_dir: Path) -> dict[str, Any]:
    """Transcribe only Turbo-flagged intervals and restore global timestamps."""
    all_segments: list[dict[str, Any]] = []
    texts: list[str] = []
    total_transcription = 0.0
    language_probabilities: list[float] = []
    for interval_index, interval in enumerate(intervals):
        clip = work_dir / source / f"clip-{interval_index + 1:03d}.wav"
        extract_interval(ffmpeg, audio, clip, float(interval["start"]),
                         float(interval["end"]), log)
        temp_base = work_dir / "hypotheses" / source / f"clip-{interval_index + 1:03d}"
        payload = transcribe_one(
            model_path, clip, temp_base, model_name, source, device,
            compute_type, threads, log, loaded_model=loaded_model,
            model_load_seconds=(model_load_seconds if interval_index == 0 else 0.0))
        offset = float(interval["start"])
        for segment in payload["segments"]:
            adjusted = dict(segment)
            adjusted["id"] = len(all_segments)
            adjusted["start"] = float(adjusted["start"]) + offset
            adjusted["end"] = float(adjusted["end"]) + offset
            adjusted["words"] = [
                {**word, "start": float(word["start"]) + offset,
                 "end": float(word["end"]) + offset}
                for word in adjusted.get("words") or []
            ]
            all_segments.append(adjusted)
        if payload.get("text"):
            texts.append(str(payload["text"]).strip())
        total_transcription += float(payload.get("transcription_seconds") or 0.0)
        if payload.get("detected_language_probability") is not None:
            language_probabilities.append(float(payload["detected_language_probability"]))

    text = " ".join(texts).strip()
    result = {
        "model": model_name, "source": source, "model_path": str(model_path),
        "audio": str(audio), "hardware": platform.processor(), "device": device,
        "compute_type": compute_type, "cpu_threads": threads,
        "detected_language": "fa",
        "detected_language_probability": (
            sum(language_probabilities) / len(language_probabilities)
            if language_probabilities else None),
        "duration": sum(float(row["end"]) - float(row["start"]) for row in intervals),
        "duration_after_vad": None,
        "load_seconds": model_load_seconds,
        "transcription_seconds": total_transcription,
        "warning": None, "error": None, "text": text, "segments": all_segments,
        "selective_intervals": intervals, "selective_secondary_asr": True,
    }
    out_base.parent.mkdir(parents=True, exist_ok=True)
    out_base.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
    out_base.with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_subtitles(all_segments, out_base)
    return result


def words_of(payload: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for seg in payload["segments"]:
        for word in seg["words"]:
            item = dict(word)
            item["avg_logprob"] = seg["avg_logprob"]
            item["model"] = payload["model"]
            item["source"] = payload["source"]
            result.append(item)
    return result


def overlap(a: dict[str, Any], b: dict[str, Any]) -> float:
    return max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def temporal_similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    intersection = overlap(a, b)
    longest = max(a["end"] - a["start"], b["end"] - b["start"], 0.001)
    return intersection / longest


def locked(base_words: list[dict[str, Any]], index: int) -> tuple[bool, str]:
    token = norm(base_words[index]["word"])
    nearby = {norm(w["word"]) for w in base_words[max(0, index-2):index+3]}
    if re.search(r"\d|[۰-۹]", token):
        return True, "number"
    if token in SENSITIVE:
        return True, "medical-unit-negation"
    if token.startswith(("نمی", "نخوا", "نباید", "ندار", "نبود")):
        return True, "negation"
    if nearby & DRUG_CONTEXT or token.endswith(DRUG_SUFFIXES):
        return True, "possible-drug-context"
    return False, ""


def merge(hypotheses: dict[str, dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    base_key = "large-v3__enhanced"
    base_words = words_of(hypotheses[base_key])
    all_words = {key: words_of(value) for key, value in hypotheses.items()}
    decisions = []
    output = []
    for idx, base in enumerate(base_words):
        is_locked, lock_reason = locked(base_words, idx)
        family_votes: dict[str, dict[str, Any]] = {}
        for key, candidates in all_words.items():
            model = hypotheses[key]["model"]
            best = max(candidates, key=lambda w: temporal_similarity(base, w), default=None)
            if best and temporal_similarity(base, best) >= 0.55 and best["probability"] >= 0.55 and best["avg_logprob"] >= -0.8:
                current = family_votes.get(model)
                if current is None or temporal_similarity(base, best) > temporal_similarity(base, current):
                    family_votes[model] = best
        grouped: dict[str, list[dict[str, Any]]] = {}
        for vote in family_votes.values():
            grouped.setdefault(norm(vote["word"]), []).append(vote)
        ranked = sorted(grouped.items(), key=lambda item: (len(item[1]), sum(v["probability"] for v in item[1])), reverse=True)
        chosen = base["word"]
        action = "keep-base"
        support = []
        if ranked:
            candidate_norm, votes = ranked[0]
            support = [{"model": v["model"], "source": v["source"], "word": v["word"],
                        "probability": v["probability"], "avg_logprob": v["avg_logprob"]} for v in votes]
            if not is_locked and candidate_norm != norm(base["word"]) and len({v["model"] for v in votes}) >= 2:
                chosen = max(votes, key=lambda v: v["probability"])["word"]
                action = "replace-consensus"
                if output and norm(base["word"]) != norm(chosen):
                    previous = norm(output[-1])
                    candidate = norm(chosen)
                    if previous == candidate or (len(previous) >= 2 and candidate.startswith(previous)):
                        chosen = base["word"]
                        action = "keep-base-prevent-temporal-reuse"
        output.append(chosen)
        decisions.append({"start": base["start"], "end": base["end"], "base": base["word"],
                          "chosen": chosen, "action": action, "locked": is_locked,
                          "lock_reason": lock_reason, "support": support})
    if len(output) != len(base_words):
        raise AssertionError("Merge invariant failed: base-word count changed.")
    return "".join(output).strip(), decisions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--compute-type", required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--run-id", help="Optional safe run identifier supplied by the local UI.")
    parser.add_argument(
        "--adaptive-turbo", action="store_true",
        help=("Transcribe Turbo raw+enhanced first and run medium/large-v3 only "
              "on uncertain or medically sensitive intervals."),
    )
    args = parser.parse_args()
    root, audio = args.root.resolve(), args.audio.resolve()
    if not audio.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio}")
    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id):
        raise ValueError("run-id may contain only letters, digits, underscore, and hyphen")
    run_dir = root / "outputs" / run_id
    delivery = run_dir / "final-delivery"
    run_dir.mkdir(parents=True)
    with (run_dir / "run-log.txt").open("w", encoding="utf-8") as log:
        log.write(f"run_id={run_id}\nsource={audio}\n")
        ffmpeg = root / "runtime" / "ffmpeg" / "ffmpeg.exe"
        ffprobe = root / "runtime" / "ffmpeg" / "ffprobe.exe"
        deepfilter = root / "runtime" / "deepfilternet" / "deep-filter.exe"
        for tool in (ffmpeg, ffprobe, deepfilter):
            if not tool.is_file():
                raise FileNotFoundError(f"Required local tool missing: {tool}. Run setup.ps1.")
        raw_meta = probe(ffprobe, audio, log)
        input_copy = root / "inputs" / f"{run_id}_{audio.name}"
        shutil.copy2(audio, input_copy)
        normalized = run_dir / "normalized_mono_48k.wav"
        start = time.perf_counter()
        run([str(ffmpeg), "-hide_banner", "-y", "-i", str(input_copy), "-map", "0:a:0",
             "-vn", "-map_metadata", "-1",
             "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(normalized)], log)
        normalize_seconds = time.perf_counter() - start
        normalized_meta = probe(ffprobe, normalized, log)
        df_temp = run_dir / "deepfilter-work"
        df_temp.mkdir()
        start = time.perf_counter()
        df_result = run([str(deepfilter), "--compensate-delay", "--output-dir", str(df_temp), str(normalized)], log)
        deepfilter_seconds = time.perf_counter() - start
        produced = [p for p in df_temp.glob("*.wav") if p.is_file()]
        if len(produced) != 1:
            raise RuntimeError(f"DeepFilterNet did not produce exactly one WAV: {produced}\n{df_result.stdout}")
        enhanced = run_dir / "enhanced_deepfilternet.wav"
        shutil.move(str(produced[0]), enhanced)
        enhanced_meta = probe(ffprobe, enhanced, log)
        hypotheses: dict[str, dict[str, Any]] = {}
        timings = []
        audio_sources = {"raw": normalized, "enhanced": enhanced}
        adaptive_plan: dict[str, Any] | None = None
        if args.adaptive_turbo:
            turbo_name = "large-v3-turbo"
            model_path = root / "models" / turbo_name
            load_started = time.perf_counter()
            try:
                loaded_model = WhisperModel(str(model_path), device=args.device,
                                             compute_type=args.compute_type,
                                             cpu_threads=args.threads, num_workers=1,
                                             local_files_only=True)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed loading {turbo_name} on {args.device}/{args.compute_type}; "
                    f"no silent CPU fallback. {exc}"
                ) from exc
            shared_load_seconds = time.perf_counter() - load_started
            for source in SOURCE_ORDER:
                key = f"{turbo_name}__{source}"
                base = run_dir / "hypotheses" / key / key
                try:
                    payload = transcribe_one(model_path, audio_sources[source], base, turbo_name, source,
                                             args.device, args.compute_type, args.threads, log,
                                             loaded_model=loaded_model,
                                             model_load_seconds=(shared_load_seconds if source == SOURCE_ORDER[0] else 0.0))
                except Exception as exc:
                    error = {"model": turbo_name, "source": source, "error": str(exc),
                             "device": args.device, "compute_type": args.compute_type}
                    base.parent.mkdir(parents=True, exist_ok=True)
                    base.with_suffix(".error.json").write_text(json.dumps(error, ensure_ascii=False, indent=2), encoding="utf-8")
                    log.write(f"FATAL: {exc}\n")
                    raise
                hypotheses[key] = payload
                timings.append({"stage": key, "load_seconds": payload["load_seconds"],
                                "processing_seconds": payload["transcription_seconds"]})
            del loaded_model
            gc.collect()

            duration = float((enhanced_meta.get("format") or {}).get("duration") or 0.0)
            intervals, adaptive_plan = turbo_uncertainty_intervals(
                hypotheses["large-v3-turbo__raw"],
                hypotheses["large-v3-turbo__enhanced"], duration)
            adaptive_plan.update({
                "enabled": True,
                "turbo_hypotheses": ["large-v3-turbo__raw", "large-v3-turbo__enhanced"],
                "secondary_models": SECONDARY_MODEL_ORDER,
                "secondary_sources": SOURCE_ORDER,
                "review_intervals": intervals,
            })
            (run_dir / "adaptive-turbo-plan.json").write_text(
                json.dumps(adaptive_plan, ensure_ascii=False, indent=2), encoding="utf-8")

            for model_name in SECONDARY_MODEL_ORDER:
                model_path = root / "models" / model_name
                if not intervals:
                    for source in SOURCE_ORDER:
                        key = f"{model_name}__{source}"
                        base = run_dir / "hypotheses" / key / key
                        payload = empty_hypothesis(
                            audio_sources[source], base, model_name, source,
                            args.device, args.compute_type, args.threads,
                            "Turbo confidence gate found no interval requiring this family.")
                        hypotheses[key] = payload
                        timings.append({"stage": key, "load_seconds": 0.0,
                                        "processing_seconds": 0.0,
                                        "selective_interval_count": 0})
                    continue
                load_started = time.perf_counter()
                try:
                    loaded_model = WhisperModel(
                        str(model_path), device=args.device,
                        compute_type=args.compute_type, cpu_threads=args.threads,
                        num_workers=1, local_files_only=True)
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed loading {model_name} on {args.device}/{args.compute_type}; "
                        f"no silent CPU fallback. {exc}"
                    ) from exc
                shared_load_seconds = time.perf_counter() - load_started
                selective_work = run_dir / "adaptive-secondary-work" / model_name
                for source in SOURCE_ORDER:
                    key = f"{model_name}__{source}"
                    base = run_dir / "hypotheses" / key / key
                    try:
                        payload = transcribe_selective(
                            model_path, audio_sources[source], base, model_name, source,
                            args.device, args.compute_type, args.threads, log,
                            loaded_model, (shared_load_seconds if source == SOURCE_ORDER[0] else 0.0),
                            intervals, ffmpeg, selective_work)
                    except Exception as exc:
                        error = {"model": model_name, "source": source, "error": str(exc),
                                 "device": args.device, "compute_type": args.compute_type,
                                 "selective_intervals": intervals}
                        base.parent.mkdir(parents=True, exist_ok=True)
                        base.with_suffix(".error.json").write_text(
                            json.dumps(error, ensure_ascii=False, indent=2), encoding="utf-8")
                        log.write(f"FATAL: {exc}\n")
                        raise
                    hypotheses[key] = payload
                    timings.append({
                        "stage": key, "load_seconds": payload["load_seconds"],
                        "processing_seconds": payload["transcription_seconds"],
                        "selective_interval_count": len(intervals),
                    })
                del loaded_model
                gc.collect()
            adaptive_work_root = run_dir / "adaptive-secondary-work"
            if adaptive_work_root.exists():
                shutil.rmtree(adaptive_work_root)
        else:
            for model_name in MODEL_ORDER:
                model_path = root / "models" / model_name
                load_started = time.perf_counter()
                try:
                    loaded_model = WhisperModel(str(model_path), device=args.device,
                                                 compute_type=args.compute_type,
                                                 cpu_threads=args.threads, num_workers=1,
                                                 local_files_only=True)
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed loading {model_name} on {args.device}/{args.compute_type}; "
                        f"no silent CPU fallback. {exc}"
                    ) from exc
                shared_load_seconds = time.perf_counter() - load_started
                for source in SOURCE_ORDER:
                    key = f"{model_name}__{source}"
                    base = run_dir / "hypotheses" / key / key
                    try:
                        payload = transcribe_one(
                            model_path, audio_sources[source], base, model_name, source,
                            args.device, args.compute_type, args.threads, log,
                            loaded_model=loaded_model,
                            model_load_seconds=(shared_load_seconds if source == SOURCE_ORDER[0] else 0.0))
                    except Exception as exc:
                        error = {"model": model_name, "source": source, "error": str(exc),
                                 "device": args.device, "compute_type": args.compute_type}
                        base.parent.mkdir(parents=True, exist_ok=True)
                        base.with_suffix(".error.json").write_text(
                            json.dumps(error, ensure_ascii=False, indent=2), encoding="utf-8")
                        log.write(f"FATAL: {exc}\n")
                        raise
                    hypotheses[key] = payload
                    timings.append({"stage": key, "load_seconds": payload["load_seconds"],
                                    "processing_seconds": payload["transcription_seconds"]})
                del loaded_model
                gc.collect()

        if args.adaptive_turbo:
            base_key = "large-v3-turbo__enhanced"
            final_text = hypotheses[base_key]["text"]
            decisions = [{
                "start": row["start"], "end": row["end"], "base": row["word"],
                "chosen": row["word"], "action": "keep-turbo-base",
                "locked": True, "lock_reason": "adaptive-turbo-first",
                "support": [],
            } for row in words_of(hypotheses[base_key])]
        else:
            base_key = "large-v3__enhanced"
            final_text, decisions = merge(hypotheses)
        base_payload = hypotheses[base_key]
        large_payload = hypotheses.get("large-v3__enhanced") or base_payload
        large_dir = delivery / "01-whisper-large-v3"
        turbo_dir = delivery / "01-whisper-large-v3-turbo"
        algo_dir = delivery / "02-after-algorithm"
        audio_dir = delivery / "03-denoised-audio"
        for folder in (large_dir, turbo_dir, algo_dir, audio_dir): folder.mkdir(parents=True, exist_ok=True)
        (large_dir / "large-v3.txt").write_text(large_payload["text"] + "\n", encoding="utf-8")
        (large_dir / "large-v3.json").write_text(json.dumps(large_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (turbo_dir / "large-v3-turbo.txt").write_text(
            hypotheses["large-v3-turbo__enhanced"]["text"] + "\n", encoding="utf-8")
        (turbo_dir / "large-v3-turbo.json").write_text(
            json.dumps(hypotheses["large-v3-turbo__enhanced"], ensure_ascii=False, indent=2),
            encoding="utf-8")
        (algo_dir / "final.txt").write_text(final_text + "\n", encoding="utf-8")
        final_json = {"base": base_key,
                      "method": ("adaptive Turbo-first before MiniLM V8" if args.adaptive_turbo
                                 else "timestamp consensus without LLM"),
                      "text": final_text, "decisions": decisions}
        (algo_dir / "final.json").write_text(json.dumps(final_json, ensure_ascii=False, indent=2), encoding="utf-8")
        (algo_dir / "decisions.json").write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")
        shutil.copy2(enhanced, audio_dir / "enhanced.wav")
        system_info = {
            "platform": platform.platform(), "python": sys.version, "processor": platform.processor(),
            "logical_cpu_threads": os.cpu_count(), "used_cpu_threads": args.threads,
            "device": args.device, "compute_type": args.compute_type,
            "adaptive_turbo": args.adaptive_turbo,
            "packages": {name: importlib.metadata.version(name) for name in
                         ["faster-whisper", "ctranslate2", "huggingface-hub", "numpy"]},
            "deepfilternet": "0.5.6 official x86_64-pc-windows-msvc binary",
            "deepfilternet_command": subprocess.list2cmdline([str(deepfilter), "--compensate-delay", "--output-dir", str(df_temp), str(normalized)]),
        }
        (run_dir / "system-info.json").write_text(json.dumps(system_info, ensure_ascii=False, indent=2), encoding="utf-8")
        rows = []
        for key, p in hypotheses.items():
            rows.append({"hypothesis": key, "model": p["model"], "source": p["source"],
                         "load_seconds": p["load_seconds"], "transcription_seconds": p["transcription_seconds"],
                         "device": p["device"], "compute_type": p["compute_type"], "text": p["text"]})
        with (run_dir / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
        md = ["# Comparison", "", "No human reference transcript was supplied; no accuracy ranking is claimed.", ""]
        for row in rows:
            md += [f"## {row['hypothesis']}", "", row["text"], ""]
        (run_dir / "comparison.md").write_text("\n".join(md), encoding="utf-8")
        sensitive_changes = [d for d in decisions if d["locked"] or norm(d["base"]) != norm(d["chosen"])]
        report = [
            "# گزارش اجرای محلی", "",
            f"- شناسه اجرا: `{run_id}`", f"- دستگاه: `{args.device}` / `{args.compute_type}` / {args.threads} CPU threads",
            f"- تبدیل FFmpeg: {normalize_seconds:.2f} ثانیه", f"- نویزگیری DeepFilterNet: {deepfilter_seconds:.2f} ثانیه",
            "- DeepFilterNet: نسخه 0.5.6، باینری رسمی Windows x64، با `--compensate-delay`",
            "", "## مشخصات صدا", "", "### خام", "", "```json", json.dumps(raw_meta, ensure_ascii=False, indent=2), "```",
            "", "### WAV نرمال‌شده", "", "```json", json.dumps(normalized_meta, ensure_ascii=False, indent=2), "```",
            "", "### نویزگیری‌شده", "", "```json", json.dumps(enhanced_meta, ensure_ascii=False, indent=2), "```",
            "", "## زمان‌ها", "",
            "| فرضیه | بارگذاری (ثانیه) | تبدیل (ثانیه) |", "|---|---:|---:|",
        ]
        report += [f"| {r['stage']} | {r['load_seconds']:.2f} | {r['processing_seconds']:.2f} |" for r in timings]
        report += ["", "## شش فرضیه", ""]
        for key, p in hypotheses.items(): report += [f"### {key}", "", p["text"], ""]
        if adaptive_plan:
            report += ["## برنامهٔ تطبیقی Turbo-first", "",
                       "```json", json.dumps(adaptive_plan, ensure_ascii=False, indent=2),
                       "```", ""]
        report += [f"## متن پایه {base_key}", "", base_payload["text"], "",
                   "## متن نهایی الگوریتمی بدون LLM", "", final_text, "",
                   "## اختلاف‌های حساس نام/عدد/دوز/دارو/اصطلاح پزشکی", "",
                   "```json", json.dumps(sensitive_changes, ensure_ascii=False, indent=2), "```", "",
                   "> هشدار: نویزگیری نمی‌تواند صدایی را که ثبت نشده، به‌شدت clipping شده، یا زیر صدای گویندهٔ دیگری پوشیده شده بازیابی کند.", "",
                   "> برای هرگونه بررسی پزشکی، صدای خام مرجع اصلی و نهایی است؛ این رونویسی صحت پزشکی را تضمین نمی‌کند."]
        (run_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
        summary = {"run_id": run_id, "run_dir": str(run_dir), "delivery": str(delivery),
                   "raw_metadata": raw_meta, "enhanced_metadata": enhanced_meta,
                   "timings": timings, "device": args.device, "compute_type": args.compute_type,
                   "adaptive_turbo": args.adaptive_turbo,
                   "adaptive_turbo_plan": adaptive_plan,
                   "hypotheses": {k: v["text"] for k, v in hypotheses.items()},
                   "final_text": final_text}
        (run_dir / "run-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
