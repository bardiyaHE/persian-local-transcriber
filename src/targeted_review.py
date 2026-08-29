from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel
from rapidfuzz import fuzz

from consensus_v2 import norm


def decode_full(model: WhisperModel, audio: Path,
                hotwords: str | None) -> tuple[str, list[dict[str, Any]], float]:
    started = time.perf_counter()
    segments, _ = model.transcribe(
        str(audio), language="fa", task="transcribe", beam_size=3,
        temperature=0.0, condition_on_previous_text=False,
        vad_filter=True, vad_parameters={"threshold": 0.5, "min_speech_duration_ms": 250,
                                         "min_silence_duration_ms": 500, "speech_pad_ms": 300},
        word_timestamps=True,
        hotwords=hotwords,
    )
    materialized = []
    for segment in segments:
        materialized.append({
            "start": segment.start, "end": segment.end, "text": segment.text,
            "words": [{"start": word.start, "end": word.end, "word": word.word}
                      for word in segment.words or []],
        })
    text = "".join(segment["text"] for segment in materialized).strip()
    return text, materialized, time.perf_counter() - started


def window_text(segments: list[dict[str, Any]], start: float, end: float) -> str:
    words = []
    for segment in segments:
        for word in segment["words"]:
            middle = (word["start"] + word["end"]) / 2.0
            if start <= middle <= end:
                words.append(word["word"])
    if words:
        return "".join(words).strip()
    return "".join(segment["text"] for segment in segments
                   if segment["end"] >= start and segment["start"] <= end).strip()


def polarity(text: str) -> str:
    compact = norm(text)
    if any(token in compact for token in ("نمیشه", "نیمیشه", "نمیگرده", "نخواهد")):
        return "negative"
    if "میشه" in compact:
        return "positive"
    return "unknown"


def decode_quality(text: str) -> dict[str, Any]:
    tokens = [norm(token) for token in text.split() if norm(token)]
    counts = Counter(tokens)
    dominance = (counts.most_common(1)[0][1] / len(tokens)) if tokens else 1.0
    unique_ratio = len(counts) / len(tokens) if tokens else 0.0
    valid = bool(tokens) and not (len(tokens) >= 8 and (dominance > 0.35 or unique_ratio < 0.35))
    return {"valid": valid, "token_count": len(tokens),
            "dominance": round(dominance, 3), "unique_ratio": round(unique_ratio, 3)}


def adapt_v4_sensitive(item: dict[str, Any]) -> dict[str, Any] | None:
    """Build generic targeted rivals from v4 evidence; no sample-specific hotwords."""
    categories = set(item.get("medical_categories") or [])
    alternatives = [row.get("candidate") for row in item.get("alternatives") or []
                    if row.get("candidate")]
    aliases = list(dict.fromkeys([item.get("candidate"), *alternatives]))
    aliases = [alias for alias in aliases if alias]
    if categories & {"drug", "medication", "drug_class"}:
        kind = "drug"
        evidence = {"aliases": aliases, "preferred_alias": item.get("candidate")}
    else:
        polarity_values = {polarity(alias) for alias in aliases}
        if not polarity_values & {"positive", "negative"}:
            return None
        kind = "negation-phrase-lock"
        family_rows: dict[str, set[str]] = defaultdict(set)
        for row in item.get("observations") or []:
            value = polarity(row.get("normalized") or row.get("word") or "")
            if value in {"positive", "negative"}:
                family_rows[row["family"]].add(value)
        family_polarity = {family: next(iter(values)) if len(values) == 1 else "abstain"
                           for family, values in family_rows.items()}
        evidence = {"aliases": aliases, "family_polarity": family_polarity}
    return {**item, "kind": kind, "evidence": evidence}


def assess_drug(item: dict[str, Any], free_text: str, rival_text: str,
                rival_quality: dict[str, Any]) -> dict[str, Any]:
    evidence = item.get("evidence") or {}
    aliases = evidence.get("aliases") or [evidence.get("preferred_alias")]
    aliases = [alias for alias in aliases if alias]

    def rank(text: str) -> dict[str, Any] | None:
        if not aliases:
            return None
        rows = [{"alias": alias, "score": max(fuzz.ratio(norm(text), norm(alias)),
                                                fuzz.partial_ratio(norm(text), norm(alias)))}
                for alias in aliases]
        return max(rows, key=lambda row: row["score"])

    free, rival = rank(free_text), rank(rival_text)
    free_score = free["score"] if free else 0.0
    rival_score = rival["score"] if rival else 0.0
    if not rival_quality["valid"]:
        verdict = "RIVAL_DECODE_REJECTED_REPETITION"
    elif free_score >= 65 and rival_score >= 75:
        verdict = "SUPPORTS_PROBABLE_CANDIDATE"
    elif free_score < 65 and rival_score >= 75:
        verdict = "HOTWORD_ONLY_REJECT_AS_CONFIRMATION"
    else:
        verdict = "INSUFFICIENT_ACOUSTIC_SUPPORT"
    return {
        "verdict": verdict, "free_best": free, "rival_best": rival,
        "rival_decode_quality": rival_quality,
        "english_identity": evidence.get("english_identity"),
        "policy": "Targeted decoding never silently confirms a drug name.",
    }


def assess_negation(item: dict[str, Any], free_text: str, rival_text: str,
                    rival_quality: dict[str, Any]) -> dict[str, Any]:
    evidence = item.get("evidence") or {}
    family_polarity = evidence.get("family_polarity") or {}
    counts = Counter(value for value in family_polarity.values()
                     if value in {"positive", "negative"})
    family_value, family_support = counts.most_common(1)[0] if counts else ("unknown", 0)
    free_value, rival_value = polarity(free_text), polarity(rival_text)
    if not rival_quality["valid"]:
        verdict = "RIVAL_DECODE_REJECTED_REPETITION"
    elif (family_support >= 2 and family_value == free_value == rival_value
            and family_value in {"positive", "negative"}):
        verdict = "POLARITY_VERIFIED"
    elif free_value == rival_value and free_value in {"positive", "negative"}:
        verdict = "TARGETED_AGREEMENT_BUT_FAMILY_SUPPORT_INSUFFICIENT"
    else:
        verdict = "POLARITY_REMAINS_REVIEW"
    return {
        "verdict": verdict, "family_polarity": family_polarity,
        "rival_decode_quality": rival_quality,
        "free_polarity": free_value, "rival_polarity": rival_value,
        "policy": "Polarity changes require two model families plus targeted agreement.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted free/rival re-decode; no LLM.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--compute-type", required=True)
    parser.add_argument("--threads", type=int, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    run_dir, root = args.run_dir.resolve(), args.root.resolve()
    candidates = [
        (run_dir / "final-delivery" / "02-after-algorithm-v5-domain-corpus", "review-v5.json"),
        (run_dir / "final-delivery" / "02-after-algorithm-v4-ngram-lexicon", "review-v4.json"),
    ]
    selected_output = next(((directory, filename) for directory, filename in candidates
                            if (directory / filename).is_file()), None)
    if selected_output is None:
        raise FileNotFoundError("No V5/V4 consensus review file was found.")
    out_dir, review_filename = selected_output
    review = json.loads((out_dir / review_filename).read_text(encoding="utf-8"))
    sensitive = [adapted for item in review if (adapted := adapt_v4_sensitive(item)) is not None]
    if not sensitive:
        result = {"method": "targeted free/rival decode", "llm_used": False,
                  "runtime_seconds": 0.0, "items": []}
    else:
        load_started = time.perf_counter()
        model = WhisperModel(str(root / "models" / "large-v3-turbo"),
                             device=args.device, compute_type=args.compute_type,
                             cpu_threads=args.threads, num_workers=1, local_files_only=True)
        load_seconds = time.perf_counter() - load_started
        audio = run_dir / "normalized_mono_48k.wav"
        hotwords_by_kind: dict[str, list[str]] = defaultdict(list)
        for item in sensitive:
            hotwords_by_kind[item["kind"]].extend((item.get("evidence") or {}).get("aliases") or [])
        free_full, free_segments, free_seconds = decode_full(model, audio, None)
        rival_decodes = {}
        for kind, hotword_rows in hotwords_by_kind.items():
            hotword_text = " ".join(dict.fromkeys(hotword_rows))
            full_text, segments, seconds = decode_full(model, audio, hotword_text)
            rival_decodes[kind] = {"hotwords": hotword_text, "full_text": full_text,
                                   "segments": segments, "seconds": seconds,
                                   "quality": decode_quality(full_text)}
        rows = []
        for item in sensitive:
            start, end = max(0.0, float(item["start"]) - 0.35), float(item["end"]) + 0.35
            rival_run = rival_decodes[item["kind"]]
            hotwords = rival_run["hotwords"]
            free_text = window_text(free_segments, start, end)
            rival_text = window_text(rival_run["segments"], start, end)
            assessment = (assess_drug(item, free_text, rival_text, rival_run["quality"])
                          if item["kind"] == "drug"
                          else assess_negation(item, free_text, rival_text, rival_run["quality"]))
            rows.append({
                "kind": item["kind"], "start": start, "end": end,
                "free_decode": free_text, "rival_hotwords": hotwords,
                "rival_decode": rival_text,
                "assessment": assessment,
            })
        result = {
            "method": "targeted free decode plus multi-rival hotword decode",
            "llm_used": False, "model": "large-v3-turbo",
            "device": args.device, "compute_type": args.compute_type,
            "load_seconds": round(load_seconds, 3),
            "free_full_decode_seconds": round(free_seconds, 3),
            "free_full_text": free_full,
            "rival_full_decodes": {
                kind: {"seconds": round(row["seconds"], 3), "full_text": row["full_text"],
                       "quality": row["quality"]}
                for kind, row in rival_decodes.items()
            },
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "items": rows,
        }
    (out_dir / "targeted-review.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# بازشنوی هدفمند — بدون LLM", "",
             "هر بازه یک‌بار آزاد و یک‌بار با چند رقیب decode شده است. نتیجهٔ hotword-only تأیید محسوب نمی‌شود.", ""]
    for row in result["items"]:
        lines += [f"## {row['kind']} — {row['start']:.2f} تا {row['end']:.2f}", "",
                  f"- آزاد: {row['free_decode']}", f"- با رقبا: {row['rival_decode']}",
                  f"- حکم: `{row['assessment']['verdict']}`", ""]
    (out_dir / "targeted-review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
