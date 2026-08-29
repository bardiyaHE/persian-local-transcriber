from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from consensus_v2 import (
    DRUG_CONTEXT,
    NEGATIONS,
    NUMBERS,
    UNITS,
    USER_BLOCKLIST,
    active_word,
    norm,
)


ACCEPT_REASONS = {
    "two-family-exact-consensus",
    "six-vote-score-plus-phrase-context",
    "merge-duplicate-suffix-with-previous",
}
REVIEW_REASONS = {
    "two-family-medical-fuzzy-consensus",
    "two-family-general-lexicon-fuzzy-consensus",
    "turbo-base-dictionary-fallback",
}
DRUG_NAME_MARKERS = {"اسم", "اسمی", "دارو", "داروی", "قرص"}
NUMBER_VALUES = {
    "صفر": 0, "یک": 1, "دو": 2, "سه": 3, "چهار": 4, "پنج": 5,
    "شش": 6, "هفت": 7, "هشت": 8, "نه": 9, "ده": 10, "بیست": 20,
    "بیس": 20, "پنجاه": 50, "پنجا": 50, "پنجام": 50, "صد": 100,
}
CANONICAL_NUMBER_WORDS = {0: "صفر", 1: "یک", 2: "دو", 3: "سه", 4: "چهار", 5: "پنج",
                          6: "شش", 7: "هفت", 8: "هشت", 9: "نه", 10: "ده",
                          20: "بیست", 25: "بیست و پنج", 50: "پنجاه", 100: "صد"}


def sensitive_kind(decisions: list[dict[str, Any]], index: int,
                   medical_map: dict[str, dict[str, Any]]) -> list[str]:
    kinds: list[str] = []
    window = decisions[max(0, index - 2):index + 3]
    tokens = {norm(item["base"]) for item in window} | {norm(item["chosen"]) for item in window}
    if any(re.search(r"\d|[۰-۹]", token) or token in NUMBERS or token in UNITS for token in tokens):
        kinds.append("number-or-dose-span")
    if any(token in NEGATIONS or token.startswith(("نمی", "نخوا", "نباید", "ندار", "نبود"))
           for token in tokens):
        kinds.append("negation-phrase-span")
    medical_rows = [medical_map[token] for token in tokens if token in medical_map]
    previous = norm(decisions[index - 1]["base"]) if index else ""
    if previous in DRUG_NAME_MARKERS and norm(decisions[index]["base"]) not in medical_map:
        kinds.append("possible-drug-name")
    if tokens & DRUG_CONTEXT or any(row.get("category") in {"drug", "medication", "drug_class"}
                                    for row in medical_rows):
        kinds.append("drug-span")
    elif medical_rows:
        kinds.append("medical-term")
    return kinds


def classify(decision: dict[str, Any], decisions: list[dict[str, Any]], index: int,
             medical_map: dict[str, dict[str, Any]]) -> tuple[str, list[str]]:
    base_norm, chosen_norm = norm(decision["base"]), norm(decision["chosen"])
    kinds = sensitive_kind(decisions, index, medical_map)
    if "possible-drug-name" in kinds:
        return "REVIEW", kinds
    if base_norm == chosen_norm:
        return "KEEP_SOURCE", kinds
    if decision["reason"].startswith("keep-"):
        return "KEEP_SOURCE", kinds
    if chosen_norm in USER_BLOCKLIST:
        return "REJECT", kinds + ["blocked-vocabulary"]
    if kinds:
        # Correlated ASR votes cannot automatically authorize a medical, dose, or negation edit.
        return "REVIEW", kinds
    if decision["reason"] in ACCEPT_REASONS:
        return "ACCEPT", kinds
    if decision["reason"] in REVIEW_REASONS:
        return "REVIEW", kinds
    return "REVIEW", kinds + ["unclassified-change"]


def numeric_value(token: str) -> int | None:
    token = norm(token)
    if re.fullmatch(r"[0-9۰-۹]+", token):
        translated = token.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789"))
        return int(translated)
    return NUMBER_VALUES.get(token)


def drug_candidate_for_index(decisions: list[dict[str, Any]], index: int,
                             medical_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if index == 0 or norm(decisions[index - 1]["base"]) not in DRUG_NAME_MARKERS:
        return None
    current_votes = {item["hypothesis"]: item.get("normalized")
                     for item in decisions[index].get("six_hypothesis_votes") or []}
    next_votes = ({item["hypothesis"]: item.get("normalized")
                   for item in decisions[index + 1].get("six_hypothesis_votes") or []}
                  if index + 1 < len(decisions) else {})
    observations = []
    for hypothesis, current in current_votes.items():
        if not current:
            continue
        family = hypothesis.split("__", 1)[0]
        # Only repair an obvious split inside the same hypothesis. Never concatenate
        # alternative sources or a normal following word such as «که».
        if len(current) < 4 and next_votes.get(hypothesis):
            current += next_votes[hypothesis]
        if len(current) >= 4:
            observations.append((family, current))
    if len({family for family, _ in observations}) < 2:
        return None

    def skeleton(text: str) -> str:
        return re.sub(r"[اآوی]", "", norm(text))

    concepts: dict[str, dict[str, Any]] = {}
    for row in medical_rows:
        term = row["normalized"]
        if " " in term or not re.search(r"[؀-ۿ]", term):
            continue
        per_family: dict[str, float] = {}
        for family, observed in observations:
            score = (0.30 * fuzz.ratio(observed, term)
                     + 0.50 * fuzz.partial_ratio(observed, term)
                     + 0.20 * fuzz.ratio(skeleton(observed), skeleton(term)))
            per_family[family] = max(per_family.get(family, 0.0), score)
        top_family_scores = sorted(per_family.values(), reverse=True)[:3]
        concept_score = sum(top_family_scores) / max(1, len(top_family_scores))
        english = str(row.get("english") or term).strip()
        alias = row.get("term") or term
        concept = concepts.setdefault(english, {"english": english, "score": 0.0,
                                                "aliases": [], "alias_scores": [],
                                                "family_scores": {}})
        concept["aliases"].append(alias)
        concept["alias_scores"].append({"alias": alias, "score": concept_score})
        if concept_score > concept["score"]:
            concept["score"] = concept_score
            concept["family_scores"] = per_family
    ranked = sorted(concepts.values(), key=lambda item: item["score"], reverse=True)
    if not ranked or ranked[0]["score"] < 68:
        return None
    best = ranked[0]
    runner = ranked[1] if len(ranked) > 1 else None
    ranked_aliases = sorted(
        ({"alias": item["alias"], "score": round(item["score"], 3)}
         for item in best["alias_scores"] if re.search(r"[؀-ۿ]", item["alias"])),
        key=lambda item: item["score"], reverse=True,
    )
    unique_aliases = []
    seen_aliases = set()
    for item in ranked_aliases:
        if item["alias"] not in seen_aliases:
            unique_aliases.append(item)
            seen_aliases.add(item["alias"])
    return {
        "status": "REVIEW",
        "english_identity": best["english"],
        "top_persian_aliases": unique_aliases[:5],
        "concept_score_heuristic": round(best["score"], 3),
        "runner_up": ({"english_identity": runner["english"], "score": round(runner["score"], 3)}
                      if runner else None),
        "score_gap": round(best["score"] - (runner["score"] if runner else 0.0), 3),
        "ambiguous": bool(runner and best["score"] - runner["score"] < 5.0),
        "ranked_concepts": [
            {"english_identity": item["english"], "score": round(item["score"], 3)}
            for item in ranked[:3]
        ],
        "observations": [{"family": family, "text": text} for family, text in observations],
        "note": "Dose is not used as identification evidence; candidate is never auto-applied.",
    }


def numeric_evidence(decision: dict[str, Any]) -> dict[str, Any] | None:
    """Summarise numeric readings by model family; never use them to identify a drug."""
    by_value: dict[int, dict[str, Any]] = {}
    for item in decision.get("six_hypothesis_votes") or []:
        value = numeric_value(item.get("normalized") or "")
        if value is None:
            continue
        family = item["hypothesis"].split("__", 1)[0]
        row = by_value.setdefault(value, {"value": value, "families": set(), "observations": []})
        row["families"].add(family)
        row["observations"].append({"hypothesis": item["hypothesis"], "text": item.get("normalized")})
    if not by_value:
        return None
    ranked = sorted(by_value.values(), key=lambda row: (len(row["families"]), len(row["observations"])),
                    reverse=True)
    for row in ranked:
        row["families"] = sorted(row["families"])
        row["canonical_fa"] = CANONICAL_NUMBER_WORDS.get(row["value"], str(row["value"]))
    return {
        "status": "REVIEW",
        "candidates": ranked,
        "note": "Correlated ASR evidence only; no number is auto-replaced.",
    }


def load_hypothesis_texts(run_dir: Path) -> list[dict[str, str]]:
    rows = []
    for path in sorted((run_dir / "hypotheses").glob("*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"hypothesis": f"{payload['model']}__{payload['source']}",
                     "family": payload["model"], "text": str(payload.get("text") or "")})
    return rows


def audit_sensitive_phrases(hypotheses: list[dict[str, str]]) -> dict[str, Any]:
    """Extract dose and negation evidence without changing the transcript."""
    dose_pattern = re.compile(
        r"(?P<number>\d+|[۰-۹]+|صد|پنجاه|پنجا\w*|بیست(?:\s*(?:و\s*)?پنج(?:اه|ا\w*)?)?)"
        r"\s*میلی\s*گر(?:م|ام)"
    )
    dose_mentions = []
    negation_mentions = []
    for row in hypotheses:
        for match in dose_pattern.finditer(row["text"]):
            surface = match.group("number")
            compact = norm(surface)
            if "صد" in compact or compact == "100":
                value = 100
            elif compact == "25" or ("بیست" in compact and "پنج" in compact):
                value = 25
            elif compact == "50" or compact.startswith("پنجا"):
                value = 50
            else:
                value = numeric_value(compact)
            dose_mentions.append({"hypothesis": row["hypothesis"], "family": row["family"],
                                  "surface": match.group(0), "numeric_candidate": value,
                                  "ambiguous_surface": bool("بیست" in compact and "پنجا" in compact)})
        tokens = row["text"].split()
        for index, token in enumerate(tokens):
            normalized = norm(token)
            if normalized in NEGATIONS or normalized.startswith(("نمی", "نخوا", "نباید", "ندار", "نبود")):
                negation_mentions.append({"hypothesis": row["hypothesis"], "family": row["family"],
                                          "token": token, "phrase": " ".join(tokens[max(0, index - 2):index + 3])})
    values_by_family: dict[str, set[int]] = {}
    for mention in dose_mentions:
        if mention["numeric_candidate"] is not None:
            values_by_family.setdefault(mention["family"], set()).add(mention["numeric_candidate"])
    common_values = sorted({value for values in values_by_family.values() for value in values})
    arithmetic = []
    if {100, 50}.issubset(common_values):
        arithmetic.append({"expression": "half(100)=50", "consistent": True,
                           "warning": "Arithmetic consistency does not prove the audio was heard correctly."})
    if {50, 25}.issubset(common_values):
        arithmetic.append({"expression": "half(50)=25", "consistent": True,
                           "warning": "Arithmetic consistency does not prove the audio was heard correctly."})
    return {
        "dose_mentions": dose_mentions,
        "negation_mentions": negation_mentions,
        "arithmetic_checks": arithmetic,
        "policy": "audit-only; sensitive phrases are never auto-corrected",
    }


def evidence_summary(decision: dict[str, Any], chosen_norm: str) -> dict[str, Any]:
    six = decision.get("six_hypothesis_votes") or []
    exact = [item for item in six if item.get("normalized") == chosen_norm]
    families = {item["hypothesis"].split("__", 1)[0] for item in exact}
    scores = decision.get("candidate_scores") or []
    candidate_score = next((row for row in scores
                            if row.get("canonical") == chosen_norm or row.get("candidate") == chosen_norm), None)
    return {
        "exact_hypothesis_votes": len(exact),
        "supporting_model_families": sorted(families),
        "candidate_score": candidate_score,
        "dictionary": decision.get("dictionary") or {},
    }


def group_review_intervals(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    intervals = []
    for item in items:
        start, end = max(0.0, float(item["start"]) - 1.5), float(item["end"]) + 1.5
        if intervals and start <= intervals[-1]["end"] + 0.75:
            intervals[-1]["end"] = max(intervals[-1]["end"], end)
            intervals[-1]["indices"].append(item["index"])
        else:
            intervals.append({"start": start, "end": end, "indices": [item["index"]]})
    return intervals


def make_review_clips(run_dir: Path, intervals: list[dict[str, Any]], ffmpeg: Path) -> list[str]:
    audio = run_dir / "normalized_mono_48k.wav"
    if not audio.is_file() or not ffmpeg.is_file():
        return []
    clip_dir = run_dir / "final-delivery" / "04-safe-no-llm" / "review-clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    result = []
    for number, interval in enumerate(intervals, 1):
        target = clip_dir / f"review-{number:03d}-{interval['start']:.2f}-{interval['end']:.2f}.wav"
        command = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
                   "-ss", f"{interval['start']:.3f}", "-to", f"{interval['end']:.3f}",
                   "-i", str(audio), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(target)]
        subprocess.run(command, check=True, capture_output=True)
        interval["clip"] = str(target)
        result.append(str(target))
    return result


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description="Fast deterministic safety gate; no LLM.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--medical-index", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run_dir, root = args.run_dir.resolve(), args.root.resolve()
    v2_dir = run_dir / "final-delivery" / "02-after-algorithm-v2-turbo-lexicon"
    payload = json.loads((v2_dir / "final-v2.json").read_text(encoding="utf-8"))
    decisions = payload["decisions"]
    medical_payload = json.loads(args.medical_index.read_text(encoding="utf-8"))
    medical_map = {row["normalized"]: row for row in medical_payload["terms"]}
    medical_rows = [row for row in medical_payload["terms"]
                    if row.get("category") in {"drug", "medication", "drug_class"}]
    sensitive_audit = audit_sensitive_phrases(load_hypothesis_texts(run_dir))

    safe_parts: list[str] = []
    suggested_parts: list[str] = []
    verdicts: list[dict[str, Any]] = []
    for index, decision in enumerate(decisions):
        verdict, risks = classify(decision, decisions, index, medical_map)
        drug_candidate = drug_candidate_for_index(decisions, index, medical_rows)
        number_evidence = numeric_evidence(decision)
        if drug_candidate:
            verdict = "REVIEW"
            if "possible-drug-name" not in risks:
                risks.append("possible-drug-name")
        proposed = decision["chosen"]
        source = decision["base"]
        safe_word = proposed if verdict == "ACCEPT" else source
        safe_parts.append(safe_word)
        suggested_parts.append(proposed)
        verdicts.append({
            "index": decision["index"], "start": decision["start"], "end": decision["end"],
            "source": source, "proposed": proposed, "safe_output": safe_word,
            "verdict": verdict, "risk_labels": risks, "reason": decision["reason"],
            "evidence": evidence_summary(decision, norm(proposed)),
            "drug_candidate": drug_candidate,
            "numeric_evidence": number_evidence,
        })

    safe_text = "".join(safe_parts).strip()
    suggested_text = "".join(suggested_parts).strip()
    review_items = [item for item in verdicts if item["verdict"] in {"REVIEW", "REJECT"}]
    intervals = group_review_intervals(review_items)
    clips = make_review_clips(run_dir, intervals, root / "runtime" / "ffmpeg" / "ffmpeg.exe")
    counts = Counter(item["verdict"] for item in verdicts)
    elapsed = time.perf_counter() - started

    out_dir = run_dir / "final-delivery" / "04-safe-no-llm"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "safe-final.txt").write_text(safe_text + "\n", encoding="utf-8")
    (out_dir / "suggested-for-review.txt").write_text(suggested_text + "\n", encoding="utf-8")
    (out_dir / "verdicts.json").write_text(json.dumps(verdicts, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "sensitive-phrase-audit.json").write_text(
        json.dumps(sensitive_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "method": "deterministic safety gate without LLM",
        "uncalibrated_heuristics": True,
        "runtime_seconds": round(elapsed, 3),
        "verdict_counts": dict(counts),
        "review_count": len(review_items),
        "review_clip_count": len(clips),
        "possible_drug_candidates": sum(bool(item["drug_candidate"]) for item in verdicts),
        "dose_mentions_across_six": len(sensitive_audit["dose_mentions"]),
        "negation_mentions_across_six": len(sensitive_audit["negation_mentions"]),
        "safe_text": safe_text,
        "suggested_text": suggested_text,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = [
        "# موارد نیازمند بازبینی — بدون LLM", "",
        "`REVIEW` یعنی پیشنهاد محتمل است اما خودکار وارد متن امن نشده است. وزن‌ها heuristic و کالیبره‌نشده‌اند.", "",
        "| زمان | متن منبع | پیشنهاد | وضعیت | ریسک | دلیل |",
        "|---:|---|---|---|---|---|",
    ]
    drug_notes = []
    for item in review_items:
        cells = [f"{item['start']:.2f}", item["source"].strip(), item["proposed"].strip(),
                 item["verdict"], ", ".join(item["risk_labels"]), item["reason"]]
        cells = [str(cell).replace("|", "\\|").replace("\n", " ") for cell in cells]
        rows.append("| " + " | ".join(cells) + " |")
        candidate = item.get("drug_candidate")
        if candidate:
            aliases = "، ".join(alias["alias"] for alias in candidate["top_persian_aliases"])
            drug_notes.append(f"- زمان {item['start']:.2f}: نام داروی محتمل فقط برای بازبینی: "
                              f"`{candidate['english_identity']}`؛ املای واژه‌نامه: {aliases or '—'}؛ "
                              f"فاصله با گزینهٔ بعدی: {candidate['score_gap']:.2f}. "
                              "**این مورد خودکار جایگزین نشده و دوز در شناسایی دخالت ندارد.**")
    if drug_notes:
        rows += ["", "## گزینه‌های نام دارو", "", *drug_notes]
    rows += ["", "## بازه‌های صوتی", ""]
    for number, interval in enumerate(intervals, 1):
        rows.append(f"- کلیپ {number}: {interval['start']:.2f} تا {interval['end']:.2f} ثانیه؛ "
                    f"واژه‌ها: {', '.join(map(str, interval['indices']))}")
    rows += ["", "## ممیزی عبارت‌های حساس", "",
             f"- اشاره‌های دوز در شش متن: {len(sensitive_audit['dose_mentions'])}",
             f"- عبارت‌های نفی در شش متن: {len(sensitive_audit['negation_mentions'])}",
             "- سازگاری حسابی فقط گزارش می‌شود و اثبات نمی‌کند صوت درست شنیده شده است.",
             "- جزئیات کامل در `sensitive-phrase-audit.json` است."]
    (out_dir / "review-items.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps({"output": str(out_dir), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
