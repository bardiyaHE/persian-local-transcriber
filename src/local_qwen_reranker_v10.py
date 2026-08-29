from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import consensus_v4 as v4
import consensus_v5 as v5


OUTPUT_RELATIVE = Path("final-delivery") / "10-local-qwen-reranker"
V9_RELATIVE = Path("final-delivery") / "09-medical-drug-dictionary"
HYPOTHESES = (
    "large-v3-turbo__enhanced",
    "large-v3-turbo__raw",
    "large-v3__enhanced",
    "large-v3__raw",
    "medium__enhanced",
    "medium__raw",
)
FAMILY_BY_HYPOTHESIS = {
    "large-v3-turbo__enhanced": "large-v3-turbo",
    "large-v3-turbo__raw": "large-v3-turbo",
    "large-v3__enhanced": "large-v3",
    "large-v3__raw": "large-v3",
    "medium__enhanced": "medium",
    "medium__raw": "medium",
    "v9": "v9",
}
DRUG_CATEGORIES = {"drug", "medication", "drug_class"}
GENERIC_DRUG_TERMS = {
    "دارو", "داروی", "قرص", "کپسول", "شربت", "آمپول", "پماد", "قطره",
    "ویتامین", "آنتیبیوتیک", "دارونما",
}
NUMBER_WORDS = set(v4.NUMBER_WORDS) | {
    "بیست", "سی", "چهل", "پنجاه", "شصت", "هفتاد", "هشتاد", "نود",
    "دویست", "سیصد", "چهارصد", "پانصد", "هزار", "نصف", "ربع",
}
UNITS = set(v4.UNITS) | {
    "میلی‌گرم", "میلیگرم", "میکروگرم", "سی‌سی", "سیسی", "قرص",
    "کپسول", "قطره", "واحد", "ساعت", "روز", "هفته",
}
TOKEN_RE = re.compile(r"[0-9۰-۹]+|[A-Za-z]+|[\u0600-\u06ff]+", re.UNICODE)
PUNCTUATION_RE = re.compile(r"\s+([،؛.!؟?,:])")
ASCII_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._+-]*$")
COMMON_MEDICAL_ASCII = {"ivig", "jak", "b1", "b6", "hiv", "hcv", "hbv", "tsh", "cbc", "crp", "esr"}
LOCAL_MODEL_ALIAS = "local-qwen3.5-35b-a3b"
LOCAL_MODEL_REPOSITORY = "unsloth/Qwen3.5-35B-A3B-GGUF"
LOCAL_MODEL_REVISION = "bc014a17be43adabd7066b7a86075ff935c6a4e2"
LOCAL_MODEL_FILE = "Qwen3.5-35B-A3B-UD-Q4_K_L.gguf"
LOCAL_MODEL_QUANTIZATION = "UD-Q4_K_L"


def normalize_text(text: str) -> str:
    text = str(text or "").replace("\u200f", " ").replace("\u200e", " ")
    text = text.replace("ي", "ی").replace("ك", "ک")
    text = re.sub(r"\s+", " ", text).strip()
    return PUNCTUATION_RE.sub(r"\1", text)


def normalized_key(text: str) -> str:
    return " ".join(v4.phrase_tokens(normalize_text(text)))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def hypothesis_json_path(run_dir: Path, key: str) -> Path:
    folder = run_dir / "hypotheses" / key
    matches = sorted(folder.glob("*.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one hypothesis JSON in {folder}; found {len(matches)}")
    return matches[0]


def flatten_words(payload: dict[str, Any]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for segment in payload.get("segments") or []:
        segment_words = segment.get("words") or []
        if segment_words:
            for item in segment_words:
                word = normalize_text(item.get("word") or "")
                if not word:
                    continue
                words.append({
                    "word": word,
                    "start": float(item.get("start", segment.get("start", 0.0))),
                    "end": float(item.get("end", segment.get("end", 0.0))),
                    "probability": float(item.get("probability") or 0.0),
                })
            continue
        text = normalize_text(segment.get("text") or "")
        if text:
            words.append({
                "word": text,
                "start": float(segment.get("start") or 0.0),
                "end": float(segment.get("end") or 0.0),
                "probability": 0.0,
            })
    return words


def load_hypotheses(run_dir: Path) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for key in HYPOTHESES:
        payload = load_json(hypothesis_json_path(run_dir, key))
        loaded[key] = {
            "text": normalize_text(payload.get("text") or ""),
            "duration": float(payload.get("duration") or 0.0),
            "words": flatten_words(payload),
        }
    return loaded


def tokens_of(text: str) -> list[str]:
    tokens: list[str] = []
    for match in TOKEN_RE.finditer(text):
        token = v4.norm(match.group(0)).strip("،؛.!؟?,:()[]{}«»\"'")
        if token:
            tokens.append(token)
    return tokens


def build_drug_terms(medical_index: Path) -> set[str]:
    if not medical_index.is_file():
        return set()
    payload = load_json(medical_index)
    terms: set[str] = set()
    for row in payload.get("terms") or []:
        if str(row.get("category") or "") not in DRUG_CATEGORIES:
            continue
        term = v4.norm(row.get("normalized") or row.get("term") or "")
        if term and term not in GENERIC_DRUG_TERMS and len(v4.phrase_tokens(term)) == 1:
            terms.add(term)
    return terms


def build_exact_medical_terms(medical_index: Path) -> set[str]:
    """Return only standalone entries, not words occurring inside a medical phrase.

    The V4 lattice can correctly mark a candidate as medical for ranking purposes,
    but that flag is not strong enough to bypass V10's conservative review.  For
    example, a common word can occur inside a lab-test phrase without being a
    medical term by itself.  The exemption therefore uses this stricter set.
    """
    if not medical_index.is_file():
        return set()
    payload = load_json(medical_index)
    terms: set[str] = set()
    for row in payload.get("terms") or []:
        # phrase_tokens intentionally ignores Latin material.  That is useful
        # for Persian matching, but would turn "PSA آزاد" into the false
        # standalone term "آزاد" here. Count every script before exempting.
        tokens = [
            v4.norm(match.group(0))
            for match in TOKEN_RE.finditer(normalize_text(
                row.get("normalized") or row.get("term") or ""))
            if v4.norm(match.group(0))
        ]
        if len(tokens) == 1:
            terms.add(tokens[0])
    return terms


def build_allowed_ascii_terms(medical_index: Path) -> set[str]:
    allowed = set(COMMON_MEDICAL_ASCII)
    if not medical_index.is_file():
        return allowed
    payload = load_json(medical_index)
    for row in payload.get("terms") or []:
        term = normalize_text(row.get("normalized") or row.get("term") or "").casefold()
        if term and " " not in term and ASCII_TOKEN_RE.fullmatch(term):
            allowed.add(term)
    return allowed


def cleanup_unknown_ascii(text: str, allowed: set[str]) -> tuple[str, list[dict[str, Any]]]:
    tokens = normalize_text(text).split()
    output: list[str] = []
    audit: list[dict[str, Any]] = []
    pending: list[str] = []

    def flush() -> None:
        if not pending:
            return
        output.append("[نامفهوم]")
        audit.append({
            "removed": list(pending),
            "replacement": "[نامفهوم]",
            "reason": "unlicensed-unknown-ascii-token-in-persian-transcript",
        })
        pending.clear()

    for surface in tokens:
        bare = surface.strip("،؛.!؟?,:()[]{}\"'").casefold()
        if ASCII_TOKEN_RE.fullmatch(bare) and bare not in allowed:
            pending.append(surface)
            continue
        flush()
        output.append(surface)
    flush()
    return normalize_text(" ".join(output)), audit


def sensitive_signature(text: str, drug_terms: set[str]) -> dict[str, list[str]]:
    numbers: list[str] = []
    units: list[str] = []
    negations: list[str] = []
    drugs: list[str] = []
    for token in tokens_of(text):
        if token.isdigit() or token in NUMBER_WORDS:
            numbers.append(token)
        if token in UNITS:
            units.append(token)
        if v4.is_negative_token(token):
            negations.append(token)
        if token in drug_terms:
            drugs.append(token)
    return {
        "numbers": sorted(numbers),
        "units": sorted(units),
        "negations": sorted(negations),
        "drugs": sorted(drugs),
    }


def signature_changed(left: dict[str, list[str]], right: dict[str, list[str]]) -> bool:
    return any(left.get(key) != right.get(key) for key in ("numbers", "units", "negations", "drugs"))


def creates_adjacent_prefix_overlap(task: dict[str, Any], selected: str) -> bool:
    """Reject a slot repair that repeats a neighbouring split-word prefix.

    A time-aligned ASR lattice can split a word such as ``گلوتریو`` into
    ``گلو تیریو``. Replacing only the second slot with the full word would
    create ``گلو گلوتریو``. Phrase-level repair may replace the whole span,
    but an independent slot repair must not introduce this duplication.
    """
    selected_tokens = tokens_of(selected)
    if not selected_tokens:
        return False
    left_tokens = tokens_of(task.get("left_context") or "")
    right_tokens = tokens_of(task.get("right_context") or "")

    def overlaps(left: str, right: str) -> bool:
        left_key = compact_phrase([left])
        right_key = compact_phrase([right])
        return bool(
            min(len(left_key), len(right_key)) >= 3
            and (left_key.startswith(right_key) or right_key.startswith(left_key))
        )

    return bool(
        (left_tokens and overlaps(left_tokens[-1], selected_tokens[0]))
        or (right_tokens and overlaps(selected_tokens[-1], right_tokens[0]))
    )


def lexical_anomaly(text: str, drug_terms: set[str]) -> dict[str, float | int]:
    tokens = tokens_of(text)
    content = [token for token in tokens if len(token) >= 3 and not token.isdigit()]
    unknown = [
        token for token in content
        if token not in drug_terms and not v4.active_general_word(token)
    ]
    adjacent_repeats = sum(
        compact_phrase([tokens[index - 1]]) == compact_phrase([tokens[index]])
        for index in range(1, len(tokens))
    )
    return {
        "content_tokens": len(content),
        "unknown_tokens": len(unknown),
        "unknown_ratio": round(len(unknown) / max(1, len(content)), 4),
        "adjacent_repeats": adjacent_repeats,
    }


def slot_is_uncertain(row: dict[str, Any]) -> bool:
    token = normalize_text(row.get("candidate") or "")
    if not token:
        return row.get("status") in {"REVIEW", "OMIT"}
    strong_families = set(row.get("strong_families") or [])
    exact_families = set(row.get("exact_families") or [])
    probability = float(row.get("acoustic_probability") or 0.0)
    oov = (
        len(v4.phrase_tokens(token)) == 1
        and len(v4.norm(token)) >= 3
        and float(row.get("zipf_frequency_fa") or 0.0) <= 0.0
        and not row.get("general_lexicon")
        and not row.get("medical_lexicon")
        and not row.get("modern_spoken")
    )
    return bool(
        re.search(r"[A-Za-z]", token)
        or
        row.get("status") in {"REVIEW", "OMIT"}
        or row.get("reason") == "turbo-first-preserved-ordinary-review"
        or row.get("medication_entity_ambiguous")
        or row.get("terminal_oov")
        or probability < 0.70
        or len(strong_families) < 2
        or (oov and len(exact_families) < 2)
    )


def slot_severity(row: dict[str, Any]) -> int:
    token = normalize_text(row.get("candidate") or "")
    return int(bool(
        re.search(r"[A-Za-z]", token)
        or row.get("status") in {"REVIEW", "OMIT"}
        or row.get("medication_entity_ambiguous")
        or row.get("sensitive") and len(set(row.get("strong_families") or [])) < 2
    ))


def span_base_text(slots: list[dict[str, Any]], start_slot: int, end_slot: int,
                   placeholders: list[dict[str, Any]]) -> str:
    placeholder_starts = {
        int(row["start_slot"]): row for row in placeholders
        if int(row.get("start_slot", -1)) >= start_slot
        and int(row.get("end_slot", -1)) <= end_slot
    }
    rows: list[dict[str, Any]] = []
    index = start_slot
    while index <= end_slot:
        placeholder = placeholder_starts.get(index)
        if placeholder:
            rows.append({"candidate_tokens": [str(placeholder["placeholder"])]})
            index = int(placeholder["end_slot"]) + 1
            continue
        rows.append({"candidate_tokens": list(slots[index].get("candidate_tokens") or [])})
        index += 1
    return normalize_text(v4.render(rows)[0])


def slot_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}

    def add(text: str, origin: str, families: list[str] | set[str] | None = None,
            sources: list[str] | None = None, score: float = 0.0) -> None:
        text = normalize_text(text)
        key = normalized_key(text) if text else "__EMPTY__"
        item = grouped.setdefault(key, {
            "text": text,
            "origins": [],
            "families": [],
            "sources": [],
            "score": float(score),
        })
        if origin not in item["origins"]:
            item["origins"].append(origin)
        for family in families or []:
            if family and family not in item["families"]:
                item["families"].append(family)
        for source in sources or []:
            if source and source not in item["sources"]:
                item["sources"].append(source)
        item["score"] = max(float(item["score"]), float(score))

    base = normalize_text(row.get("candidate") or "")
    add(base, "v9", row.get("exact_families") or row.get("strong_families") or [], ["v9"], 1e9)
    for observation in row.get("observations") or []:
        add(
            observation.get("normalized") or observation.get("word") or "",
            "observed",
            [str(observation.get("family") or "")],
            [str(observation.get("hypothesis") or "")],
            float(observation.get("probability") or 0.0) * 10.0,
        )
    for alternative in row.get("alternatives") or []:
        text = alternative.get("candidate") or ""
        if text or row.get("status") == "OMIT" or float(row.get("acoustic_probability") or 0.0) < 0.60:
            families = alternative.get("exact_families") or alternative.get("strong_families") or []
            add(
                text,
                str(alternative.get("origin") or "lattice"),
                families,
                [],
                float(alternative.get("emission_score") or 0.0),
            )
            if alternative.get("medical_lexicon"):
                add(text, "medical-lexicon", families, [],
                    float(alternative.get("emission_score") or 0.0))
    rows = list(grouped.values())
    rows.sort(key=lambda item: (
        normalized_key(item["text"]) != normalized_key(base),
        -len(item["families"]),
        -float(item["score"]),
        item["text"],
    ))
    return rows[:7]


def make_slot_tasks(slots: list[dict[str, Any]], max_tasks: int = 18) -> list[dict[str, Any]]:
    ranked: list[tuple[int, int, dict[str, Any]]] = []
    for index, row in enumerate(slots):
        if not slot_is_uncertain(row):
            continue
        candidates = slot_candidates(row)
        base = normalize_text(row.get("candidate") or "")
        filtered: list[dict[str, Any]] = []
        for candidate in candidates:
            text = normalize_text(candidate.get("text") or "")
            # Independent empty-slot insertions were the largest source of
            # hallucinated dose units and function words.  Omissions may only
            # be recovered by the time-aligned phrase stage, never by a token
            # that has no base anchor.
            if not base and text:
                continue
            if base and not text:
                continue
            if (base and v4.active_general_word(base) and text
                    and not v4.active_general_word(text)
                    and len(set(candidate.get("families") or [])) < 2):
                continue
            filtered.append(candidate)
        candidates = filtered
        if len(candidates) < 2:
            continue
        task = {
            "id": f"S{index}",
            "slot": index,
            "base_text": normalize_text(row.get("candidate") or ""),
            "left_context": span_base_text(slots, max(0, index - 5), index - 1, []) if index else "",
            "right_context": span_base_text(slots, index + 1, min(len(slots) - 1, index + 5), [])
            if index + 1 < len(slots) else "",
            "candidates": candidates,
        }
        ranked.append((slot_severity(row), int(row.get("status") in {"REVIEW", "OMIT"}), task))
    ranked.sort(key=lambda item: (-item[0], -item[1], int(item[2]["slot"])))
    selected = [item[2] for item in ranked[:max_tasks]]
    return sorted(selected, key=lambda item: int(item["slot"]))


def build_slot_prompt(tasks: list[dict[str, Any]]) -> tuple[str, dict[str, dict[str, Any]]]:
    lines = [
        "برای هر جایگاه مبهم در رونویسی پزشکی فارسی، مناسب‌ترین گزینه آوایی و زبانی را انتخاب کن.",
        "فقط شناسه را برگردان. هیچ کلمه‌ای نساز. گزینه‌ها از شش شنیده و لغت‌نامه محلی آمده‌اند.",
        "ترتیب گزینه‌ها هیچ امتیاز یا اولویتی ندارد؛ گزینه A پیش‌فرض نیست.",
        "به جمله قبل/بعد دقت کن؛ صورت رایج امروزی را بر واژه مهجور ترجیح بده.",
        "برای نام دارو، عدد، مقدار و نفی محافظه‌کار باش. U یعنی گزینه فعلی را نگه دار.",
    ]
    maps: dict[str, dict[str, Any]] = {}
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    for task in tasks:
        lines.extend([
            "",
            f"جایگاه {task['id']}: ... {task['left_context'] or '—'} [[؟]] {task['right_context'] or '—'} ...",
        ])
        mapping: dict[str, Any] = {}
        # Counter the grammar's first-option bias only when the challenger has
        # direct observations from at least two independent ASR families.
        # Lexicon-only and single-family guesses retain their original order.
        base_key = normalized_key(task["base_text"])
        strong_challengers = [
            candidate for candidate in task["candidates"]
            if normalized_key(candidate.get("text") or "") != base_key
            and "observed" in set(candidate.get("origins") or [])
            and len(set(candidate.get("families") or [])) >= 2
            and "medical-lexicon" not in set(candidate.get("origins") or [])
            and not signature_changed(
                sensitive_signature(task["base_text"], set()),
                sensitive_signature(candidate.get("text") or "", set()),
            )
        ]
        ordered_candidates = strong_challengers + [
            candidate for candidate in task["candidates"]
            if candidate not in strong_challengers
        ]
        for index, candidate in enumerate(ordered_candidates):
            option = alphabet[index]
            mapping[option] = candidate
            display = candidate["text"] if candidate["text"] else "[حذف]"
            lines.append(f"{option}) {display}")
        lines.append("U) نگه‌داشتن گزینه فعلی")
        maps[task["id"]] = mapping
    lines.append("\nفقط JSON مطابق قالب برگردان.")
    return "\n".join(lines), maps


def call_local_slot_qwen(server_url: str, tasks: list[dict[str, Any]], timeout: float,
                         option_maps: dict[str, dict[str, Any]], prompt: str) -> tuple[dict[str, str], dict[str, Any]]:
    properties = {
        task["id"]: {"type": "string", "enum": [*option_maps[task["id"]].keys(), "U"]}
        for task in tasks
    }
    schema = {
        "type": "object",
        "properties": properties,
        "required": [task["id"] for task in tasks],
        "additionalProperties": False,
    }
    body = {
        "model": LOCAL_MODEL_ALIAS,
        "messages": [
            {"role": "system", "content": (
                "تو انتخاب‌گر محدود جایگاه واژه برای رونویسی پزشکی فارسی هستی؛ فقط شناسه گزینه‌ها را تولید کن.")},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "seed": 42,
        "max_tokens": max(64, len(tasks) * 12),
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "slot_choices", "strict": True, "schema": schema},
        },
    }
    request = urllib.request.Request(
        server_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    latency = time.perf_counter() - started
    content = payload["choices"][0]["message"]["content"]
    choices = json.loads(content)
    if set(choices) != set(properties):
        raise ValueError("Local model slot response omitted or added identifiers")
    for task_id, choice in choices.items():
        if choice not in properties[task_id]["enum"]:
            raise ValueError(f"Invalid slot option {choice!r} for {task_id}")
    usage = payload.get("usage") or {}
    return choices, {
        "stage": "slot-lattice-rerank",
        "latency_seconds": round(latency, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "model": payload.get("model"),
    }


def validate_slot_choice(task: dict[str, Any], row: dict[str, Any], choice: str,
                         option_maps: dict[str, dict[str, Any]],
                         drug_terms: set[str]) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "task_id": task["id"],
        "slot": task["slot"],
        "base_text": task["base_text"],
        "model_choice": choice,
        "applied": False,
    }
    if choice == "U":
        audit.update({"selected_text": task["base_text"], "reason": "model-kept-current"})
        return audit
    candidate = option_maps[task["id"]].get(choice)
    if candidate is None:
        audit.update({"selected_text": task["base_text"], "reason": "invalid-option"})
        return audit
    selected = normalize_text(candidate["text"])
    if normalized_key(selected) == normalized_key(task["base_text"]):
        audit.update({"selected_text": task["base_text"], "reason": "model-selected-current"})
        return audit
    observations = [
        normalize_text(item.get("normalized") or item.get("word") or "")
        for item in row.get("observations") or []
    ]
    acoustic_similarity = max(
        (v4.token_similarity(selected, observed) for observed in observations if observed),
        default=0.0,
    )
    families = sorted(set(candidate.get("families") or []))
    base_signature = sensitive_signature(task["base_text"], drug_terms)
    selected_signature = sensitive_signature(selected, drug_terms)
    changed_sensitive = signature_changed(base_signature, selected_signature)

    if not selected and task["base_text"]:
        audit.update({"selected_text": task["base_text"], "reason": "rejected-independent-nonempty-slot-deletion"})
        return audit
    if selected and not task["base_text"]:
        audit.update({
            "selected_text": task["base_text"],
            "reason": "rejected-independent-gap-insertion",
            "supporting_families": families,
        })
        return audit
    if creates_adjacent_prefix_overlap(task, selected):
        audit.update({
            "selected_text": task["base_text"],
            "reason": "rejected-adjacent-prefix-duplication",
            "supporting_families": sorted(set(candidate.get("families") or [])),
        })
        return audit
    elif acoustic_similarity < 0.58 and not families:
        audit.update({
            "selected_text": task["base_text"],
            "reason": "rejected-weak-acoustic-link",
            "acoustic_similarity": round(acoustic_similarity, 4),
        })
        return audit
    base_known = bool(
        task["base_text"] in drug_terms or v4.active_general_word(task["base_text"]))
    selected_known = bool(selected in drug_terms or v4.active_general_word(selected))
    if base_known and selected and not selected_known and len(families) < 2:
        audit.update({
            "selected_text": task["base_text"],
            "reason": "rejected-known-base-to-single-family-oov",
            "supporting_families": families,
        })
        return audit
    if changed_sensitive and (len(families) < 2 or "observed" not in candidate.get("origins", [])):
        audit.update({
            "selected_text": task["base_text"],
            "reason": "rejected-sensitive-slot-change-without-two-observed-families",
            "supporting_families": families,
            "base_sensitive_signature": base_signature,
            "selected_sensitive_signature": selected_signature,
        })
        return audit
    if (selected and selected not in drug_terms and not v4.active_general_word(selected)
            and not any(normalized_key(selected) == normalized_key(item) for item in observations)):
        audit.update({
            "selected_text": task["base_text"],
            "reason": "rejected-unsupported-oov-slot-choice",
            "acoustic_similarity": round(acoustic_similarity, 4),
        })
        return audit
    audit.update({
        "selected_text": selected,
        "supporting_families": families,
        "origins": candidate.get("origins") or [],
        "acoustic_similarity": round(acoustic_similarity, 4),
        "base_sensitive_signature": base_signature,
        "selected_sensitive_signature": selected_signature,
        "applied": True,
        "reason": "constrained-slot-choice-passed-hard-validation",
    })
    return audit


def apply_slot_audits(slots: list[dict[str, Any]], audits: list[dict[str, Any]]) -> None:
    for audit in audits:
        if not audit.get("applied"):
            continue
        row = slots[int(audit["slot"])]
        selected = normalize_text(audit.get("selected_text") or "")
        row["candidate"] = selected
        row["candidate_tokens"] = v4.phrase_tokens(selected)
        row["v10_slot_origin"] = "local-qwen-constrained-lattice-choice"
        row["v10_slot_audit"] = audit


def reject_unconfirmed_single_family_slots(slot_audits: list[dict[str, Any]],
                                           phrase_audits: list[dict[str, Any]],
                                           original_slots: list[dict[str, Any]],
                                           slots: list[dict[str, Any]]) -> None:
    for slot_audit in slot_audits:
        if not slot_audit.get("applied") or len(slot_audit.get("supporting_families") or []) >= 2:
            continue
        slot_index = int(slot_audit["slot"])
        selected = normalize_text(slot_audit.get("selected_text") or "")
        confirmed = False
        for phrase in phrase_audits:
            if (not phrase.get("applied")
                    or not int(phrase["start_slot"]) <= slot_index <= int(phrase["end_slot"])):
                continue
            phrase_text = normalize_text(
                phrase.get("selected_text_before_slot_overlay") or phrase.get("selected_text") or "")
            for source in phrase.get("selected_sources") or []:
                observation = next((
                    item for item in original_slots[slot_index].get("observations") or []
                    if str(item.get("hypothesis") or "") == str(source)
                ), None)
                observed = normalize_text(
                    (observation or {}).get("normalized") or (observation or {}).get("word") or "")
                if (observed and normalized_key(observed) == normalized_key(selected)
                        and normalized_key(selected) in {normalized_key(token) for token in phrase_text.split()}):
                    confirmed = True
                    break
            if confirmed:
                break
        if confirmed:
            slot_audit["single_family_phrase_confirmation"] = True
            continue
        original = original_slots[slot_index]
        slots[slot_index] = copy.deepcopy(original)
        slot_audit["applied"] = False
        slot_audit["selected_text_before_rejection"] = selected
        slot_audit["selected_text"] = normalize_text(original.get("candidate") or "")
        slot_audit["reason"] = "rejected-single-family-slot-without-selected-phrase-confirmation"


def overlay_slot_repairs_on_phrases(phrase_audits: list[dict[str, Any]],
                                    slot_audits: list[dict[str, Any]],
                                    original_slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    overlays: list[dict[str, Any]] = []
    applied_slots = [row for row in slot_audits if row.get("applied")]
    for phrase in phrase_audits:
        if not phrase.get("applied") or not phrase.get("selected_sources"):
            continue
        source = str(phrase["selected_sources"][0])
        surfaces = normalize_text(phrase["selected_text"]).split()
        before = normalize_text(phrase["selected_text"])
        for slot_audit in applied_slots:
            slot_index = int(slot_audit["slot"])
            if not int(phrase["start_slot"]) <= slot_index <= int(phrase["end_slot"]):
                continue
            original = original_slots[slot_index]
            source_observation = next((
                item for item in original.get("observations") or []
                if str(item.get("hypothesis") or "") == source
            ), None)
            if source_observation is None:
                continue
            target = normalize_text(
                source_observation.get("normalized") or source_observation.get("word") or "")
            if not target:
                continue
            matched_index = next((
                index for index, surface in enumerate(surfaces)
                if normalized_key(surface) == normalized_key(target)
            ), None)
            if matched_index is None:
                continue
            replacement = normalize_text(slot_audit.get("selected_text") or "").split()
            old_surface = surfaces[matched_index]
            surfaces[matched_index:matched_index + 1] = replacement
            overlays.append({
                "region_id": phrase["region_id"],
                "slot": slot_index,
                "source_hypothesis": source,
                "from": old_surface,
                "to": normalize_text(slot_audit.get("selected_text") or ""),
                "policy": "validated-slot-choice-overrides-matching-source-token",
            })
        after = normalize_text(" ".join(surfaces))
        if after != before:
            phrase["selected_text_before_slot_overlay"] = before
            phrase["selected_text"] = after
            phrase["slot_overlay_applied"] = True
    return overlays


def expand_for_placeholders(start: int, end: int,
                            placeholders: list[dict[str, Any]]) -> tuple[int, int]:
    changed = True
    while changed:
        changed = False
        for row in placeholders:
            left, right = int(row.get("start_slot", -1)), int(row.get("end_slot", -1))
            if right < start or left > end:
                continue
            new_start, new_end = min(start, left), max(end, right)
            changed = changed or new_start != start or new_end != end
            start, end = new_start, new_end
    return start, end


def compact_phrase(tokens: list[str]) -> str:
    return "".join(v4.phonetic_key(token) for token in tokens)


def trim_boundary_overlap(text: str, left_context: str, right_context: str) -> str:
    candidate = v4.phrase_tokens(text)
    left = v4.phrase_tokens(left_context)
    right = v4.phrase_tokens(right_context)
    if not candidate:
        return ""

    best_prefix = 0
    for candidate_count in range(1, min(3, len(candidate)) + 1):
        for context_count in range(1, min(3, len(left)) + 1):
            candidate_part = candidate[:candidate_count]
            context_part = left[-context_count:]
            exact_compact = compact_phrase(candidate_part) == compact_phrase(context_part)
            close_single = (
                candidate_count == context_count == 1
                and v4.token_similarity(candidate_part[0], context_part[0]) >= 0.84
            )
            if exact_compact or close_single:
                best_prefix = max(best_prefix, candidate_count)
    if best_prefix:
        candidate = candidate[best_prefix:]
    if not candidate:
        return ""

    best_suffix = 0
    for candidate_count in range(1, min(3, len(candidate)) + 1):
        for context_count in range(1, min(3, len(right)) + 1):
            candidate_part = candidate[-candidate_count:]
            context_part = right[:context_count]
            exact_compact = compact_phrase(candidate_part) == compact_phrase(context_part)
            close_single = (
                candidate_count == context_count == 1
                and v4.token_similarity(candidate_part[0], context_part[0]) >= 0.84
            )
            if exact_compact or close_single:
                best_suffix = max(best_suffix, candidate_count)
    if best_suffix:
        candidate = candidate[:-best_suffix]
    return normalize_text(" ".join(candidate))


def region_candidates(hypotheses: dict[str, dict[str, Any]], base_text: str,
                      start_time: float, end_time: float, full_audio: bool,
                      left_context: str, right_context: str) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}

    def add(text: str, source: str, trim: bool = False) -> None:
        text = normalize_text(text)
        if trim:
            text = trim_boundary_overlap(text, left_context, right_context)
        key = normalized_key(text)
        if not key:
            return
        row = grouped.setdefault(key, {"text": text, "sources": [], "families": []})
        if source not in row["sources"]:
            row["sources"].append(source)
        family = FAMILY_BY_HYPOTHESIS[source]
        if family not in row["families"]:
            row["families"].append(family)

    add(base_text, "v9")
    for key, payload in hypotheses.items():
        if full_audio:
            add(payload["text"], key)
            continue
        selected = []
        for word in payload["words"]:
            midpoint = (word["start"] + word["end"]) / 2.0
            overlaps = word["end"] > start_time and word["start"] < end_time
            if start_time <= midpoint <= end_time or overlaps:
                selected.append(word["word"])
        add(" ".join(selected), key, trim=True)
    rows = list(grouped.values())
    rows.sort(key=lambda row: ("v9" not in row["sources"], -len(row["families"]), row["text"]))
    return rows


def make_regions(slots: list[dict[str, Any]], hypotheses: dict[str, dict[str, Any]],
                 placeholders: list[dict[str, Any]], max_regions: int) -> list[dict[str, Any]]:
    if not slots:
        return []
    duration = max(payload["duration"] for payload in hypotheses.values())
    uncertain = [index for index, row in enumerate(slots) if slot_is_uncertain(row)]
    # Short messages benefit from one coherent utterance-level decision. This rule
    # is based only on duration, never on a filename, reference transcript or voice.
    full_audio = duration <= 25.0 and bool(uncertain)
    if full_audio:
        ranges = [(0, len(slots) - 1)]
    else:
        clusters: list[list[int]] = []
        for index in uncertain:
            if not clusters:
                clusters.append([index])
                continue
            previous = clusters[-1][-1]
            time_gap = float(slots[index].get("start") or 0.0) - float(slots[previous].get("end") or 0.0)
            if index - previous <= 3 and time_gap <= 1.4:
                clusters[-1].append(index)
            else:
                clusters.append([index])
        ranges = []
        for cluster in clusters:
            start = max(0, cluster[0] - 2)
            end = min(len(slots) - 1, cluster[-1] + 2)
            start, end = expand_for_placeholders(start, end, placeholders)
            if ranges and start <= ranges[-1][1] + 1:
                ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
            else:
                ranges.append((start, end))
        # Prefer the spans with the largest uncertainty density when the audio is long.
        ranges.sort(key=lambda pair: (
            -sum(slot_severity(row) for row in slots[pair[0]:pair[1] + 1]),
            -sum(slot_is_uncertain(row) for row in slots[pair[0]:pair[1] + 1]),
            pair[0],
        ))
        ranges = sorted(ranges[:max_regions])

    regions: list[dict[str, Any]] = []
    for region_index, (start_slot, end_slot) in enumerate(ranges):
        start_time = max(0.0, float(slots[start_slot].get("start") or 0.0) - 0.06)
        end_time = float(slots[end_slot].get("end") or start_time) + 0.06
        base_text = span_base_text(slots, start_slot, end_slot, placeholders)
        left_context = span_base_text(slots, max(0, start_slot - 5), start_slot - 1, placeholders) \
            if start_slot > 0 else ""
        right_context = span_base_text(
            slots, end_slot + 1, min(len(slots) - 1, end_slot + 5), placeholders) \
            if end_slot + 1 < len(slots) else ""
        candidates = region_candidates(
            hypotheses, base_text, start_time, end_time, full_audio=full_audio,
            left_context=left_context, right_context=right_context)
        if len(candidates) < 2:
            continue
        regions.append({
            "id": f"R{region_index}",
            "start_slot": start_slot,
            "end_slot": end_slot,
            "start": round(start_time, 3),
            "end": round(end_time, 3),
            "base_text": base_text,
            "left_context": left_context,
            "right_context": right_context,
            "candidates": candidates,
            "full_audio": full_audio,
        })
    return regions


def build_prompt(regions: list[dict[str, Any]]) -> tuple[str, dict[str, dict[str, Any]]]:
    lines = [
        "از میان رونویسی‌های یک مکالمه پزشکی فارسی، برای هر بخش نزدیک‌ترین گزینه به گفتار طبیعی و معنادار را انتخاب کن.",
        "فقط شناسه گزینه را بده؛ کلمه تازه نساز، گزینه‌ها را ترکیب نکن و بازنویسی نکن.",
        "ترتیب گزینه‌ها هیچ امتیاز یا اولویتی ندارد؛ گزینه A پیش‌فرض و امن‌تر از بقیه نیست.",
        "تکرار ناخواسته، پسوند جداافتاده، واژه شکسته و عبارت ناتمام را خطای جدی بدان و گزینه روان‌تر و کامل‌تر را انتخاب کن.",
        "معنا، دستور و گفتار روزمره فارسی، بافت پزشکی و متن قبل/بعد را بسنج.",
        "لازم نیست گزینه بی‌نقص باشد: اگر یکی از بقیه روشن‌تر و منسجم‌تر است همان را انتخاب کن.",
        "U فقط وقتی مجاز است که همه گزینه‌ها واقعاً نامفهوم باشند و هیچ‌کدام برتری معناداری نداشته باشد.",
        "در نام دارو، دوز، عدد و نفی محافظه‌کار باش. خروجی فقط شناسه است.",
    ]
    option_maps: dict[str, dict[str, Any]] = {}
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    for region in regions:
        lines.extend([
            "",
            f"بخش {region['id']}:",
            f"قبل: {region['left_context'] or '—'}",
            f"بعد: {region['right_context'] or '—'}",
        ])
        mapping: dict[str, Any] = {}
        # The current V9/V10 fallback is intentionally placed last. Some local
        # quantizations exhibit a strong first-option bias under JSON grammar;
        # keeping the fallback first would silently turn that bias into a
        # preference for preserving even visibly broken or duplicated text.
        ordered_candidates = sorted(
            region["candidates"],
            key=lambda candidate: "v9" in set(candidate.get("sources") or []),
        )
        for index, candidate in enumerate(ordered_candidates):
            option = alphabet[index]
            mapping[option] = candidate
            lines.append(f"{option}) {candidate['text']}")
        lines.append("U) هیچ گزینه‌ای قابل اعتماد نیست")
        option_maps[region["id"]] = mapping
    lines.append("\nفقط شیء JSON مطابق قالب خواسته‌شده را برگردان.")
    return "\n".join(lines), option_maps


def call_local_qwen(server_url: str, regions: list[dict[str, Any]], timeout: float,
                    option_maps: dict[str, dict[str, Any]], prompt: str) -> tuple[dict[str, str], dict[str, Any]]:
    properties = {
        region["id"]: {"type": "string", "enum": [*option_maps[region["id"]].keys(), "U"]}
        for region in regions
    }
    schema = {
        "type": "object",
        "properties": properties,
        "required": [region["id"] for region in regions],
        "additionalProperties": False,
    }
    body = {
        "model": LOCAL_MODEL_ALIAS,
        "messages": [
            {"role": "system", "content": (
                "تو انتخاب‌گر محدود عبارت برای رونویسی پزشکی فارسی هستی. "
                "هیچ متن آزادی تولید نکن و دقیقاً از گزینه‌ها انتخاب کن.")},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "seed": 42,
        "max_tokens": max(48, len(regions) * 12),
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "phrase_choices", "strict": True, "schema": schema},
        },
    }
    endpoint = server_url.rstrip("/") + "/v1/chat/completions"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    latency = time.perf_counter() - started
    content = payload["choices"][0]["message"]["content"]
    choices = json.loads(content)
    if set(choices) != set(properties):
        raise ValueError("Local model response omitted or added region identifiers")
    for region_id, choice in choices.items():
        if choice not in properties[region_id]["enum"]:
            raise ValueError(f"Invalid option {choice!r} for {region_id}")
    usage = payload.get("usage") or {}
    return choices, {
        "latency_seconds": round(latency, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "model": payload.get("model"),
    }


def call_conservative_reviewer(server_url: str, items: list[dict[str, Any]], timeout: float,
                               stage: str) -> tuple[dict[str, str], dict[str, Any]]:
    """Verify weakly supported choices without allowing any generated text."""
    if not items:
        return {}, {"stage": stage, "latency_seconds": 0.0, "prompt_tokens": 0,
                    "completion_tokens": 0, "model": None}
    properties = {
        item["id"]: {"type": "string", "enum": ["A", "U"]}
        for item in items
    }
    schema = {
        "type": "object",
        "properties": properties,
        "required": [item["id"] for item in items],
        "additionalProperties": False,
    }
    lines = [
        "این تأیید ثانویه و مستقلِ تصمیم‌های پرریسک در رونویسی پزشکی فارسی است.",
        "A یعنی پیشنهاد را فقط وقتی بپذیر که به طور روشن طبیعی‌تر، از نظر آوایی نزدیک و در همین بافت درست‌تر باشد.",
        "U یعنی متن پایه حفظ شود. اگر هر دو ناقص‌اند، معنا مبهم است یا تغییر فقط یک حدس محتمل است، حتماً U را انتخاب کن.",
        "پیشنهاد حق ندارد واقعیت پزشکی، نام، نفی، عدد یا مقدار را حدس بزند. هیچ متن تازه‌ای تولید نکن.",
    ]
    for item in items:
        lines.extend([
            "",
            f"{item['id']} [{item['risk']}]:",
            f"قبل: {item.get('left_context') or '—'}",
            f"پایه: {item['base_text']}",
            f"پیشنهاد: {item['selected_text']}",
            f"بعد: {item.get('right_context') or '—'}",
        ])
    body = {
        "model": LOCAL_MODEL_ALIAS,
        "messages": [
            {"role": "system", "content": (
                "تو ممیز سخت‌گیر انتخاب محدود برای رونویسی پزشکی فارسی هستی. "
                "تصمیم قبلی را مهر تأیید نزن؛ در تردید گزینه پایه را نگه دار و فقط JSON شناسه‌ها را بده.")},
            {"role": "user", "content": "\n".join(lines)},
        ],
        "temperature": 0,
        "seed": 42,
        "max_tokens": max(48, len(items) * 10),
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": stage.replace("-", "_"), "strict": True, "schema": schema},
        },
    }
    request = urllib.request.Request(
        server_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    latency = time.perf_counter() - started
    choices = json.loads(payload["choices"][0]["message"]["content"])
    if set(choices) != set(properties):
        raise ValueError("Local conservative-review response omitted or added identifiers")
    if any(choice not in {"A", "U"} for choice in choices.values()):
        raise ValueError("Local conservative-review response contained an invalid decision")
    usage = payload.get("usage") or {}
    return choices, {
        "stage": stage,
        "latency_seconds": round(latency, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "model": payload.get("model"),
    }


def make_slot_review_items(slot_tasks: list[dict[str, Any]],
                           slot_audits: list[dict[str, Any]],
                           exact_medical_terms: set[str] | None = None) -> list[dict[str, Any]]:
    tasks = {task["id"]: task for task in slot_tasks}
    exact_medical_terms = exact_medical_terms or set()
    items: list[dict[str, Any]] = []
    for audit in slot_audits:
        if not audit.get("applied"):
            continue
        task = tasks[audit["task_id"]]
        observed_choice = "observed" in (audit.get("origins") or [])
        base_key = normalized_key(audit.get("base_text") or "")
        selected_key = normalized_key(audit.get("selected_text") or "")
        function_keys = {normalized_key(word) for word in v4.FUNCTION_WORDS}
        function_word_swap = (
            len(base_key) <= 3 and len(selected_key) <= 3
            and (base_key in function_keys or selected_key in function_keys)
        )
        changed_sensitive = signature_changed(
            audit.get("base_sensitive_signature") or {},
            audit.get("selected_sensitive_signature") or {},
        )
        if observed_choice and not function_word_swap and not changed_sensitive:
            continue
        right_first = next(iter(v4.phrase_tokens(task.get("right_context") or "")), "")
        supported_vocative_name = (
            normalized_key(right_first) == normalized_key("جان")
            and len(audit.get("supporting_families") or []) >= 2
            and float(audit.get("acoustic_similarity") or 0.0) >= 0.80
        )
        if supported_vocative_name:
            audit["conservative_review_exemption"] = "two-family-vocative-name-before-jan"
            continue
        supported_productive_repair = (
            "productive-prefix-repair" in (audit.get("origins") or [])
            and len(audit.get("supporting_families") or []) >= 2
            and float(audit.get("acoustic_similarity") or 0.0) >= 0.90
        )
        if supported_productive_repair:
            audit["conservative_review_exemption"] = "high-similarity-two-family-productive-repair"
            continue
        supported_medical_canonicalization = (
            "medical-lexicon" in (audit.get("origins") or [])
            and selected_key in exact_medical_terms
            and len(audit.get("supporting_families") or []) >= 2
            and float(audit.get("acoustic_similarity") or 0.0) >= 0.88
        )
        if supported_medical_canonicalization:
            audit["conservative_review_exemption"] = "high-similarity-two-family-medical-canonicalization"
            continue
        canonicalizes_supported_observation = any(
            "observed" in (candidate.get("origins") or [])
            and len(candidate.get("families") or []) >= 2
            and normalized_key(candidate.get("text") or "") != base_key
            and v4.token_similarity(audit.get("selected_text") or "", candidate.get("text") or "") >= 0.92
            for candidate in task.get("candidates") or []
        )
        if not observed_choice and canonicalizes_supported_observation:
            audit["conservative_review_exemption"] = "canonicalizes-two-family-observation"
            continue
        if not observed_choice:
            audit["selected_text_before_review_rejection"] = audit.get("selected_text")
            audit["selected_text"] = audit.get("base_text") or ""
            audit["applied"] = False
            audit["reason"] = "rejected-unobserved-lexicon-without-safe-exemption"
            continue
        items.append({
            "id": audit["task_id"],
            "risk": (
                "تغییر نفی، عدد، مقدار یا نام دارو با وجود شاهد صوتی"
                if changed_sensitive else
                "جابه‌جایی دو واژهٔ نقشیِ کوتاه" if function_word_swap else
                "واژهٔ لغت‌نامه‌ای که عیناً در هیچ فرضیه شنیده نشده"),
            "left_context": task.get("left_context") or "",
            "right_context": task.get("right_context") or "",
            "base_text": audit["base_text"],
            "selected_text": audit["selected_text"],
        })
    return items


def make_phrase_review_items(regions: list[dict[str, Any]],
                             audits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    region_map = {region["id"]: region for region in regions}
    items: list[dict[str, Any]] = []
    for audit in audits:
        if not audit.get("applied") or len(audit.get("supporting_families") or []) >= 2:
            continue
        region = region_map[audit["region_id"]]
        base_count = len(v4.phrase_tokens(audit["base_text"]))
        selected_count = len(v4.phrase_tokens(audit["selected_text"]))
        # A time-aligned source phrase may legitimately omit duplicated filler.
        # The extra review is reserved for one-family insertions, plus any
        # length change that would replace an entire short recording.
        if selected_count == base_count:
            continue
        if selected_count < base_count and (
                not region.get("full_audio") or base_count > 4):
            continue
        items.append({
            "id": audit["region_id"],
            "risk": "عبارت تک‌خانواده‌ای که تعداد واژه‌ها را تغییر داده",
            "left_context": region.get("left_context") or "",
            "right_context": region.get("right_context") or "",
            "base_text": audit["base_text"],
            "selected_text": audit["selected_text"],
        })
    return items


def apply_conservative_review(audits: list[dict[str, Any]],
                              choices: dict[str, str], id_key: str) -> None:
    for audit in audits:
        item_id = str(audit.get(id_key) or "")
        if item_id not in choices:
            continue
        audit["conservative_review"] = choices[item_id]
        if choices[item_id] == "A":
            audit["conservative_review_passed"] = True
            continue
        audit["selected_text_before_review_rejection"] = audit.get("selected_text")
        audit["selected_text"] = audit.get("base_text") or ""
        audit["applied"] = False
        audit["reason"] = "rejected-by-conservative-second-pass"


def region_is_sensitive(region: dict[str, Any], slots: list[dict[str, Any]],
                        drug_terms: set[str]) -> bool:
    if any(row.get("sensitive") for row in slots[region["start_slot"]:region["end_slot"] + 1]):
        return True
    base_signature = sensitive_signature(region["base_text"], drug_terms)
    return any(base_signature.values())


def validate_choice(region: dict[str, Any], choice: str, option_maps: dict[str, dict[str, Any]],
                    slots: list[dict[str, Any]], drug_terms: set[str]) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "region_id": region["id"],
        "start_slot": region["start_slot"],
        "end_slot": region["end_slot"],
        "start": region["start"],
        "end": region["end"],
        "base_text": region["base_text"],
        "model_choice": choice,
        "applied": False,
    }
    if choice == "U":
        audit.update({"selected_text": region["base_text"], "reason": "model-abstained"})
        return audit
    candidate = option_maps[region["id"]].get(choice)
    if not candidate:
        audit.update({"selected_text": region["base_text"], "reason": "invalid-option"})
        return audit
    selected_text = normalize_text(candidate["text"])
    source_texts = {normalized_key(item["text"]) for item in region["candidates"]}
    if normalized_key(selected_text) not in source_texts:
        audit.update({"selected_text": region["base_text"], "reason": "not-verbatim-source-candidate"})
        return audit

    base_count = len(v4.phrase_tokens(region["base_text"]))
    selected_count = len(v4.phrase_tokens(selected_text))
    ratio = selected_count / max(1, base_count)
    minimum_count = 3 if region.get("full_audio") else 4
    lower_ratio, upper_ratio = (0.45, 2.20) if region.get("full_audio") else (0.55, 1.80)
    if base_count >= minimum_count and not lower_ratio <= ratio <= upper_ratio:
            audit.update({
                "selected_text": region["base_text"],
                "reason": "rejected-temporal-span-length-mismatch",
                "base_token_count": base_count,
                "selected_token_count": selected_count,
            })
            return audit

    base_anomaly = lexical_anomaly(region["base_text"], drug_terms)
    selected_anomaly = lexical_anomaly(selected_text, drug_terms)
    independent_families = sorted(family for family in candidate["families"] if family != "v9")
    anomaly_increased = (
        int(selected_anomaly["unknown_tokens"]) > int(base_anomaly["unknown_tokens"])
        and float(selected_anomaly["unknown_ratio"]) > float(base_anomaly["unknown_ratio"])
    )
    if (
        anomaly_increased
        and (
            len(independent_families) < 2
            or float(selected_anomaly["unknown_ratio"])
            > float(base_anomaly["unknown_ratio"]) + 0.12
        )
    ):
        audit.update({
            "selected_text": region["base_text"],
            "reason": "rejected-increased-lexical-anomaly",
            "base_lexical_anomaly": base_anomaly,
            "selected_lexical_anomaly": selected_anomaly,
        })
        return audit

    base_signature = sensitive_signature(region["base_text"], drug_terms)
    selected_signature = sensitive_signature(selected_text, drug_terms)
    changed_sensitive = signature_changed(base_signature, selected_signature)
    if changed_sensitive and len(independent_families) < 2:
        audit.update({
            "selected_text": region["base_text"],
            "reason": "rejected-sensitive-change-without-two-independent-families",
            "base_sensitive_signature": base_signature,
            "selected_sensitive_signature": selected_signature,
            "supporting_families": independent_families,
        })
        return audit
    audit.update({
        "selected_text": selected_text,
        "selected_sources": candidate["sources"],
        "supporting_families": independent_families,
        "base_sensitive_signature": base_signature,
        "selected_sensitive_signature": selected_signature,
        "base_lexical_anomaly": base_anomaly,
        "selected_lexical_anomaly": selected_anomaly,
        "sensitive_region": region_is_sensitive(region, slots, drug_terms),
        "applied": normalized_key(selected_text) != normalized_key(region["base_text"]),
        "reason": "verbatim-source-choice-passed-hard-validation",
    })
    return audit


def merge_regions(slots: list[dict[str, Any]], placeholders: list[dict[str, Any]],
                  audits: list[dict[str, Any]]) -> str:
    audit_by_start = {int(row["start_slot"]): row for row in audits}
    placeholder_by_start = {int(row["start_slot"]): row for row in placeholders}
    rows: list[dict[str, Any]] = []
    index = 0
    while index < len(slots):
        audit = audit_by_start.get(index)
        if audit:
            rows.append({"candidate_tokens": normalize_text(audit["selected_text"]).split()})
            index = int(audit["end_slot"]) + 1
            continue
        placeholder = placeholder_by_start.get(index)
        if placeholder:
            rows.append({"candidate_tokens": [str(placeholder["placeholder"])]})
            index = int(placeholder["end_slot"]) + 1
            continue
        rows.append({"candidate_tokens": list(slots[index].get("candidate_tokens") or [])})
        index += 1
    text = normalize_text(v4.render(rows)[0])
    protected_text, _protected = v5.protect_honorific_names(text)
    return normalize_text(protected_text)


def project_bounded_choice(text: str, base_text: str, selected_text: str,
                           left_context: str = "", right_context: str = "") -> tuple[str, dict[str, Any]]:
    """Patch one bounded choice onto V9; never rebuild untouched transcript spans."""
    surfaces = normalize_text(text).split()
    base_tokens = v4.phrase_tokens(base_text)
    selected = normalize_text(selected_text)
    if not surfaces or not base_tokens:
        return text, {"projected": False, "reason": "empty-projection-anchor"}
    normalized_surfaces = [normalized_key(surface) for surface in surfaces]
    left_tokens = v4.phrase_tokens(left_context)[-4:]
    right_tokens = v4.phrase_tokens(right_context)[:4]
    candidates: list[tuple[float, float, int, int]] = []
    minimum = max(1, len(base_tokens) - 2)
    maximum = min(len(surfaces), len(base_tokens) + 2)
    for start in range(len(surfaces)):
        for width in range(minimum, maximum + 1):
            end = start + width
            if end > len(surfaces):
                break
            window = [token for token in normalized_surfaces[start:end] if token]
            if not window:
                continue
            lexical = v4.fuzz.ratio(" ".join(base_tokens), " ".join(window)) / 100.0
            left_window = [token for token in normalized_surfaces[max(0, start - len(left_tokens)):start] if token]
            right_window = [token for token in normalized_surfaces[end:end + len(right_tokens)] if token]
            left_score = (
                v4.fuzz.ratio(" ".join(left_tokens), " ".join(left_window)) / 100.0
                if left_tokens and left_window else 0.5
            )
            right_score = (
                v4.fuzz.ratio(" ".join(right_tokens), " ".join(right_window)) / 100.0
                if right_tokens and right_window else 0.5
            )
            total = 0.78 * lexical + 0.11 * left_score + 0.11 * right_score
            candidates.append((total, lexical, start, end))
    candidates.sort(reverse=True)
    if not candidates:
        return text, {"projected": False, "reason": "no-projection-window"}
    best = candidates[0]
    runner_up = candidates[1] if len(candidates) > 1 else (0.0, 0.0, -1, -1)
    exact_count = sum(candidate[1] >= 0.999 for candidate in candidates)
    exact = best[1] >= 0.999
    # An exact surface match is still ambiguous when the same anchor is repeated.
    # Context must then separate the intended occurrence by a meaningful margin.
    unambiguous = (exact and exact_count == 1) or best[0] - runner_up[0] >= 0.025
    if best[1] < 0.72 or best[0] < 0.70 or not unambiguous:
        return text, {
            "projected": False,
            "reason": "low-or-ambiguous-v9-projection",
            "best_score": round(best[0], 4),
            "lexical_score": round(best[1], 4),
            "margin": round(best[0] - runner_up[0], 4),
        }
    replacement = selected.split() if selected else []
    projected = surfaces[:best[2]] + replacement + surfaces[best[3]:]
    return normalize_text(" ".join(projected)), {
        "projected": True,
        "reason": "bounded-choice-projected-onto-v9",
        "base_text": normalize_text(base_text),
        "selected_text": selected,
        "matched_text": " ".join(surfaces[best[2]:best[3]]),
        "best_score": round(best[0], 4),
        "lexical_score": round(best[1], 4),
    }


def project_audited_changes_onto_v9(v9_text: str, original_slots: list[dict[str, Any]],
                                    placeholders: list[dict[str, Any]],
                                    regions: list[dict[str, Any]], audits: list[dict[str, Any]],
                                    slot_tasks: list[dict[str, Any]],
                                    slot_audits: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    text = normalize_text(v9_text)
    projection_audit: list[dict[str, Any]] = []
    covered_slots: set[int] = set()
    region_map = {region["id"]: region for region in regions}
    task_map = {task["id"]: task for task in slot_tasks}
    for audit in sorted(
            (row for row in audits if row.get("applied")),
            key=lambda row: int(row["start_slot"]), reverse=True):
        start_slot, end_slot = int(audit["start_slot"]), int(audit["end_slot"])
        original_base = span_base_text(original_slots, start_slot, end_slot, placeholders)
        region = region_map[audit["region_id"]]
        text, result = project_bounded_choice(
            text, original_base, audit["selected_text"],
            region.get("left_context") or "", region.get("right_context") or "")
        result.update({"kind": "phrase", "id": audit["region_id"]})
        projection_audit.append(result)
        if result["projected"]:
            covered_slots.update(range(start_slot, end_slot + 1))
            for slot_audit in slot_audits:
                if slot_audit.get("applied") and start_slot <= int(slot_audit["slot"]) <= end_slot:
                    projection_audit.append({
                        "projected": True,
                        "reason": "included-in-projected-verbatim-phrase",
                        "kind": "slot",
                        "id": slot_audit["task_id"],
                        "phrase_id": audit["region_id"],
                    })
        else:
            audit["applied"] = False
            audit["reason"] = "rejected-unprojectable-onto-v9"
    for audit in sorted(
            (row for row in slot_audits if row.get("applied") and int(row["slot"]) not in covered_slots),
            key=lambda row: int(row["slot"]), reverse=True):
        task = task_map[audit["task_id"]]
        text, result = project_bounded_choice(
            text, audit["base_text"], audit["selected_text"],
            task.get("left_context") or "", task.get("right_context") or "")
        result.update({"kind": "slot", "id": audit["task_id"]})
        projection_audit.append(result)
        if not result["projected"]:
            audit["applied"] = False
            audit["reason"] = "rejected-unprojectable-onto-v9"
    protected_text, _protected = v5.protect_honorific_names(text)
    return normalize_text(protected_text), projection_audit


def cleanup_repetitions(text: str) -> tuple[str, list[dict[str, Any]]]:
    tokens = normalize_text(text).split()
    audit: list[dict[str, Any]] = []
    collapsed: list[str] = []
    for token in tokens:
        if collapsed:
            previous = v4.norm(collapsed[-1]).replace("‌", "")
            current = v4.norm(token).replace("‌", "")
            if previous == current:
                audit.append({"removed": token, "reason": "adjacent-exact-duplicate"})
                continue
            if (previous == "می" + current or previous == "نمی" + current
                    or current == "می" + previous or current == "نمی" + previous):
                keep_previous = len(previous) >= len(current)
                audit.append({
                    "removed": token if keep_previous else collapsed[-1],
                    "reason": "boundary-inflection-duplicate",
                })
                if keep_previous:
                    continue
                collapsed[-1] = token
                continue
        collapsed.append(token)

    index = 0
    output: list[str] = []
    while index < len(collapsed):
        matched = False
        for width in range(min(6, (len(collapsed) - index) // 3), 0, -1):
            unit = [normalized_key(item) for item in collapsed[index:index + width]]
            repeats = 1
            while index + (repeats + 1) * width <= len(collapsed):
                following = [
                    normalized_key(item)
                    for item in collapsed[index + repeats * width:index + (repeats + 1) * width]
                ]
                if following != unit:
                    break
                repeats += 1
            if repeats >= 3:
                output.extend(collapsed[index:index + width])
                audit.append({
                    "removed_repeat_count": repeats - 1,
                    "phrase": " ".join(collapsed[index:index + width]),
                    "reason": "three-or-more-consecutive-exact-phrase-repetitions",
                })
                index += repeats * width
                matched = True
                break
        if not matched:
            output.append(collapsed[index])
            index += 1
    return normalize_text(" ".join(output)), audit


def cleanup_honorific_titles(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Normalize high-precision ASR variants only when followed by a title."""
    audit: list[dict[str, Any]] = []
    patterns = (
        (r"(?<!\S)(?:خانون|خانوم|خام)(?=\s+(?:دکتر|پزشک)\b)", "خانم"),
        (r"(?<!\S)آقایی(?=\s+(?:دکتر|پزشک|مهندس)\b)", "آقای"),
    )
    result = normalize_text(text)
    for pattern, replacement in patterns:
        matches = re.findall(pattern, result)
        if not matches:
            continue
        result = re.sub(pattern, replacement, result)
        audit.append({
            "matched": matches,
            "replacement": replacement,
            "reason": "high-precision-honorific-before-title",
        })
    return normalize_text(result), audit


def write_outputs(run_dir: Path, v9: dict[str, Any], final_text: str,
                  regions: list[dict[str, Any]], audits: list[dict[str, Any]],
                  slot_tasks: list[dict[str, Any]], slot_audits: list[dict[str, Any]],
                  slot_overlays: list[dict[str, Any]],
                  model_call: dict[str, Any], elapsed: float, fallback_reason: str | None,
                  ascii_cleanup: list[dict[str, Any]],
                  repetition_cleanup: list[dict[str, Any]],
                  honorific_cleanup: list[dict[str, Any]],
                  projection_audit: list[dict[str, Any]]) -> dict[str, Any]:
    output_dir = run_dir / OUTPUT_RELATIVE
    output_dir.mkdir(parents=True, exist_ok=True)
    applied = [row for row in audits if row.get("applied")]
    rejected = [row for row in audits if str(row.get("reason") or "").startswith("rejected-")]
    abstained = [row for row in audits if row.get("reason") == "model-abstained"]
    slot_applied = [row for row in slot_audits if row.get("applied")]
    slot_rejected = [
        row for row in slot_audits if str(row.get("reason") or "").startswith("rejected-")]
    projected_choices = {
        (str(row.get("kind") or ""), str(row.get("id") or ""))
        for row in projection_audit if row.get("projected")
    }
    serializable_regions = []
    for region in regions:
        row = dict(region)
        row["candidates"] = [dict(candidate) for candidate in region["candidates"]]
        serializable_regions.append(row)
    payload = {
        "algorithm": "v10 local Qwen3.5-35B-A3B constrained slot-lattice and phrase reranker",
        "text": final_text,
        "v9_text": normalize_text(v9.get("text") or ""),
        "runtime_seconds": round(elapsed, 3),
        "local_model_used": fallback_reason is None and bool(regions or slot_tasks),
        "external_api_used_at_runtime": False,
        "free_text_generation_enters_output": False,
        "candidate_policy": "slot options from the V9 acoustic/lexicon lattice, verbatim phrases from six ASR hypotheses, and unchanged V9 fallback only",
        "safety_policy": "drug/dose/number/negation changes require exact support from at least two independent ASR model families; unobserved lexicon choices and one-family length changes require a separate conservative accept/revert pass",
        "failure_policy": "any server, schema, timeout or validation failure returns unchanged V9 text",
        "model": {
            "repository": LOCAL_MODEL_REPOSITORY,
            "revision": LOCAL_MODEL_REVISION,
            "file": LOCAL_MODEL_FILE,
            "quantization": LOCAL_MODEL_QUANTIZATION,
            **model_call,
        },
        "fallback_reason": fallback_reason,
        "region_count": len(regions),
        "applied_region_count": len(applied),
        "rejected_region_count": len(rejected),
        "abstained_region_count": len(abstained),
        "slot_task_count": len(slot_tasks),
        "applied_slot_count": len(slot_applied),
        "rejected_slot_count": len(slot_rejected),
        "slot_tasks": slot_tasks,
        "slot_audit": slot_audits,
        "slot_overlays_on_selected_phrases": slot_overlays,
        "regions": serializable_regions,
        "audit": audits,
        "unknown_ascii_cleanup": ascii_cleanup,
        "repetition_cleanup": repetition_cleanup,
        "honorific_cleanup": honorific_cleanup,
        "v9_bounded_projection": projection_audit,
        "hard_validation": {
            "all_applied_phrases_are_source_candidates": all(
                row.get("reason") == "verbatim-source-choice-passed-hard-validation"
                for row in applied),
            "all_applied_slots_are_lattice_candidates": all(
                row.get("reason") == "constrained-slot-choice-passed-hard-validation"
                for row in slot_applied),
            "unsafe_sensitive_changes_applied": 0,
            "model_free_text_accepted": False,
            "all_applied_changes_projected_onto_v9": (
                all(("phrase", str(row["region_id"])) in projected_choices for row in applied)
                and all(("slot", str(row["task_id"])) in projected_choices for row in slot_applied)
            ),
        },
    }
    summary = {
        key: payload[key] for key in (
            "algorithm", "runtime_seconds", "local_model_used", "external_api_used_at_runtime",
            "free_text_generation_enters_output", "fallback_reason", "region_count",
            "applied_region_count", "rejected_region_count", "abstained_region_count",
            "slot_task_count", "applied_slot_count", "rejected_slot_count")
    }
    summary["model"] = payload["model"]
    (output_dir / "final-v10.txt").write_text(final_text + "\n", encoding="utf-8")
    (output_dir / "final-v10.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary-v10.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    comparison_lines = [
        "# مقایسهٔ V10 با V9",
        "",
        f"- زمان انتخاب‌گر محلی: {model_call.get('latency_seconds', 0)} ثانیه",
        f"- جایگاه‌های واژه‌ای تغییرکرده: {len(slot_applied)} از {len(slot_tasks)}",
        f"- ناحیه‌های عبارتی تغییرکرده: {len(applied)} از {len(regions)}",
        f"- تغییرهای پرخطر ردشده: {len(rejected) + len(slot_rejected)}",
        "",
        "## V9",
        "",
        normalize_text(v9.get("text") or ""),
        "",
        "## V10",
        "",
        final_text,
    ]
    (output_dir / "comparison-v10.md").write_text("\n".join(comparison_lines) + "\n", encoding="utf-8")
    review_lines = ["# بازبینی V10", ""]
    if not rejected and not slot_rejected and not abstained and not fallback_reason:
        review_lines.append("موردی برای بازبینی اجباری ثبت نشد.")
    if fallback_reason:
        review_lines.append(f"- مدل محلی اعمال نشد و V9 حفظ شد: `{fallback_reason}`")
    for row in slot_rejected:
        review_lines.append(
            f"- {row['task_id']} (slot {row['slot']}): {row['reason']} — `{row['base_text']}`")
    for row in rejected + abstained:
        review_lines.append(
            f"- {row['region_id']} ({row['start']:.2f}–{row['end']:.2f}s): "
            f"{row['reason']} — `{row['base_text']}`")
    (output_dir / "review-v10.md").write_text("\n".join(review_lines) + "\n", encoding="utf-8")
    return {"output": str(output_dir), **summary, "text": final_text}


def run(run_dir: Path, medical_index: Path, server_url: str, timeout: float,
        max_regions: int, dry_run: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    v9_path = run_dir / V9_RELATIVE / "final-v9.json"
    if not v9_path.is_file():
        raise FileNotFoundError(f"V9 result is missing: {v9_path}")
    v9 = load_json(v9_path)
    slots = copy.deepcopy(list(v9.get("slots") or []))
    original_slots = copy.deepcopy(slots)
    placeholders = list(v9.get("uncertainty_placeholders") or [])
    hypotheses = load_hypotheses(run_dir)
    drug_terms = build_drug_terms(medical_index)
    exact_medical_terms = build_exact_medical_terms(medical_index)
    allowed_ascii = build_allowed_ascii_terms(medical_index)
    slot_tasks = make_slot_tasks(slots)
    regions: list[dict[str, Any]] = []
    fallback_reason: str | None = None
    model_calls: list[dict[str, Any]] = []
    slot_audits: list[dict[str, Any]] = []
    slot_overlays: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []

    if dry_run:
        fallback_reason = "dry-run"
        regions = make_regions(slots, hypotheses, placeholders, max_regions=max_regions)
    else:
        try:
            if slot_tasks:
                slot_prompt, slot_option_maps = build_slot_prompt(slot_tasks)
                slot_choices, slot_call = call_local_slot_qwen(
                    server_url, slot_tasks, timeout, slot_option_maps, slot_prompt)
                model_calls.append(slot_call)
                slot_audits = [
                    validate_slot_choice(
                        task, slots[int(task["slot"])], slot_choices[task["id"]],
                        slot_option_maps, drug_terms)
                    for task in slot_tasks
                ]
                slot_review_items = make_slot_review_items(
                    slot_tasks, slot_audits, exact_medical_terms)
                if slot_review_items:
                    slot_review_choices, slot_review_call = call_conservative_reviewer(
                        server_url, slot_review_items, timeout, "slot-risk-review")
                    model_calls.append(slot_review_call)
                    apply_conservative_review(slot_audits, slot_review_choices, "task_id")
                apply_slot_audits(slots, slot_audits)
            regions = make_regions(slots, hypotheses, placeholders, max_regions=max_regions)
            if regions:
                prompt, option_maps = build_prompt(regions)
                choices, phrase_call = call_local_qwen(
                    server_url, regions, timeout, option_maps, prompt)
                phrase_call["stage"] = "verbatim-phrase-rerank"
                model_calls.append(phrase_call)
                audits = [
                    validate_choice(region, choices[region["id"]], option_maps, slots, drug_terms)
                    for region in regions
                ]
                phrase_review_items = make_phrase_review_items(regions, audits)
                if phrase_review_items:
                    phrase_review_choices, phrase_review_call = call_conservative_reviewer(
                        server_url, phrase_review_items, timeout, "phrase-risk-review")
                    model_calls.append(phrase_review_call)
                    apply_conservative_review(audits, phrase_review_choices, "region_id")
            reject_unconfirmed_single_family_slots(
                slot_audits, audits, original_slots, slots)
            slot_overlays = overlay_slot_repairs_on_phrases(
                audits, slot_audits, original_slots)
        except (OSError, TimeoutError, ValueError, KeyError, json.JSONDecodeError,
                urllib.error.URLError) as error:
            fallback_reason = f"{type(error).__name__}: {error}"
    model_call: dict[str, Any] = {
        "calls": model_calls,
        "call_count": len(model_calls),
        "latency_seconds": round(sum(float(row.get("latency_seconds") or 0.0) for row in model_calls), 3),
        "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in model_calls),
        "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in model_calls),
        "model": next((row.get("model") for row in model_calls if row.get("model")), None),
    }
    applied_any_choice = (
        any(row.get("applied") for row in slot_audits)
        or any(row.get("applied") for row in audits)
    )
    if fallback_reason:
        final_text = normalize_text(v9.get("text") or "")
        ascii_cleanup: list[dict[str, Any]] = []
        repetition_cleanup: list[dict[str, Any]] = []
        honorific_cleanup: list[dict[str, Any]] = []
        projection_audit: list[dict[str, Any]] = []
    elif not applied_any_choice:
        # Re-rendering unchanged slots is not guaranteed to reproduce V9
        # byte-for-byte.  With no accepted model choice, preserve the actual
        # V9 text and run only deterministic cleanup passes.
        final_text = normalize_text(v9.get("text") or "")
        final_text, repetition_cleanup = cleanup_repetitions(final_text)
        final_text, ascii_cleanup = cleanup_unknown_ascii(final_text, allowed_ascii)
        final_text, honorific_cleanup = cleanup_honorific_titles(final_text)
        projection_audit = []
    else:
        final_text, projection_audit = project_audited_changes_onto_v9(
            v9.get("text") or "", original_slots, placeholders, regions, audits,
            slot_tasks, slot_audits)
        final_text, repetition_cleanup = cleanup_repetitions(final_text)
        final_text, ascii_cleanup = cleanup_unknown_ascii(final_text, allowed_ascii)
        final_text, honorific_cleanup = cleanup_honorific_titles(final_text)
    return write_outputs(
        run_dir, v9, final_text, regions, audits, slot_tasks, slot_audits, slot_overlays, model_call,
        time.perf_counter() - started, fallback_reason, ascii_cleanup, repetition_cleanup,
        honorific_cleanup, projection_audit)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V10 local Qwen constrained phrase reranker for Persian medical ASR.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--medical-index", type=Path, required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:18080")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-regions", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run(
        args.run_dir.resolve(), args.medical_index.resolve(), args.server_url,
        max(5.0, args.timeout), max(1, min(8, args.max_regions)), args.dry_run)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
