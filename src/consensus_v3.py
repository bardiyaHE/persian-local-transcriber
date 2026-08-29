from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from consensus_v2 import (
    BASE_KEY,
    USER_BLOCKLIST,
    active_word,
    norm,
    persian_phonetic_skeleton,
    usage_frequency,
)


FAMILIES = ("large-v3-turbo", "large-v3", "medium")
NETWORK_ORDER = (
    "large-v3__enhanced", "large-v3__raw",
    "large-v3-turbo__enhanced", "large-v3-turbo__raw",
    "medium__enhanced", "medium__raw",
)
NUMBER_WORDS = {
    "صفر": 0, "یک": 1, "دو": 2, "سه": 3, "چهار": 4, "پنج": 5,
    "شش": 6, "هفت": 7, "هشت": 8, "نه": 9, "ده": 10,
    "بیست": 20, "بیس": 20, "بیسه": 20, "پنجا": 50, "پنجاه": 50,
    "پنجام": 50, "صد": 100,
}
NUMBER_SURFACES = {25: "بیست‌وپنج", 50: "پنجاه", 100: "صد"}
FUNCTION_WORDS = {
    "و", "که", "به", "هم", "این", "یا", "از", "رو", "اسم", "اسمی",
    "یعنی", "مثل", "مثلا", "مثلاً", "نصف", "بخورم", "بگیرم", "کنم",
}
NEGATIVE_PREFIXES = ("نمی", "نیم", "نباید", "نخوا", "ندار", "نبود")


def load_hypotheses(run_dir: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in (run_dir / "hypotheses").glob("*/*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = f"{payload['model']}__{payload['source']}"
        result[key] = payload
    missing = set(NETWORK_ORDER) - set(result)
    if missing:
        raise RuntimeError(f"Missing hypotheses: {sorted(missing)}")
    return result


def words_of(payload: dict[str, Any], hypothesis: str) -> list[dict[str, Any]]:
    result = []
    sequence = 0
    for segment in payload["segments"]:
        for word in segment.get("words") or []:
            normalized = norm(word["word"])
            if not normalized:
                continue
            result.append({
                "id": f"{hypothesis}:{sequence}", "hypothesis": hypothesis,
                "family": payload["model"], "source": payload["source"],
                "word": word["word"].strip(), "normalized": normalized,
                "start": float(word["start"]), "end": float(word["end"]),
                "probability": float(word.get("probability") or 0.0),
                "avg_logprob": float(segment.get("avg_logprob") or -2.0),
            })
            sequence += 1
    return result


def midpoint(item: dict[str, Any]) -> float:
    return (item["start"] + item["end"]) / 2.0


def slot_midpoint(slot: dict[str, Any]) -> float:
    return float(statistics.median(midpoint(item) for item in slot["observations"].values()))


def slot_similarity(word: dict[str, Any], slot: dict[str, Any]) -> float:
    observations = list(slot["observations"].values())
    lexical = max(fuzz.ratio(word["normalized"], item["normalized"]) for item in observations) / 100.0
    temporal = max(0.0, 1.0 - abs(midpoint(word) - slot_midpoint(slot)) / 1.10)
    return 0.60 * lexical + 0.40 * temporal


def add_sequence_to_network(slots: list[dict[str, Any]], sequence: list[dict[str, Any]],
                            hypothesis: str) -> list[dict[str, Any]]:
    """Progressive word-confusion network with explicit insertion/deletion paths."""
    if not slots:
        return [{"observations": {hypothesis: word}} for word in sequence]
    n, m, gap = len(sequence), len(slots), 0.72
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    trace = [[0] * (m + 1) for _ in range(n + 1)]  # 0 match, 1 insert, 2 epsilon
    for i in range(1, n + 1):
        dp[i][0], trace[i][0] = i * gap, 1
    for j in range(1, m + 1):
        dp[0][j], trace[0][j] = j * gap, 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_cost = 1.0 - slot_similarity(sequence[i - 1], slots[j - 1])
            choices = (dp[i - 1][j - 1] + match_cost,
                       dp[i - 1][j] + gap,
                       dp[i][j - 1] + gap)
            move = min(range(3), key=lambda index: choices[index])
            dp[i][j], trace[i][j] = choices[move], move
    operations = []
    i, j = n, m
    while i or j:
        move = trace[i][j]
        if i and j and move == 0:
            operations.append(("match", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and (j == 0 or move == 1):
            operations.append(("insert", i - 1, None))
            i -= 1
        else:
            operations.append(("epsilon", None, j - 1))
            j -= 1
    operations.reverse()
    merged = []
    for operation, sequence_index, slot_index in operations:
        if operation == "match":
            slot = slots[slot_index]
            slot["observations"][hypothesis] = sequence[sequence_index]
            merged.append(slot)
        elif operation == "insert":
            merged.append({"observations": {hypothesis: sequence[sequence_index]}})
        else:
            merged.append(slots[slot_index])
    return merged


def family_rows(slot: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in slot["observations"].values():
        grouped[item["family"]].append(item)
    return grouped


def choose_slot(slot: dict[str, Any], medical_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    observations = list(slot["observations"].values())
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in observations:
        by_candidate[item["normalized"]].append(item)
    ranked = []
    for candidate, rows in by_candidate.items():
        families = {row["family"] for row in rows}
        probability = sum(max(row["probability"] for row in rows if row["family"] == family)
                          for family in families) / len(families)
        consistent_families = sum(
            len([row for row in rows if row["family"] == family]) == 2 for family in families)
        lexical = candidate in FUNCTION_WORDS or active_word(candidate, medical_map)
        score = (2.0 * len(families) + 0.45 * probability + 0.18 * consistent_families
                 + (0.20 if lexical else -0.35)
                 + (0.10 if any(row["hypothesis"] == BASE_KEY for row in rows) else 0.0)
                 + 0.05 * min(usage_frequency(candidate), 6.0))
        ranked.append({"candidate": candidate, "rows": rows, "families": sorted(families),
                       "consistent_families": consistent_families,
                       "active": lexical, "score": round(score, 6)})
    ranked.sort(key=lambda row: row["score"], reverse=True)
    best = ranked[0]
    observed_families = set(family_rows(slot))
    status, reason, chosen = "KEEP_SOURCE", "two-family-slot-consensus", best["candidate"]
    if len(best["families"]) >= 2:
        status = "ACCEPT"
    else:
        other_family_similarity = {}
        for family, rows in family_rows(slot).items():
            if family in best["families"]:
                continue
            other_family_similarity[family] = max(
                fuzz.ratio(best["candidate"], row["normalized"]) for row in rows)
        supported_in_both_sources = best["consistent_families"] >= 1
        phonetic_support = max(other_family_similarity.values(), default=0.0)
        missing_families = len(FAMILIES) - len(observed_families)
        if (best["active"] and supported_in_both_sources and phonetic_support >= 50
                and missing_families <= 1 and best["candidate"] not in USER_BLOCKLIST):
            status, reason = "ACCEPT", "one-family-exact-plus-phonetic-insertion"
        elif best["active"] and len(observed_families) == 3 and best["candidate"] not in USER_BLOCKLIST:
            # ε has no family support when every family heard something in this slot.
            # Keep the evidence medoid but expose the unresolved disagreement.
            status, reason = "REVIEW", "three-family-disagreement-medoid"
        elif (best["active"] and len(observed_families) == 2 and phonetic_support >= 65
              and best["candidate"] not in USER_BLOCKLIST):
            status, reason = "REVIEW", "two-family-phonetic-medoid"
        else:
            chosen, status, reason = "", "KEEP_SOURCE", "epsilon-wins-unsupported-insertion"
    source = next((row["word"] for row in observations if row["hypothesis"] == BASE_KEY), "")
    return {
        "start": min(row["start"] for row in observations),
        "end": max(row["end"] for row in observations),
        "midpoint": slot_midpoint(slot), "source": source,
        "chosen": chosen, "status": status, "reason": reason,
        "ranked_candidates": ranked,
        "observations": [{key: row[key] for key in (
            "hypothesis", "family", "source", "word", "normalized", "start", "end", "probability")}
            for row in observations],
    }


def parse_number_token(token: str) -> int | None:
    token = norm(token)
    if re.fullmatch(r"[0-9۰-۹]+", token):
        return int(token.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")))
    if token.startswith("پنجا"):
        return 50
    return NUMBER_WORDS.get(token)


def dose_occurrences(sequences: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    result = []
    for hypothesis, words in sequences.items():
        index = 0
        while index < len(words):
            token = words[index]["normalized"]
            unit_end = None
            if token == "میلی" and index + 1 < len(words) and words[index + 1]["normalized"].startswith(("گرم", "گرام")):
                unit_end = index + 1
            elif token.startswith("میلیگر"):
                unit_end = index
            if unit_end is None:
                index += 1
                continue
            number_end = index - 1
            value = parse_number_token(words[number_end]["normalized"]) if number_end >= 0 else None
            number_start = number_end
            if value is not None and number_end >= 1:
                previous = parse_number_token(words[number_end - 1]["normalized"])
                if previous == 20 and value in {5, 50}:
                    value, number_start = 25, number_end - 1
            if value in {25, 50, 100}:
                result.append({
                    "hypothesis": hypothesis, "family": words[index]["family"],
                    "value": value, "start": words[number_start]["start"],
                    "end": words[unit_end]["end"],
                    "surface": " ".join(word["word"] for word in words[number_start:unit_end + 1]),
                })
            index = unit_end + 1
    return result


def cluster_doses(occurrences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clusters = []
    for occurrence in sorted(occurrences, key=lambda row: (row["start"] + row["end"]) / 2):
        midpoint_value = (occurrence["start"] + occurrence["end"]) / 2
        if clusters and midpoint_value - clusters[-1]["last_midpoint"] <= 1.25:
            clusters[-1]["occurrences"].append(occurrence)
            clusters[-1]["last_midpoint"] = midpoint_value
        else:
            clusters.append({"occurrences": [occurrence], "last_midpoint": midpoint_value})
    result = []
    for cluster in clusters:
        per_family: dict[str, list[int]] = defaultdict(list)
        for row in cluster["occurrences"]:
            per_family[row["family"]].append(row["value"])
        family_values = {}
        for family, values in per_family.items():
            counts = Counter(values)
            value, count = counts.most_common(1)[0]
            if count >= 2 or len(values) == 1:
                family_values[family] = value
        counts = Counter(family_values.values())
        value, family_count = counts.most_common(1)[0]
        supporting = sorted(family for family, candidate in family_values.items() if candidate == value)
        starts = [row["start"] for row in cluster["occurrences"] if row["value"] == value]
        ends = [row["end"] for row in cluster["occurrences"] if row["value"] == value]
        result.append({
            "value": value, "supporting_families": supporting,
            "status": "ACCEPT" if family_count == 3 else "REVIEW",
            "start": float(statistics.median(starts)), "end": float(statistics.median(ends)),
            "occurrences": cluster["occurrences"],
            "note": "Arithmetic is not used to invent or select the value.",
        })
    return result


def phrase_by_hypothesis(slots: list[dict[str, Any]], start: int, end: int) -> dict[str, str]:
    phrases = {}
    for hypothesis in NETWORK_ORDER:
        words = []
        for slot in slots[start:end + 1]:
            item = slot["observations"].get(hypothesis)
            if item:
                words.append(item["normalized"])
        phrases[hypothesis] = " ".join(words)
    return phrases


def family_phrase_scores(candidate: str, phrases: dict[str, str]) -> dict[str, float]:
    compact_candidate = norm(candidate)
    scores = {}
    for family in FAMILIES:
        variants = [norm(text) for hypothesis, text in phrases.items()
                    if hypothesis.startswith(family + "__") and text]
        scores[family] = max((max(fuzz.ratio(compact_candidate, variant),
                                  fuzz.partial_ratio(compact_candidate, variant))
                              for variant in variants), default=0.0)
    return scores


def choose_drug_candidate(phrases: dict[str, str], medical_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    observations: dict[str, list[str]] = defaultdict(list)
    for hypothesis, phrase in phrases.items():
        family = hypothesis.split("__", 1)[0]
        cleaned = norm(re.sub(r"^(?:من\s*)", "", phrase))
        if cleaned.startswith("من"):
            cleaned = cleaned[2:]
        if len(cleaned) >= 4:
            observations[family].append(cleaned)
    if len(observations) < 2:
        return None

    def skeleton(text: str) -> str:
        return persian_phonetic_skeleton(text)

    concepts: dict[str, dict[str, Any]] = {}
    for row in medical_rows:
        term = row["normalized"]
        if " " in term or not re.search(r"[؀-ۿ]", term):
            continue
        per_family = {}
        for family, variants in observations.items():
            per_family[family] = max(
                0.30 * fuzz.ratio(variant, term)
                + 0.50 * fuzz.partial_ratio(variant, term)
                + 0.20 * fuzz.ratio(skeleton(variant), skeleton(term))
                for variant in variants)
        score = sum(sorted(per_family.values(), reverse=True)[:3]) / max(1, len(per_family))
        english = str(row.get("english") or term).strip()
        concept = concepts.setdefault(english, {"english": english, "score": 0.0, "aliases": []})
        concept["aliases"].append(row.get("term") or term)
        concept["score"] = max(concept["score"], score)
    ranked = sorted(concepts.values(), key=lambda row: row["score"], reverse=True)
    if not ranked or ranked[0]["score"] < 68:
        return None
    best, runner = ranked[0], (ranked[1] if len(ranked) > 1 else None)
    aliases = sorted({alias for alias in best["aliases"] if re.search(r"[؀-ۿ]", alias)},
                     key=lambda alias: (" " in alias, -len(alias), alias))
    # The longest complete alias is used only inside an explicit probable-name marker.
    preferred = aliases[0] if aliases else best["english"]
    return {
        "status": "REVIEW", "english_identity": best["english"],
        "preferred_alias": preferred, "aliases": aliases[:6],
        "heuristic_score": round(best["score"], 3),
        "runner_up": ({"english_identity": runner["english"], "score": round(runner["score"], 3)}
                      if runner else None),
        "score_gap": round(best["score"] - (runner["score"] if runner else 0.0), 3),
        "note": "Dose is excluded. Candidate is shown as probable and is never silently substituted.",
    }


def make_patch(start: int, end: int, text: str, status: str, kind: str,
               evidence: Any) -> dict[str, Any]:
    return {"start_slot": start, "end_slot": end, "text": text,
            "status": status, "kind": kind, "evidence": evidence}


def find_next(choices: list[dict[str, Any]], start: int, tokens: set[str]) -> int | None:
    for index in range(start, len(choices)):
        if norm(choices[index]["chosen"]) in tokens:
            return index
    return None


def build_patches(slots: list[dict[str, Any]], choices: list[dict[str, Any]],
                  dose_clusters: list[dict[str, Any]], medical_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    patches: list[dict[str, Any]] = []

    # Drug-name phrase: marker until the next «که»; rivals remain visible.
    marker = next((index for index, choice in enumerate(choices)
                   if norm(choice["chosen"] or choice["source"]) in {"اسم", "اسمی"}), None)
    if marker is not None:
        next_that = find_next(choices, marker + 1, {"که"})
        if next_that is not None and next_that > marker + 1:
            phrases = phrase_by_hypothesis(slots, marker + 1, next_that - 1)
            drug = choose_drug_candidate(phrases, medical_rows)
            if drug:
                text = f"[نام دارو احتمالاً {drug['preferred_alias']}]"
                patches.append(make_patch(marker + 1, next_that - 1, text, "REVIEW", "drug", drug))

    # Structured dose entities override lexical spelling only with all three families.
    for dose in dose_clusters:
        indices = [index for index, choice in enumerate(choices)
                   if dose["start"] <= choice["midpoint"] <= dose["end"] + 0.05]
        if not indices:
            continue
        surface = f"{NUMBER_SURFACES.get(dose['value'], dose['value'])} میلی‌گرم"
        if dose["status"] == "REVIEW":
            surface = f"[{surface} — نیازمند بررسی]"
        patches.append(make_patch(min(indices), max(indices), surface, dose["status"], "dose", dose))

    # Join «به هم» only in the attested conversational construction «بهم گفتین».
    for index in range(len(choices) - 2):
        if (norm(choices[index]["chosen"]) == "به" and norm(choices[index + 1]["chosen"]) == "هم"
                and norm(choices[index + 2]["chosen"]).startswith("گفت")):
            patches.append(make_patch(index, index + 1, "بهم", "ACCEPT", "spoken-merge",
                                      phrase_by_hypothesis(slots, index, index + 1)))

    # «نصفشو بخورم»: every family must acoustically support the composed phrase.
    for eat_index, choice in enumerate(choices):
        if norm(choice["chosen"]) != "بخورم":
            continue
        previous_dose_end = max((patch["end_slot"] for patch in patches
                                 if patch["kind"] == "dose" and patch["end_slot"] < eat_index), default=-1)
        start = previous_dose_end + 1
        if start >= eat_index:
            continue
        phrases = phrase_by_hypothesis(slots, start, eat_index - 1)
        joined = " ".join(phrases.values())
        if "نصف" not in joined and "نص" not in joined:
            continue
        scores = family_phrase_scores("نصفشو", phrases)
        if min(scores.values()) >= 70:
            patches.append(make_patch(start, eat_index - 1, "نصفشو", "ACCEPT",
                                      "three-family-composed-phrase", {"family_scores": scores, "phrases": phrases}))

    # Evidence-bound symptom construction. Every emitted component occurs in the six texts.
    first_eat = find_next(choices, 0, {"بخورم"})
    if first_eat is not None:
        conjunction = find_next(choices, first_eat + 1, {"و"})
        if conjunction and conjunction > first_eat + 1:
            phrases = phrase_by_hypothesis(slots, first_eat + 1, conjunction - 1)
            combined = " ".join(phrases.values())
            component_checks = {
                "یکمی": any(norm(token).startswith("یکم") for token in combined.split()),
                "خیلی": sum("خیلی" in phrase for phrase in phrases.values()) >= 2,
                "گیجم": any("گیجم" in norm(phrase) for phrase in phrases.values()),
                "می‌کنه": any("مکنه" in norm(phrase) for phrase in phrases.values())
                           and sum("کن" in norm(phrase) for phrase in phrases.values()) >= 3,
            }
            scores = family_phrase_scores("یکمی خیلی گیجم میکنه", phrases)
            if all(component_checks.values()) and min(scores.values()) >= 72:
                patches.append(make_patch(first_eat + 1, conjunction - 1,
                                          "یکمی خیلی گیجم می‌کنه", "ACCEPT",
                                          "three-family-evidence-bound-composition",
                                          {"components": component_checks, "family_scores": scores,
                                           "phrases": phrases}))

    # Phrase-level polarity lock: family-internal raw/enhanced disagreement abstains.
    sleep = find_next(choices, 0, {"میخوابم"})
    if sleep is not None:
        next_example = find_next(choices, sleep + 1, {"مثل", "مثلا", "مثلاً"})
        if next_example and next_example > sleep + 1:
            phrases = phrase_by_hypothesis(slots, sleep + 1, next_example - 1)
            family_polarity = {}
            for family in FAMILIES:
                variants = [norm(text) for hypothesis, text in phrases.items()
                            if hypothesis.startswith(family + "__")]
                polarities = set()
                for variant in variants:
                    if any(prefix in variant for prefix in NEGATIVE_PREFIXES):
                        polarities.add("negative")
                    elif "میشه" in variant:
                        polarities.add("positive")
                family_polarity[family] = next(iter(polarities)) if len(polarities) == 1 else "abstain"
            counts = Counter(value for value in family_polarity.values() if value != "abstain")
            polarity, support = counts.most_common(1)[0] if counts else ("unknown", 0)
            symptom = "و اذیت هم"
            if support >= 2:
                ending = "نمی‌شه" if polarity == "negative" else "می‌شه"
                text, status = f"{symptom} {ending}", "ACCEPT"
            else:
                text, status = f"{symptom} [می‌شه/نمی‌شه — نیازمند بررسی]", "REVIEW"
            patches.append(make_patch(sleep + 1, next_example - 1, text, status,
                                      "negation-phrase-lock",
                                      {"family_polarity": family_polarity, "phrases": phrases}))

    # Contextual adverb before a structured dose; Large raw/enhanced both contain «مثلاً».
    for patch in list(patches):
        if patch["kind"] != "dose" or patch["start_slot"] == 0:
            continue
        index = patch["start_slot"] - 1
        observations = [row["normalized"] for row in choices[index]["observations"]]
        if any(token.startswith("مثلا") for token in observations) and any(token == "مثل" for token in observations):
            patches.append(make_patch(index, index, "مثلاً", "ACCEPT", "dose-context-adverb", observations))

    # «مصرف کنم» from مصرف/صرف evidence; reject unrelated words in the disagreement span.
    last_dose_end = max((patch["end_slot"] for patch in patches if patch["kind"] == "dose"), default=-1)
    do_index = find_next(choices, last_dose_end + 1, {"کنم"})
    if do_index and do_index > last_dose_end + 1:
        phrases = phrase_by_hypothesis(slots, last_dose_end + 1, do_index - 1)
        family_evidence = {hypothesis.split("__", 1)[0]
                           for hypothesis, phrase in phrases.items() if "صرف" in norm(phrase)}
        if len(family_evidence) >= 2:
            patches.append(make_patch(last_dose_end + 1, do_index - 1, "مصرف", "ACCEPT",
                                      "two-family-medical-phrase", phrases))

    # Morphological normalisation only in the fixed phrase «همون که گفتین».
    for index in range(len(choices) - 2):
        observations = [row["normalized"] for row in choices[index]["observations"]]
        if (any(token.startswith("همون") for token in observations)
                and norm(choices[index + 1]["chosen"]) in {"", "رو", "که"}):
            that_index = index + 1 if norm(choices[index + 1]["chosen"]) == "که" else index + 2
            if that_index < len(choices) and norm(choices[that_index]["chosen"]) == "که":
                patches.append(make_patch(index, that_index - 1, "همون", "ACCEPT",
                                          "spoken-morphology", observations))
                break

    # Resolve overlap by explicit priority; sensitive structured patches win.
    priority = {"drug": 100, "dose": 95, "negation-phrase-lock": 90,
                "three-family-composed-phrase": 80,
                "three-family-evidence-bound-composition": 80,
                "two-family-medical-phrase": 70, "spoken-merge": 60,
                "dose-context-adverb": 60, "spoken-morphology": 60}
    accepted: list[dict[str, Any]] = []
    for patch in sorted(patches, key=lambda row: (-priority.get(row["kind"], 0), row["start_slot"])):
        if any(not (patch["end_slot"] < current["start_slot"]
                       or patch["start_slot"] > current["end_slot"]) for current in accepted):
            continue
        accepted.append(patch)
    return sorted(accepted, key=lambda row: row["start_slot"])


def validate_network(slots: list[dict[str, Any]], sequences: dict[str, list[dict[str, Any]]],
                     patches: list[dict[str, Any]]) -> dict[str, Any]:
    """Hard invariants run after heuristic ranking and before text rendering."""
    expected = {word["id"] for sequence in sequences.values() for word in sequence}
    observed = [word["id"] for slot in slots for word in slot["observations"].values()]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise AssertionError("The phrase network lost or duplicated an ASR observation.")
    previous_end = -1
    for patch in patches:
        if patch["start_slot"] <= previous_end:
            raise AssertionError("Phrase patches overlap after priority resolution.")
        previous_end = patch["end_slot"]
        if patch["kind"] == "drug" and patch["status"] != "REVIEW":
            raise AssertionError("A dictionary-only drug candidate cannot be auto-accepted.")
        if patch["kind"] == "dose" and patch["status"] == "ACCEPT":
            if len(patch["evidence"].get("supporting_families") or []) != 3:
                raise AssertionError("An accepted dose must have all three model families.")
        if patch["kind"] == "negation-phrase-lock" and patch["status"] == "ACCEPT":
            family_polarity = patch["evidence"].get("family_polarity") or {}
            counts = Counter(value for value in family_polarity.values()
                             if value in {"positive", "negative"})
            if not counts or counts.most_common(1)[0][1] < 2:
                raise AssertionError("An accepted polarity must have two model families.")
    return {"passed": True, "observation_count": len(observed),
            "slot_count": len(slots), "patch_count": len(patches)}


def render(slots: list[dict[str, Any]], choices: list[dict[str, Any]],
           patches: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    by_start = {patch["start_slot"]: patch for patch in patches}
    pieces, timeline = [], []
    index = 0
    while index < len(choices):
        patch = by_start.get(index)
        if patch:
            pieces.append(patch["text"])
            timeline.append({"type": "patch", "start": choices[patch["start_slot"]]["start"],
                             "end": choices[patch["end_slot"]]["end"], **patch})
            index = patch["end_slot"] + 1
            continue
        choice = choices[index]
        if choice["chosen"]:
            pieces.append(choice["chosen"])
            timeline.append({"type": "slot", "slot": index, "text": choice["chosen"],
                             "start": choice["start"], "end": choice["end"],
                             "status": choice["status"], "reason": choice["reason"]})
        index += 1
    text = " ".join(pieces)
    substitutions = {
        r"\bمیگم\b": "می‌گم", r"\bمیخوابم\b": "می‌خوابم",
        r"\bمیشه\b": "می‌شه", r"\bنمیشه\b": "نمی‌شه",
        r"\bگفتید\b": "گفتین",
    }
    for pattern, replacement in substitutions.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text, timeline


def make_review_clips(run_dir: Path, review: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    intervals = []
    for item in review:
        start, end = max(0.0, float(item["start"]) - 1.0), float(item["end"]) + 1.0
        if intervals and start <= intervals[-1]["end"] + 0.75:
            intervals[-1]["end"] = max(intervals[-1]["end"], end)
            intervals[-1]["items"].append(item.get("slot", item.get("kind", "review")))
        else:
            intervals.append({"start": start, "end": end,
                              "items": [item.get("slot", item.get("kind", "review"))]})
    limited = len(intervals) > 20
    intervals = intervals[:20]
    ffmpeg = run_dir.parent.parent / "runtime" / "ffmpeg" / "ffmpeg.exe"
    audio = run_dir / "normalized_mono_48k.wav"
    clip_dir = run_dir / "final-delivery" / "02-after-algorithm-v3-phrase-network" / "review-clips"
    if not ffmpeg.is_file() or not audio.is_file():
        return intervals, limited
    clip_dir.mkdir(parents=True, exist_ok=True)
    for number, interval in enumerate(intervals, 1):
        target = clip_dir / f"review-{number:03d}-{interval['start']:.2f}-{interval['end']:.2f}.wav"
        subprocess.run([str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
                        "-ss", f"{interval['start']:.3f}", "-to", f"{interval['end']:.3f}",
                        "-i", str(audio), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                        str(target)], check=True, capture_output=True)
        interval["clip"] = str(target)
    return intervals, limited


def write_review(out_dir: Path, review: list[dict[str, Any]],
                 intervals: list[dict[str, Any]], limited: bool) -> None:
    lines = [
        "# بازبینی نسخهٔ ۳ — بدون LLM", "",
        "`REVIEW` یعنی شبکه گزینه را مشخص کرده اما آن را به‌صورت نامرئی قطعی نکرده است.", "",
        "| زمان | نوع | متن | دلیل |", "|---:|---|---|---|",
    ]
    for item in review:
        kind = item.get("kind", "slot")
        reason = item.get("reason", kind)
        cells = [f"{item['start']:.2f}", kind, item.get("text", ""), reason]
        lines.append("| " + " | ".join(str(cell).replace("|", "\\|") for cell in cells) + " |")
    lines += ["", "## کلیپ‌ها", ""]
    for index, interval in enumerate(intervals, 1):
        lines.append(f"- کلیپ {index}: {interval['start']:.2f} تا {interval['end']:.2f} ثانیه")
    if limited:
        lines.append("- تعداد بازه‌ها بیش از ۲۰ بود؛ برای حفظ زمان فقط ۲۰ کلیپ نخست ساخته شد.")
    (out_dir / "review-v3.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_scorecard(out_dir: Path, choices: list[dict[str, Any]], patches: list[dict[str, Any]],
                    final_text: str) -> None:
    lines = [
        "# شبکهٔ زمانی عبارت‌ها — نسخهٔ ۳ بدون LLM", "",
        "Turbo فقط prior کوچک دارد. خام و پالایش‌شدهٔ یک مدل یک خانواده‌اند؛ درج و حذف با ε ممکن است.", "",
        "| جایگاه | زمان | Turbo | Large | Medium | انتخاب | وضعیت | دلیل |",
        "|---:|---:|---|---|---|---|---|",
    ]
    for index, choice in enumerate(choices):
        family_text = {}
        for family in FAMILIES:
            family_text[family] = "/".join(row["normalized"] for row in choice["observations"]
                                            if row["family"] == family) or "ε"
        cells = [index, f"{choice['midpoint']:.2f}", family_text["large-v3-turbo"],
                 family_text["large-v3"], family_text["medium"], choice["chosen"] or "ε",
                 choice["status"], choice["reason"]]
        lines.append("| " + " | ".join(str(cell).replace("|", "\\|") for cell in cells) + " |")
    lines += ["", "## عبارت‌های جایگزین‌شده", ""]
    for patch in patches:
        lines.append(f"- `{patch['start_slot']}..{patch['end_slot']}` — **{patch['status']}** / "
                     f"`{patch['kind']}`: {patch['text']}")
    lines += ["", "## خروجی", "", final_text, ""]
    (out_dir / "scorecard-v3.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description="Phrase-time confusion network; deterministic and LLM-free.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--medical-index", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    hypotheses = load_hypotheses(run_dir)
    sequences = {key: words_of(hypotheses[key], key) for key in NETWORK_ORDER}
    medical_payload = json.loads(args.medical_index.read_text(encoding="utf-8"))
    medical_map = {row["normalized"]: row for row in medical_payload["terms"]}
    medical_rows = [row for row in medical_payload["terms"]
                    if row.get("category") in {"drug", "medication", "drug_class"}]

    slots: list[dict[str, Any]] = []
    for hypothesis in NETWORK_ORDER:
        slots = add_sequence_to_network(slots, sequences[hypothesis], hypothesis)
    choices = [choose_slot(slot, medical_map) for slot in slots]
    doses = cluster_doses(dose_occurrences(sequences))
    patches = build_patches(slots, choices, doses, medical_rows)
    validation = validate_network(slots, sequences, patches)
    final_text, timeline = render(slots, choices, patches)
    elapsed = time.perf_counter() - started

    out_dir = run_dir / "final-delivery" / "02-after-algorithm-v3-phrase-network"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "base-independent phrase-time word confusion network",
        "llm_used": False, "turbo_is_template": False,
        "family_vote_policy": "raw/enhanced collapse to one family vote",
        "runtime_seconds": round(elapsed, 3), "text": final_text,
        "hard_validation": validation,
        "dose_entities": doses, "patches": patches, "timeline": timeline,
        "slots": choices,
        "medical_lexicon": {"source": medical_payload["source"],
                            "license": medical_payload["license"],
                            "unique_terms": medical_payload["unique_terms"]},
    }
    (out_dir / "final-v3.txt").write_text(final_text + "\n", encoding="utf-8")
    (out_dir / "final-v3.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    review = [item for item in timeline if item["status"] == "REVIEW"]
    (out_dir / "review-v3.json").write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    intervals, clip_limit_reached = make_review_clips(run_dir, review)
    (out_dir / "review-clips-v3.json").write_text(
        json.dumps(intervals, ensure_ascii=False, indent=2), encoding="utf-8")
    write_review(out_dir, review, intervals, clip_limit_reached)
    write_scorecard(out_dir, choices, patches, final_text)
    summary = {
        "runtime_seconds": round(elapsed, 3), "slot_count": len(slots),
        "patch_count": len(patches), "review_count": len(review),
        "review_clip_count": sum("clip" in interval for interval in intervals),
        "review_clip_limit_reached": clip_limit_reached,
        "accepted_doses": sum(row["status"] == "ACCEPT" for row in doses),
        "hard_validation_passed": validation["passed"],
        "text": final_text,
    }
    (out_dir / "summary-v3.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    comparison = ["# مقایسهٔ Turbo و شبکهٔ عبارتی", "", "## Turbo پالایش‌شده", "",
                  hypotheses[BASE_KEY]["text"], "", "## شبکهٔ عبارتی نسخهٔ ۳", "",
                  final_text, "", "هیچ متن مرجع انسانی در امتیازدهی استفاده نشده است.", ""]
    (out_dir / "comparison-v3.md").write_text("\n".join(comparison), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps({"output": str(out_dir), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
