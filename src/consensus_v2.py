from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import mnk_persian_words as persian_words
from rapidfuzz import fuzz, process
from wordfreq import zipf_frequency


MODELS = ["medium", "large-v3-turbo", "large-v3"]
SOURCES = ["raw", "enhanced"]
BASE_KEY = "large-v3-turbo__enhanced"
MODEL_PRIOR = {"medium": 0.00, "large-v3-turbo": 0.10, "large-v3": 0.08}
UNITS = {"میلیگرم", "میکروگرم", "گرم", "کیلوگرم", "میلیلیتر", "لیتر", "واحد", "درصد", "دوز"}
NUMBERS = {"صفر", "یک", "دو", "سه", "چهار", "پنج", "شش", "هفت", "هشت", "نه", "ده", "یازده", "دوازده", "سی", "صد"}
NEGATIONS = {"نه", "نیست", "نبود", "نشد", "ندارد", "نداریم", "نخورید", "نکنید", "منفی"}
DRUG_CONTEXT = {"دارو", "داروی", "قرص", "کپسول", "شربت", "آمپول", "دوز", "مصرف"}
MIN_ACTIVE_ZIPF = 3.50


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).replace("ي", "ی").replace("ك", "ک")
    text = text.replace("ۀ", "ه").replace("ة", "ه")
    text = re.sub(r"[\u064b-\u065f\u0670]", "", text)
    return re.sub(r"[^\w\u0600-\u06ff]+", "", text).casefold()


def render_like(original: str, canonical: str) -> str:
    match = re.match(r"^(\s*).*?([^\w\u0600-\u06ff]*)$", original, flags=re.DOTALL)
    if not match:
        return " " + canonical
    return match.group(1) + canonical + match.group(2)


def load_overlay(name: str) -> set[str]:
    path = Path(__file__).resolve().parent.parent / "offline-lexicon" / name
    if not path.is_file():
        return set()
    return {norm(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")}


USER_BLOCKLIST = load_overlay("fa_user_blocklist.txt")
MODERN_SPOKEN = load_overlay("fa_modern_spoken_allowlist.txt")


def usage_frequency(token: str) -> float:
    return float(zipf_frequency(token, "fa"))


@lru_cache(maxsize=100_000)
def general_contains(token: str) -> bool:
    return bool(persian_words.contains_word(token))


def productive_stems(token: str) -> set[str]:
    """Generate conservative Persian stems before declaring an inflected form rare."""
    stems = {token}
    prefixes = ("نمی", "می")
    suffixes = ("هایشون", "هایتون", "هایمون", "هاشون", "هاتون", "هامون",
                "شون", "تون", "مون", "های", "هاش", "هات", "هام", "ها",
                "اش", "ات", "ام", "شو", "تو", "مو", "ش", "ت", "م", "ید", "یم", "ند", "ی")
    for prefix in prefixes:
        if token.startswith(prefix) and len(token) - len(prefix) >= 3:
            stems.add(token[len(prefix):])
    for current in list(stems):
        for suffix in suffixes:
            if current.endswith(suffix) and len(current) - len(suffix) >= 3:
                stems.add(current[:-len(suffix)])
    return stems


def morphology_supported(token: str) -> bool:
    for stem in productive_stems(token):
        if stem == token:
            continue
        if usage_frequency(stem) >= 4.0 and general_contains(stem):
            return True
    return False


def persian_phonetic_skeleton(token: str) -> str:
    """Conservative consonant skeleton for accent-driven Persian vowel variation."""
    normalized = norm(token).replace("آ", "ا")
    return re.sub(r"[اوی]", "", normalized)


def active_word(token: str, medical_map: dict[str, dict[str, Any]]) -> bool:
    if token in USER_BLOCKLIST:
        return False
    if token in medical_map or token in MODERN_SPOKEN:
        return True
    return usage_frequency(token) >= MIN_ACTIVE_ZIPF or morphology_supported(token)


def words_of(payload: dict[str, Any]) -> list[dict[str, Any]]:
    words = []
    for segment in payload["segments"]:
        for word in segment.get("words") or []:
            n = norm(word["word"])
            if not n:
                continue
            words.append({"word": word["word"], "normalized": n, "start": float(word["start"]),
                          "end": float(word["end"]), "probability": float(word.get("probability") or 0),
                          "avg_logprob": float(segment.get("avg_logprob") or -2),
                          "model": payload["model"], "source": payload["source"]})
    return words


def midpoint(word: dict[str, Any]) -> float:
    return (word["start"] + word["end"]) / 2


def align_monotonic(base: list[dict[str, Any]], other: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
    """Global one-to-one alignment using lexical and timestamp evidence."""
    n, m = len(base), len(other)
    gap = 0.72
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    trace = [[0] * (m + 1) for _ in range(n + 1)]  # 0 diagonal, 1 up, 2 left
    for i in range(1, n + 1):
        dp[i][0] = i * gap
        trace[i][0] = 1
    for j in range(1, m + 1):
        dp[0][j] = j * gap
        trace[0][j] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            lexical = 1.0 - fuzz.ratio(base[i - 1]["normalized"], other[j - 1]["normalized"]) / 100.0
            time_distance = min(abs(midpoint(base[i - 1]) - midpoint(other[j - 1])) / 1.25, 1.0)
            match_cost = 0.72 * lexical + 0.62 * time_distance
            choices = (dp[i - 1][j - 1] + match_cost, dp[i - 1][j] + gap, dp[i][j - 1] + gap)
            best = min(range(3), key=lambda k: choices[k])
            dp[i][j] = choices[best]
            trace[i][j] = best
    result: list[dict[str, Any] | None] = [None] * n
    i, j = n, m
    while i > 0 or j > 0:
        move = trace[i][j]
        if i > 0 and j > 0 and move == 0:
            if abs(midpoint(base[i - 1]) - midpoint(other[j - 1])) <= 1.25:
                result[i - 1] = other[j - 1]
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or move == 1):
            i -= 1
        else:
            j -= 1
    return result


def protected(base: list[dict[str, Any]], index: int, medical_map: dict[str, dict[str, Any]]) -> tuple[bool, str]:
    token = base[index]["normalized"]
    nearby = {w["normalized"] for w in base[max(0, index - 2):index + 3]}
    if re.search(r"\d|[۰-۹]", token) or token in NUMBERS:
        return True, "number"
    if token in UNITS:
        return True, "dose-or-unit"
    if token in NEGATIONS or token.startswith(("نمی", "نخوا", "نباید", "ندار", "نبود")):
        return True, "negation"
    med = medical_map.get(token)
    if (nearby & DRUG_CONTEXT) or (med and med.get("category") == "drug"):
        return True, "drug-or-drug-context"
    return False, ""


def evidence_score(vote: dict[str, Any], medical_map: dict[str, dict[str, Any]]) -> float:
    score = vote["probability"] + MODEL_PRIOR[vote["model"]]
    score += max(-0.15, min(0.10, (vote["avg_logprob"] + 1.0) * 0.12))
    if vote["source"] == "enhanced":
        score += 0.025
    if vote["normalized"] in medical_map:
        score += 0.18
    elif general_contains(vote["normalized"]):
        score += 0.07
    return score


def lexicon_valid(token: str, medical_map: dict[str, dict[str, Any]]) -> bool:
    return active_word(token, medical_map) and (token in medical_map or token in MODERN_SPOKEN
                                               or general_contains(token))


def score_six_candidates(raw_evidence: list[dict[str, Any]], base_norm: str,
                         previous: list[str], medical_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Transparent scoring over all six hypotheses; family votes are not collapsed here."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in raw_evidence:
        grouped[item["normalized"]].append(item)
    previous_norms = [norm(item) for item in previous[-2:]]
    scored = []
    for candidate, items in grouped.items():
        exact_votes = len(items)
        family_count = len({item["model"] for item in items})
        confidence = sum(item["probability"] for item in items) / max(1, exact_votes)
        general_bonus = 0.15 if general_contains(candidate) else 0.0
        medical_bonus = 0.35 if candidate in medical_map else 0.0
        frequency = usage_frequency(candidate)
        modern_bonus = 0.65 if candidate in MODERN_SPOKEN else 0.0
        frequency_bonus = max(0.0, min(1.20, (frequency - MIN_ACTIVE_ZIPF) * 0.45))
        rare_penalty = -3.0 if (candidate in USER_BLOCKLIST or not active_word(candidate, medical_map)) else 0.0
        base_bonus = 0.20 if candidate == base_norm else 0.0
        context_bonus = 0.0
        canonical = candidate
        context_rule = ""
        # A word dictionary validates both «خیل» and «بخیر». This collocation rule
        # disambiguates the conventional greeting without using an LLM.
        if previous_norms[-2:] and previous_norms[-1] == "به" and candidate == "بخیر":
            canonical = "خیر"
            context_bonus = 4.0 if len(previous_norms) < 2 or previous_norms[-2] not in {"وقت", "وقتتون", "وقتیتون"} else 4.5
            context_rule = "common-greeting: وقتتون/وقتیتون به خیر"
        components = {
            "six_vote_count": exact_votes * 1.0,
            "model_family_count": family_count * 0.35,
            "mean_confidence": confidence * 0.50,
            "general_lexicon": general_bonus,
            "medical_lexicon": medical_bonus,
            "modern_spoken": modern_bonus,
            "usage_frequency": frequency_bonus,
            "rare_or_blocked": rare_penalty,
            "turbo_base": base_bonus,
            "phrase_context": context_bonus,
        }
        scored.append({
            "candidate": candidate,
            "canonical": canonical,
            "votes": exact_votes,
            "families": family_count,
            "mean_confidence": round(confidence, 6),
            "general_lexicon_match": bool(general_bonus),
            "medical_lexicon_match": bool(medical_bonus),
            "modern_spoken_match": bool(modern_bonus),
            "zipf_frequency_fa": round(frequency, 4),
            "active_vocabulary": active_word(candidate, medical_map),
            "context_rule": context_rule,
            "components": {key: round(value, 6) for key, value in components.items()},
            "total_score": round(sum(components.values()), 6),
        })
    return sorted(scored, key=lambda row: (row["total_score"], row["votes"]), reverse=True)


def family_vote(items: list[dict[str, Any]], medical_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_word: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_word[item["normalized"]].append(item)
    ranked = []
    for normalized, variants in by_word.items():
        best = max(variants, key=lambda v: evidence_score(v, medical_map)).copy()
        best["family_internal_support"] = len(variants)
        best["score"] = evidence_score(best, medical_map) + (0.08 if len(variants) == 2 else 0)
        ranked.append(best)
    return max(ranked, key=lambda v: v["score"])


def general_choices(cache: dict[int, list[str]], length: int, radius: int) -> list[str]:
    """Build the general dictionary length index once instead of querying SQLite per token."""
    if not cache:
        for word in persian_words.iter_words(mode="clean", min_length=2, max_length=40,
                                              order="alpha"):
            cache.setdefault(len(word), []).append(word)
    result = []
    for candidate_length in range(max(2, length - radius), length + radius + 1):
        result.extend(cache.get(candidate_length, []))
    return result


def choose_medical_canonical(votes: list[dict[str, Any]], medical_choices: list[str],
                             medical_map: dict[str, dict[str, Any]],
                             medical_skeletons: dict[str, list[str]]) -> tuple[str | None, list[dict[str, Any]]]:
    matches = []
    for vote in votes:
        observed = vote["normalized"]
        scored = {term: float(score) for term, score, _ in process.extract(
            observed, medical_choices, scorer=fuzz.ratio, score_cutoff=88, limit=5)}
        # Add the two conservative phonetic exceptions without invoking a Python
        # scorer for every one of the tens of thousands of medical terms.
        if observed.startswith("ا") and len(observed) > 1:
            ayn_variant = "ع" + observed[1:]
            if ayn_variant in medical_map:
                scored[ayn_variant] = max(scored.get(ayn_variant, 0.0), 96.0)
        skeleton = persian_phonetic_skeleton(observed)
        if len(skeleton) >= 5:
            for term in medical_skeletons.get(skeleton, []):
                if abs(len(observed) - len(term)) <= 2:
                    scored[term] = max(scored.get(term, 0.0), 96.0)
        candidates = sorted(scored.items(), key=lambda item: item[1], reverse=True)[:5]
        if candidates:
            match = candidates[0]
            close = [item for item in candidates if match[1] >= 95 and match[1] - item[1] < 2]
            if len(close) > 1:
                english_ids = {str(medical_map[item[0]].get("english") or "").strip().casefold()
                               for item in close}
                # Alternate Persian spellings with the same English identity are aliases,
                # not competing medicines. Prefer the more frequent contemporary spelling.
                if len(english_ids) == 1 and "" not in english_ids:
                    match = max(close, key=lambda item: usage_frequency(item[0]))
                    ambiguous = False
                else:
                    ambiguous = True
            else:
                ambiguous = False
            runner_up = next((item for item in candidates if item[0] != match[0]), None)
            if not ambiguous:
                matches.append({"family": vote["model"], "candidate": vote["normalized"],
                                "term": match[0], "similarity": match[1],
                                "runner_up": ({"term": runner_up[0], "similarity": runner_up[1]}
                                              if runner_up else None)})
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for match in matches:
        grouped[match["term"]].append(match)
    if not grouped:
        return None, matches
    term, support = max(grouped.items(), key=lambda item: (len({m["family"] for m in item[1]}), sum(m["similarity"] for m in item[1])))
    families = {m["family"] for m in support}
    category = medical_map[term].get("category")
    threshold = 95 if category in {"drug", "medication", "drug_class"} else 88
    if len(families) >= 2 and min(m["similarity"] for m in support) >= threshold:
        return term, support
    return None, support


def choose_general_canonical(votes: list[dict[str, Any]], cache: dict[int, list[str]],
                             medical_map: dict[str, dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    def choices_for(length: int) -> list[str]:
        return general_choices(cache, length, radius=1)

    canonicals = {v["normalized"] for v in votes
                  if general_contains(v["normalized"]) and active_word(v["normalized"], medical_map)}
    for vote in votes:
        if vote["normalized"] in canonicals:
            continue
        match = process.extractOne(vote["normalized"], choices_for(len(vote["normalized"])),
                                   scorer=fuzz.ratio, score_cutoff=82)
        if match and active_word(match[0], medical_map):
            canonicals.add(match[0])
    ranked = []
    for canonical in canonicals:
        support = [{"family": v["model"], "candidate": v["normalized"],
                    "term": canonical, "similarity": fuzz.ratio(v["normalized"], canonical),
                    "exact": v["normalized"] == canonical,
                    "evidence_score": v.get("score", 0.0)}
                   for v in votes if fuzz.ratio(v["normalized"], canonical) >= 80]
        families = {s["family"] for s in support}
        if len(families) >= 2:
            exact = [s for s in support if s["exact"]]
            ranked.append((len(families), len(exact), sum(s["evidence_score"] for s in exact),
                           sum(s["similarity"] for s in support), canonical, support))
    if not ranked:
        return None, []
    _, _, _, _, canonical, support = max(ranked)
    return canonical, support


def choose_turbo_dictionary_fallback(base_norm: str, raw_evidence: list[dict[str, Any]],
                                     cache: dict[int, list[str]],
                                     medical_map: dict[str, dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    """Find a valid replacement when voting failed and the Turbo base is not a real active word."""
    if len(base_norm) < 3:
        return None, []
    evidence_counts: dict[str, int] = defaultdict(int)
    for item in raw_evidence:
        evidence_counts[item["normalized"]] += 1
    pool = {item["normalized"] for item in raw_evidence
            if (lexicon_valid(item["normalized"], medical_map)
                or (evidence_counts[item["normalized"]] >= 2
                    and usage_frequency(item["normalized"]) >= 3.0
                    and item["normalized"] not in USER_BLOCKLIST))}
    nearby = process.extract(base_norm, general_choices(cache, len(base_norm), radius=2), scorer=fuzz.ratio,
                             score_cutoff=60, limit=60)
    pool.update(word for word, _, _ in nearby if active_word(word, medical_map))
    ranked = []
    evidence_tokens = [item["normalized"] for item in raw_evidence]
    for candidate in pool:
        base_similarity = fuzz.ratio(base_norm, candidate)
        evidence_similarity = (sum(fuzz.ratio(token, candidate) for token in evidence_tokens)
                               / max(1, len(evidence_tokens)))
        exact_votes = sum(token == candidate for token in evidence_tokens)
        suffix_bonus = 0.75 if len(base_norm) >= 2 and candidate.endswith(base_norm[-2:]) else 0.0
        frequency = usage_frequency(candidate)
        score = (1.5 * base_similarity / 100 + 3.0 * evidence_similarity / 100
                 + 0.35 * exact_votes + suffix_bonus + 0.25 * min(frequency, 6.0) / 6.0)
        ranked.append({"candidate": candidate, "score": round(score, 6),
                       "base_similarity": round(base_similarity, 3),
                       "six_evidence_similarity": round(evidence_similarity, 3),
                       "exact_votes": exact_votes, "zipf_frequency_fa": round(frequency, 3),
                       "suffix_agreement": bool(suffix_bonus)})
    ranked.sort(key=lambda row: (row["score"], row["exact_votes"]), reverse=True)
    if not ranked:
        return None, []
    best = ranked[0]
    runner_up = ranked[1]["score"] if len(ranked) > 1 else 0.0
    reliable = (best["base_similarity"] >= 60 and best["six_evidence_similarity"] >= 68
                and (best["score"] - runner_up >= 0.12 or best["exact_votes"] >= 2))
    return (best["candidate"] if reliable else None), ranked[:10]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--medical-index", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    medical_payload = json.loads(args.medical_index.read_text(encoding="utf-8"))
    medical_map = {r["normalized"]: r for r in medical_payload["terms"]}
    medical_single = [key for key in medical_map if " " not in key]
    medical_skeletons: dict[str, list[str]] = defaultdict(list)
    for key in medical_single:
        if medical_map[key].get("category") in {"drug", "medication", "drug_class"}:
            medical_skeletons[persian_phonetic_skeleton(key)].append(key)
    general_cache: dict[int, list[str]] = {}
    hypotheses = {}
    for path in (run_dir / "hypotheses").glob("*/*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        hypotheses[f"{payload['model']}__{payload['source']}"] = payload
    expected = {f"{m}__{s}" for m in MODELS for s in SOURCES}
    if set(hypotheses) != expected:
        raise RuntimeError(f"Expected six hypotheses, found {sorted(hypotheses)}")
    base = words_of(hypotheses[BASE_KEY])
    aligned = {key: align_monotonic(base, words_of(payload)) for key, payload in hypotheses.items()}
    decisions, output = [], []
    for index, base_word in enumerate(base):
        family_items: dict[str, list[dict[str, Any]]] = defaultdict(list)
        raw_evidence = []
        for key in sorted(aligned):
            item = aligned[key][index]
            if item is not None:
                family_items[item["model"]].append(item)
                raw_evidence.append(item)
        family_votes = [family_vote(items, medical_map) for items in family_items.values()]
        six_votes = []
        for key in sorted(aligned):
            item = aligned[key][index]
            six_votes.append({"hypothesis": key, "word": item["word"] if item else None,
                              "normalized": item["normalized"] if item else None,
                              "probability": item["probability"] if item else None})
        candidate_scores = score_six_candidates(raw_evidence, base_word["normalized"], output, medical_map)
        exact_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for vote in family_votes:
            exact_groups[vote["normalized"]].append(vote)
        ranked = sorted(exact_groups.items(), key=lambda entry: (len(entry[1]), sum(v["score"] for v in entry[1])), reverse=True)
        chosen_norm = base_word["normalized"]
        chosen_word = base_word["word"]
        reason = "keep-turbo-base"
        dictionary_detail: dict[str, Any] = {}
        is_protected, protection_reason = protected(base, index, medical_map)
        if ranked and len({v["model"] for v in ranked[0][1]}) >= 2:
            candidate_norm, supporting = ranked[0]
            if candidate_norm != chosen_norm:
                family_support = len({v["model"] for v in supporting})
                next_norm = base[index + 1]["normalized"] if index + 1 < len(base) else ""
                morphological_merge = (next_norm in {"ها", "هات", "ات"}
                                       and candidate_norm.startswith(chosen_norm)
                                       and len(candidate_norm) > len(chosen_norm))
                candidate_is_sensitive = (candidate_norm in NUMBERS or candidate_norm in UNITS
                                          or candidate_norm in NEGATIONS)
                candidate_active = active_word(candidate_norm, medical_map)
                both_valid = (lexicon_valid(chosen_norm, medical_map)
                              and lexicon_valid(candidate_norm, medical_map))
                medical_upgrade = candidate_norm in medical_map and chosen_norm not in medical_map
                turbo_supports = any(v["model"] == "large-v3-turbo" for v in supporting)
                # Turbo is the requested base. In a lexical tie, two model families are not
                # enough to replace another valid word, a number, a dose, or a negation.
                allow_exact = candidate_active and (
                    family_support == 3 or medical_upgrade or morphological_merge
                    or (turbo_supports and not candidate_is_sensitive)
                    or (not both_valid and not candidate_is_sensitive)
                )
                if allow_exact:
                    chosen_norm = candidate_norm
                    chosen_word = max(supporting, key=lambda v: v["score"])["word"]
                    reason = "two-family-exact-consensus"
                else:
                    reason = "keep-turbo-base-lexical-tie"
        if chosen_norm not in medical_map and not general_contains(chosen_norm):
            medical_term, medical_support = choose_medical_canonical(
                family_votes, medical_single, medical_map, medical_skeletons)
            dictionary_detail = {"medical_candidate": medical_term, "medical_support": medical_support}
            if medical_term and medical_term != chosen_norm:
                chosen_norm = medical_term
                chosen_word = " " + medical_map[medical_term]["term"]
                reason = "two-family-medical-fuzzy-consensus"
            else:
                general_term, general_support = choose_general_canonical(family_votes, general_cache, medical_map)
                dictionary_detail["general_candidate"] = general_term
                dictionary_detail["general_support"] = general_support
                if general_term and general_term != chosen_norm:
                    current_exact_support = len({v["model"] for v in family_votes if v["normalized"] == chosen_norm})
                    general_exact_support = len({s["family"] for s in general_support if s["exact"]})
                    unseen_unanimous = (general_exact_support == 0 and len({s["family"] for s in general_support}) == 3
                                        and min(s["similarity"] for s in general_support) >= 92)
                    if current_exact_support < 2 or general_exact_support >= 2 or unseen_unanimous:
                        chosen_norm = general_term
                        chosen_word = " " + general_term
                        reason = "two-family-general-lexicon-fuzzy-consensus"
        if candidate_scores and candidate_scores[0]["context_rule"]:
            current_score = next((row["total_score"] for row in candidate_scores
                                  if row["candidate"] == chosen_norm), 0.0)
            contextual = candidate_scores[0]
            if contextual["total_score"] > current_score and contextual["canonical"] != chosen_norm:
                chosen_norm = contextual["canonical"]
                chosen_word = render_like(base_word["word"], chosen_norm)
                reason = "six-vote-score-plus-phrase-context"
                dictionary_detail["phrase_context"] = contextual["context_rule"]
        base_exact_votes = sum(item["normalized"] == base_word["normalized"] for item in raw_evidence)
        voted_surface_is_plausible = (usage_frequency(base_word["normalized"]) >= 3.0
                                     or base_word["normalized"] in MODERN_SPOKEN
                                     or base_word["normalized"] in medical_map)
        voting_unresolved = base_exact_votes < 2 or not voted_surface_is_plausible
        if (chosen_norm == base_word["normalized"] and voting_unresolved
                and not lexicon_valid(chosen_norm, medical_map)):
            fallback, fallback_ranked = choose_turbo_dictionary_fallback(
                chosen_norm, raw_evidence, general_cache, medical_map)
            dictionary_detail["turbo_dictionary_fallback"] = fallback
            dictionary_detail["turbo_dictionary_candidates"] = fallback_ranked
            if fallback:
                chosen_norm = fallback
                chosen_word = render_like(base_word["word"], fallback)
                reason = "turbo-base-dictionary-fallback"
        if is_protected and chosen_norm != base_word["normalized"]:
            supporters = [v for v in family_votes if v["normalized"] == chosen_norm]
            medical_fuzzy_support = dictionary_detail.get("medical_support") or []
            safe_phonetic_medical = (reason == "two-family-medical-fuzzy-consensus"
                                     and len({s["family"] for s in medical_fuzzy_support}) >= 2
                                     and min((s["similarity"] for s in medical_fuzzy_support), default=0) >= 95)
            if len({v["model"] for v in supporters}) < 3 and not safe_phonetic_medical:
                chosen_norm = base_word["normalized"]
                chosen_word = base_word["word"]
                reason = "keep-protected-turbo-base"
        if chosen_norm != base_word["normalized"] and base_word["normalized"].endswith(chosen_norm) and len(base_word["normalized"]) - len(chosen_norm) <= 3:
            chosen_norm = base_word["normalized"]
            chosen_word = base_word["word"]
            reason = "keep-prevent-prefix-loss"
        if output:
            previous = norm(output[-1])
            if chosen_norm != base_word["normalized"] and (chosen_norm == previous or (len(previous) >= 2 and chosen_norm.startswith(previous))):
                chosen_norm = base_word["normalized"]
                chosen_word = base_word["word"]
                reason = "keep-prevent-duplicate"
        if index + 1 < len(base) and chosen_norm != base_word["normalized"] and chosen_norm == base[index + 1]["normalized"]:
            chosen_norm = base_word["normalized"]
            chosen_word = base_word["word"]
            reason = "keep-prevent-lookahead-duplicate"
        if output and base_word["normalized"] in {"ها", "هات", "ات"}:
            previous_norm = norm(output[-1])
            suffix = base_word["normalized"].lstrip("ه")
            if suffix and previous_norm.endswith(suffix):
                chosen_norm = ""
                chosen_word = ""
                reason = "merge-duplicate-suffix-with-previous"
        output.append(chosen_word)
        decisions.append({"index": index, "start": base_word["start"], "end": base_word["end"],
                          "base": base_word["word"], "chosen": chosen_word, "reason": reason,
                          "protected": is_protected, "protection_reason": protection_reason,
                          "family_votes": family_votes, "raw_evidence": raw_evidence,
                          "six_hypothesis_votes": six_votes, "candidate_scores": candidate_scores,
                          "dictionary": dictionary_detail})
    final_text = "".join(output).strip()
    out_dir = run_dir / "final-delivery" / "02-after-algorithm-v2-turbo-lexicon"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"base": BASE_KEY, "method": "six-hypothesis monotonic alignment, family voting, general and medical lexicons",
               "general_lexicon": {"package": "mnk-persian-words", "version": "1.5.0", "clean_words": persian_words.count_words()},
               "medical_lexicon": {"source": medical_payload["source"], "license": medical_payload["license"],
                                   "unique_terms": medical_payload["unique_terms"]},
               "text": final_text, "decisions": decisions}
    (out_dir / "final-v2.txt").write_text(final_text + "\n", encoding="utf-8")
    (out_dir / "final-v2.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "decisions-v2.json").write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")
    scorecard = [{"index": d["index"], "start": d["start"], "base": d["base"],
                  "chosen": d["chosen"], "reason": d["reason"],
                  "six_hypothesis_votes": d["six_hypothesis_votes"],
                  "candidate_scores": d["candidate_scores"]} for d in decisions]
    (out_dir / "scorecard-v2.json").write_text(json.dumps(scorecard, ensure_ascii=False, indent=2), encoding="utf-8")
    score_lines = [
        "# امتیاز شش رونویسی در هر جایگاه",
        "",
        "هر رأی مستقیم ۱ امتیاز، هر خانوادهٔ مدل ۰٫۳۵، میانگین اطمینان تا ۰٫۵، "
        "لغت‌نامهٔ عمومی ۰٫۱۵، پزشکی ۰٫۳۵، فارسی محاوره ۰٫۶۵ و پایهٔ Turbo مقدار ۰٫۲۰ امتیاز دارد. "
        "فراوانی فارسی تا ۱٫۲۰ امتیاز می‌گیرد و واژهٔ مسدود یا زیر آستانه ۳ امتیاز جریمه می‌شود. "
        "امتیاز بافت فقط هنگام تطبیق یک عبارت ثبت‌شده اعمال می‌شود.",
        "",
        "| ردیف | زمان | پایه | شش رأی (Turbo پالایش/خام؛ Large پالایش/خام؛ Medium پالایش/خام) | گزینه‌ها و امتیاز | انتخاب | دلیل |",
        "|---:|---:|---|---|---|---|---|",
    ]
    for decision in decisions:
        vote_map = {v["hypothesis"]: (v["normalized"] or "—") for v in decision["six_hypothesis_votes"]}
        vote_order = [("large-v3-turbo", "enhanced"), ("large-v3-turbo", "raw"),
                      ("large-v3", "enhanced"), ("large-v3", "raw"),
                      ("medium", "enhanced"), ("medium", "raw")]
        ordered_votes = [vote_map.get(f"{model}__{source}", "—") for model, source in vote_order]
        scores = "؛ ".join(f"{row['candidate']}={row['total_score']:.2f} ({row['votes']}/6)"
                            for row in decision["candidate_scores"])
        cells = [str(decision["index"]), f"{decision['start']:.2f}", decision["base"].strip(),
                 "، ".join(ordered_votes), scores, decision["chosen"].strip(), decision["reason"]]
        cells = [str(cell).replace("|", "\\|").replace("\n", " ") for cell in cells]
        score_lines.append("| " + " | ".join(cells) + " |")
    (out_dir / "scorecard-v2.md").write_text("\n".join(score_lines) + "\n", encoding="utf-8")
    summary = defaultdict(int)
    for decision in decisions:
        summary[decision["reason"]] += 1
    (out_dir / "summary-v2.json").write_text(json.dumps(dict(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    report_lines = [
        "# گزارش ادغام شش رونویسی",
        "",
        f"- پایه: `{BASE_KEY}`",
        "- ورودی تصمیم‌گیری: سه خانوادهٔ مدل × صدای خام و پالایش‌شده (شش فرضیه)",
        "- روش: هم‌ترازی زمانی/واژگانی، یک رأی برای هر خانوادهٔ مدل، سپس اجماع و امتیاز اطمینان",
        "- شفافیت: فایل `scorecard-v2.md` رأی هر شش فرضیه و امتیاز هر گزینه را در همهٔ جایگاه‌ها نشان می‌دهد.",
        f"- واژه‌نامهٔ عمومی: `mnk-persian-words 1.5.0` ({persian_words.count_words():,} واژه)",
        f"- واژه‌نامهٔ پزشکی: `PersianMedQA` ({medical_payload['unique_terms']:,} مدخل یکتا، {medical_payload['license']})",
        f"- پالایش واژگان امروز: آستانهٔ فراوانی فارسی Zipf برابر `{MIN_ACTIVE_ZIPF:.2f}`؛ "
        f"{len(USER_BLOCKLIST)} واژهٔ منع‌شده و {len(MODERN_SPOKEN)} واژهٔ محاوره‌ای افزوده‌شده.",
        "- محافظت: عدد، دوز/واحد، نفی و نام دارو بدون اجماع هر سه خانواده تغییر نمی‌کنند.",
        "",
        "## خلاصهٔ تصمیم‌ها",
        "",
    ]
    report_lines.extend(f"- `{key}`: {value}" for key, value in sorted(summary.items()))
    report_lines.extend([
        "",
        "## متن نهایی الگوریتمی",
        "",
        final_text,
        "",
        "> این خروجی مرجع انسانی ندارد و برای تصمیم پزشکی باید با صوت اصلی بازبینی شود.",
        "",
    ])
    (out_dir / "report-v2.md").write_text("\n".join(report_lines), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps({"output": str(out_dir), "decision_summary": dict(summary), "text": final_text}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
