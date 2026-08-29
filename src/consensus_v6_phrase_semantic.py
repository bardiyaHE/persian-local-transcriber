from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import consensus_v4 as v4
import consensus_v5 as v5
from rapidfuzz import fuzz, process
from consensus_v2 import norm
from consensus_v3 import NETWORK_ORDER, cluster_doses, dose_occurrences, load_hypotheses, words_of


OUTPUT_RELATIVE = Path("final-delivery") / "06-phrase-semantic-no-llm"
FAMILY_COUNT = 3
MAX_PHRASE_ORDER = 5
DOSE_CUES = {"دارو", "داروی", "قرص", "کپسول", "شربت", "آمپول", "دوز", "مصرف"}
VITAL_CUES = {"فشار", "فشارتون", "فشارتان", "نبض", "نبز", "نبضتون", "نبضتان"}
TIME_CUES = {"صبح", "ظهر", "عصر", "شب", "روزی", "هفته", "ماه", "ساعت"}
ROUTE_PHRASES = {("زیر", "زبان"), ("صبح", "و", "عصر"), ("صبح", "و", "شب")}
NUMBER_RE = re.compile(r"^[0-9۰-۹]+(?:[./٫][0-9۰-۹]+)?$")
INFLECTION_SUFFIXES = ("هاشون", "هاتون", "هایمون", "شون", "تون", "مون", "ها", "اش", "ات", "ام")
PERSIAN_DIGIT_TRANSLATION = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")


class PhraseEvidence:
    """Two-to-five-token evidence collapsed by model family, never by decode count."""

    def __init__(self, sequences: dict[str, list[dict[str, Any]]]) -> None:
        self.exact: dict[tuple[str, ...], set[str]] = defaultdict(set)
        self.phonetic: dict[tuple[str, ...], set[str]] = defaultdict(set)
        self.counts: Counter[tuple[str, ...]] = Counter()
        for words in sequences.values():
            if not words:
                continue
            family = words[0]["family"]
            tokens = [word["normalized"] for word in words]
            for size in range(2, MAX_PHRASE_ORDER + 1):
                for index in range(len(tokens) - size + 1):
                    gram = tuple(tokens[index:index + size])
                    self.exact[gram].add(family)
                    self.phonetic[tuple(v4.phonetic_key(token) for token in gram)].add(family)
                    self.counts[gram] += 1

    def trailing_score(self, history: tuple[str, ...], parts: tuple[str, ...]) -> tuple[float, list[dict[str, Any]]]:
        combined = (*history[-(MAX_PHRASE_ORDER - 1):], *parts)
        score = 0.0
        details = []
        for size in range(2, min(MAX_PHRASE_ORDER, len(combined)) + 1):
            gram = tuple(combined[-size:])
            phonetic = tuple(v4.phonetic_key(token) for token in gram)
            exact_families = len(self.exact.get(gram, set()))
            phonetic_families = len(self.phonetic.get(phonetic, set()))
            support = max(exact_families, 0.68 * phonetic_families)
            contribution = 0.11 * size * support
            if max(exact_families, phonetic_families) >= 2:
                contribution += 0.13 * size
            if max(exact_families, phonetic_families) == 3:
                contribution += 0.08 * size
            score += contribution
            details.append({
                "gram": " ".join(gram), "order": size,
                "exact_families": exact_families,
                "phonetic_families": phonetic_families,
                "occurrences": self.counts.get(gram, 0),
                "score": round(contribution, 6),
            })
        return score, details


def is_number(token: str) -> bool:
    return token in v4.NUMBER_WORDS or bool(NUMBER_RE.fullmatch(token))


def is_drug(token: str, resolver: v4.LexiconResolver) -> bool:
    return bool(resolver.medical_categories(token) & {"drug", "medication", "drug_class"})


def effective_family_count(row: dict[str, Any]) -> int:
    return max(
        len(row.get("strong_families") or []),
        len(row.get("template_families") or []),
        len(row.get("entity_consistency_families") or []),
    )


def future_anchor_tokens(lattice: list[list[dict[str, Any]]], index: int,
                         limit: int = 6) -> tuple[str, ...]:
    result: list[str] = []
    for candidates in lattice[index + 1:]:
        nonempty = [row for row in candidates if row.get("candidate")]
        if not nonempty:
            continue
        best = max(nonempty, key=lambda row: (
            len(row.get("exact_families") or []),
            len(row.get("strong_families") or []),
            float(row.get("acoustic_probability") or 0.0),
            float(row.get("emission_score") or 0.0),
        ))
        result.extend(best.get("candidate_tokens") or [])
        if len(result) >= limit:
            break
    return tuple(result[:limit])


def entity_structure_score(history: tuple[str, ...], parts: tuple[str, ...],
                           future: tuple[str, ...], row: dict[str, Any],
                           resolver: v4.LexiconResolver) -> tuple[float, list[str]]:
    context = (*history[-4:], *parts, *future[:6])
    score = 0.0
    reasons: list[str] = []
    strong_families = effective_family_count(row)
    semantic_medical_repair = row.get("origin") == "semantic-lexicon-repair"
    for offset, token in enumerate(parts):
        before = (*history[-3:], *parts[:offset])
        after = (*parts[offset + 1:], *future[:6])
        if is_number(token):
            structured_frame = (
                any(item in DOSE_CUES | VITAL_CUES for item in before[-3:])
                or any(is_drug(item, resolver) for item in before[-3:])
                or any(item in v4.UNITS for item in after[:2])
            )
            if structured_frame:
                score += 0.85
                reasons.append("number-fits-dose-or-vital-frame")
                if row.get("structured_numeric_candidate"):
                    # An observed digit supported exactly by two model families is
                    # safer here than a unanimous dictionary-invalid sound-alike.
                    score += 7.20
                    reasons.append("two-family-structured-number")
            if strong_families < 2:
                score -= 1.65
                reasons.append("single-family-number")
        if token in v4.UNITS:
            if any(is_number(item) for item in before[-2:]):
                score += 1.05
                reasons.append("unit-follows-number")
            elif strong_families < 2:
                score -= 1.15
                reasons.append("unsupported-unit")
        categories = set(row.get("medical_categories") or [])
        drug_candidate = (len(token) >= 3 and token not in v4.FUNCTION_WORDS and (
            is_drug(token, resolver)
            or bool(categories & {"drug", "medication", "drug_class"})))
        if drug_candidate:
            route_frame = ("زیر" in (*before[-4:], *after[:6]) and any(
                item.startswith("زبان") or item.startswith("زبون")
                for item in (*before[-4:], *after[:6])))
            if any(item in DOSE_CUES | {"نصف"} for item in (*before[-3:], *after[:4])):
                score += 0.75
                reasons.append("drug-fits-medication-frame")
            if route_frame:
                score += 2.70
                reasons.append("drug-fits-sublingual-frame")
            if len(row.get("entity_consistency_families") or []) >= 2:
                score += 3.20
                reasons.append("repeated-medication-entity")
            if strong_families < 2:
                score -= 1.70
                reasons.append("single-family-drug")
        if semantic_medical_repair and categories & {"symptom", "disease", "anatomical_term"}:
            if any(item in {"حمله", "شدید", "پانیک", "دریچه", "فشار", "بالای"}
                   for item in (*before[-3:], *after[:4])):
                score += 4.00
                reasons.append("medical-entity-fits-clinical-frame")
            if "symptom" in categories and "دریچه" in after[:4]:
                score += 2.40
                reasons.append("symptom-fits-valve-frame")
            if categories & {"symptom", "disease"} and any(
                    item in {"حمله", "پانیک", "شدید"} for item in (*before[-3:], *after[:4])):
                score += 2.20
                reasons.append("symptom-fits-episode-frame")
        if token == "قرص" and "نصف" in before[-3:] and (
                "زیر" in after[:6] and any(item.startswith("زبان") or item.startswith("زبون")
                                           for item in after[:6])):
            score += 4.40
            reasons.append("dosage-form-fits-sublingual-frame")
        if token == "قرص" and row.get("precedes_resolved_medication_entity"):
            score += 6.00
            reasons.append("dosage-form-precedes-resolved-drug")
        if v4.is_negative_token(token) and strong_families < 2:
            score -= 2.50
            reasons.append("single-family-negation")
    for phrase in ROUTE_PHRASES:
        joined = " ".join(context)
        if " ".join(phrase) in joined:
            score += 0.55
            reasons.append("recognized-instruction-phrase")
    if any(token in TIME_CUES for token in context) and any(
            is_number(token) for token in context):
        score += 0.25
        reasons.append("time-quantity-frame")
    lexical = bool(row.get("general_lexicon") or row.get("medical_lexicon")
                   or row.get("modern_spoken") or any(is_number(part) for part in parts))
    if parts and not lexical and strong_families < 2:
        score -= 0.85
        reasons.append("rare-single-family-token")
    if row.get("medical_semantic_rival") and any(
            item in {"دریچه", "حمله", "شدید", "پانیک", "فشار", "بالای"}
            for item in context):
        score -= 3.60
        reasons.append("shared-oov-loses-to-clinical-medical-rival")
    for size in (3, 2):
        if not semantic_medical_repair:
            break
        for start in range(max(0, len(history[-2:]) - 1), len(context) - size + 1):
            gram = tuple(context[start:start + size])
            if not any(part in parts for part in gram):
                continue
            phonetic = tuple(v4.phonetic_key(part) for part in gram)
            if gram in resolver.medical_phrase_ngrams:
                score += 1.25 if size == 2 else 2.10
                reasons.append(f"medical-{size}gram-exact")
                break
            if phonetic in resolver.medical_phonetic_ngrams:
                score += 0.90 if size == 2 else 1.55
                reasons.append(f"medical-{size}gram-phonetic")
                break
    return score, reasons


def mark_turbo_anchors(slots: list[dict[str, Any]],
                       lattice: list[list[dict[str, Any]]]) -> None:
    """Record the user-requested Turbo base prior without suppressing other families."""
    for slot, candidates in zip(slots, lattice):
        turbo = [row for row in slot["observations"].values()
                 if row.get("family") == "large-v3-turbo"]
        for row in candidates:
            candidate = row.get("candidate") or ""
            similarities = [v4.token_similarity(candidate, item["normalized"]) for item in turbo]
            row["turbo_exact_anchor"] = bool(candidate and any(
                candidate == item["normalized"] for item in turbo))
            row["turbo_anchor_similarity"] = round(max(similarities, default=0.0), 6)


def calibrated_transition_score(ngrams: v5.DomainNgramEvidence,
                                history: tuple[str, ...],
                                token: str) -> tuple[float, dict[str, Any]]:
    """Keep corpus evidence independent and discount duplicated local-six evidence."""
    raw_score, detail = ngrams.transition_score(history, token)
    local_score = float(detail["local_six_hypothesis_score"]
                        if "local_six_hypothesis_score" in detail else raw_score)
    corpus_score = raw_score - local_score
    calibrated = 0.28 * local_score + corpus_score
    detail["raw_combined_score"] = round(raw_score, 6)
    detail["calibrated_local_six_score"] = round(0.28 * local_score, 6)
    detail["independent_corpus_score"] = round(corpus_score, 6)
    detail["calibrated_score"] = round(calibrated, 6)
    return calibrated, detail


def prune_lattice(lattice: list[list[dict[str, Any]]], limit: int = 8) -> list[list[dict[str, Any]]]:
    result = []
    for candidates in lattice:
        eligible = []
        for row in candidates:
            if not row.get("candidate"):
                eligible.append(row)
                continue
            if (row.get("observed") or len(row.get("strong_families") or []) >= 2
                    or row.get("origin") in {"repeated-text-bigram-split", "productive-prefix-repair",
                                             "medication-template", "repeated-medication-entity"}):
                eligible.append(row)
        eligible.sort(key=lambda row: (
            bool(row.get("candidate")), len(row.get("strong_families") or []),
            len(row.get("exact_families") or []), float(row.get("emission_score") or 0.0),
        ), reverse=True)
        epsilon = next((row for row in eligible if not row.get("candidate")), None)
        nonempty = [row for row in eligible if row.get("candidate")][:limit]
        if epsilon is not None:
            nonempty.append(epsilon)
        result.append(nonempty)
    return result


def augment_semantic_candidates(slots: list[dict[str, Any]],
                                lattice: list[list[dict[str, Any]]],
                                resolver: v4.LexiconResolver,
                                ngrams: v5.DomainNgramEvidence) -> dict[str, int]:
    """Admit broad but acoustically gated modern/medical repairs for shared ASR errors."""
    resolver._ensure_general()
    cache: dict[str, list[tuple[str, float, str, str]]] = {}

    def ranked_pool(token: str, pool: list[str], cutoff: int = 76,
                    limit: int = 18) -> list[tuple[str, float]]:
        """Search spelling and phonetic forms using RapidFuzz's indexed loops."""
        if not pool:
            return []
        scored: dict[str, float] = {}
        for word, _score, _index in process.extract(
                token, pool, scorer=fuzz.ratio, score_cutoff=cutoff, limit=limit):
            scored[word] = v4.token_similarity(token, word)
        phonetic_to_words: dict[str, list[str]] = defaultdict(list)
        for word in pool:
            phonetic_to_words[v4.phonetic_key(word)].append(word)
        phonetic_keys = list(phonetic_to_words)
        for key, _score, _index in process.extract(
                v4.phonetic_key(token), phonetic_keys, scorer=fuzz.ratio,
                score_cutoff=cutoff, limit=limit):
            for word in phonetic_to_words[key]:
                scored[word] = max(scored.get(word, 0.0), v4.token_similarity(token, word))
        return sorted(scored.items(), key=lambda item: (
            item[1], v4.usage_frequency(item[0])), reverse=True)[:limit]

    def nearby(token: str) -> list[tuple[str, float, str, str]]:
        token = norm(token)
        if token in cache:
            return cache[token]
        found: dict[str, tuple[float, str, str]] = {}
        medical_pool = resolver._pool(resolver._medical_buckets, token, 3)
        for word, similarity in ranked_pool(token, medical_pool, 76, 16):
            if similarity >= 0.78:
                found[word] = max(found.get(word, (0.0, "", "")),
                                  (similarity, "medical", ""))
        general_pool = resolver._pool(resolver._general_buckets or {}, token, 3)
        for word, similarity in ranked_pool(token, general_pool, 76, 16):
            if similarity >= 0.78 and v4.active_general_word(word):
                found[word] = max(found.get(word, (0.0, "", "")),
                                  (similarity, "general", ""))
        for suffix in INFLECTION_SUFFIXES:
            if not token.endswith(suffix) or len(token) <= len(suffix) + 2:
                continue
            stem = token[:-len(suffix)]
            stem_pool = resolver._pool(resolver._medical_buckets, stem, 2)
            stem_pool += resolver._pool(resolver._general_buckets or {}, stem, 2)
            for word, stem_similarity in ranked_pool(stem, stem_pool, 76, 12):
                candidate = word + suffix
                similarity = v4.token_similarity(token, candidate)
                if stem_similarity >= 0.78 and similarity >= 0.78 and (
                        resolver.is_medical(word) or v4.active_general_word(word)):
                    source = "medical" if resolver.is_medical(word) else "general"
                    found[candidate] = max(found.get(candidate, (0.0, "", "")),
                                           (similarity, source, word))
        result = sorted(
            ((word, value[0], value[1], value[2]) for word, value in found.items()),
            key=lambda item: (item[1], v4.usage_frequency(item[0])), reverse=True)[:10]
        cache[token] = result
        return result

    added = 0
    penalized_shared_errors = 0
    numeric_marked = 0
    for slot_index, candidates in enumerate(lattice):
        observations = list(slots[slot_index]["observations"].values())
        existing = {row.get("candidate") for row in candidates}
        observed_tokens = {row["normalized"] for row in observations}
        for observed in observed_tokens:
            if v4.active_general_word(observed) or resolver.is_medical(observed):
                continue
            for candidate, similarity, source, stem in nearby(observed):
                if candidate in existing or candidate == observed:
                    continue
                similarities = v4.family_similarity(candidate, observations)
                strong = {family for family, value in similarities.items() if value >= 0.78}
                loose = {family for family, value in similarities.items() if value >= 0.66}
                if len(strong) < 2:
                    continue
                probability_by_family = []
                for family in strong:
                    family_rows = [row for row in observations if row["family"] == family]
                    probability_by_family.append(max(
                        row["probability"] * v4.token_similarity(candidate, row["normalized"])
                        for row in family_rows))
                probability = sum(probability_by_family) / max(1, len(probability_by_family))
                base_token = stem or candidate
                general = source == "general" or v4.active_general_word(base_token)
                medical = source == "medical" or resolver.is_medical(base_token)
                categories = resolver.medical_categories(base_token) if medical else set()
                medical_bonus = 0.78 if categories & {"drug", "medication", "drug_class"} else 0.38 if medical else 0.0
                repetition_families, repetition_count = ngrams.token_repetition(candidate)
                emission = (
                    1.45 * len(strong) + 0.26 * max(0, len(loose) - len(strong))
                    + 0.42 * probability + 0.32 * repetition_families
                    + 0.11 * math.log1p(repetition_count)
                    + (0.50 if general else 0.0) + medical_bonus
                    + 0.055 * min(6.0, v4.usage_frequency(base_token)) - 0.64
                )
                candidates.append({
                    "candidate": candidate, "candidate_tokens": [candidate],
                    "origin": "semantic-lexicon-repair", "dictionary_similarity": similarity,
                    "emission_score": round(emission, 6), "exact_families": [],
                    "strong_families": sorted(strong), "loose_families": sorted(loose),
                    "family_similarity": similarities, "exact_sources": 0,
                    "repetition_families": repetition_families,
                    "repetition_count": repetition_count,
                    "general_dictionary_member": general, "general_lexicon": general,
                    "medical_lexicon": medical, "modern_spoken": False,
                    "medical_categories": sorted(categories),
                    "acoustic_probability": round(probability, 6),
                    "zipf_frequency_fa": round(v4.usage_frequency(base_token), 4),
                    "observed": False, "derived_from": observed,
                })
                existing.add(candidate)
                added += 1
        numeric_rows = [row for row in candidates if row.get("candidate") and NUMBER_RE.fullmatch(row["candidate"])]
        if numeric_rows:
            for row in numeric_rows:
                if len(row.get("exact_families") or []) >= 2:
                    row["structured_numeric_candidate"] = True
                    numeric_marked += 1
        valid_repairs = [row for row in candidates
                         if row.get("candidate")
                         and (row.get("general_lexicon") or row.get("medical_lexicon")
                              or row.get("modern_spoken"))
                         and len(row.get("strong_families") or []) >= 2]
        for row in candidates:
            if (not row.get("candidate") or row.get("general_lexicon")
                    or row.get("medical_lexicon") or row.get("modern_spoken")):
                continue
            rivals = [other for other in valid_repairs
                      if v4.token_similarity(row["candidate"], other["candidate"]) >= 0.78]
            if not rivals:
                continue
            penalty = 2.20 + 0.75 * len(row.get("exact_families") or [])
            row["emission_score"] = round(float(row["emission_score"]) - penalty, 6)
            row["shared-oov-semantic-rival-penalty"] = round(penalty, 6)
            medical_rivals = [other for other in rivals if other.get("medical_lexicon")]
            if medical_rivals:
                row["medical_semantic_rival"] = [other["candidate"] for other in medical_rivals]
            penalized_shared_errors += 1
    return {"added_candidates": added, "penalized_shared_errors": penalized_shared_errors,
            "structured_numeric_candidates": numeric_marked}


def augment_medication_frames(slots: list[dict[str, Any]],
                              lattice: list[list[dict[str, Any]]],
                              resolver: v4.LexiconResolver,
                              ngrams: v5.DomainNgramEvidence) -> dict[str, int]:
    """Resolve dosage form/drug identity only inside a strict sublingual template."""
    slot_tokens = [
        {item["normalized"] for item in slot["observations"].values()}
        for slot in slots
    ]
    routes: list[int] = []
    for index, tokens in enumerate(slot_tokens):
        if "زیر" not in tokens:
            continue
        lookahead = set().union(*slot_tokens[index + 1:min(len(slot_tokens), index + 4)])
        if any(token.startswith("زبان") or token.startswith("زبون") for token in lookahead):
            routes.append(index)
    frames: list[tuple[int, int, int]] = []
    for route in routes:
        halves = [index for index in range(max(0, route - 5), route)
                  if "نصف" in slot_tokens[index]]
        if halves:
            frames.append((max(0, route - 8), halves[-1], route))
    if not frames:
        return {"medication_template_candidates": 0,
                "repeated_medication_candidates": 0, "dosage_form_templates": 0}

    dosage_forms = 0
    for _start, half, route in frames:
        form_slot = half + 1
        if form_slot >= route:
            continue
        observations = list(slots[form_slot]["observations"].values())
        similarities = v4.family_similarity("قرص", observations)
        template_families = {family for family, value in similarities.items() if value >= 0.57}
        if len(template_families) < 2:
            continue
        existing = next((row for row in lattice[form_slot] if row.get("candidate") == "قرص"), None)
        probability = sum(max(
            item["probability"] * v4.token_similarity("قرص", item["normalized"])
            for item in observations if item["family"] == family)
            for family in template_families) / len(template_families)
        if existing is None:
            existing = {
                "candidate": "قرص", "candidate_tokens": ["قرص"],
                "origin": "medication-template", "dictionary_similarity": max(similarities.values()),
                "emission_score": round(1.35 * len(template_families) + 0.75 * probability + 0.8, 6),
                "exact_families": [], "strong_families": [],
                "loose_families": sorted(template_families), "family_similarity": similarities,
                "exact_sources": 0, "general_dictionary_member": True,
                "general_lexicon": True, "medical_lexicon": True, "modern_spoken": False,
                "medical_categories": ["drug"], "acoustic_probability": round(probability, 6),
                "zipf_frequency_fa": round(v4.usage_frequency("قرص"), 4),
                "observed": False,
            }
            lattice[form_slot].append(existing)
        existing["template_families"] = sorted(template_families)
        existing["medication_template_frame"] = True
        dosage_forms += 1

    drug_terms = [term for term in resolver.medical_exact
                  if resolver.medical_categories(term) & {"drug", "medication", "drug_class"}]
    primary_drug_terms = [term for term in drug_terms
                          if resolver.medical_source_priority(term) >= 2]
    drug_phonetic_keys: dict[str, list[str]] = defaultdict(list)
    for term in drug_terms:
        drug_phonetic_keys[v4.phonetic_key(term)].append(term)
    added_entities: list[tuple[int, str, str]] = []
    template_added = 0
    frame_slots = sorted({index for start, _half, route in frames for index in range(start, route)})
    for slot_index in frame_slots:
        observations = list(slots[slot_index]["observations"].values())
        observed_tokens = {row["normalized"] for row in observations}
        if not observed_tokens or all(len(token) < 5 for token in observed_tokens):
            continue
        existing_names = {row.get("candidate") for row in lattice[slot_index]}
        for observed in observed_tokens:
            suffix = next((item for item in INFLECTION_SUFFIXES
                           if observed.endswith(item) and len(observed) > len(item) + 3), "")
            stem = observed[:-len(suffix)] if suffix else observed
            direct = process.extract(stem, drug_terms, scorer=fuzz.ratio,
                                     score_cutoff=54, limit=14)
            # An expanded supplemental list must not crowd a known base entry
            # out of the short candidate pool (for example لورازپام among many
            # similarly spelled benzodiazepines).
            primary_direct = process.extract(
                stem, primary_drug_terms, scorer=fuzz.ratio,
                score_cutoff=54, limit=14)
            phonetic = process.extract(v4.phonetic_key(stem), list(drug_phonetic_keys),
                                       scorer=fuzz.ratio, score_cutoff=54, limit=14)
            terms = {word for word, _score, _index in [*direct, *primary_direct]}
            for key, _score, _index in phonetic:
                terms.update(drug_phonetic_keys[key])
            for drug in terms:
                candidate = drug + suffix
                if candidate in existing_names:
                    continue
                similarities = v4.family_similarity(candidate, observations)
                strong = {family for family, value in similarities.items() if value >= 0.78}
                loose = {family for family, value in similarities.items() if value >= 0.58}
                if len(strong) < 2 or max(similarities.values()) < 0.78:
                    continue
                probability = sum(max(
                    item["probability"] * v4.token_similarity(candidate, item["normalized"])
                    for item in observations if item["family"] == family)
                    for family in strong) / len(strong)
                repetition_families, repetition_count = ngrams.token_repetition(candidate)
                lattice[slot_index].append({
                    "candidate": candidate, "candidate_tokens": [candidate],
                    "origin": "medication-template", "dictionary_similarity": max(similarities.values()),
                    "emission_score": round(1.55 * len(strong) + 0.35 * len(loose)
                                             + 0.8 * probability + 1.0, 6),
                    "exact_families": [], "strong_families": sorted(strong),
                    "loose_families": sorted(loose), "family_similarity": similarities,
                    "exact_sources": 0, "repetition_families": repetition_families,
                    "repetition_count": repetition_count, "general_dictionary_member": False,
                    "general_lexicon": False, "medical_lexicon": True, "modern_spoken": False,
                    "medical_categories": sorted(resolver.medical_categories(drug)),
                    "acoustic_probability": round(probability, 6),
                    "zipf_frequency_fa": round(v4.usage_frequency(drug), 4),
                    "observed": False, "derived_from": observed, "medical_base_term": drug,
                    "medication_template_frame": True,
                })
                existing_names.add(candidate)
                added_entities.append((slot_index, candidate, drug))
                template_added += 1

    repeated_added = 0
    unique_entities = sorted({candidate for _slot, candidate, _drug in added_entities})
    ranked_entities: list[tuple[str, int, float]] = []
    for candidate in unique_entities:
        aggregate = 0.0
        consistent_occurrences = 0
        for slot_index in frame_slots:
            observations = list(slots[slot_index]["observations"].values())
            if not any(len(item["normalized"]) >= 5 for item in observations):
                continue
            family_scores = []
            for family in v4.FAMILIES:
                family_scores.append(max((v4.token_similarity(candidate, item["normalized"])
                                          for item in observations if item["family"] == family),
                                         default=0.0))
            strict = sum(value >= 0.78 for value in family_scores)
            consistent = sum(value >= 0.58 for value in family_scores)
            if strict >= 1 and consistent >= 2:
                consistent_occurrences += 1
                aggregate += sum(family_scores)
        ranked_entities.append((candidate, consistent_occurrences, aggregate))
    ranked_entities.sort(key=lambda item: (item[1], item[2]), reverse=True)
    repeated_entity = None
    entity_margin = 0.0
    if ranked_entities:
        if len(ranked_entities) > 1 and ranked_entities[0][1] == ranked_entities[1][1]:
            entity_margin = ranked_entities[0][2] - ranked_entities[1][2]
        else:
            entity_margin = float(ranked_entities[0][1] - (
                ranked_entities[1][1] if len(ranked_entities) > 1 else 0))
        if ranked_entities[0][1] >= 2 and entity_margin >= 0.08:
            repeated_entity = ranked_entities[0][0]
    entity_scores = {candidate: score for candidate, _count, score in ranked_entities}
    entity_occurrences = {candidate: count for candidate, count, _score in ranked_entities}
    for candidates in lattice:
        for row in candidates:
            candidate = row.get("candidate")
            if candidate not in unique_entities:
                continue
            row["cross_occurrence_drug_score"] = round(entity_scores[candidate], 6)
            row["cross_occurrence_drug_count"] = entity_occurrences[candidate]
            row["cross_occurrence_drug_margin"] = round(entity_margin, 6)
            if repeated_entity and candidate == repeated_entity:
                row["emission_score"] = round(float(row["emission_score"]) + 1.65, 6)
                row["cross_occurrence_entity_winner"] = True
            elif repeated_entity:
                row["emission_score"] = round(float(row["emission_score"]) - 0.85, 6)
                row["cross_occurrence_entity_competitor"] = True
            else:
                row["medication_entity_ambiguous"] = True
    if repeated_entity:
        added_entities = [item for item in added_entities if item[1] == repeated_entity]
        for source_slot, _candidate, _drug in added_entities:
            form_slot = source_slot - 1
            while form_slot >= max(0, source_slot - 3) and slot_tokens[form_slot] <= {"و"}:
                form_slot -= 1
            anchors = set().union(*slot_tokens[max(0, form_slot - 3):form_slot])
            if form_slot < 0 or not anchors & {"اون", "آن", "این", "از"}:
                continue
            observations = list(slots[form_slot]["observations"].values())
            similarities = v4.family_similarity("قرص", observations)
            template_families = {family for family, value in similarities.items() if value >= 0.57}
            if len(template_families) < 2:
                continue
            existing = next((row for row in lattice[form_slot]
                             if row.get("candidate") == "قرص"), None)
            probability = sum(max(
                item["probability"] * v4.token_similarity("قرص", item["normalized"])
                for item in observations if item["family"] == family)
                for family in template_families) / len(template_families)
            if existing is None:
                existing = {
                    "candidate": "قرص", "candidate_tokens": ["قرص"],
                    "origin": "medication-template", "dictionary_similarity": max(similarities.values()),
                    "emission_score": round(1.35 * len(template_families)
                                             + 0.75 * probability + 0.8, 6),
                    "exact_families": [], "strong_families": [],
                    "loose_families": sorted(template_families),
                    "family_similarity": similarities, "exact_sources": 0,
                    "general_dictionary_member": True, "general_lexicon": True,
                    "medical_lexicon": True, "modern_spoken": False,
                    "medical_categories": ["drug"],
                    "acoustic_probability": round(probability, 6),
                    "zipf_frequency_fa": round(v4.usage_frequency("قرص"), 4),
                    "observed": False,
                }
                lattice[form_slot].append(existing)
            existing["template_families"] = sorted(template_families)
            existing["medication_template_frame"] = True
            existing["precedes_resolved_medication_entity"] = repeated_entity
            dosage_forms += 1
    else:
        added_entities = []
    for source_slot, candidate, drug in added_entities:
        for slot_index in frame_slots:
            if slot_index == source_slot or any(
                    row.get("candidate") == candidate for row in lattice[slot_index]):
                continue
            observations = list(slots[slot_index]["observations"].values())
            similarities = v4.family_similarity(candidate, observations)
            strict = {family for family, value in similarities.items() if value >= 0.78}
            consistent = {family for family, value in similarities.items() if value >= 0.58}
            if len(strict) < 1 or len(consistent) < 2:
                continue
            probability = sum(max(
                item["probability"] * v4.token_similarity(candidate, item["normalized"])
                for item in observations if item["family"] == family)
                for family in consistent) / len(consistent)
            lattice[slot_index].append({
                "candidate": candidate, "candidate_tokens": [candidate],
                "origin": "repeated-medication-entity", "dictionary_similarity": max(similarities.values()),
                "emission_score": round(1.4 * len(strict) + 0.62 * len(consistent)
                                         + 0.75 * probability + 1.0, 6),
                "exact_families": [], "strong_families": sorted(strict),
                "loose_families": sorted(consistent), "entity_consistency_families": sorted(consistent),
                "family_similarity": similarities, "exact_sources": 0,
                "general_dictionary_member": False, "general_lexicon": False,
                "medical_lexicon": True, "modern_spoken": False,
                "medical_categories": sorted(resolver.medical_categories(drug)),
                "acoustic_probability": round(probability, 6),
                "zipf_frequency_fa": round(v4.usage_frequency(drug), 4),
                "observed": False, "medical_base_term": drug,
                "medication_template_frame": True,
            })
            repeated_added += 1
    return {"medication_template_candidates": template_added,
            "repeated_medication_candidates": repeated_added,
            "dosage_form_templates": dosage_forms,
            "cross_occurrence_drug_resolved": int(repeated_entity is not None)}


def right_context_score(row: dict[str, Any], future: tuple[str, ...],
                        ngrams: v5.DomainNgramEvidence) -> tuple[float, list[dict[str, Any]]]:
    history = tuple(row.get("candidate_tokens") or [])[-2:]
    score = 0.0
    details = []
    for token in future[:2]:
        value, detail = calibrated_transition_score(ngrams, history, token)
        score += max(-0.30, min(2.4, value))
        details.append(detail)
        history = (*history[-1:], token)
    return score, details


def decode_phrase_lattice(lattice: list[list[dict[str, Any]]],
                          ngrams: v5.DomainNgramEvidence,
                          phrase_evidence: PhraseEvidence,
                          resolver: v4.LexiconResolver,
                          beam_size: int = 96) -> tuple[list[int], list[dict[str, Any]]]:
    futures = [future_anchor_tokens(lattice, index) for index in range(len(lattice))]
    right_cache: dict[tuple[int, int], tuple[float, list[dict[str, Any]]]] = {}
    for slot_index, candidates in enumerate(lattice):
        for candidate_index, row in enumerate(candidates):
            right_cache[(slot_index, candidate_index)] = right_context_score(
                row, futures[slot_index], ngrams)

    beam = [{"score": 0.0, "history": tuple(), "path": [], "transitions": []}]
    for slot_index, candidates in enumerate(lattice):
        expanded = []
        for state in beam:
            for candidate_index, row in enumerate(candidates):
                parts = tuple(row.get("candidate_tokens") or [])
                history = state["history"]
                forward_score = 0.0
                redundancy = 0.0
                forward_details = []
                next_history = history
                for part in parts:
                    value, detail = calibrated_transition_score(ngrams, next_history[-2:], part)
                    forward_score += max(-0.50, min(4.5, value))
                    penalty, reason = v4.redundancy_penalty(next_history[-2:], part)
                    redundancy += penalty
                    forward_details.append({"ngram": detail, "redundancy_reason": reason})
                    next_history = (*next_history[-(MAX_PHRASE_ORDER - 2):], part)
                phrase_score, phrase_details = phrase_evidence.trailing_score(history, parts)
                # Phrase evidence comes from the same six ASR hypotheses as the
                # emissions. Cap and scale it so a shared ASR hallucination is not
                # counted repeatedly as independent linguistic confirmation.
                applied_phrase_score = 0.34 * min(4.0, phrase_score)
                entity_score, entity_reasons = entity_structure_score(
                    history, parts, futures[slot_index], row, resolver)
                right_score, right_details = right_cache[(slot_index, candidate_index)]
                acoustic = 0.52 * float(row.get("emission_score") or 0.0)
                family_bonus = 0.55 * effective_family_count(row)
                source_bonus = 0.08 * min(6, int(row.get("exact_sources") or 0))
                drug_categories = set(row.get("medical_categories") or [])
                turbo_component = (3.00 if row.get("turbo_exact_anchor") and
                                   drug_categories & {"drug", "medication", "drug_class"}
                                   else 0.75 if row.get("turbo_exact_anchor")
                                   else 0.12 if float(row.get("turbo_anchor_similarity") or 0.0) >= 0.78
                                   else 0.0)
                lexical_valid = bool(row.get("general_lexicon") or row.get("medical_lexicon")
                                     or row.get("modern_spoken")
                                     or all(is_number(part) for part in parts))
                lexical_component = 0.68 if parts and lexical_valid else -0.44 if parts else 0.0
                repair_fidelity_component = 0.0
                if parts and not row.get("observed"):
                    similarity = float(row.get("dictionary_similarity") or 0.0)
                    repair_fidelity_component = 4.0 * max(0.0, similarity - 0.78)
                grammar_component = 0.0
                if len(parts) == 1 and history and history[-1] in {"می", "نمی"}:
                    joined = history[-1] + parts[0]
                    if v4.general_contains(joined) and v4.usage_frequency(joined) >= 3.0:
                        grammar_component += 1.80
                    elif row.get("general_lexicon") and v4.usage_frequency(joined) == 0.0:
                        grammar_component -= 0.60
                increment = (acoustic + family_bonus + source_bonus + turbo_component
                             + 0.72 * forward_score + applied_phrase_score
                             + 0.34 * right_score + entity_score
                             + lexical_component + repair_fidelity_component
                             + grammar_component + redundancy)
                if not parts:
                    next_history = history
                transition = {
                    "increment": round(increment, 6),
                    "acoustic_component": round(acoustic + family_bonus + source_bonus, 6),
                    "turbo_base_component": round(turbo_component, 6),
                    "forward_ngram_component": round(0.72 * forward_score, 6),
                    "phrase_component": round(applied_phrase_score, 6),
                    "raw_phrase_evidence": round(phrase_score, 6),
                    "right_context_component": round(0.34 * right_score, 6),
                    "entity_component": round(entity_score, 6),
                    "lexical_component": round(lexical_component, 6),
                    "repair_fidelity_component": round(repair_fidelity_component, 6),
                    "grammar_component": round(grammar_component, 6),
                    "redundancy_component": round(redundancy, 6),
                    "forward_details": forward_details,
                    "phrase_details": phrase_details,
                    "right_details": right_details,
                    "entity_reasons": entity_reasons,
                    "future_anchor": list(futures[slot_index]),
                }
                expanded.append({
                    "score": state["score"] + increment,
                    "history": next_history[-(MAX_PHRASE_ORDER - 1):],
                    "path": [*state["path"], candidate_index],
                    "transitions": [*state["transitions"], transition],
                })
        best_by_history: dict[tuple[str, ...], dict[str, Any]] = {}
        for state in expanded:
            current = best_by_history.get(state["history"])
            if current is None or state["score"] > current["score"]:
                best_by_history[state["history"]] = state
        beam = sorted(best_by_history.values(), key=lambda item: item["score"], reverse=True)[:beam_size]
    best = max(beam, key=lambda item: item["score"])
    return best["path"], best["transitions"]


def classify(slots: list[dict[str, Any]], lattice: list[list[dict[str, Any]]],
             path: list[int], transitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected_rows = [lattice[index][choice] for index, choice in enumerate(path)]
    selected: list[dict[str, Any]] = []
    for index, row in enumerate(selected_rows):
        families = effective_family_count(row)
        exact_families = len(row.get("exact_families") or [])
        phrase_cross_family = max((max(
            int(item.get("exact_families") or 0), int(item.get("phonetic_families") or 0))
            for item in transitions[index].get("phrase_details") or []), default=0)
        if not row.get("candidate"):
            observed_families = {item["family"] for item in slots[index]["observations"].values()}
            status = "REVIEW" if len(observed_families) >= 2 else "OMIT"
            reason = "phrase-decoder-epsilon"
        elif exact_families == 3 or (families >= 2 and phrase_cross_family >= 2):
            status, reason = "ACCEPT", "cross-family-phrase-consensus"
        elif families >= 2:
            status, reason = "ACCEPT", "cross-family-word-consensus"
        else:
            status, reason = "REVIEW", "single-family-or-weak-phrase"
        if row.get("medication_entity_ambiguous"):
            status, reason = "REVIEW", "ambiguous-medication-entity"
        alternatives = sorted(lattice[index], key=lambda item: item["emission_score"], reverse=True)
        selected.append({
            **copy.deepcopy(row), "slot": index,
            "start": min(item["start"] for item in slots[index]["observations"].values()),
            "end": max(item["end"] for item in slots[index]["observations"].values()),
            "midpoint": v4.slot_midpoint(slots[index]),
            "status": status, "reason": reason,
            "phrase_cross_family_support": phrase_cross_family,
            "transition": transitions[index],
            "alternatives": [{key: item.get(key) for key in (
                "candidate", "origin", "emission_score", "exact_families", "strong_families",
                "general_lexicon", "medical_lexicon", "zipf_frequency_fa")}
                for item in alternatives[:6]],
            "observations": [{key: item[key] for key in (
                "hypothesis", "family", "source", "word", "normalized",
                "start", "end", "probability")}
                for item in slots[index]["observations"].values()],
        })
    for index, row in enumerate(selected):
        row["sensitive"] = v4.is_sensitive(index, selected)
        if row["sensitive"] and effective_family_count(row) < 2:
            row["status"] = "REVIEW"
            row["reason"] = "sensitive-single-family"
    terminal = next((row for row in reversed(selected) if row.get("candidate")), None)
    if terminal and not (terminal.get("general_lexicon") or terminal.get("medical_lexicon")
                         or terminal.get("modern_spoken")
                         or is_number(terminal.get("candidate") or "")):
        terminal["status"] = "REVIEW"
        terminal["reason"] = "terminal-out-of-lexicon-without-right-context"
        terminal["terminal_oov"] = True
    return selected


def cleanup_entity_fragments(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    operations: list[dict[str, Any]] = []
    for index in range(len(selected) - 2):
        row, following, after = selected[index:index + 3]
        drug = bool(set(row.get("medical_categories") or [])
                    & {"drug", "medication", "drug_class"})
        if (drug and str(row.get("candidate") or "").endswith("تون")
                and following.get("candidate") in {"میتون", "تون"}
                and after.get("candidate") == "نصف"
                and effective_family_count(following) < 2):
            operations.append({"slot": following.get("slot"),
                               "removed": following.get("candidate"),
                               "reason": "duplicate-drug-possessive-fragment"})
            following.update({
                "candidate": "", "candidate_tokens": [], "status": "OMIT",
                "origin": "entity-fragment-cleanup",
                "reason": "duplicate-drug-possessive-fragment",
            })
    for index in range(len(selected) - 2):
        form, conjunction, drug = selected[index:index + 3]
        drug_identity = bool(set(drug.get("medical_categories") or [])
                             & {"drug", "medication", "drug_class"})
        if (form.get("candidate") == "قرص" and conjunction.get("candidate") == "و"
                and drug_identity and drug.get("origin") in {
                    "medication-template", "repeated-medication-entity"}):
            operations.append({"slot": conjunction.get("slot"), "removed": "و",
                               "reason": "spurious-conjunction-inside-medication-entity"})
            conjunction.update({
                "candidate": "", "candidate_tokens": [], "status": "OMIT",
                "origin": "entity-fragment-cleanup",
                "reason": "spurious-conjunction-inside-medication-entity",
            })
    for index in range(len(selected) - 1):
        base, suffix = selected[index:index + 2]
        base_token = str(base.get("candidate") or "")
        suffix_token = str(suffix.get("candidate") or "")
        if (suffix_token in {"تون", "تان", "شون", "شان", "مون", "مان", "ای"}
                and len(base_token) >= 3 and not base_token.endswith(suffix_token)
                and (base.get("general_lexicon") or base.get("medical_lexicon"))):
            joined = base_token + ("‌ای" if suffix_token == "ای" else suffix_token)
            joined_families = sorted({item["family"] for item in base.get("observations") or []
                                      if item.get("normalized") == joined})
            operations.append({"slot": suffix.get("slot"), "removed": suffix_token,
                               "joined_to_slot": base.get("slot"), "result": joined,
                               "reason": "productive-spoken-suffix-join"})
            base.update({
                "candidate": joined, "candidate_tokens": [joined],
                "origin_before_suffix_join": base.get("origin"),
                "origin": "productive-suffix-join", "joined_productive_suffix": True,
                "strong_families": sorted(set(base.get("strong_families") or [])
                                           | set(joined_families)),
                "exact_families": sorted(set(base.get("exact_families") or [])
                                          | set(joined_families)),
                "status": "ACCEPT" if effective_family_count(base) >= 1 else base.get("status"),
                "reason": "productive-spoken-suffix-join",
            })
            suffix.update({
                "candidate": "", "candidate_tokens": [], "status": "OMIT",
                "origin": "productive-suffix-join-continuation",
                "reason": "consumed-by-productive-spoken-suffix-join",
            })
    return operations


def validate_v6(slots: list[dict[str, Any]], sequences: dict[str, list[dict[str, Any]]],
                selected: list[dict[str, Any]], text: str) -> dict[str, Any]:
    """Extend V4's safety checks for explicit, licensed V6 candidate origins."""
    validation = v4.validate(slots, sequences, selected, text)
    by_slot = {int(row["slot"]): row for row in selected}
    unsupported = []
    for item in validation.get("unsupported_candidates") or []:
        row = by_slot.get(int(item["slot"]), {})
        licensed_v6 = (
            row.get("origin") in {"medication-template", "repeated-medication-entity"}
            and bool(row.get("medical_lexicon"))
            and effective_family_count(row) >= 2
        )
        if row.get("joined_productive_suffix"):
            licensed_v6 = True
        if not licensed_v6:
            unsupported.append(item)
    sensitive = []
    for item in validation.get("sensitive_single_family") or []:
        row = by_slot.get(int(item["slot"]), {})
        if row.get("status") != "ACCEPT" or effective_family_count(row) >= 2:
            continue
        sensitive.append(item)
    validation["unsupported_candidates"] = unsupported
    validation["sensitive_single_family"] = sensitive
    validation["checks"]["no_unseen_unlicensed_candidate"] = not unsupported
    validation["checks"]["no_sensitive_single_family_accept"] = not sensitive
    validation["passed"] = all(validation["checks"].values())
    validation["policy_extension"] = (
        "V6 medication-template candidates require a licensed medical entry, "
        "strict route frame, and at least two effective acoustic/entity families."
    )
    return validation


def integer_to_persian_words(value: int) -> str | None:
    """Render common non-negative clinical integers without another dependency."""
    if not 0 <= value <= 9999:
        return None
    ones = ("صفر", "یک", "دو", "سه", "چهار", "پنج", "شش", "هفت", "هشت", "نه")
    teens = {
        10: "ده", 11: "یازده", 12: "دوازده", 13: "سیزده", 14: "چهارده",
        15: "پانزده", 16: "شانزده", 17: "هفده", 18: "هجده", 19: "نوزده",
    }
    tens = {20: "بیست", 30: "سی", 40: "چهل", 50: "پنجاه",
            60: "شصت", 70: "هفتاد", 80: "هشتاد", 90: "نود"}
    hundreds = {100: "صد", 200: "دویست", 300: "سیصد", 400: "چهارصد",
                500: "پانصد", 600: "ششصد", 700: "هفتصد", 800: "هشتصد", 900: "نهصد"}

    def below_thousand(number: int) -> str:
        pieces: list[str] = []
        hundred = (number // 100) * 100
        if hundred:
            pieces.append(hundreds[hundred])
        remainder = number % 100
        if 10 <= remainder <= 19:
            pieces.append(teens[remainder])
        else:
            ten = (remainder // 10) * 10
            one = remainder % 10
            if ten:
                pieces.append(tens[ten])
            if one:
                pieces.append(ones[one])
        return "‌و‌".join(pieces) or ones[0]

    if value < 1000:
        return below_thousand(value)
    thousands, remainder = divmod(value, 1000)
    prefix = "هزار" if thousands == 1 else below_thousand(thousands) + "‌هزار"
    return prefix if not remainder else prefix + "‌و‌" + below_thousand(remainder)


def canonicalize_structured_numbers(selected: list[dict[str, Any]],
                                    resolver: v4.LexiconResolver) -> list[dict[str, Any]]:
    operations: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        token = str(row.get("candidate") or "")
        ascii_token = token.translate(PERSIAN_DIGIT_TRANSLATION)
        if not row.get("structured_numeric_candidate") or not ascii_token.isdigit():
            continue
        nearby: list[str] = []
        for other in selected[max(0, index - 3):min(len(selected), index + 4)]:
            if other is row:
                continue
            nearby.extend(other.get("candidate_tokens") or [])
        structured_frame = (
            any(item in DOSE_CUES | VITAL_CUES | v4.UNITS for item in nearby)
            or any(is_drug(item, resolver) for item in nearby)
        )
        rendered = integer_to_persian_words(int(ascii_token)) if structured_frame else None
        if not rendered:
            continue
        before = token
        row.update({
            "candidate": rendered, "candidate_tokens": [rendered],
            "origin_before_numeric_canonicalization": row.get("origin"),
            "origin": "structured-number-canonicalization",
        })
        operations.append({"slot": index, "from": before, "to": rendered})
    return operations


def placeholder_render(selected: list[dict[str, Any]], resolver: v4.LexiconResolver) -> tuple[str, list[dict[str, Any]]]:
    rendered_rows: list[dict[str, Any]] = []
    placeholders: list[dict[str, Any]] = []
    index = 0
    while index < len(selected):
        row = selected[index]
        weak = (row.get("candidate") and row.get("status") == "REVIEW"
                and (row.get("medication_entity_ambiguous") or row.get("terminal_oov") or (
                    effective_family_count(row) < 2
                    and float(row.get("acoustic_probability") or 0.0) < 0.70)))
        if not weak:
            rendered_rows.append({"candidate_tokens": list(row.get("candidate_tokens") or [])})
            index += 1
            continue
        end = index + 1
        while end < len(selected):
            following = selected[end]
            following_weak = (following.get("candidate") and following.get("status") == "REVIEW"
                              and (following.get("medication_entity_ambiguous")
                                   or following.get("terminal_oov") or (
                                  effective_family_count(following) < 2
                                  and float(following.get("acoustic_probability") or 0.0) < 0.70)))
            if not following_weak:
                break
            end += 1
        span = selected[index:end]
        medical = any(
            set(item.get("medical_categories") or []) & {"drug", "medication", "drug_class"}
            or is_drug(item.get("candidate") or "", resolver) for item in span)
        sensitive = any(item.get("sensitive") for item in span)
        label = "[نام دارو نامفهوم]" if medical else "[مقدار نامفهوم]" if sensitive else "[نامفهوم]"
        rendered_rows.append({"candidate_tokens": [label]})
        placeholders.append({
            "start_slot": index, "end_slot": end - 1, "placeholder": label,
            "original_candidates": [item.get("candidate") for item in span],
            "reason": "consecutive-low-acoustic-single-family-phrase",
        })
        index = end
    text, _operations = v4.render(rendered_rows)
    return text, placeholders


def write_outputs(run_dir: Path, selected: list[dict[str, Any]], final_text: str,
                  raw_selected_text: str, placeholders: list[dict[str, Any]],
                  protected_names: list[dict[str, Any]], validation: dict[str, Any],
                  corpus: v5.DomainCorpus, ngrams: v5.DomainNgramEvidence,
                  elapsed: float, dose_locks: list[dict[str, Any]],
                  semantic_candidate_stats: dict[str, int],
                  numeric_canonicalizations: list[dict[str, Any]]) -> dict[str, Any]:
    out_dir = run_dir / OUTPUT_RELATIVE
    out_dir.mkdir(parents=True, exist_ok=True)
    review = [row for row in selected if row.get("status") == "REVIEW"]
    clips, clip_limit = v4.make_review_clips(run_dir, out_dir, review)
    payload = {
        "algorithm": "v6 deterministic cross-family phrase lattice with bidirectional context",
        "llm_used": False, "pretrained_generator_used": False,
        "old_pipeline_modified": False, "output_replaces_v5": False,
        "candidate_policy": "six ASR observations plus licensed general/medical lexicon candidates only",
        "family_policy": "raw/enhanced collapse to one vote per model family",
        "context_policy": "2-5 token cross-family phrases, forward n-grams, fixed future anchors",
        "uncertainty_policy": "low-acoustic single-family spans become explicit placeholders",
        "runtime_seconds": round(elapsed, 3), "text": final_text,
        "raw_selected_text": raw_selected_text,
        "domain_detection": ngrams.domain_detection,
        "domain_corpus": corpus.describe(),
        "dose_locks": dose_locks, "uncertainty_placeholders": placeholders,
        "semantic_candidate_stats": semantic_candidate_stats,
        "numeric_canonicalizations": numeric_canonicalizations,
        "protected_name_slots": protected_names,
        "hard_validation": validation, "slots": selected,
    }
    (out_dir / "final-v6.txt").write_text(final_text + "\n", encoding="utf-8")
    (out_dir / "final-v6.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "review-v6.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "review-clips-v6.json").write_text(
        json.dumps(clips, ensure_ascii=False, indent=2), encoding="utf-8")
    review_lines = [
        "# بازبینی موتور عبارتی نسخهٔ ۶", "",
        "این خروجی بدون LLM ساخته شده و جایگزین خروجی قبلی نیست.", "",
        "| زمان | انتخاب | دلیل | خانواده‌های پشتیبان | پشتیبانی عبارت |",
        "|---:|---|---|---|---:|",
    ]
    for row in review:
        review_lines.append(
            f"| {row['midpoint']:.2f} | {row.get('candidate') or 'ε'} | {row['reason']} | "
            f"{', '.join(row.get('strong_families') or []) or '—'} | "
            f"{row.get('phrase_cross_family_support', 0)} |")
    review_lines += ["", "## جای‌خالی‌های عدم قطعیت", ""]
    for item in placeholders:
        review_lines.append(
            f"- اسلات {item['start_slot']} تا {item['end_slot']}: {item['placeholder']} "
            f"(به‌جای: {'، '.join(item['original_candidates'])})")
    if clip_limit:
        review_lines += ["", "تعداد بازه‌های بازبینی بیش از سقف کلیپ بود."]
    (out_dir / "review-v6.md").write_text("\n".join(review_lines) + "\n", encoding="utf-8")
    previous = run_dir / v5.OUTPUT_RELATIVE / "final-v5.txt"
    comparison = ["# مقایسهٔ مستقل V5 و V6", ""]
    if previous.is_file():
        comparison += ["## V5 قبلی — بدون تغییر", "", previous.read_text(encoding="utf-8").strip(), ""]
    comparison += ["## V6 عبارتی — بدون LLM", "", final_text, ""]
    (out_dir / "comparison-v6.md").write_text("\n".join(comparison), encoding="utf-8")
    summary = {
        "runtime_seconds": round(elapsed, 3), "review_count": len(review),
        "placeholder_count": len(placeholders), "protected_name_count": len(protected_names),
        "dose_lock_count": len(dose_locks), "hard_validation_passed": validation["passed"],
        "semantic_candidate_stats": semantic_candidate_stats,
        "numeric_canonicalization_count": len(numeric_canonicalizations),
        "llm_used": False, "pretrained_generator_used": False,
        "old_pipeline_modified": False, "text": final_text,
    }
    (out_dir / "summary-v6.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output": str(out_dir), **summary}


def run(run_dir: Path, medical_index: Path, corpus_index: Path) -> dict[str, Any]:
    started = time.perf_counter()
    hypotheses = load_hypotheses(run_dir)
    sequences = {key: words_of(hypotheses[key], key) for key in NETWORK_ORDER}
    medical_payload = json.loads(medical_index.read_text(encoding="utf-8"))
    resolver = v4.LexiconResolver(medical_payload)
    corpus = v5.DomainCorpus(corpus_index)
    try:
        ngrams = v5.DomainNgramEvidence(sequences, resolver, corpus)
        phrase_evidence = PhraseEvidence(sequences)
        slots: list[dict[str, Any]] = []
        for hypothesis in NETWORK_ORDER:
            slots = v4.add_sequence_to_network(slots, sequences[hypothesis], hypothesis)
        lattice = [v4.build_slot_candidates(slot, resolver, ngrams) for slot in slots]
        v5.augment_adjacent_acoustic_support(slots, lattice)
        semantic_candidate_stats = augment_semantic_candidates(slots, lattice, resolver, ngrams)
        semantic_candidate_stats.update(augment_medication_frames(slots, lattice, resolver, ngrams))
        mark_turbo_anchors(slots, lattice)
        lattice = prune_lattice(lattice)
        path, transitions = decode_phrase_lattice(lattice, ngrams, phrase_evidence, resolver)
        selected = classify(slots, lattice, path, transitions)
        entity_fragment_cleanups = cleanup_entity_fragments(selected)
        semantic_candidate_stats["entity_fragment_cleanups"] = len(entity_fragment_cleanups)
        doses = cluster_doses(dose_occurrences(sequences))
        dose_locks = v4.apply_dose_locks(selected, doses)
        numeric_canonicalizations = canonicalize_structured_numbers(selected, resolver)
        raw_selected_text, _render_operations = v4.render(selected)
        validation = validate_v6(slots, sequences, selected, raw_selected_text)
        placeholder_text, placeholders = placeholder_render(selected, resolver)
        final_text, protected_names = v5.protect_honorific_names(placeholder_text)
        return write_outputs(
            run_dir, selected, final_text, raw_selected_text, placeholders,
            protected_names, validation, corpus, ngrams,
            time.perf_counter() - started, dose_locks, semantic_candidate_stats,
            numeric_canonicalizations)
    finally:
        corpus.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independent deterministic phrase-semantic six-ASR decoder; no LLM.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--medical-index", type=Path, required=True)
    parser.add_argument("--corpus-index", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.run_dir.resolve(), args.medical_index.resolve(), args.corpus_index.resolve())
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
