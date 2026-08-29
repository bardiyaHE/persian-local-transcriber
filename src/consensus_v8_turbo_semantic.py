from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import consensus_v4 as v4
import consensus_v5 as v5
import consensus_v6_phrase_semantic as v6
import consensus_v7_semantic_agent as v7
from consensus_v3 import cluster_doses, dose_occurrences, load_hypotheses, words_of
from semantic_encoder_onnx import OnnxSentenceEncoder


OUTPUT_RELATIVE = Path("final-delivery") / "08-turbo-first-minilm"
TURBO_ENHANCED = "large-v3-turbo__enhanced"
TURBO_RAW = "large-v3-turbo__raw"
TURBO_FAMILY = "large-v3-turbo"
TURBO_FIRST_ORDER = (
    TURBO_ENHANCED, TURBO_RAW,
    "large-v3__enhanced", "large-v3__raw",
    "medium__enhanced", "medium__raw",
)
CRITICAL_MEDICAL_CATEGORIES = {"drug", "medication", "drug_class", "medical_device"}


def candidate_token(row: dict[str, Any]) -> str:
    return str(row.get("candidate") or "")


def is_critical_candidate(row: dict[str, Any], resolver: v4.LexiconResolver) -> bool:
    token = candidate_token(row)
    categories = set(row.get("medical_categories") or resolver.medical_categories(token))
    return (v6.is_number(token) or token in v4.NUMBER_WORDS
            or token in v4.UNITS or v4.is_negative_token(token)
            or bool(categories & CRITICAL_MEDICAL_CATEGORIES))


def observation_quality(row: dict[str, Any] | None, prefer_enhanced: bool = False) -> float:
    if not row:
        return -10.0
    probability = float(row.get("probability") or 0.0)
    avg_logprob = max(-2.0, min(0.0, float(row.get("avg_logprob") or -2.0)))
    return probability + 0.18 * (avg_logprob + 2.0) / 2.0 + (0.025 if prefer_enhanced else 0.0)


def find_exact_candidate(candidates: list[dict[str, Any]], token: str) -> int | None:
    for index, row in enumerate(candidates):
        if candidate_token(row) == token:
            return index
    return None


def preserve_exact_turbo_candidate(
        slot: dict[str, Any], candidates: list[dict[str, Any]],
        resolver: v4.LexiconResolver, ngrams: v5.DomainNgramEvidence,
) -> None:
    """Restore an exact observed Turbo surface removed by V4 vocabulary pruning."""
    enhanced = slot["observations"].get(TURBO_ENHANCED)
    raw = slot["observations"].get(TURBO_RAW)
    preferred = (enhanced if observation_quality(enhanced, True)
                 >= observation_quality(raw) else raw)
    token = str((preferred or {}).get("normalized") or "")
    if not token or find_exact_candidate(candidates, token) is not None:
        return
    observations = list(slot["observations"].values())
    similarities = v4.family_similarity(token, observations)
    exact_families = {row["family"] for row in observations if row["normalized"] == token}
    strong_families = {family for family, score in similarities.items() if score >= 0.78}
    loose_families = {family for family, score in similarities.items() if score >= 0.66}
    probability_by_family = []
    for family in strong_families:
        matching = [row for row in observations if row["family"] == family]
        if matching:
            probability_by_family.append(max(
                float(row.get("probability") or 0.0)
                * v4.token_similarity(token, row["normalized"])
                for row in matching))
    probability = sum(probability_by_family) / max(1, len(probability_by_family))
    exact_sources = sum(row["normalized"] == token for row in observations)
    repetition_families, repetition_count = ngrams.token_repetition(token)
    general_member = v4.general_contains(token)
    general = v4.active_general_word(token)
    medical = resolver.is_medical(token)
    categories = resolver.medical_categories(token)
    modern = token in v4.MODERN_SPOKEN
    frequency = v4.usage_frequency(token)
    fuzzy_only = max(0, len(strong_families) - len(exact_families))
    medical_bonus = (0.78 if categories & {"drug", "medication", "drug_class"}
                     else 0.38 if medical else 0.0)
    score = (
        2.20 * len(exact_families) + 1.45 * fuzzy_only
        + 0.26 * max(0, len(loose_families) - len(strong_families))
        + 0.16 * exact_sources + 0.42 * probability
        + 0.32 * repetition_families + 0.11 * math.log1p(repetition_count)
        + (0.50 if general else 0.0) + medical_bonus
        + (0.22 if general and medical else 0.0)
        + (0.30 if modern else 0.0) + 0.055 * min(6.0, frequency) + 0.10
    )
    candidates.append({
        "candidate": token, "candidate_tokens": [token],
        "origin": "turbo-observed-preserved", "dictionary_similarity": 1.0,
        "emission_score": round(score, 6),
        "exact_families": sorted(exact_families),
        "strong_families": sorted(strong_families),
        "loose_families": sorted(loose_families),
        "family_similarity": similarities, "exact_sources": exact_sources,
        "repetition_families": repetition_families,
        "repetition_count": repetition_count,
        "general_dictionary_member": general_member,
        "general_lexicon": general, "medical_lexicon": medical,
        "modern_spoken": modern, "medical_categories": sorted(categories),
        "acoustic_probability": round(probability, 6),
        "zipf_frequency_fa": round(frequency, 4), "observed": True,
        "turbo_surface_preserved": True,
    })


def interval_reasons(run_dir: Path, midpoint: float) -> list[str]:
    plan_path = run_dir / "adaptive-turbo-plan.json"
    if not plan_path.is_file():
        return []
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    reasons: set[str] = set()
    for row in plan.get("review_intervals") or []:
        if float(row["start"]) <= midpoint <= float(row["end"]):
            reasons.update(str(value) for value in row.get("reasons") or [])
    return sorted(reasons)


def build_turbo_base(run_dir: Path, slots: list[dict[str, Any]],
                     lattice: list[list[dict[str, Any]]],
                     resolver: v4.LexiconResolver) -> tuple[list[int], list[dict[str, Any]]]:
    """Choose a Turbo-only base path and label the slots MiniLM may inspect."""
    path: list[int] = []
    audit: list[dict[str, Any]] = []
    for slot_index, (slot, candidates) in enumerate(zip(slots, lattice)):
        enhanced = slot["observations"].get(TURBO_ENHANCED)
        raw = slot["observations"].get(TURBO_RAW)
        enhanced_token = str((enhanced or {}).get("normalized") or "")
        raw_token = str((raw or {}).get("normalized") or "")
        if enhanced and raw and enhanced_token == raw_token:
            chosen_observation = enhanced
            source_reason = "turbo-raw-enhanced-agree"
        elif observation_quality(enhanced, True) >= observation_quality(raw):
            chosen_observation = enhanced
            source_reason = "turbo-enhanced-higher-quality"
        else:
            chosen_observation = raw
            source_reason = "turbo-raw-higher-quality"
        chosen_token = str((chosen_observation or {}).get("normalized") or "")
        choice = find_exact_candidate(candidates, chosen_token)
        if choice is None:
            choice = find_exact_candidate(candidates, "")
        if choice is None:
            # This should be unreachable because the V6 lattice explicitly
            # represents epsilon.  Keep the failure deterministic if a future
            # lattice implementation changes that invariant.
            choice = min(range(len(candidates)), key=lambda index: (
                bool(candidate_token(candidates[index])),
                float(candidates[index].get("emission_score") or 0.0),
            ))
        base = candidates[choice]
        reasons: set[str] = set(interval_reasons(run_dir, float(base.get("midpoint") or 0.0)))
        if not enhanced and not raw:
            reasons.add("turbo-missing")
        if not enhanced or not raw:
            reasons.add("one-turbo-source-missing")
        if enhanced and raw and enhanced_token != raw_token:
            similarity = v4.token_similarity(enhanced_token, raw_token)
            if similarity < 0.94:
                reasons.add("turbo-raw-enhanced-disagreement")
        probability = float((chosen_observation or {}).get("probability") or 0.0)
        avg_logprob = float((chosen_observation or {}).get("avg_logprob") or -2.0)
        if chosen_observation and probability < 0.64:
            reasons.add("low-turbo-word-probability")
        if chosen_observation and avg_logprob < -0.70:
            reasons.add("low-turbo-segment-logprob")
        token = candidate_token(base)
        lexical = bool(base.get("general_lexicon") or base.get("medical_lexicon")
                       or base.get("modern_spoken") or token in v4.FUNCTION_WORDS
                       or v6.is_number(token))
        if token in v4.USER_BLOCKLIST:
            reasons.add("turbo-user-blocklist")
        if token and not lexical:
            reasons.add("turbo-out-of-lexicon")
        critical = is_critical_candidate(base, resolver)
        if critical:
            reasons.add("critical-medical-number-or-negation")

        # A confident ordinary Turbo word is immutable.  On an uncertain slot
        # it still receives a prior, so an alternative needs substantial
        # independent and semantic evidence.
        if reasons:
            base_prior = 4.0 if critical and enhanced_token == raw_token else 1.8
            if token in v4.USER_BLOCKLIST:
                base_prior = 0.0
            if not token:
                base_prior = 0.0
            if base_prior:
                before = float(base.get("emission_score") or 0.0)
                base["emission_score_before_turbo_first_prior"] = round(before, 6)
                base["emission_score"] = round(before + base_prior, 6)
                base["turbo_first_base_prior"] = base_prior
        path.append(choice)
        audit.append({
            "slot": slot_index,
            "turbo_base_candidate": token,
            "turbo_source_reason": source_reason,
            "turbo_enhanced": enhanced_token,
            "turbo_raw": raw_token,
            "selected_probability": round(probability, 6),
            "selected_avg_logprob": round(avg_logprob, 6),
            "critical": critical,
            "uncertain": bool(reasons),
            "uncertainty_reasons": sorted(reasons),
            "turbo_locked": not reasons,
        })
    return path, audit


def independent_candidate(row: dict[str, Any]) -> bool:
    exact = set(row.get("exact_families") or [])
    strong = set(row.get("strong_families") or [])
    families = exact | strong
    return len(families) >= 2 and any(family != TURBO_FAMILY for family in families)


def mark_critical_independent_checks(
        lattice: list[list[dict[str, Any]]], turbo_path: list[int],
        turbo_audit: list[dict[str, Any]], resolver: v4.LexiconResolver,
) -> list[dict[str, Any]]:
    """Open a valid Turbo word when two independent families report a critical value."""
    audits: list[dict[str, Any]] = []
    turbo_tokens = [candidate_token(lattice[index][choice])
                    for index, choice in enumerate(turbo_path)]
    for slot_index, candidates in enumerate(lattice):
        base = candidates[turbo_path[slot_index]]
        base_token = candidate_token(base)
        nearby = set(turbo_tokens[max(0, slot_index - 3):slot_index + 4])
        clinical_frame = bool(nearby & (v6.VITAL_CUES | v6.DOSE_CUES | v6.TIME_CUES
                                        | v4.UNITS | {"بالای", "پایین", "نصف"}))
        if not clinical_frame:
            clinical_frame = any(
                resolver.medical_categories(token) & CRITICAL_MEDICAL_CATEGORIES
                for token in nearby if token)
        if not clinical_frame:
            continue
        eligible = []
        for row in candidates:
            token = candidate_token(row)
            if row is base or not token or not row.get("observed"):
                continue
            exact_families = set(row.get("exact_families") or [])
            if len(exact_families) < 2 or TURBO_FAMILY in exact_families:
                continue
            if not (v6.is_number(token) or token in v4.NUMBER_WORDS
                    or token in v4.UNITS or v4.is_negative_token(token)):
                continue
            eligible.append(row)
        if not eligible:
            continue
        winner = max(eligible, key=lambda row: (
            len(set(row.get("exact_families") or [])),
            float(row.get("acoustic_probability") or 0.0),
            float(row.get("emission_score") or 0.0),
        ))
        before = float(winner.get("emission_score") or 0.0)
        prior = 5.0
        winner["emission_score_before_critical_verification_prior"] = round(before, 6)
        winner["emission_score"] = round(before + prior, 6)
        winner["turbo_critical_verification_prior"] = prior
        if turbo_audit[slot_index]["turbo_locked"]:
            turbo_audit[slot_index]["turbo_locked"] = False
            turbo_audit[slot_index]["uncertain"] = True
            turbo_audit[slot_index]["uncertainty_reasons"].append(
                "two-independent-families-report-critical-value")
            base_before = float(base.get("emission_score") or 0.0)
            base["emission_score_before_turbo_first_prior"] = round(base_before, 6)
            base["emission_score"] = round(base_before + 1.8, 6)
            base["turbo_first_base_prior"] = 1.8
        audits.append({
            "slot": slot_index, "from": base_token,
            "favored_candidate": candidate_token(winner),
            "reason": "two-independent-families-critical-context",
            "prior": prior,
            "exact_families": sorted(winner.get("exact_families") or []),
            "nearby_turbo_tokens": sorted(nearby),
        })
    return audits


def apply_turbo_oov_minilm_repairs(
        slots: list[dict[str, Any]], lattice: list[list[dict[str, Any]]],
        turbo_path: list[int], turbo_audit: list[dict[str, Any]],
        agent: v7.SemanticRetrievalAgent, resolver: v4.LexiconResolver,
) -> list[dict[str, Any]]:
    """Use MiniLM plus corpus context to repair only close non-critical Turbo OOVs."""
    baseline = [lattice[index][choice] for index, choice in enumerate(turbo_path)]
    audits: list[dict[str, Any]] = []
    spoken_clitics = ("تون", "تان", "شون", "شان", "مون", "مان")

    def unigram_count(token: str) -> int:
        exact = int(agent.corpus.connection.execute(
            "SELECT COALESCE(SUM(count),0) FROM unigrams WHERE w1=?", (token,)
        ).fetchone()[0] or 0)
        suffix = next((value for value in spoken_clitics
                       if token.endswith(value) and len(token) > len(value) + 1), "")
        if not suffix:
            return exact
        stem = token[:-len(suffix)]
        stem_count = int(agent.corpus.connection.execute(
            "SELECT COALESCE(SUM(count),0) FROM unigrams WHERE w1=?", (stem,)
        ).fetchone()[0] or 0)
        return max(exact, stem_count)
    for slot_index, candidates in enumerate(lattice):
        reasons = turbo_audit[slot_index]["uncertainty_reasons"]
        base_blocklisted = "turbo-user-blocklist" in reasons
        if "turbo-out-of-lexicon" not in reasons and not base_blocklisted:
            continue
        base = baseline[slot_index]
        base_token = candidate_token(base)
        if not base_token or is_critical_candidate(base, resolver):
            continue
        base_clitic = next((suffix for suffix in spoken_clitics
                            if base_token.endswith(suffix)), "")
        alternatives = []
        for row in candidates:
            token = candidate_token(row)
            exact_families = set(row.get("exact_families") or [])
            independent_blocklist_observation = bool(
                base_blocklisted and row.get("observed")
                and int(row.get("exact_sources") or 0) >= 2
                and any(family != TURBO_FAMILY for family in exact_families))
            if (row is base or not token or token in v4.USER_BLOCKLIST
                    or not (independent_candidate(row)
                            or independent_blocklist_observation)
                    or is_critical_candidate(row, resolver)
                    or len(row.get("candidate_tokens") or []) != 1
                    or not (row.get("general_lexicon") or row.get("medical_lexicon")
                            or row.get("modern_spoken"))
                    or (not base_blocklisted
                        and v4.token_similarity(base_token, token) < 0.78)
                    or (base_clitic and not token.endswith(base_clitic))):
                continue
            alternatives.append(row)
        if not alternatives:
            continue
        left, right = max(0, slot_index - 6), min(len(lattice), slot_index + 7)
        prefix = [candidate_token(row) for row in baseline[left:slot_index]
                  if candidate_token(row)]
        suffix = [candidate_token(row) for row in baseline[slot_index + 1:right]
                  if candidate_token(row)]
        rows = [base, *alternatives[:6]]
        variants = [v7.text_from_parts([*prefix, candidate_token(row), *suffix], limit=40)
                    for row in rows]
        analyses = agent.score_texts(variants)
        previous_token = prefix[-1] if prefix else ""
        following_token = suffix[0] if suffix else ""

        def corpus_context(token: str) -> tuple[float, dict[str, int]]:
            left_count = int(agent.corpus.connection.execute(
                "SELECT COALESCE(SUM(count),0) FROM bigrams WHERE w1=? AND w2=?",
                (previous_token, token),
            ).fetchone()[0] or 0) if previous_token else 0
            right_count = int(agent.corpus.connection.execute(
                "SELECT COALESCE(SUM(count),0) FROM bigrams WHERE w1=? AND w2=?",
                (token, following_token),
            ).fetchone()[0] or 0) if following_token else 0
            trigram_count = int(agent.corpus.connection.execute(
                "SELECT COALESCE(SUM(count),0) FROM trigrams WHERE w1=? AND w2=? AND w3=?",
                (previous_token, token, following_token),
            ).fetchone()[0] or 0) if previous_token and following_token else 0
            value = (4.0 * math.log1p(trigram_count)
                     + 1.5 * min(math.log1p(left_count), math.log1p(right_count))
                     + 0.5 * max(math.log1p(left_count), math.log1p(right_count)))
            return value, {"left_bigram": left_count, "right_bigram": right_count,
                           "trigram": trigram_count}

        base_context, base_counts = corpus_context(base_token)
        base_semantic = float(analyses[0]["semantic_score"])
        base_unigram = unigram_count(base_token)
        ranked = []
        for position, row in enumerate(alternatives[:6], 1):
            token = candidate_token(row)
            context, counts = corpus_context(token)
            unigram = unigram_count(token)
            semantic = float(analyses[position]["semantic_score"])
            similarity = v4.token_similarity(base_token, token)
            score = (0.32 * semantic + 0.24 * min(1.0, context / 8.0)
                     + 0.22 * min(1.0, math.log1p(unigram) / 8.0)
                     + 0.22 * similarity)
            ranked.append((score, row, semantic, context, counts, unigram, similarity))
        ranked.sort(key=lambda item: item[0], reverse=True)
        medical_ranked = [item for item in ranked if item[1].get("medical_lexicon")]
        if medical_ranked:
            ranked = medical_ranked
        winner_score, winner, winner_semantic, winner_context, winner_counts, winner_unigram, similarity = ranked[0]
        winner_medical = bool(winner.get("medical_lexicon"))
        context_improvement = winner_context > base_context + 0.05
        # Close licensed medical repairs can be slightly less similar at the
        # sentence-vector level because MiniLM is intentionally insensitive to
        # a one-token spelling change.  MiniLM still selects among the repair
        # variants; the wider floor only applies to a zero-corpus Turbo OOV.
        semantic_floor = base_semantic - (0.25 if winner_medical else -0.01)
        if base_blocklisted:
            safe_blocklist_repair = bool(
                winner_unigram >= 20
                and winner_semantic >= base_semantic - 0.12
                and (context_improvement or winner_unigram >= base_unigram + 20))
            if not safe_blocklist_repair:
                continue
        elif (base_unigram > 0 or winner_unigram == 0 or similarity < 0.80
              or winner_semantic < semantic_floor
              or not (winner_medical or context_improvement)):
            continue
        before = float(winner.get("emission_score") or 0.0)
        prior = min(7.0, 6.2 + 0.2 * min(4.0, max(0.0, winner_context - base_context)))
        winner["emission_score_before_turbo_oov_prior"] = round(before, 6)
        winner["emission_score"] = round(before + prior, 6)
        winner["turbo_oov_minilm_prior"] = round(prior, 6)
        audits.append({
            "slot": slot_index, "from": base_token,
            "favored_candidate": candidate_token(winner),
            "reason": ("turbo-blocklist-minilm-corpus-repair" if base_blocklisted
                       else "turbo-oov-minilm-dictionary-repair"),
            "prior": round(prior, 6),
            "semantic_before": round(base_semantic, 6),
            "semantic_favored": round(winner_semantic, 6),
            "corpus_context_before": base_counts,
            "corpus_context_favored": winner_counts,
            "unigram_before": base_unigram,
            "unigram_favored": winner_unigram,
            "surface_similarity": round(similarity, 6),
            "combined_score_favored": round(winner_score, 6),
        })
    return audits


def constrain_turbo_first(
        lattice: list[list[dict[str, Any]]], turbo_path: list[int],
        turbo_audit: list[dict[str, Any]], local_priors: list[dict[str, Any]],
        resolver: v4.LexiconResolver,
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    """Expose alternatives only at audited Turbo uncertainty slots."""
    favored_by_slot = {
        int(row["slot"]): str(row.get("favored_candidate") or "")
        for row in local_priors
    }
    constrained: list[list[dict[str, Any]]] = []
    gate_audit: list[dict[str, Any]] = []
    for slot_index, candidates in enumerate(lattice):
        base = candidates[turbo_path[slot_index]]
        base_token = candidate_token(base)
        uncertainty = turbo_audit[slot_index]
        if uncertainty["turbo_locked"]:
            constrained.append([base])
            gate_audit.append({
                "slot": slot_index, "turbo_base_candidate": base_token,
                "allowed_candidates": [base_token], "rejected_candidates": [],
                "unresolved_critical_alternative": False,
                "reason": "high-confidence-turbo-locked",
            })
            continue

        favored = favored_by_slot.get(slot_index, "")
        base_blocklisted = base_token in v4.USER_BLOCKLIST
        base_critical = is_critical_candidate(base, resolver)
        base_lexical = bool(base.get("general_lexicon") or base.get("medical_lexicon")
                            or base.get("modern_spoken"))
        allowed = [] if base_blocklisted else [base]
        rejected: list[dict[str, str]] = []
        for row in candidates:
            if row is base:
                continue
            token = candidate_token(row)
            if not token:
                rejected.append({"candidate": token, "reason": "non-base-epsilon"})
                continue
            exact_families = set(row.get("exact_families") or [])
            independent_blocklist_observation = bool(
                base_blocklisted and row.get("observed")
                and int(row.get("exact_sources") or 0) >= 2
                and any(family != TURBO_FAMILY for family in exact_families))
            if not independent_candidate(row) and not independent_blocklist_observation:
                rejected.append({"candidate": token, "reason": "no-independent-family"})
                continue
            lexical = bool(row.get("general_lexicon") or row.get("medical_lexicon")
                           or row.get("modern_spoken") or v6.is_number(token))
            if not lexical:
                rejected.append({"candidate": token, "reason": "unlicensed-token"})
                continue
            tokens = tuple(row.get("candidate_tokens") or [])
            base_tokens = tuple(base.get("candidate_tokens") or [])
            if base_tokens and len(tokens) != len(base_tokens):
                rejected.append({"candidate": token, "reason": "token-count-change"})
                continue
            similarity = v4.token_similarity(base_token, token) if base_token else 1.0
            critical_exact_alternative = bool(
                len(exact_families) >= 2 and row.get("observed")
                and (v6.is_number(token) or token in v4.NUMBER_WORDS or token in v4.UNITS
                     or v4.is_negative_token(token)))
            if (base_token and not base_blocklisted and base_lexical and similarity < 0.72
                    and not critical_exact_alternative):
                rejected.append({"candidate": token, "reason": "too-far-from-valid-turbo"})
                continue
            row_critical = is_critical_candidate(row, resolver)
            if base_critical or row_critical:
                categories = set(
                    row.get("medical_categories") or resolver.medical_categories(token))
                licensed_entity = bool(
                    categories & CRITICAL_MEDICAL_CATEGORIES
                    and (row.get("medical_lexicon") or row.get("general_lexicon")))
                observed_number_or_negation = bool(
                    row.get("observed") and (v6.is_number(token)
                                             or token in v4.NUMBER_WORDS
                                             or token in v4.UNITS
                                             or v4.is_negative_token(token)))
                if len(exact_families) < 2 or not (licensed_entity or observed_number_or_negation):
                    rejected.append({"candidate": token,
                                     "reason": "critical-needs-two-exact-families"})
                    continue
            semantic_favored = token == favored
            exact_consensus = len(set(row.get("exact_families") or [])) >= 2
            strict_medical_rescue = bool(
                set(row.get("medical_categories") or resolver.medical_categories(token))
                & CRITICAL_MEDICAL_CATEGORIES
                and row.get("origin") == "general-lexicon"
                and float(row.get("dictionary_similarity") or 0.0) >= 0.84)
            if (base_token and not base_blocklisted and similarity < 0.72 and not semantic_favored
                    and not critical_exact_alternative):
                rejected.append({"candidate": token,
                                 "reason": "far-consensus-needs-minilm-favor"})
                continue
            if not (semantic_favored or exact_consensus or strict_medical_rescue):
                rejected.append({"candidate": token,
                                 "reason": "not-semantic-or-two-family-consensus"})
                continue
            allowed.append(row)
        if not allowed:
            epsilon = next((row for row in candidates if not candidate_token(row)), None)
            if epsilon is None:
                raise AssertionError("A blocklisted Turbo slot has no safe alternative or epsilon.")
            allowed.append(epsilon)
        constrained.append(allowed)
        unresolved_critical = any(
            row is not base and is_critical_candidate(row, resolver) and row not in allowed
            for row in candidates)
        gate_audit.append({
            "slot": slot_index,
            "turbo_base_candidate": base_token,
            "allowed_candidates": [candidate_token(row) for row in allowed],
            "rejected_candidates": rejected,
            "unresolved_critical_alternative": unresolved_critical,
            "reason": ("uncertain-turbo-with-independent-alternatives"
                       if len(allowed) > 1 else "uncertain-turbo-no-safe-alternative"),
        })
    return constrained, gate_audit


def write_outputs(
        run_dir: Path, selected: list[dict[str, Any]], final_text: str,
        raw_selected_text: str, placeholders: list[dict[str, Any]],
        protected_names: list[dict[str, Any]], validation: dict[str, Any],
        corpus: v5.DomainCorpus, ngrams: v5.DomainNgramEvidence,
        encoder: OnnxSentenceEncoder, agent: v7.SemanticRetrievalAgent,
        semantic_decode: dict[str, Any], elapsed: float,
        dose_locks: list[dict[str, Any]], semantic_candidate_stats: dict[str, int],
        numeric_canonicalizations: list[dict[str, Any]],
        path_changes: list[dict[str, Any]], local_semantic_priors: list[dict[str, Any]],
        turbo_audit: list[dict[str, Any]], change_gate_audit: list[dict[str, Any]],
) -> dict[str, Any]:
    out_dir = run_dir / OUTPUT_RELATIVE
    out_dir.mkdir(parents=True, exist_ok=True)
    review = [row for row in selected if row.get("status") == "REVIEW"]
    clips, _clip_limit = v4.make_review_clips(run_dir, out_dir, review)
    adaptive_plan_path = run_dir / "adaptive-turbo-plan.json"
    adaptive_plan = (json.loads(adaptive_plan_path.read_text(encoding="utf-8"))
                     if adaptive_plan_path.is_file() else None)
    locked_count = sum(bool(row["turbo_locked"]) for row in turbo_audit)
    uncertain_count = len(turbo_audit) - locked_count
    payload = {
        "algorithm": "v8 Turbo-first adaptive ASR plus local non-generative MiniLM",
        "generative_llm_used": False,
        "pretrained_semantic_encoder_used": True,
        "encoder_generates_text": False,
        "external_api_used_at_runtime": False,
        "turbo_is_immutable_default": True,
        "candidate_policy": (
            "MiniLM may rerank only uncertain Turbo slots with independent acoustic support"
        ),
        "runtime_seconds": round(elapsed, 3),
        "text": final_text,
        "raw_selected_text": raw_selected_text,
        "encoder": encoder.describe(),
        "semantic_agent": agent.describe(),
        "semantic_decode": semantic_decode,
        "adaptive_turbo_plan": adaptive_plan,
        "turbo_base_audit": turbo_audit,
        "change_gate_audit": change_gate_audit,
        "local_semantic_priors": local_semantic_priors,
        "path_changes_from_turbo_base": path_changes,
        "domain_detection": ngrams.domain_detection,
        "domain_corpus": corpus.describe(),
        "dose_locks": dose_locks,
        "uncertainty_placeholders": placeholders,
        "semantic_candidate_stats": semantic_candidate_stats,
        "numeric_canonicalizations": numeric_canonicalizations,
        "protected_name_slots": protected_names,
        "hard_validation": validation,
        "slots": selected,
    }
    (out_dir / "final-v8.txt").write_text(final_text + "\n", encoding="utf-8")
    (out_dir / "final-v8.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "review-v8.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "review-clips-v8.json").write_text(
        json.dumps(clips, ensure_ascii=False, indent=2), encoding="utf-8")

    review_lines = [
        "# بازبینی V8 Turbo-first + MiniLM", "",
        "Turbo پایهٔ تغییرناپذیر است؛ فقط اسلات‌های نامطمئن با شاهد خانوادهٔ مستقل باز شده‌اند.", "",
        "| زمان | انتخاب | وضعیت | دلیل |", "|---:|---|---|---|",
    ]
    for row in review:
        review_lines.append(
            f"| {row['midpoint']:.2f} | {row.get('candidate') or 'ε'} | "
            f"{row['status']} | {row['reason']} |")
    (out_dir / "review-v8.md").write_text(
        "\n".join(review_lines) + "\n", encoding="utf-8")

    turbo_path = (run_dir / "hypotheses" / TURBO_ENHANCED
                  / f"{TURBO_ENHANCED}.txt")
    comparison = ["# مقایسهٔ Turbo و V8", ""]
    if turbo_path.is_file():
        comparison += ["## Turbo enhanced", "", turbo_path.read_text(
            encoding="utf-8").strip(), ""]
    comparison += ["## V8 Turbo-first + MiniLM", "", final_text, "",
                   "## تغییرهای مجاز", "", f"تعداد: {len(path_changes)}", ""]
    (out_dir / "comparison-v8.md").write_text(
        "\n".join(comparison), encoding="utf-8")

    summary = {
        "runtime_seconds": round(elapsed, 3),
        "review_count": len(review),
        "placeholder_count": len(placeholders),
        "protected_name_count": len(protected_names),
        "dose_lock_count": len(dose_locks),
        "hard_validation_passed": validation["passed"],
        "retrieved_sentence_count": len(agent.retrieved),
        "semantic_checkpoint_count": semantic_decode["checkpoint_count"],
        "path_change_count": len(path_changes),
        "turbo_locked_slot_count": locked_count,
        "turbo_uncertain_slot_count": uncertain_count,
        "turbo_retention_ratio": round(1.0 - len(path_changes) / max(1, len(turbo_audit)), 6),
        "local_semantic_prior_count": len(local_semantic_priors),
        "adaptive_secondary_asr": adaptive_plan is not None,
        "adaptive_review_coverage_ratio": (
            adaptive_plan.get("review_coverage_ratio") if adaptive_plan else None),
        "generative_llm_used": False,
        "pretrained_semantic_encoder_used": True,
        "encoder_generates_text": False,
        "text": final_text,
    }
    (out_dir / "summary-v8.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output": str(out_dir), **summary}


def run(run_dir: Path, medical_index: Path, corpus_index: Path,
        encoder_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    hypotheses = load_hypotheses(run_dir)
    sequences = {key: words_of(hypotheses[key], key) for key in TURBO_FIRST_ORDER}
    medical_payload = json.loads(medical_index.read_text(encoding="utf-8"))
    resolver = v4.LexiconResolver(medical_payload)
    corpus = v5.DomainCorpus(corpus_index)
    try:
        ngrams = v5.DomainNgramEvidence(sequences, resolver, corpus)
        phrase_evidence = v6.PhraseEvidence(sequences)
        slots: list[dict[str, Any]] = []
        for hypothesis in TURBO_FIRST_ORDER:
            slots = v4.add_sequence_to_network(slots, sequences[hypothesis], hypothesis)
        lattice = [v4.build_slot_candidates(slot, resolver, ngrams) for slot in slots]
        for slot, candidates in zip(slots, lattice):
            preserve_exact_turbo_candidate(slot, candidates, resolver, ngrams)
        v5.augment_adjacent_acoustic_support(slots, lattice)
        semantic_candidate_stats = v6.augment_semantic_candidates(
            slots, lattice, resolver, ngrams)
        semantic_candidate_stats.update(v6.augment_medication_frames(
            slots, lattice, resolver, ngrams))
        v6.mark_turbo_anchors(slots, lattice)
        # V6 pruning is optimized for consensus and can discard a frequent but
        # colloquial Turbo surface (for example «فشارتون»).  A Turbo-first
        # decoder must retain the exact base row even when a dictionary repair
        # has a higher generic score.
        exact_turbo_rows: list[dict[str, Any] | None] = []
        for slot, candidates in zip(slots, lattice):
            enhanced = slot["observations"].get(TURBO_ENHANCED)
            raw = slot["observations"].get(TURBO_RAW)
            enhanced_token = str((enhanced or {}).get("normalized") or "")
            raw_token = str((raw or {}).get("normalized") or "")
            preferred = (enhanced_token if observation_quality(enhanced, True)
                         >= observation_quality(raw) else raw_token)
            exact_turbo_rows.append(next(
                (row for row in candidates if candidate_token(row) == preferred), None))
        lattice = v6.prune_lattice(lattice)
        for candidates, turbo_row in zip(lattice, exact_turbo_rows):
            if turbo_row is not None and all(row is not turbo_row for row in candidates):
                candidates.append(turbo_row)
        lattice = copy.deepcopy(lattice)

        turbo_path, turbo_audit = build_turbo_base(
            run_dir, slots, lattice, resolver)
        turbo_candidates = [candidate_token(lattice[index][choice])
                            for index, choice in enumerate(turbo_path)]
        encoder = OnnxSentenceEncoder(encoder_dir)
        agent = v7.SemanticRetrievalAgent(
            encoder, hypotheses, sequences, resolver, corpus,
            ngrams.medical_mix, lattice)
        critical_priors = mark_critical_independent_checks(
            lattice, turbo_path, turbo_audit, resolver)
        oov_priors = apply_turbo_oov_minilm_repairs(
            slots, lattice, turbo_path, turbo_audit, agent, resolver)
        v7_local_priors = v7.apply_local_semantic_priors(
            slots, lattice, turbo_candidates, agent, resolver)
        local_semantic_priors = [*critical_priors, *oov_priors, *v7_local_priors]
        constrained_lattice, change_gate_audit = constrain_turbo_first(
            lattice, turbo_path, turbo_audit, local_semantic_priors, resolver)
        constrained_path, transitions, semantic_decode = v7.decode_semantic_lattice(
            constrained_lattice, ngrams, phrase_evidence, resolver, agent)
        path: list[int] = []
        for slot_index, choice in enumerate(constrained_path):
            chosen = constrained_lattice[slot_index][choice]
            path.append(next(index for index, row in enumerate(lattice[slot_index])
                             if row is chosen))
        semantic_decode["turbo_change_gate"] = change_gate_audit
        semantic_decode["opened_slot_count"] = sum(
            len(row["allowed_candidates"]) > 1 for row in change_gate_audit)

        selected = v6.classify(slots, lattice, path, transitions)
        # Entity-fragment cleanup belongs to the older consensus-first path and
        # can delete a locked Turbo word after decoding.  V8 leaves that job to
        # its explicit change gate instead of applying an unaudited rewrite.
        semantic_candidate_stats["entity_fragment_cleanups"] = 0
        doses = cluster_doses(dose_occurrences(sequences))
        dose_locks = v4.apply_dose_locks(selected, doses)
        numeric_canonicalizations = v6.canonicalize_structured_numbers(selected, resolver)
        path_changes: list[dict[str, Any]] = []
        for slot_index, row in enumerate(selected):
            before = turbo_candidates[slot_index]
            after = candidate_token(row)
            if before != after:
                path_changes.append({
                    "slot": slot_index,
                    "turbo_base_candidate": before,
                    "semantic_v8_candidate": after,
                    "turbo_uncertainty_reasons": turbo_audit[slot_index][
                        "uncertainty_reasons"],
                })
        raw_selected_text, _render_operations = v4.render(selected)
        validation = v6.validate_v6(slots, sequences, selected, raw_selected_text)
        preserved_ordinary_reviews = 0
        for slot_index, row in enumerate(selected):
            if row.get("status") != "REVIEW":
                continue
            audit = turbo_audit[slot_index]
            severe_audio = (
                "turbo-missing" in audit["uncertainty_reasons"]
                or float(audit.get("selected_probability") or 0.0) < 0.45
                or float(audit.get("selected_avg_logprob") or -2.0) < -1.0)
            unresolved_critical = bool(
                change_gate_audit[slot_index].get("unresolved_critical_alternative"))
            if (not severe_audio and not unresolved_critical
                    and not is_critical_candidate(row, resolver)):
                row["status"] = "ACCEPT"
                row["reason"] = "turbo-first-preserved-ordinary-review"
                preserved_ordinary_reviews += 1
        semantic_candidate_stats[
            "ordinary_turbo_reviews_preserved"] = preserved_ordinary_reviews
        placeholder_text, placeholders = v6.placeholder_render(selected, resolver)
        final_text, protected_names = v5.protect_honorific_names(placeholder_text)
        return write_outputs(
            run_dir, selected, final_text, raw_selected_text, placeholders,
            protected_names, validation, corpus, ngrams, encoder, agent,
            semantic_decode, time.perf_counter() - started, dose_locks,
            semantic_candidate_stats, numeric_canonicalizations, path_changes,
            local_semantic_priors, turbo_audit, change_gate_audit)
    finally:
        corpus.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Turbo-first adaptive consensus with local MiniLM semantic reranking; no text generation."
        ))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--medical-index", type=Path, required=True)
    parser.add_argument("--corpus-index", type=Path, required=True)
    parser.add_argument("--encoder-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.run_dir.resolve(), args.medical_index.resolve(),
        args.corpus_index.resolve(), args.encoder_dir.resolve())
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
