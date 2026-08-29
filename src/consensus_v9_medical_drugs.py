from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import consensus_v4 as v4
import consensus_v5 as v5
import consensus_v6_phrase_semantic as v6
import consensus_v7_semantic_agent as v7
import consensus_v8_turbo_semantic as v8
from consensus_v3 import cluster_doses, dose_occurrences
from semantic_encoder_onnx import OnnxSentenceEncoder


OUTPUT_RELATIVE = Path("final-delivery") / "09-medical-drug-dictionary"
DRUG_CATEGORIES = {"drug", "medication"}
ALLOWED_DRUG_LICENSES = {"cc by 4.0", "mit"}
PERSIAN_TOKEN_RE = re.compile(r"[\u0600-\u06ff]+", re.UNICODE)
POSSESSIVE_SUFFIXES = ("هایتون", "هاتون", "هاشون", "هاشان", "تون", "تان", "شون", "شان", "مون", "مان")
GENERIC_DRUG_TERMS = {
    "دارو", "داروی", "داروها", "داروهاتون", "دارواتون", "قرص", "آمپول",
    "شربت", "پماد", "سم", "سرکه", "روی", "ویتامین", "آنتیبیوتیک",
    "آنتییوتیک", "دارونما",
}
GENERIC_ENGLISH = {
    "drug", "medicine", "medication", "poison", "vinegar", "zinc", "vitamin",
    "antibiotic", "tablet", "pill", "capsule", "injection", "placebo",
}
MEDICATION_CUES = {
    "اسم", "دارو", "داروی", "دارواتون", "قرص", "آمپول", "کپسول", "شربت",
    "دوز", "میلی", "گرم", "میلیگرم", "مصرف", "بخورید", "بخوره", "بخورم",
    "تزریق", "نسخه", "روزی", "صبح", "ظهر", "عصر", "شب", "نصف", "زیر",
    "زبون", "زبان", "تجویز", "شروع", "ادامه",
}
DRUG_LIKE_SUFFIXES = (
    "ماب", "زپام", "پام", "زولون", "تیزون", "زون", "تینیب", "سیب",
    "کسیب", "گابالین", "پرازول", "ستاتین", "مایسین", "سایکلین",
)


def candidate_token(row: dict[str, Any]) -> str:
    return str(row.get("candidate") or "")


def strip_possessive(token: str) -> tuple[str, str]:
    token = v4.norm(token).strip("،,.؛:!?؟…«»\"'()[]{}")
    persian_parts = PERSIAN_TOKEN_RE.findall(token)
    if persian_parts:
        token = "".join(persian_parts)
    for suffix in POSSESSIVE_SUFFIXES:
        if token.endswith(suffix) and len(token) >= len(suffix) + 4:
            return token[:-len(suffix)], suffix
    return token, ""


class DrugDictionaryV2:
    """Canonical Persian drug spellings derived only from the licensed local index."""

    def __init__(self, medical_payload: dict[str, Any], corpus: v5.DomainCorpus) -> None:
        self.source = str(medical_payload.get("source") or "")
        self.license = str(medical_payload.get("license") or "")
        self.sources = list(medical_payload.get("sources") or [{
            "name": self.source, "license": self.license,
        }])
        self.alias_to_identity: dict[str, str] = {}
        self.alias_to_row: dict[str, dict[str, Any]] = {}
        self.identity_attribution: dict[str, set[tuple[str, str]]] = defaultdict(set)
        grouped: dict[str, set[str]] = defaultdict(set)
        for row in medical_payload.get("terms") or []:
            if str(row.get("category") or "") not in DRUG_CATEGORIES:
                continue
            row_source = str(row.get("source") or self.source)
            row_license = str(row.get("license") or self.license)
            if row_license.strip().casefold() not in ALLOWED_DRUG_LICENSES:
                continue
            term = v4.norm(row.get("normalized") or row.get("term") or "")
            if (not term or " " in term or not PERSIAN_TOKEN_RE.fullmatch(term)
                    or term in GENERIC_DRUG_TERMS):
                continue
            english = str(row.get("english") or "").strip().casefold()
            if not english or english in GENERIC_ENGLISH:
                continue
            identity = english
            # The first vetted source wins an exact alias collision. Other
            # identities remain available through their distinct spellings.
            self.alias_to_identity.setdefault(term, identity)
            self.alias_to_row.setdefault(term, row)
            identity = self.alias_to_identity[term]
            self.identity_attribution[identity].add((row_source, row_license))
            grouped[identity].add(term)

        self.canonical_by_identity: dict[str, str] = {}
        for identity, aliases in grouped.items():
            scored = []
            for alias in aliases:
                count = int(corpus.connection.execute(
                    "SELECT COALESCE(SUM(count),0) FROM unigrams WHERE w1=?", (alias,)
                ).fetchone()[0] or 0)
                # Corpus frequency chooses the current spelling; the stable
                # lexical tiebreakers keep the build deterministic offline.
                scored.append((count, "ی" in alias, "ک" in alias, len(alias), alias))
            self.canonical_by_identity[identity] = max(scored)[-1]

    def identify(self, row: dict[str, Any]) -> tuple[str, str, str] | None:
        surface = v4.norm(candidate_token(row))
        hinted = v4.norm(row.get("medical_base_term") or "")
        stem, suffix = strip_possessive(surface)
        alias = hinted or stem
        identity = self.alias_to_identity.get(alias)
        if not identity:
            return None
        canonical = self.canonical_by_identity[identity]
        return identity, canonical + suffix, canonical

    def describe(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "license": self.license,
            "sources": self.sources,
            "usable_persian_alias_count": len(self.alias_to_identity),
            "canonical_drug_count": len(self.canonical_by_identity),
            "licensed_source_count": len(self.sources),
            "policy": "canonical spelling only; dictionary is not acoustic evidence",
        }

    def attribution(self, identity: str) -> list[dict[str, str]]:
        return [
            {"source": source, "license": license_name}
            for source, license_name in sorted(self.identity_attribution.get(identity, set()))
        ]

    def is_primary_identity(self, identity: str) -> bool:
        return any(
            license_name.strip().casefold() == "cc by 4.0"
            or "persianmedqa" in source.casefold()
            for source, license_name in self.identity_attribution.get(identity, set())
        )


def base_is_unlicensed(row: dict[str, Any], dictionary: DrugDictionaryV2) -> bool:
    token = v4.norm(candidate_token(row))
    if not token:
        return True
    if dictionary.identify(row):
        return False
    return not bool(
        row.get("general_lexicon") or row.get("medical_lexicon")
        or row.get("modern_spoken") or v4.active_general_word(token)
    )


def medication_context(tokens: list[str], slot_index: int, radius: int = 9) -> tuple[bool, list[str]]:
    nearby = [v4.norm(token) for token in tokens[max(0, slot_index - radius):slot_index + radius + 1]
              if v4.norm(token)]
    cues = sorted({token for token in nearby if (
        token in MEDICATION_CUES
        or token.startswith("میلی") or token.startswith("دارو")
        or token.startswith("قرص") or token.startswith("آمپول")
        or token.startswith("زبون") or token.startswith("زبان")
    )})
    return bool(cues), cues


def corpus_context(corpus: v5.DomainCorpus, previous: str, token: str,
                   following: str) -> tuple[float, dict[str, int]]:
    left_count = int(corpus.connection.execute(
        "SELECT COALESCE(SUM(count),0) FROM bigrams WHERE w1=? AND w2=?",
        (previous, token),
    ).fetchone()[0] or 0) if previous else 0
    right_count = int(corpus.connection.execute(
        "SELECT COALESCE(SUM(count),0) FROM bigrams WHERE w1=? AND w2=?",
        (token, following),
    ).fetchone()[0] or 0) if following else 0
    trigram_count = int(corpus.connection.execute(
        "SELECT COALESCE(SUM(count),0) FROM trigrams WHERE w1=? AND w2=? AND w3=?",
        (previous, token, following),
    ).fetchone()[0] or 0) if previous and following else 0
    score = (4.0 * math.log1p(trigram_count)
             + 0.7 * math.log1p(left_count + right_count))
    return score, {"left_bigram": left_count, "right_bigram": right_count,
                   "trigram": trigram_count}


def rank_dictionary_drug_repairs(
        slots: list[dict[str, Any]], lattice: list[list[dict[str, Any]]], turbo_path: list[int],
        turbo_audit: list[dict[str, Any]], agent: v7.SemanticRetrievalAgent,
        dictionary: DrugDictionaryV2,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    baseline = [lattice[index][choice] for index, choice in enumerate(turbo_path)]
    baseline_tokens = [candidate_token(row) for row in baseline]
    winners: dict[int, dict[str, Any]] = {}
    audits: list[dict[str, Any]] = []

    for slot_index, candidates in enumerate(lattice):
        base = baseline[slot_index]
        if not base_is_unlicensed(base, dictionary):
            continue
        in_context, cues = medication_context(baseline_tokens, slot_index)
        base_stem, _base_suffix = strip_possessive(candidate_token(base))
        drug_like_surface = any(base_stem.endswith(suffix) for suffix in DRUG_LIKE_SUFFIXES)
        if drug_like_surface:
            in_context = True
            cues = [*cues, "drug-like-suffix"]
        if not in_context:
            continue

        eligible: list[tuple[dict[str, Any], tuple[str, str, str]]] = []
        for row in candidates:
            identity = dictionary.identify(row)
            if row is base or not identity:
                continue
            token = candidate_token(row)
            if (v4.norm(identity[2]) in GENERIC_DRUG_TERMS
                    or len(set(row.get("strong_families") or [])) < 2
                    or float(row.get("dictionary_similarity") or 0.0) < 0.80
                    or float(row.get("acoustic_probability") or 0.0) < 0.50
                    or row.get("origin") not in {
                        "medical-lexicon", "semantic-lexicon-repair",
                        "medication-template", "repeated-medication-entity",
                    }):
                continue
            if v4.norm(token) in GENERIC_DRUG_TERMS:
                continue
            direct_surface_similarity = v4.token_similarity(base_stem, identity[2])
            if (direct_surface_similarity < 0.55
                    and int(row.get("cross_occurrence_drug_count") or 0) < 2):
                continue
            if not dictionary.is_primary_identity(identity[0]):
                # A new single-source spelling protects an exact Turbo word,
                # but changing an OOV word to that drug requires stronger
                # acoustic agreement than the clinician-reviewed base list.
                strong_count = len(set(row.get("strong_families") or []))
                cross_count = int(row.get("cross_occurrence_drug_count") or 0)
                candidate_stem, _candidate_suffix = strip_possessive(token)
                suffix_evidence = any(candidate_stem.endswith(suffix)
                                      for suffix in DRUG_LIKE_SUFFIXES)
                if ((strong_count < 3 and cross_count < 2)
                        or float(row.get("dictionary_similarity") or 0.0) < 0.86
                        or float(row.get("acoustic_probability") or 0.0) < 0.58
                        or not (suffix_evidence or drug_like_surface)):
                    continue
            eligible.append((row, identity))
        if not eligible:
            continue

        # Keep one spelling per underlying drug, preferring the canonical
        # Persian spelling selected from the local corpus.
        by_identity: dict[str, list[tuple[dict[str, Any], tuple[str, str, str]]]] = defaultdict(list)
        for row, identity in eligible:
            by_identity[identity[0]].append((row, identity))
        identity_rows = []
        for identity, rows in by_identity.items():
            canonical_rows = [item for item in rows
                              if v4.norm(candidate_token(item[0])) == v4.norm(item[1][1])]
            pool = canonical_rows or rows
            identity_rows.append(max(pool, key=lambda item: (
                float(item[0].get("emission_score") or 0.0),
                float(item[0].get("dictionary_similarity") or 0.0),
            )))

        left, right = max(0, slot_index - 6), min(len(lattice), slot_index + 7)
        prefix = [candidate_token(row) for row in baseline[left:slot_index] if candidate_token(row)]
        suffix = [candidate_token(row) for row in baseline[slot_index + 1:right] if candidate_token(row)]
        variants = [v7.text_from_parts([*prefix, candidate_token(base), *suffix], limit=40)]
        variants.extend(v7.text_from_parts([*prefix, candidate_token(row), *suffix], limit=40)
                        for row, _identity in identity_rows)
        analyses = agent.score_texts(variants)
        previous = prefix[-1] if prefix else ""
        following = suffix[0] if suffix else ""
        ranked = []
        for position, (row, identity) in enumerate(identity_rows, 1):
            token = candidate_token(row)
            family_scores = list((row.get("family_similarity") or {}).values())
            family_score = sum(family_scores) / max(1, len(family_scores))
            semantic_score = float(analyses[position]["semantic_score"])
            context_score, counts = corpus_context(agent.corpus, previous, token, following)
            cross_count = int(row.get("cross_occurrence_drug_count") or 0)
            emission = float(row.get("emission_score") or 0.0)
            canonical_bonus = float(v4.norm(token) == v4.norm(identity[1]))
            primary_source_bonus = float(dictionary.is_primary_identity(identity[0]))
            surface_similarity = v4.token_similarity(base_stem, identity[2])
            score = (
                0.20 * float(row.get("dictionary_similarity") or 0.0)
                + 0.18 * family_score
                + 0.12 * float(row.get("acoustic_probability") or 0.0)
                + 0.08 * semantic_score
                + 0.08 * min(1.0, context_score / 8.0)
                + 0.14 * min(1.0, cross_count / 2.0)
                + 0.14 * min(1.0, emission / 7.0)
                + 0.06 * canonical_bonus
                + 0.07 * primary_source_bonus
                + 0.16 * surface_similarity
            )
            ranked.append((score, row, identity, semantic_score, context_score,
                           counts, family_score, cross_count, surface_similarity,
                           primary_source_bonus))
        ranked.sort(key=lambda item: item[0], reverse=True)
        winner = ranked[0]
        margin = winner[0] - (ranked[1][0] if len(ranked) > 1 else 0.0)
        minimum_score = 0.62 if drug_like_surface else 0.65
        if winner[0] < minimum_score or (len(ranked) > 1 and margin < 0.035):
            audits.append({
                "slot": slot_index, "from": candidate_token(base), "action": "review",
                "reason": "ambiguous-dictionary-drug-candidates",
                "top_score": round(winner[0], 6), "margin": round(margin, 6),
                "candidates": [candidate_token(item[1]) for item in ranked[:4]],
                "context_cues": cues,
            })
            continue
        row = winner[1]
        prior = 7.0
        row["emission_score_before_v9_drug_prior"] = round(
            float(row.get("emission_score") or 0.0), 6)
        row["emission_score"] = round(float(row.get("emission_score") or 0.0) + prior, 6)
        row["v9_medical_dictionary_prior"] = prior
        row["v9_drug_identity"] = winner[2][0]
        winners[slot_index] = row
        turbo_audit[slot_index]["turbo_locked"] = False
        turbo_audit[slot_index]["uncertain"] = True
        if "licensed-medical-drug-repair" not in turbo_audit[slot_index]["uncertainty_reasons"]:
            turbo_audit[slot_index]["uncertainty_reasons"].append(
                "licensed-medical-drug-repair")
        audits.append({
            "slot": slot_index, "from": candidate_token(base),
            "favored_candidate": candidate_token(row), "canonical_drug": winner[2][2],
            "identity": winner[2][0], "action": "repair",
            "reason": "two-family-phonetic-plus-medical-dictionary",
            "score": round(winner[0], 6), "margin": round(margin, 6),
            "semantic_score": round(winner[3], 6),
            "corpus_context_score": round(winner[4], 6),
            "corpus_counts": winner[5], "family_similarity": round(winner[6], 6),
            "cross_occurrence_count": winner[7],
            "surface_similarity": round(winner[8], 6), "context_cues": cues,
            "primary_source": bool(winner[9]),
            "sources": dictionary.attribution(winner[2][0]),
        })

    # A directly resolved drug may occur again with weaker acoustics. Reuse it
    # only when the V6 repeated-entity candidate has two effective families.
    identities = {str(row.get("v9_drug_identity") or "") for row in winners.values()}
    for slot_index, candidates in enumerate(lattice):
        if slot_index in winners or not base_is_unlicensed(baseline[slot_index], dictionary):
            continue
        in_context, cues = medication_context(baseline_tokens, slot_index)
        if not in_context:
            continue
        repeated = []
        for row in candidates:
            identity = dictionary.identify(row)
            if (not identity or identity[0] not in identities
                    or row.get("origin") != "repeated-medication-entity"
                    or v6.effective_family_count(row) < 2
                    or float(row.get("dictionary_similarity") or 0.0) < 0.78):
                continue
            if (not dictionary.is_primary_identity(identity[0])
                    and v6.effective_family_count(row) < 3):
                continue
            repeated.append((row, identity))
        if not repeated:
            # If a drug was confidently resolved once, admit the same canonical
            # identity at a later medication slot when two Whisper families are
            # phonetically consistent. This does not use the dictionary as
            # acoustic evidence: every qualifying family still comes from the
            # six stored hypotheses.
            observed = list(slots[slot_index]["observations"].values())
            generated = []
            for resolved_identity in sorted(identities):
                canonical = dictionary.canonical_by_identity.get(resolved_identity)
                if not canonical:
                    continue
                by_family: dict[str, float] = {}
                probability_by_family: dict[str, float] = {}
                for observation in observed:
                    family = str(observation.get("family") or "")
                    observation_stem, _suffix = strip_possessive(
                        str(observation.get("normalized") or ""))
                    similarity = v4.token_similarity(canonical, observation_stem)
                    by_family[family] = max(by_family.get(family, 0.0), similarity)
                    probability_by_family[family] = max(
                        probability_by_family.get(family, 0.0),
                        float(observation.get("probability") or 0.0) * similarity,
                    )
                consistent = {family for family, value in by_family.items() if value >= 0.60}
                base_stem, _base_suffix = strip_possessive(candidate_token(baseline[slot_index]))
                base_similarity = v4.token_similarity(canonical, base_stem)
                if len(consistent) < 2 or max(by_family.values(), default=0.0) < 0.68:
                    continue
                if base_similarity < 0.58:
                    continue
                acoustic_probability = sum(
                    probability_by_family[family] for family in consistent
                ) / len(consistent)
                generated.append((
                    sum(by_family[family] for family in consistent) / len(consistent),
                    resolved_identity,
                    canonical,
                    sorted(consistent),
                    by_family,
                    acoustic_probability,
                ))
            generated.sort(key=lambda item: item[0], reverse=True)
            if generated and (len(generated) == 1 or generated[0][0] - generated[1][0] >= 0.08):
                similarity, resolved_identity, canonical, consistent, by_family, probability = generated[0]
                base_row = baseline[slot_index]
                row = copy.deepcopy(base_row)
                row.update({
                    "candidate": canonical,
                    "candidate_tokens": [canonical],
                    "origin": "repeated-medication-entity",
                    "dictionary_similarity": round(similarity, 6),
                    "family_similarity": by_family,
                    "strong_families": [],
                    "entity_consistency_families": consistent,
                    "medical_base_term": canonical,
                    "medical_lexicon": True,
                    "medical_categories": ["drug"],
                    "acoustic_probability": round(probability, 6),
                    "observed": False,
                    "emission_score": round(float(base_row.get("emission_score") or 0.0) + 1.0, 6),
                })
                lattice[slot_index].append(row)
                repeated.append((row, (resolved_identity, canonical, canonical)))
        if not repeated:
            continue
        row, identity = max(repeated, key=lambda item: float(item[0].get("emission_score") or 0.0))
        prior = 6.0
        row["emission_score_before_v9_repeated_drug_prior"] = round(
            float(row.get("emission_score") or 0.0), 6)
        row["emission_score"] = round(float(row.get("emission_score") or 0.0) + prior, 6)
        row["v9_medical_dictionary_prior"] = prior
        row["v9_drug_identity"] = identity[0]
        winners[slot_index] = row
        audits.append({
            "slot": slot_index, "from": candidate_token(baseline[slot_index]),
            "favored_candidate": candidate_token(row), "canonical_drug": identity[2],
            "identity": identity[0], "action": "repair",
            "reason": "resolved-drug-repeated-with-two-effective-families",
            "context_cues": cues,
            "sources": dictionary.attribution(identity[0]),
        })
    return winners, audits


def open_v9_drug_gates(
        constrained: list[list[dict[str, Any]]], gate_audit: list[dict[str, Any]],
        lattice: list[list[dict[str, Any]]], winners: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    form_audits: list[dict[str, Any]] = []
    opened_forms: set[int] = set()
    for slot_index, winner in winners.items():
        # The resolver has already required a licensed canonical entry, two
        # strong acoustic families, a medication frame and a ranking margin.
        # Once that gate passes, retaining the misspelled OOV Turbo surface as
        # a competing path defeats canonicalization, so V9 locks the winner.
        constrained[slot_index] = [winner]
        gate = gate_audit[slot_index]
        token = candidate_token(winner)
        previous_allowed = list(gate.get("allowed_candidates") or [])
        previous_rejected = list(gate.get("rejected_candidates") or [])
        gate["allowed_candidates"] = [token]
        gate["rejected_candidates"] = [
            row for row in previous_rejected if row.get("candidate") != token
        ] + [
            {"candidate": candidate, "reason": "superseded-by-v9-canonical-drug-gate"}
            for candidate in previous_allowed if candidate != token
        ]
        gate["reason"] = "v9-licensed-drug-dictionary-gate"
        gate["unresolved_critical_alternative"] = False

        # Restore the dosage form only when the strict V6 sublingual template
        # and a resolved canonical drug agree. This repairs «گرس و راسپام» to
        # «قرص لورازپام» without admitting a free dictionary guess.
        for form_slot in range(max(0, slot_index - 4), min(len(lattice), slot_index + 5)):
            if form_slot == slot_index or form_slot in opened_forms:
                continue
            form = next((row for row in lattice[form_slot]
                         if candidate_token(row) == "قرص"
                         and len(set(row.get("template_families") or [])) >= 2), None)
            if form is None:
                continue
            before = float(form.get("emission_score") or 0.0)
            form["emission_score_before_v9_form_prior"] = round(before, 6)
            form["emission_score"] = round(before + 7.0, 6)
            form["v9_resolved_drug_form_prior"] = 7.0
            constrained[form_slot] = [form]
            gate_form = gate_audit[form_slot]
            gate_form["allowed_candidates"] = ["قرص"]
            gate_form["rejected_candidates"] = []
            gate_form["reason"] = "v9-dosage-form-before-resolved-drug"
            form_audits.append({
                "slot": form_slot, "favored_candidate": "قرص",
                "resolved_drug_slot": slot_index,
                "reason": "strict-sublingual-template-plus-resolved-drug",
            })
            opened_forms.add(form_slot)
    return form_audits


def cleanup_drug_conjunctions(selected: list[dict[str, Any]],
                              dictionary: DrugDictionaryV2) -> list[dict[str, Any]]:
    operations = []
    for index in range(len(selected) - 2):
        form, conjunction, drug = selected[index:index + 3]
        if (candidate_token(form) == "قرص" and candidate_token(conjunction) == "و"
                and dictionary.identify(drug)):
            operations.append({
                "slot": conjunction.get("slot"), "removed": "و",
                "reason": "spurious-conjunction-before-resolved-drug",
            })
            conjunction.update({
                "candidate": "", "candidate_tokens": [], "status": "OMIT",
                "origin": "v9-drug-phrase-cleanup",
                "reason": "spurious-conjunction-before-resolved-drug",
            })
    return operations


def write_outputs(
        run_dir: Path, selected: list[dict[str, Any]], final_text: str,
        raw_selected_text: str, placeholders: list[dict[str, Any]],
        protected_names: list[dict[str, Any]], validation: dict[str, Any],
        corpus: v5.DomainCorpus, ngrams: v5.DomainNgramEvidence,
        encoder: OnnxSentenceEncoder, agent: v7.SemanticRetrievalAgent,
        semantic_decode: dict[str, Any], elapsed: float,
        dose_locks: list[dict[str, Any]], semantic_candidate_stats: dict[str, Any],
        numeric_canonicalizations: list[dict[str, Any]], path_changes: list[dict[str, Any]],
        local_semantic_priors: list[dict[str, Any]], turbo_audit: list[dict[str, Any]],
        change_gate_audit: list[dict[str, Any]], drug_dictionary: DrugDictionaryV2,
        drug_audit: list[dict[str, Any]], drug_phrase_cleanups: list[dict[str, Any]],
) -> dict[str, Any]:
    out_dir = run_dir / OUTPUT_RELATIVE
    out_dir.mkdir(parents=True, exist_ok=True)
    review = [row for row in selected if row.get("status") == "REVIEW"]
    clips, _clip_limit = v4.make_review_clips(run_dir, out_dir, review)
    plan_path = run_dir / "adaptive-turbo-plan.json"
    adaptive_plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.is_file() else None
    dictionary_repairs = [row for row in drug_audit if row.get("action") == "repair"]
    payload = {
        "algorithm": "v9 Turbo-first MiniLM plus licensed canonical medical drug dictionary",
        "generative_llm_used": False,
        "external_api_used_at_runtime": False,
        "pretrained_semantic_encoder_used": True,
        "encoder_generates_text": False,
        "runtime_seconds": round(elapsed, 3),
        "text": final_text,
        "raw_selected_text": raw_selected_text,
        "encoder": encoder.describe(),
        "semantic_agent": agent.describe(),
        "semantic_decode": semantic_decode,
        "adaptive_turbo_plan": adaptive_plan,
        "drug_dictionary": drug_dictionary.describe(),
        "drug_dictionary_audit": drug_audit,
        "drug_phrase_cleanups": drug_phrase_cleanups,
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
    (out_dir / "final-v9.txt").write_text(final_text + "\n", encoding="utf-8")
    (out_dir / "final-v9.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "review-v9.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "review-clips-v9.json").write_text(
        json.dumps(clips, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "drug-audit-v9.json").write_text(
        json.dumps(drug_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    review_lines = [
        "# بازبینی V9 واژه‌نامهٔ دارویی", "",
        "نام دارو فقط با شاهد آوایی، بافت دارویی و مدخل مجاز پذیرفته می‌شود.", "",
        "| زمان | انتخاب | وضعیت | دلیل |", "|---:|---|---|---|",
    ]
    for row in review:
        review_lines.append(
            f"| {row['midpoint']:.2f} | {row.get('candidate') or 'ε'} | "
            f"{row['status']} | {row['reason']} |")
    (out_dir / "review-v9.md").write_text("\n".join(review_lines) + "\n", encoding="utf-8")
    turbo_path = run_dir / "hypotheses" / v8.TURBO_ENHANCED / f"{v8.TURBO_ENHANCED}.txt"
    comparison = ["# مقایسهٔ Turbo و V9 دارویی", ""]
    if turbo_path.is_file():
        comparison += ["## Turbo enhanced", "", turbo_path.read_text(encoding="utf-8").strip(), ""]
    comparison += ["## V9", "", final_text, "", "## اصلاح‌های واژه‌نامه‌ای", "",
                   f"تعداد: {len(dictionary_repairs)}", ""]
    (out_dir / "comparison-v9.md").write_text("\n".join(comparison), encoding="utf-8")
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
        "drug_dictionary_repair_count": len(dictionary_repairs),
        "drug_phrase_cleanup_count": len(drug_phrase_cleanups),
        "turbo_locked_slot_count": sum(bool(row["turbo_locked"]) for row in turbo_audit),
        "turbo_uncertain_slot_count": sum(not bool(row["turbo_locked"]) for row in turbo_audit),
        "turbo_retention_ratio": round(1.0 - len(path_changes) / max(1, len(turbo_audit)), 6),
        "adaptive_secondary_asr": adaptive_plan is not None,
        "adaptive_review_coverage_ratio": adaptive_plan.get("review_coverage_ratio") if adaptive_plan else None,
        "generative_llm_used": False,
        "pretrained_semantic_encoder_used": True,
        "encoder_generates_text": False,
        "text": final_text,
    }
    (out_dir / "summary-v9.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output": str(out_dir), **summary}


def run(run_dir: Path, medical_index: Path, corpus_index: Path,
        encoder_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    hypotheses = v8.load_hypotheses(run_dir)
    sequences = {key: v8.words_of(hypotheses[key], key) for key in v8.TURBO_FIRST_ORDER}
    medical_payload = json.loads(medical_index.read_text(encoding="utf-8"))
    resolver = v4.LexiconResolver(medical_payload)
    corpus = v5.DomainCorpus(corpus_index)
    try:
        drug_dictionary = DrugDictionaryV2(medical_payload, corpus)
        ngrams = v5.DomainNgramEvidence(sequences, resolver, corpus)
        phrase_evidence = v6.PhraseEvidence(sequences)
        slots: list[dict[str, Any]] = []
        for hypothesis in v8.TURBO_FIRST_ORDER:
            slots = v4.add_sequence_to_network(slots, sequences[hypothesis], hypothesis)
        lattice = [v4.build_slot_candidates(slot, resolver, ngrams) for slot in slots]
        for slot, candidates in zip(slots, lattice):
            v8.preserve_exact_turbo_candidate(slot, candidates, resolver, ngrams)
        v5.augment_adjacent_acoustic_support(slots, lattice)
        semantic_candidate_stats = v6.augment_semantic_candidates(slots, lattice, resolver, ngrams)
        semantic_candidate_stats.update(v6.augment_medication_frames(slots, lattice, resolver, ngrams))
        v6.mark_turbo_anchors(slots, lattice)
        exact_turbo_rows: list[dict[str, Any] | None] = []
        for slot, candidates in zip(slots, lattice):
            enhanced = slot["observations"].get(v8.TURBO_ENHANCED)
            raw = slot["observations"].get(v8.TURBO_RAW)
            preferred = (str((enhanced or {}).get("normalized") or "")
                         if v8.observation_quality(enhanced, True) >= v8.observation_quality(raw)
                         else str((raw or {}).get("normalized") or ""))
            exact_turbo_rows.append(next(
                (row for row in candidates if candidate_token(row) == preferred), None))
        lattice = v6.prune_lattice(lattice)
        for candidates, turbo_row in zip(lattice, exact_turbo_rows):
            if turbo_row is not None and all(row is not turbo_row for row in candidates):
                candidates.append(turbo_row)
        lattice = copy.deepcopy(lattice)

        turbo_path, turbo_audit = v8.build_turbo_base(run_dir, slots, lattice, resolver)
        turbo_candidates = [candidate_token(lattice[index][choice])
                            for index, choice in enumerate(turbo_path)]
        encoder = OnnxSentenceEncoder(encoder_dir)
        agent = v7.SemanticRetrievalAgent(
            encoder, hypotheses, sequences, resolver, corpus, ngrams.medical_mix, lattice)
        critical_priors = v8.mark_critical_independent_checks(
            lattice, turbo_path, turbo_audit, resolver)
        oov_priors = v8.apply_turbo_oov_minilm_repairs(
            slots, lattice, turbo_path, turbo_audit, agent, resolver)
        drug_winners, drug_audit = rank_dictionary_drug_repairs(
            slots, lattice, turbo_path, turbo_audit, agent, drug_dictionary)
        v7_local_priors = v7.apply_local_semantic_priors(
            slots, lattice, turbo_candidates, agent, resolver)
        drug_priors = [{
            "slot": slot, "from": turbo_candidates[slot],
            "favored_candidate": candidate_token(row),
            "reason": "v9-licensed-medical-drug-dictionary",
            "prior": row.get("v9_medical_dictionary_prior"),
        } for slot, row in drug_winners.items()]
        local_semantic_priors = [
            *critical_priors, *oov_priors, *drug_priors, *v7_local_priors,
        ]
        constrained_lattice, change_gate_audit = v8.constrain_turbo_first(
            lattice, turbo_path, turbo_audit, local_semantic_priors, resolver)
        form_audits = open_v9_drug_gates(
            constrained_lattice, change_gate_audit, lattice, drug_winners)
        drug_audit.extend(form_audits)
        constrained_path, transitions, semantic_decode = v7.decode_semantic_lattice(
            constrained_lattice, ngrams, phrase_evidence, resolver, agent)
        path: list[int] = []
        for slot_index, choice in enumerate(constrained_path):
            chosen = constrained_lattice[slot_index][choice]
            path.append(next(index for index, row in enumerate(lattice[slot_index]) if row is chosen))
        semantic_decode["turbo_change_gate"] = change_gate_audit
        semantic_decode["opened_slot_count"] = sum(
            len(row["allowed_candidates"]) > 1 for row in change_gate_audit)
        semantic_decode["v9_drug_gate_count"] = len(drug_winners)

        selected = v6.classify(slots, lattice, path, transitions)
        drug_phrase_cleanups = cleanup_drug_conjunctions(selected, drug_dictionary)
        semantic_candidate_stats["v9_drug_dictionary_candidates"] = len(drug_winners)
        semantic_candidate_stats["v9_drug_phrase_cleanups"] = len(drug_phrase_cleanups)
        doses = cluster_doses(v8.dose_occurrences(sequences))
        dose_locks = v4.apply_dose_locks(selected, doses)
        numeric_canonicalizations = v6.canonicalize_structured_numbers(selected, resolver)
        path_changes = []
        for slot_index, row in enumerate(selected):
            before = turbo_candidates[slot_index]
            after = candidate_token(row)
            if before != after:
                path_changes.append({
                    "slot": slot_index, "turbo_base_candidate": before,
                    "semantic_v9_candidate": after,
                    "turbo_uncertainty_reasons": turbo_audit[slot_index]["uncertainty_reasons"],
                })
        raw_selected_text, _operations = v4.render(selected)
        validation = v6.validate_v6(slots, sequences, selected, raw_selected_text)
        preserved_ordinary_reviews = 0
        for slot_index, row in enumerate(selected):
            if (row.get("status") == "REVIEW" and row.get("v9_drug_identity")
                    and v6.effective_family_count(row) >= 2):
                row["status"] = "ACCEPT"
                row["reason"] = "v9-licensed-drug-two-family-support"
                continue
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
                    and not v8.is_critical_candidate(row, resolver)):
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
            local_semantic_priors, turbo_audit, change_gate_audit,
            drug_dictionary, drug_audit, drug_phrase_cleanups)
    finally:
        corpus.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V9 Turbo-first MiniLM with licensed canonical Persian drug-name resolution.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--medical-index", type=Path, required=True)
    parser.add_argument("--corpus-index", type=Path, required=True)
    parser.add_argument("--encoder-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.run_dir.resolve(), args.medical_index.resolve(),
                 args.corpus_index.resolve(), args.encoder_dir.resolve())
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
