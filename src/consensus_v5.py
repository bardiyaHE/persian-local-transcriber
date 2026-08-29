from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import consensus_v4 as v4
from consensus_v2 import BASE_KEY, norm
from consensus_v3 import NETWORK_ORDER, cluster_doses, dose_occurrences, load_hypotheses, words_of
from ngram_voice_score import SCORE_FILENAME, score_voice_text


OUTPUT_RELATIVE = Path("final-delivery") / "02-after-algorithm-v5-domain-corpus"
MEDICAL_CUES = {
    "پزشک", "دکتر", "بیمار", "بیماری", "دارو", "داروی", "قرص", "کپسول", "شربت",
    "آمپول", "دوز", "مصرف", "میلیگرم", "میلیگرم", "درمان", "آزمایش", "قند", "خون",
    "نسخه", "صبحانه", "شام",
}
LEARNING_POLICY = {
    "persistent_learning_from_audio": False,
    "historical_transcripts_used_for_scoring": False,
    "current_six_hypotheses_context": "ephemeral_per_request",
    "domain_corpus": "versioned_frozen_read_only",
    "sample_specific_phrases": False,
}

NAME_BLANK = "________"
HONORIFIC_NAME_RE = re.compile(
    r"(?<!\w)(?P<honorific>خانم|خانوم|آقا|آقای)(?P<spacing>\s+)"
    r"(?P<candidate>[^\s،؛.!؟?]+)"
)
HONORIFIC_TITLE_EXCEPTIONS = {
    "دکتر", "پزشک", "مهندس", "پرستار", "استاد", "حاج", "حاجی",
    "جناب", "سرکار", "محترم", "محترمه",
}


def protect_honorific_names(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Hide a likely personal name after an honorific instead of trusting ASR correction."""
    protected: list[dict[str, Any]] = []

    def replace(match: re.Match[str]) -> str:
        candidate = match.group("candidate")
        if norm(candidate) in HONORIFIC_TITLE_EXCEPTIONS:
            return match.group(0)
        protected.append({
            "order": len(protected) + 1,
            "honorific": match.group("honorific"),
            "placeholder": NAME_BLANK,
            "requires_user_input": True,
        })
        return f"{match.group('honorific')}{match.group('spacing')}{NAME_BLANK}"

    return HONORIFIC_NAME_RE.sub(replace, text), protected


class DomainCorpus:
    """Read-only exact Persian corpus counts. It never generates candidate words."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Domain corpus is missing: {self.path}. Run src/build_domain_corpus.py first.")
        self.connection = sqlite3.connect(
            f"file:{self.path.as_posix()}?mode=ro", uri=True, check_same_thread=False)
        self.connection.execute("PRAGMA query_only=ON")
        if int(self.connection.execute("PRAGMA query_only").fetchone()[0]) != 1:
            raise RuntimeError("The live domain corpus did not enter SQLite query-only mode.")
        self.metadata = dict(self.connection.execute("SELECT key,value FROM metadata"))
        if self.metadata.get("schema_version") != "1":
            raise RuntimeError("Unsupported domain-corpus schema.")

    def close(self) -> None:
        self.connection.close()

    @lru_cache(maxsize=100_000)
    def bigram_count(self, left: str, right: str) -> int:
        row = self.connection.execute(
            "SELECT COALESCE(SUM(count),0) FROM bigrams WHERE w1=? AND w2=?",
            (norm(left), norm(right)),
        ).fetchone()
        return int(row[0] or 0)

    @lru_cache(maxsize=200_000)
    def _counts(self, history: tuple[str, ...], candidate: str) -> dict[str, dict[str, int]]:
        result = {
            "medical": {"unigram": 0, "bigram": 0, "trigram": 0},
            "daily": {"unigram": 0, "bigram": 0, "trigram": 0},
        }
        for domain, count in self.connection.execute(
                "SELECT domain,count FROM unigrams WHERE w1=?", (candidate,)):
            result[domain]["unigram"] = int(count)
        if history:
            for domain, count in self.connection.execute(
                    "SELECT domain,count FROM bigrams WHERE w1=? AND w2=?",
                    (history[-1], candidate)):
                result[domain]["bigram"] = int(count)
        if len(history) >= 2:
            for domain, count in self.connection.execute(
                    "SELECT domain,count FROM trigrams WHERE w1=? AND w2=? AND w3=?",
                    (history[-2], history[-1], candidate)):
                result[domain]["trigram"] = int(count)
        return result

    def score(self, history: tuple[str, ...], candidate: str,
              medical_mix: float, sensitive: bool, drug_identity: bool) -> tuple[float, dict[str, Any]]:
        counts = self._counts(tuple(history[-2:]), norm(candidate))
        weights = {"medical": medical_mix, "daily": 1.0 - 0.45 * medical_mix}
        by_domain: dict[str, float] = {}
        for domain in ("medical", "daily"):
            row = counts[domain]
            # The corpus only adds bounded positive evidence. Missing phrases never
            # penalize an acoustically supported name, dialect form, dose, or negation.
            has_context = bool(row["bigram"] or row["trigram"])
            value = ((
                0.020 * min(5.0, math.log1p(row["unigram"]))
                + 0.145 * min(5.0, math.log1p(row["bigram"]))
                + 0.235 * min(5.5, math.log1p(row["trigram"]))
            ) * weights[domain]) if has_context else 0.0
            by_domain[domain] = value
        raw = min(0.18, sum(by_domain.values()))
        if sensitive:
            raw = min(raw, 0.06)
        elif drug_identity:
            raw = min(raw, 0.12)
        details = {
            "counts": counts,
            "domain_weights": {key: round(value, 4) for key, value in weights.items()},
            "domain_scores": {key: round(value, 6) for key, value in by_domain.items()},
            "bounded_bonus": round(raw, 6),
            "policy": "positive tie-breaker only; no candidate generation or unseen-phrase penalty",
        }
        return raw, details

    @lru_cache(maxsize=4096)
    def examples(self, domain: str, phrase: str, limit: int = 2) -> tuple[tuple[str, str], ...]:
        phrase_tokens = v4.phrase_tokens(phrase)
        if len(phrase_tokens) < 2:
            return tuple()
        query = '"' + " ".join(phrase_tokens[-3:]).replace('"', "") + '"'
        rows = self.connection.execute(
            "SELECT s.source,s.text FROM sentence_fts f "
            "JOIN sentences s ON s.id=f.rowid "
            "WHERE f.normalized MATCH ? AND s.domain=? LIMIT ?",
            (query, domain, int(limit)),
        ).fetchall()
        return tuple((str(source), str(text)) for source, text in rows)

    def describe(self) -> dict[str, Any]:
        row_counts = {
            table: int(self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("sentences", "unigrams", "bigrams", "trigrams")
        }
        return {
            "path": str(self.path), "bytes": self.path.stat().st_size,
            "created_at": self.metadata.get("created_at"),
            "algorithm": self.metadata.get("algorithm"),
            "translation_used": self.metadata.get("translation_used") == "true",
            "llm_used": self.metadata.get("llm_used") == "true",
            "open_mode": "read-only", "persistent_learning": False,
            "source_stats": json.loads(self.metadata.get("source_stats") or "{}"),
            "rows": row_counts,
        }


class DomainNgramEvidence(v4.NgramEvidence):
    def __init__(self, sequences: dict[str, list[dict[str, Any]]],
                 resolver: v4.LexiconResolver, corpus: DomainCorpus) -> None:
        super().__init__(sequences, resolver)
        self.resolver = resolver
        self.corpus = corpus
        all_tokens = [word["normalized"] for words in sequences.values() for word in words]
        cue_hits = sum(token in MEDICAL_CUES for token in all_tokens)
        lexicon_hits = sum(resolver.is_medical(token) for token in all_tokens)
        family_tokens = max(1, len(all_tokens))
        if cue_hits >= 6 or lexicon_hits >= 12:
            self.medical_mix = 0.90
        elif cue_hits >= 2 or lexicon_hits >= 4:
            self.medical_mix = 0.76
        elif cue_hits or lexicon_hits:
            self.medical_mix = 0.55
        else:
            self.medical_mix = 0.25
        self.domain_detection = {
            "medical_mix": self.medical_mix,
            "daily_mix": round(1.0 - 0.45 * self.medical_mix, 4),
            "medical_cue_hits": cue_hits,
            "medical_lexicon_hits": lexicon_hits,
            "hypothesis_token_count": family_tokens,
        }

    def transition_score(self, history: tuple[str, ...], candidate: str) -> tuple[float, dict[str, Any]]:
        base_score, details = super().transition_score(history, candidate)
        if not candidate:
            return base_score, details
        sensitive = (candidate in v4.NUMBER_WORDS or candidate in v4.UNITS
                     or v4.is_negative_token(candidate))
        categories = self.resolver.medical_categories(candidate)
        drug_identity = bool(categories & {"drug", "medication", "drug_class"})
        corpus_score, corpus_details = self.corpus.score(
            history, candidate, self.medical_mix, sensitive, drug_identity)
        details["local_six_hypothesis_score"] = round(base_score, 6)
        details["domain_corpus"] = corpus_details
        return base_score + corpus_score, details


def decode(candidate_lattice: list[list[dict[str, Any]]], ngrams: DomainNgramEvidence,
           beam_size: int = 72) -> tuple[list[int], list[dict[str, Any]]]:
    """V4 beam decoder with a hard two-family gate around external corpus evidence."""
    beam = [{"score": 0.0, "history": tuple(), "path": [], "transitions": []}]
    for candidates in candidate_lattice:
        expanded = []
        for state in beam:
            for candidate_index, row in enumerate(candidates):
                token = row["candidate"]
                history = state["history"]
                context_score = 0.0
                penalty = 0.0
                context_parts = []
                penalty_reasons = []
                family_gate = len(row.get("strong_families") or []) >= 2
                for part in row.get("candidate_tokens") or []:
                    part_context_score, part_context = ngrams.transition_score(history, part)
                    corpus_detail = part_context.get("domain_corpus") or {}
                    proposed_bonus = float(corpus_detail.get("bounded_bonus") or 0.0)
                    applied_bonus = proposed_bonus if family_gate else 0.0
                    if proposed_bonus and not family_gate:
                        part_context_score -= proposed_bonus
                    if corpus_detail:
                        corpus_detail["applied_bonus"] = round(applied_bonus, 6)
                        corpus_detail["acoustic_family_gate_passed"] = family_gate
                    part_penalty, part_penalty_reason = v4.redundancy_penalty(history, part)
                    context_score += part_context_score
                    penalty += part_penalty
                    context_parts.append(part_context)
                    if part_penalty_reason:
                        penalty_reasons.append(part_penalty_reason)
                    history = (*history[-1:], part)
                if row.get("low_confidence_repeat_penalty") and any(
                        v4.is_negative_token(part) for part in row.get("candidate_tokens") or []):
                    context_score = min(context_score, 0.25)
                    penalty -= 1.0
                    penalty_reasons.append("low-confidence-negation-lock")
                expanded.append({
                    "score": state["score"] + row["emission_score"] + context_score + penalty,
                    "history": history, "path": [*state["path"], candidate_index],
                    "transitions": [*state["transitions"], {
                        "context_score": round(context_score, 6), "context": context_parts,
                        "redundancy_penalty": penalty,
                        "redundancy_reason": ",".join(penalty_reasons) or None,
                    }],
                })
        best_by_history: dict[tuple[str, ...], dict[str, Any]] = {}
        for state in expanded:
            current = best_by_history.get(state["history"])
            if current is None or state["score"] > current["score"]:
                best_by_history[state["history"]] = state
        beam = sorted(best_by_history.values(), key=lambda row: row["score"], reverse=True)[:beam_size]
    best = max(beam, key=lambda row: row["score"])
    return best["path"], best["transitions"]


def enforce_acoustic_consensus(lattice: list[list[dict[str, Any]]]) -> dict[str, int]:
    """Prevent local/corpus language priors from overruling exact independent ASR votes."""
    unanimous_slots = 0
    majority_slots = 0
    for index, candidates in enumerate(lattice):
        epsilon = [row for row in candidates if not row.get("candidate")]
        nonempty = [row for row in candidates if row.get("candidate")]
        unanimous = [row for row in nonempty if len(row.get("exact_families") or []) == 3]
        if unanimous:
            # Preserve, but do not make unanimity absolute: all Whisper families can
            # share the same spelling/fusion error. Active three-family fuzzy rivals
            # and well-supported split candidates must remain able to correct it.
            for row in unanimous:
                row["unanimous_exact_protected"] = True
            has_split_rival = any(
                row.get("origin") == "repeated-text-bigram-split"
                and len(row.get("strong_families") or []) >= 2
                for row in nonempty
            )
            if not has_split_rival:
                for row in unanimous:
                    if row.get("general_lexicon") or row.get("medical_lexicon") or row.get("modern_spoken"):
                        row["emission_score"] = round(float(row["emission_score"]) + 1.50, 6)
                        row["unanimous_exact_bonus"] = 1.50
                # Raw/enhanced are not independent families, but they are still useful
                # within-family replications. If an active word occurs in >=4 of the six
                # decodes and spans all three families, do not let a 2-of-6 inflectional
                # rival win solely through the language-model transition (اسم vs اسمی).
                replicated = [
                    row for row in unanimous
                    if int(row.get("exact_sources") or 0) >= 4
                    and (row.get("general_lexicon") or row.get("medical_lexicon")
                         or row.get("modern_spoken"))
                ]
                if replicated:
                    best_replication = max(
                        replicated,
                        key=lambda row: (int(row.get("exact_sources") or 0),
                                         float(row.get("emission_score") or 0.0)),
                    )
                    max_sources = int(best_replication.get("exact_sources") or 0)
                    eligible = [
                        row for row in nonempty
                        if row is best_replication
                        or int(row.get("exact_sources") or 0) >= max_sources
                    ]
                    lattice[index] = sorted(
                        eligible, key=lambda row: row["emission_score"], reverse=True
                    ) + epsilon
            unanimous_slots += 1
        exact_majority = [row for row in nonempty if len(row.get("exact_families") or []) >= 2]
        if exact_majority and not unanimous:
            # When two families agree exactly, a one-family word may not win merely
            # because its neighbours form a common phrase. Other two-family fuzzy
            # alternatives remain available for genuine pronunciation ambiguity.
            eligible = [row for row in nonempty if len(row.get("strong_families") or []) >= 2]
            lattice[index] = sorted(eligible, key=lambda row: row["emission_score"], reverse=True) + epsilon
            majority_slots += 1
    return {"unanimous_exact_slots": unanimous_slots, "exact_majority_slots": majority_slots}


def augment_adjacent_acoustic_support(slots: list[dict[str, Any]],
                                      lattice: list[list[dict[str, Any]]]) -> int:
    """Recover evidence shifted by one alignment slot without creating candidates.

    Insertions in one decode can place a close form one slot later than the same word
    in another decode (مصرف/صرف). Only active existing candidates, adjacent timing
    overlap, and high fuzzy similarity are accepted.
    """
    augmented = 0
    for index, candidates in enumerate(lattice):
        current_observations = list(slots[index]["observations"].values())
        current_start = min(row["start"] for row in current_observations)
        current_end = max(row["end"] for row in current_observations)
        neighbours: list[tuple[int, dict[str, Any]]] = []
        for adjacent in (index - 1, index + 1):
            if adjacent < 0 or adjacent >= len(slots):
                continue
            for observation in slots[adjacent]["observations"].values():
                overlap = max(0.0, min(current_end, observation["end"])
                              - max(current_start, observation["start"]))
                shortest = max(0.08, min(current_end - current_start,
                                         observation["end"] - observation["start"]))
                if overlap / shortest >= 0.35:
                    neighbours.append((adjacent, observation))
        for row in candidates:
            candidate = row.get("candidate") or ""
            if not candidate or not (
                    row.get("general_lexicon") or row.get("medical_lexicon")
                    or row.get("modern_spoken")):
                continue
            similarities = dict(row.get("family_similarity") or {})
            strong = set(row.get("strong_families") or [])
            loose = set(row.get("loose_families") or [])
            before_strong = len(strong)
            before_loose = len(loose)
            adjacent_evidence = []
            for adjacent, observation in neighbours:
                similarity = v4.token_similarity(candidate, observation["normalized"])
                family = observation["family"]
                similarities[family] = max(float(similarities.get(family) or 0.0), similarity)
                if similarity >= 0.82:
                    if family not in strong:
                        adjacent_evidence.append({
                            "slot": adjacent, "family": family,
                            "observation": observation["normalized"],
                            "similarity": round(similarity, 6),
                        })
                    strong.add(family)
                elif similarity >= 0.70:
                    loose.add(family)
            added_strong = len(strong) - before_strong
            added_loose = len(loose) - before_loose
            if not (added_strong or added_loose):
                continue
            row["family_similarity"] = similarities
            row["strong_families"] = sorted(strong)
            row["loose_families"] = sorted(loose | strong)
            bonus = 1.45 * added_strong + 0.26 * added_loose
            row["emission_score"] = round(float(row["emission_score"]) + bonus, 6)
            row["adjacent_alignment_bonus"] = round(bonus, 6)
            if adjacent_evidence:
                unique = {
                    (item["slot"], item["family"]): item for item in adjacent_evidence
                }
                row["adjacent_alignment_evidence"] = list(unique.values())
            augmented += 1
    return augmented


def suppress_single_family_bridge_insertions(selected: list[dict[str, Any]],
                                              corpus: DomainCorpus) -> int:
    """Drop a one-family inserted word when it breaks an attested outer bigram."""
    removed = 0
    nonempty = [index for index, row in enumerate(selected) if row.get("candidate")]
    for position in range(1, len(nonempty) - 1):
        left_index, middle_index, right_index = nonempty[position - 1:position + 2]
        left = selected[left_index].get("candidate") or ""
        middle = selected[middle_index].get("candidate") or ""
        right = selected[right_index].get("candidate") or ""
        middle_exact = set(selected[middle_index].get("exact_families") or [])
        consumed_families = {
            evidence["family"]
            for evidence in selected[left_index].get("adjacent_alignment_evidence") or []
            if int(evidence["slot"]) == middle_index
        }
        # The outer word must have explicitly borrowed the same family's close
        # observation from this middle slot; otherwise this may be a real OOV name
        # or an ordinary colloquial word and must never be deleted.
        if not consumed_families or not middle_exact or not middle_exact <= consumed_families:
            continue
        outer_count = corpus.bigram_count(left, right)
        if outer_count < 2:
            continue
        if corpus.bigram_count(left, middle) or corpus.bigram_count(middle, right):
            continue
        row = selected[middle_index]
        row["candidate"] = ""
        row["candidate_tokens"] = []
        row["origin"] = "corpus-bridge-epsilon"
        row["status"] = "OMIT"
        row["reason"] = "single-family-insertion-breaks-attested-bigram"
        row["bridge_context"] = {
            "left": left, "right": right, "outer_bigram_count": outer_count,
        }
        removed += 1
    return removed


def annotate_corpus_examples(selected: list[dict[str, Any]], corpus: DomainCorpus) -> int:
    history: tuple[str, ...] = tuple()
    supported = 0
    for row in selected:
        row_examples: list[dict[str, str]] = []
        row_bonus = 0.0
        context_parts = row.get("transition", {}).get("context") or []
        for part_index, part in enumerate(row.get("candidate_tokens") or []):
            detail = context_parts[part_index] if part_index < len(context_parts) else {}
            domain_detail = detail.get("domain_corpus") or {}
            row_bonus += float(domain_detail.get("applied_bonus") or 0.0)
            counts = domain_detail.get("counts") or {}
            phrase = " ".join((*history[-2:], part))
            domains = sorted(
                ("medical", "daily"),
                key=lambda domain: max(
                    int((counts.get(domain) or {}).get("trigram") or 0),
                    int((counts.get(domain) or {}).get("bigram") or 0)),
                reverse=True,
            )
            for domain in domains:
                domain_counts = counts.get(domain) or {}
                if not (domain_counts.get("trigram") or domain_counts.get("bigram")):
                    continue
                for source, text in corpus.examples(domain, phrase, 1):
                    row_examples.append({
                        "domain": domain, "source": source, "matched_context": phrase,
                        "sentence": text,
                    })
                    break
                if row_examples:
                    break
            history = (*history[-1:], part)
        row["domain_corpus_bonus"] = round(row_bonus, 6)
        row["domain_corpus_examples"] = row_examples[:2]
        if row_bonus > 0:
            supported += 1
    return supported


def mark_suffix_boundary_ambiguities(selected: list[dict[str, Any]]) -> int:
    """Flag cases where a final ی may actually contain the start of a following function word."""
    marked = 0
    for index, row in enumerate(selected[:-1]):
        candidate = row.get("candidate") or ""
        following = selected[index + 1].get("candidate") or ""
        if following != "که" or not candidate.endswith("ی") or len(candidate) < 4:
            continue
        stem = candidate[:-1]
        rival = next((alternative for alternative in row.get("alternatives") or []
                      if alternative.get("candidate") == stem
                      and len(alternative.get("strong_families") or []) >= 2), None)
        if rival is None or row.get("domain_corpus_bonus", 0.0) <= 0:
            continue
        row["status"] = "REVIEW"
        row["reason"] = "suffix-boundary-ambiguous-before-ke"
        row["suffix_boundary_rival"] = stem
        marked += 1
    return marked


def write_outputs(run_dir: Path, medical_payload: dict[str, Any], slots: list[dict[str, Any]],
                  selected: list[dict[str, Any]], final_text: str, render_operations: list[str],
                  doses: list[dict[str, Any]], dose_locks: list[dict[str, Any]],
                  validation: dict[str, Any], elapsed: float, resolver: v4.LexiconResolver,
                  ngrams: DomainNgramEvidence, corpus: DomainCorpus,
                  corpus_supported: int, suffix_boundary_reviews: int,
                  acoustic_consensus: dict[str, int],
                  protected_name_slots: list[dict[str, Any]],
                  voice_score: dict[str, Any]) -> dict[str, Any]:
    out_dir = run_dir / OUTPUT_RELATIVE
    out_dir.mkdir(parents=True, exist_ok=True)
    review = [row for row in selected if row["status"] == "REVIEW"]
    clips, clip_limit = v4.make_review_clips(run_dir, out_dir, review)
    corpus_description = corpus.describe()
    voice_score_summary = {key: value for key, value in voice_score.items()
                           if key != "transitions"}
    payload = {
        "method": "three-family word-confusion network with local-six and offline domain-corpus n-grams",
        "llm_used": False, "translation_used": False, "sample_specific_phrases": False,
        "turbo_is_template": False, "runtime_seconds": round(elapsed, 3), "text": final_text,
        "learning_policy": LEARNING_POLICY,
        "family_vote_policy": "raw/enhanced collapse to one family; corpus never counts as a model family",
        "dictionary_policy": "dictionary candidates still require acoustic support from at least two families",
        "domain_corpus_policy": "bounded positive tie-breaker; dose/number/negation/drug identity are capped",
        "domain_detection": ngrams.domain_detection,
        "ngram_voice_score": voice_score_summary,
        "acoustic_consensus_gate": acoustic_consensus,
        "domain_corpus": corpus_description,
        "name_protection_policy": {
            "honorifics": ["خانم", "خانوم", "آقا", "آقای"],
            "action": "replace likely following personal name with a user-filled blank",
            "dictionary_or_ngram_correction_used_for_output": False,
            "title_exceptions": sorted(HONORIFIC_TITLE_EXCEPTIONS),
        },
        "protected_name_slots": protected_name_slots,
        "general_lexicon": {"package": "mnk-persian-words", "clean_words": v4.persian_words.count_words()},
        "medical_lexicon": {
            "source": medical_payload.get("source"), "license": medical_payload.get("license"),
            "unique_terms": medical_payload.get("unique_terms"),
        },
        "medical_phrase_bigram_trigram_count": len(resolver.medical_phrase_ngrams),
        "corpus_supported_selected_slots": corpus_supported,
        "suffix_boundary_reviews": suffix_boundary_reviews,
        "dose_entities": doses, "dose_locks": dose_locks,
        "hard_validation": validation, "render_operations": render_operations,
        "slots": selected,
    }
    (out_dir / "final-v5.txt").write_text(final_text + "\n", encoding="utf-8")
    (out_dir / "final-v5.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "review-v5.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "review-clips-v5.json").write_text(
        json.dumps(clips, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "name-slots-v5.json").write_text(
        json.dumps(protected_name_slots, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / SCORE_FILENAME).write_text(
        json.dumps(voice_score, ensure_ascii=False, indent=2), encoding="utf-8")

    review_lines = [
        "# بازبینی نسخهٔ ۵ — پیکرهٔ دامنه‌ای، بدون LLM", "",
        "پیکره فقط برای شکستن تساوی به کار رفته است؛ REVIEWها و شواهد صوتی حفظ شده‌اند.", "",
        "| زمان | واژه | دلیل | خانواده‌های پشتیبان | امتیاز پیکره |",
        "|---:|---|---|---|---:|",
    ]
    for row in review:
        review_lines.append(
            f"| {row['midpoint']:.2f} | {row['candidate'] or 'ε'} | {row['reason']} | "
            f"{', '.join(row['strong_families']) or '—'} | {row.get('domain_corpus_bonus', 0.0):.3f} |")
    review_lines += ["", "## کلیپ‌ها", ""]
    for index, clip in enumerate(clips, 1):
        review_lines.append(f"- کلیپ {index}: {clip['start']:.2f} تا {clip['end']:.2f} ثانیه")
    if clip_limit:
        review_lines.append("- تعداد بازه‌ها بیش از سقف کلیپ بود؛ جزئیات کامل در JSON است.")
    (out_dir / "review-v5.md").write_text("\n".join(review_lines) + "\n", encoding="utf-8")

    score_lines = [
        "# امتیازنامهٔ شش متن + پیکرهٔ دامنه‌ای — نسخهٔ ۵", "",
        f"**امتیاز n-gram متن پایهٔ Turbo: {voice_score['score']:.1f} از ۱۰۰**", "",
        "این عدد از n-gram ترکیبی، اطمینان صوتی Turbo، اعتبار واژه و زنجیرهٔ خطا ساخته شده است؛ پنج متن دیگر در نمره دخالت ندارند و متن تغییر نمی‌کند.", "",
        "| جایگاه | زمان | انتخاب | خانواده | عمومی | پزشکی | n-gram شش متن | n-gram پیکره | وضعیت |",
        "|---:|---:|---|---:|:---:|:---:|---:|---:|---|",
    ]
    for row in selected:
        transition = row.get("transition") or {}
        context_parts = transition.get("context") or []
        local_score = sum(float(part.get("local_six_hypothesis_score") or 0.0)
                          for part in context_parts)
        score_lines.append(
            f"| {row['slot']} | {row['midpoint']:.2f} | {row['candidate'] or 'ε'} | "
            f"{len(row['strong_families'])} | {'✓' if row['general_lexicon'] else ''} | "
            f"{'✓' if row['medical_lexicon'] else ''} | {local_score:.3f} | "
            f"{row.get('domain_corpus_bonus', 0.0):.3f} | {row['status']} |")
    examples = []
    for row in selected:
        for example in row.get("domain_corpus_examples") or []:
            key = (example["source"], example["matched_context"], example["sentence"])
            if key not in examples:
                examples.append(key)
    score_lines += ["", "## نمونهٔ زمینه‌های پیدا‌شده", ""]
    for source, phrase, sentence in examples[:20]:
        score_lines.append(f"- `{phrase}` — {sentence} (`{source}`)")
    score_lines += ["", "## خروجی", "", final_text, ""]
    (out_dir / "scorecard-v5.md").write_text("\n".join(score_lines), encoding="utf-8")

    turbo = run_dir / "hypotheses" / "large-v3-turbo__enhanced" / "large-v3-turbo__enhanced.txt"
    previous = run_dir / "final-delivery" / "02-after-algorithm-v4-ngram-lexicon" / "final-v4.txt"
    comparison = ["# مقایسهٔ خروجی", ""]
    if turbo.is_file():
        comparison += ["## Turbo enhanced", "", turbo.read_text(encoding="utf-8").strip(), ""]
    if previous.is_file():
        comparison += ["## نسخهٔ ۴ (فقط n-gram شش متن)", "", previous.read_text(encoding="utf-8").strip(), ""]
    comparison += ["## نسخهٔ ۵ (پیکرهٔ پزشکی + روزمره)", "", final_text, ""]
    (out_dir / "comparison-v5.md").write_text("\n".join(comparison), encoding="utf-8")

    summary = {
        "runtime_seconds": round(elapsed, 3), "slot_count": len(slots),
        "review_count": len(review), "review_clip_count": sum("clip" in row for row in clips),
        "review_clip_limit_reached": clip_limit, "hard_validation_passed": validation["passed"],
        "llm_used": False, "translation_used": False, "sample_specific_phrases": False,
        "learning_policy": LEARNING_POLICY,
        "protected_name_count": len(protected_name_slots),
        "accepted_dose_locks": len(dose_locks),
        "corpus_supported_selected_slots": corpus_supported,
        "suffix_boundary_reviews": suffix_boundary_reviews,
        "domain_detection": ngrams.domain_detection, "corpus_rows": corpus_description["rows"],
        "acoustic_consensus_gate": acoustic_consensus,
        "ngram_voice_score": voice_score["score"],
        "ngram_voice_score_scale": "0-100",
        "ngram_voice_score_text_changed": False,
        "text": final_text,
    }
    (out_dir / "summary-v5.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output": str(out_dir), **summary}


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Deterministic six-ASR consensus plus offline Persian domain n-grams; no LLM.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--medical-index", type=Path, required=True)
    parser.add_argument("--corpus-index", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    corpus_path = args.corpus_index.resolve()
    if run_dir == corpus_path.parent or run_dir in corpus_path.parents:
        raise RuntimeError("A per-run corpus is forbidden in live mode; use the frozen offline-corpus index.")
    corpus_signature = (corpus_path.stat().st_size, corpus_path.stat().st_mtime_ns)
    hypotheses = load_hypotheses(run_dir)
    sequences = {key: words_of(hypotheses[key], key) for key in NETWORK_ORDER}
    medical_payload = json.loads(args.medical_index.read_text(encoding="utf-8"))
    resolver = v4.LexiconResolver(medical_payload)
    corpus = DomainCorpus(corpus_path)
    try:
        ngrams = DomainNgramEvidence(sequences, resolver, corpus)
        doses = cluster_doses(dose_occurrences(sequences))
        slots: list[dict[str, Any]] = []
        for hypothesis in NETWORK_ORDER:
            slots = v4.add_sequence_to_network(slots, sequences[hypothesis], hypothesis)
        lattice = [v4.build_slot_candidates(slot, resolver, ngrams) for slot in slots]
        adjacent_support = augment_adjacent_acoustic_support(slots, lattice)
        acoustic_consensus = enforce_acoustic_consensus(lattice)
        acoustic_consensus["adjacent_alignment_augmented_candidates"] = adjacent_support
        path, transitions = decode(lattice, ngrams)
        selected = v4.classify_choices(slots, lattice, path, transitions)
        acoustic_consensus["corpus_bridge_insertions_removed"] = (
            suppress_single_family_bridge_insertions(selected, corpus))
        dose_locks = v4.apply_dose_locks(selected, doses)
        corpus_supported = annotate_corpus_examples(selected, corpus)
        suffix_boundary_reviews = mark_suffix_boundary_ambiguities(selected)
        unprotected_text, render_operations = v4.render(selected)
        validation = v4.validate(slots, sequences, selected, unprotected_text)
        voice_score = score_voice_text(
            str(hypotheses[BASE_KEY].get("text") or ""), resolver, corpus.connection,
            sequences[BASE_KEY])
        final_text, protected_name_slots = protect_honorific_names(unprotected_text)
        elapsed = time.perf_counter() - started
        result = write_outputs(
            run_dir, medical_payload, slots, selected, final_text, render_operations,
            doses, dose_locks, validation, elapsed, resolver, ngrams, corpus,
            corpus_supported, suffix_boundary_reviews, acoustic_consensus,
            protected_name_slots, voice_score)
    finally:
        corpus.close()
    if corpus_signature != (corpus_path.stat().st_size, corpus_path.stat().st_mtime_ns):
        raise RuntimeError("The frozen corpus changed during a live request.")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
