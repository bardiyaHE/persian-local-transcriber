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

import numpy as np

import consensus_v4 as v4
import consensus_v5 as v5
import consensus_v6_phrase_semantic as v6
from consensus_v3 import NETWORK_ORDER, cluster_doses, dose_occurrences, load_hypotheses, words_of
from semantic_encoder_onnx import OnnxSentenceEncoder


OUTPUT_RELATIVE = Path("final-delivery") / "07-semantic-retrieval-agent"
CHECKPOINT_SLOTS = 24
SEMANTIC_WEIGHT = 7.5
FRAME_WEIGHT = 1.6
MAX_RETRIEVED_SENTENCES = 128
CONTENT_STOPWORDS = set(v4.FUNCTION_WORDS) | {
    "این", "اون", "آن", "که", "رو", "را", "هم", "برای", "اگر", "ولی", "حالا",
    "خیلی", "یکم", "یه", "من", "شما", "ایشون", "کرد", "کرده", "کنید", "بشه",
    "بود", "هست", "هستم", "دارم", "داره", "دارید", "میشه", "می‌کنه", "میکنه",
    "چه",
}
TOKEN_RE = re.compile(r"[0-9۰-۹]+|[\w\u0600-\u06ff]+", re.UNICODE)


def tokens_of_text(text: str) -> list[str]:
    return [v4.norm(item) for item in TOKEN_RE.findall(str(text)) if v4.norm(item)]


def text_from_parts(parts: list[str] | tuple[str, ...], limit: int = 72) -> str:
    return " ".join(parts[-limit:]).strip()


def family_anchor_tokens(sequences: dict[str, list[dict[str, Any]]],
                         resolver: v4.LexiconResolver,
                         corpus: v5.DomainCorpus,
                         lattice: list[list[dict[str, Any]]] | None = None,
                         limit: int = 14) -> list[dict[str, Any]]:
    families: dict[str, set[str]] = defaultdict(set)
    occurrences: Counter[str] = Counter()
    for words in sequences.values():
        for row in words:
            token = row["normalized"]
            if token:
                families[token].add(row["family"])
                occurrences[token] += 1
    for candidates in lattice or []:
        for row in candidates:
            token = str(row.get("candidate") or "")
            strong = set(row.get("strong_families") or [])
            if token and len(strong) >= 2:
                families[token].update(strong)
                occurrences[token] += max(1, len(strong))
    ranked: list[dict[str, Any]] = []
    for token, token_families in families.items():
        if (len(token) < 3 or token in CONTENT_STOPWORDS or token.isdigit()
                or v6.is_number(token)):
            continue
        categories = sorted(resolver.medical_categories(token))
        corpus_count = int(corpus.connection.execute(
            "SELECT COALESCE(SUM(count),0) FROM unigrams WHERE w1=?", (token,)
        ).fetchone()[0] or 0)
        # Retrieval anchors must actually exist in the sentence database.  A
        # unanimous ASR error with zero corpus occurrences is evidence about
        # acoustics, not a useful FTS query.
        if corpus_count == 0:
            continue
        rarity = 1.0 / max(1.0, math.log1p(corpus_count)) if corpus_count else 1.0
        score = (2.2 * len(token_families) + 0.22 * min(6, occurrences[token])
                 + (2.0 if categories else 0.0) + rarity)
        ranked.append({
            "token": token,
            "families": sorted(token_families),
            "occurrences": occurrences[token],
            "medical_categories": categories,
            "corpus_count": corpus_count,
            "anchor_score": round(score, 6),
        })
    ranked.sort(key=lambda row: (
        row["anchor_score"], len(row["medical_categories"]),
        len(row["families"]), -row["corpus_count"], row["token"]), reverse=True)
    return ranked[:limit]


def retrieve_corpus_sentences(corpus: v5.DomainCorpus,
                              anchors: list[dict[str, Any]],
                              medical_mix: float,
                              limit: int = MAX_RETRIEVED_SENTENCES) -> list[dict[str, Any]]:
    if not anchors:
        return []
    rows: dict[int, dict[str, Any]] = {}
    anchor_tokens = {row["token"] for row in anchors}
    medical_per_anchor = 12 if medical_mix >= 0.65 else 8
    daily_per_anchor = 5 if medical_mix >= 0.65 else 10
    for anchor in anchors[:10]:
        token = str(anchor["token"]).replace('"', "")
        query = f'"{token}"'
        for domain, quota in (("medical", medical_per_anchor), ("daily", daily_per_anchor)):
            found = corpus.connection.execute(
                "SELECT s.id,s.domain,s.source,s.text,s.normalized,bm25(sentence_fts) "
                "FROM sentence_fts JOIN sentences s ON s.id=sentence_fts.rowid "
                "WHERE sentence_fts MATCH ? AND s.domain=? "
                "ORDER BY bm25(sentence_fts) LIMIT ?",
                (query, domain, quota),
            ).fetchall()
            for sentence_id, row_domain, source, text, normalized, bm25 in found:
                normalized_tokens = set(str(normalized).split())
                if not 5 <= len(str(normalized).split()) <= 96:
                    continue
                overlap = sorted(anchor_tokens & normalized_tokens)
                if not overlap:
                    continue
                current = rows.get(int(sentence_id))
                candidate = {
                    "id": int(sentence_id), "domain": str(row_domain),
                    "source": str(source), "text": str(text),
                    "normalized": str(normalized), "anchor_overlap": overlap,
                    "anchor_overlap_count": len(overlap), "bm25": round(float(bm25), 6),
                }
                if current is None or (
                        candidate["anchor_overlap_count"], -candidate["bm25"]
                ) > (current["anchor_overlap_count"], -current["bm25"]):
                    rows[int(sentence_id)] = candidate
    # Pair queries strongly prefer sentences that contain a local relation
    # instead of short dictionary-like rows containing only one anchor.
    pair_anchors = [row["token"] for row in anchors[:8]]
    for left_index, left in enumerate(pair_anchors):
        for right in pair_anchors[left_index + 1:]:
            query = f'"{left.replace(chr(34), "")}" AND "{right.replace(chr(34), "")}"'
            for domain in ("medical", "daily"):
                found = corpus.connection.execute(
                    "SELECT s.id,s.domain,s.source,s.text,s.normalized,bm25(sentence_fts) "
                    "FROM sentence_fts JOIN sentences s ON s.id=sentence_fts.rowid "
                    "WHERE sentence_fts MATCH ? AND s.domain=? "
                    "ORDER BY bm25(sentence_fts) LIMIT 6",
                    (query, domain),
                ).fetchall()
                for sentence_id, row_domain, source, text, normalized, bm25 in found:
                    normalized_tokens = set(str(normalized).split())
                    if not 5 <= len(str(normalized).split()) <= 96:
                        continue
                    overlap = sorted(anchor_tokens & normalized_tokens)
                    candidate = {
                        "id": int(sentence_id), "domain": str(row_domain),
                        "source": str(source), "text": str(text),
                        "normalized": str(normalized), "anchor_overlap": overlap,
                        "anchor_overlap_count": len(overlap), "bm25": round(float(bm25), 6),
                    }
                    current = rows.get(int(sentence_id))
                    if current is None or candidate["anchor_overlap_count"] > current["anchor_overlap_count"]:
                        rows[int(sentence_id)] = candidate
    ranked = sorted(rows.values(), key=lambda row: (
        row["anchor_overlap_count"], row["domain"] == "medical", -row["bm25"]
    ), reverse=True)
    # Do not let a single source dominate the semantic target set.
    result: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    for row in ranked:
        if source_counts[row["source"]] >= max(24, limit // 2):
            continue
        result.append(row)
        source_counts[row["source"]] += 1
        if len(result) >= limit:
            break
    return result


class SemanticRetrievalAgent:
    """Retrieval plus encoder scorer; it cannot create a candidate token."""

    def __init__(self, encoder: OnnxSentenceEncoder,
                 hypotheses: dict[str, dict[str, Any]],
                 sequences: dict[str, list[dict[str, Any]]],
                 resolver: v4.LexiconResolver,
                 corpus: v5.DomainCorpus,
                 medical_mix: float,
                 lattice: list[list[dict[str, Any]]]) -> None:
        self.encoder = encoder
        self.resolver = resolver
        self.corpus = corpus
        self.anchors = family_anchor_tokens(sequences, resolver, corpus, lattice)
        self.retrieved = retrieve_corpus_sentences(
            corpus, self.anchors, medical_mix, MAX_RETRIEVED_SENTENCES)

        hypothesis_texts: list[str] = []
        hypothesis_families: list[str] = []
        for key in NETWORK_ORDER:
            words = sequences[key]
            text = str((hypotheses.get(key) or {}).get("text") or "").strip()
            if not text:
                text = " ".join(row["normalized"] for row in words)
            if text:
                hypothesis_texts.append(text)
                hypothesis_families.append(words[0]["family"] if words else key.split("__")[0])
        hypothesis_vectors = encoder.encode(hypothesis_texts)
        family_vectors: list[np.ndarray] = []
        for family in sorted(set(hypothesis_families)):
            indexes = [index for index, value in enumerate(hypothesis_families) if value == family]
            vector = hypothesis_vectors[indexes].mean(axis=0)
            vector /= max(float(np.linalg.norm(vector)), 1e-9)
            family_vectors.append(vector)
        self.family_vectors = np.stack(family_vectors) if family_vectors else np.empty((0, 384), np.float32)
        if len(self.family_vectors):
            self.consensus_vector = self.family_vectors.mean(axis=0)
            self.consensus_vector /= max(float(np.linalg.norm(self.consensus_vector)), 1e-9)
        else:
            self.consensus_vector = np.zeros(384, dtype=np.float32)
        self.corpus_vectors = encoder.encode([row["text"] for row in self.retrieved])
        self.vector_cache: dict[str, np.ndarray] = {}

    def frame_score(self, text: str) -> tuple[float, list[str]]:
        tokens = tokens_of_text(text)
        raw = 0.0
        reasons: list[str] = []
        for index, token in enumerate(tokens):
            nearby = tokens[max(0, index - 3):min(len(tokens), index + 4)]
            if v6.is_number(token):
                if any(item in v4.UNITS | v6.DOSE_CUES | v6.VITAL_CUES | v6.TIME_CUES
                       for item in nearby):
                    raw += 0.45
                    reasons.append("number-has-clinical-or-time-role")
                else:
                    raw -= 0.30
            categories = self.resolver.medical_categories(token)
            if categories & {"drug", "medication", "drug_class"}:
                if any(item in v6.DOSE_CUES | v4.UNITS | v6.TIME_CUES | {"نصف"}
                       for item in nearby):
                    raw += 0.55
                    reasons.append("drug-has-instruction-role")
            if index and token == tokens[index - 1] and token not in {"خیلی"}:
                raw -= 0.65
                reasons.append("adjacent-duplicate")
        joined = " ".join(tokens)
        for phrase in ("صبح و عصر", "زیر زبان", "یک ماه دیگر", "دو هفته بعد",
                       "آزمایش بدید", "مصرف کنید", "ادامه بدید"):
            if phrase in joined:
                raw += 0.35
                reasons.append("complete-clinical-frame")
        return math.tanh(raw / 4.0), sorted(set(reasons))

    def score_texts(self, texts: list[str]) -> list[dict[str, Any]]:
        missing = [text for text in dict.fromkeys(texts) if text not in self.vector_cache]
        if missing:
            vectors = self.encoder.encode(missing)
            self.vector_cache.update(zip(missing, vectors))
        vectors = np.stack([self.vector_cache[text] for text in texts])
        family_similarity = (vectors @ self.family_vectors.T if len(self.family_vectors)
                             else np.zeros((len(texts), 0), np.float32))
        consensus_similarity = vectors @ self.consensus_vector
        corpus_similarity = (vectors @ self.corpus_vectors.T if len(self.corpus_vectors)
                             else np.zeros((len(texts), 0), np.float32))
        result: list[dict[str, Any]] = []
        for index, text in enumerate(texts):
            family_mean = (float(np.mean(family_similarity[index]))
                           if family_similarity.shape[1] else 0.0)
            family_max = (float(np.max(family_similarity[index]))
                          if family_similarity.shape[1] else 0.0)
            if corpus_similarity.shape[1]:
                top = np.sort(corpus_similarity[index])[-min(3, corpus_similarity.shape[1]):]
                corpus_top3 = float(np.mean(top))
                corpus_best_index = int(np.argmax(corpus_similarity[index]))
                corpus_best = float(corpus_similarity[index, corpus_best_index])
            else:
                corpus_top3, corpus_best, corpus_best_index = 0.0, 0.0, -1
            semantic = (0.46 * float(consensus_similarity[index])
                        + 0.28 * family_mean + 0.12 * family_max
                        + 0.14 * max(0.0, corpus_top3))
            frame, frame_reasons = self.frame_score(text)
            best_row = self.retrieved[corpus_best_index] if corpus_best_index >= 0 else None
            result.append({
                "semantic_score": round(semantic, 6),
                "consensus_similarity": round(float(consensus_similarity[index]), 6),
                "family_mean_similarity": round(family_mean, 6),
                "family_max_similarity": round(family_max, 6),
                "corpus_top3_similarity": round(corpus_top3, 6),
                "corpus_best_similarity": round(corpus_best, 6),
                "corpus_best_id": best_row["id"] if best_row else None,
                "frame_score": round(frame, 6),
                "frame_reasons": frame_reasons,
            })
        return result

    def describe(self) -> dict[str, Any]:
        return {
            "candidate_generation": False,
            "external_api_at_runtime": False,
            "persistent_learning": False,
            "anchors": self.anchors,
            "retrieved_sentence_count": len(self.retrieved),
            "retrieved_sentences": self.retrieved,
        }


def apply_local_semantic_priors(slots: list[dict[str, Any]],
                                lattice: list[list[dict[str, Any]]],
                                previous_candidates: list[str],
                                agent: SemanticRetrievalAgent,
                                resolver: v4.LexiconResolver) -> list[dict[str, Any]]:
    """Inspect ambiguous local clauses before the global beam search.

    The encoder may add a bounded score only to an existing, acoustically
    supported lattice row.  It cannot insert a token or bypass sensitive gates.
    """
    baseline_rows: list[dict[str, Any]] = []
    for index, candidates in enumerate(lattice):
        previous = previous_candidates[index] if index < len(previous_candidates) else ""
        matched = next((row for row in candidates
                        if str(row.get("candidate") or "") == previous), None)
        baseline_rows.append(matched or max(
            candidates, key=lambda row: float(row.get("emission_score") or 0.0)))

    audits: list[dict[str, Any]] = []
    for index, candidates in enumerate(lattice):
        current = baseline_rows[index]
        current_token = str(current.get("candidate") or "")
        if not current_token:
            continue
        eligible = [row for row in sorted(
            candidates, key=lambda item: float(item.get("emission_score") or 0.0), reverse=True)
            if row.get("candidate") and v6.effective_family_count(row) >= 2][:6]
        if len(eligible) < 2:
            continue

        left, right = max(0, index - 6), min(len(lattice), index + 7)
        base_parts: list[str] = []
        slot_offsets: dict[int, tuple[int, int]] = {}
        for slot_index in range(left, right):
            start = len(base_parts)
            base_parts.extend(baseline_rows[slot_index].get("candidate_tokens") or [])
            slot_offsets[slot_index] = (start, len(base_parts))
        replace_start, replace_end = slot_offsets[index]
        variants = []
        for row in eligible:
            parts = [*base_parts[:replace_start],
                     *(row.get("candidate_tokens") or []),
                     *base_parts[replace_end:]]
            variants.append(text_from_parts(parts, limit=40))

        hypothesis_names = sorted({
            observation["hypothesis"]
            for slot in slots[left:right]
            for observation in slot["observations"].values()
        })
        context_texts: list[str] = []
        for hypothesis in hypothesis_names:
            context_tokens: list[str] = []
            for slot in slots[left:right]:
                observations = [row for row in slot["observations"].values()
                                if row["hypothesis"] == hypothesis]
                if observations:
                    best = max(observations, key=lambda row: float(row.get("probability") or 0.0))
                    context_tokens.append(best["normalized"])
            if context_tokens:
                context_texts.append(" ".join(context_tokens))
        if not context_texts:
            continue
        variant_vectors = agent.encoder.encode(variants)
        context_vectors = agent.encoder.encode(context_texts)
        centroid = context_vectors.mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-9)
        local_similarities = variant_vectors @ centroid
        global_analyses = agent.score_texts(variants)

        current_index = next((row_index for row_index, row in enumerate(eligible)
                              if row is current), None)
        if current_index is None:
            # The previous path can be outside the top six emissions. Compare
            # against its own local text without giving it an artificial prior.
            current_parts = [*base_parts[:replace_start],
                             *(current.get("candidate_tokens") or []),
                             *base_parts[replace_end:]]
            current_text = text_from_parts(current_parts, limit=40)
            current_vector = agent.encoder.encode([current_text])[0]
            current_local = float(current_vector @ centroid)
            current_global = float(agent.score_texts([current_text])[0]["semantic_score"])
            current_emission = float(current.get("emission_score") or 0.0)
        else:
            current_local = float(local_similarities[current_index])
            current_global = float(global_analyses[current_index]["semantic_score"])
            current_emission = float(current.get("emission_score") or 0.0)

        drug_rows = [row_index for row_index, row in enumerate(eligible)
                     if set(row.get("medical_categories") or resolver.medical_categories(
                         str(row.get("candidate") or "")))
                     & {"drug", "medication", "drug_class"}]
        current_drug = bool(resolver.medical_categories(current_token)
                            & {"drug", "medication", "drug_class"})
        # A complete licensed drug supported by two families can beat a
        # one-family function fragment.  Semantic evidence selects among drug
        # candidates but cannot waive the two-family requirement.
        if drug_rows and not current_drug and v6.effective_family_count(current) < 2:
            drug_index = max(drug_rows, key=lambda row_index: (
                0.55 * float(local_similarities[row_index])
                + 0.10 * float(global_analyses[row_index]["semantic_score"])
                + 0.08 * float(eligible[row_index].get("emission_score") or 0.0)
            ))
            winner = eligible[drug_index]
            before = float(winner.get("emission_score") or 0.0)
            prior = 4.8
            winner["emission_score_before_semantic_prior"] = round(before, 6)
            winner["emission_score"] = round(before + prior, 6)
            winner["local_semantic_prior"] = prior
            winner["local_semantic_prior_reason"] = "two-family-licensed-drug-over-one-family-fragment"
            audits.append({
                "slot": index, "from": current_token,
                "favored_candidate": winner.get("candidate"),
                "reason": winner["local_semantic_prior_reason"],
                "prior": prior,
                "local_similarity_before": round(current_local, 6),
                "local_similarity_favored": round(float(local_similarities[drug_index]), 6),
            })
            continue

        # Numbers, units, negation, and drug identity remain entirely under the
        # hard V6 rules unless the licensed drug rule above was satisfied.
        def sensitive(row: dict[str, Any]) -> bool:
            token = str(row.get("candidate") or "")
            categories = set(row.get("medical_categories") or resolver.medical_categories(token))
            return (v6.is_number(token) or token in v4.UNITS or v4.is_negative_token(token)
                    or bool(categories & {"drug", "medication", "drug_class"}))

        ordinary = [(row_index, row) for row_index, row in enumerate(eligible)
                    if (not sensitive(row)
                        and len(row.get("candidate_tokens") or []) == 1
                        and (row.get("general_lexicon") or row.get("medical_lexicon"))
                        and (float(row.get("zipf_frequency_fa") or 0.0) >= 2.0
                             or row.get("medical_lexicon")))]
        if (len(ordinary) < 2 or sensitive(current)
                or current_token in CONTENT_STOPWORDS
                or len(current.get("candidate_tokens") or []) != 1
                or (current.get("turbo_exact_anchor")
                    and v6.effective_family_count(current) >= 2
                    and current.get("general_lexicon"))):
            continue
        previous_token = next((str(baseline_rows[slot_index].get("candidate") or "")
                               for slot_index in range(index - 1, left - 1, -1)
                               if baseline_rows[slot_index].get("candidate")), "")
        following_token = next((str(baseline_rows[slot_index].get("candidate") or "")
                                for slot_index in range(index + 1, right)
                                if baseline_rows[slot_index].get("candidate")), "")

        def corpus_context_evidence(token: str) -> tuple[float, dict[str, int]]:
            left_count = 0
            right_count = 0
            trigram_count = 0
            if previous_token:
                left_count = int(agent.corpus.connection.execute(
                    "SELECT COALESCE(SUM(count),0) FROM bigrams WHERE w1=? AND w2=?",
                    (previous_token, token),
                ).fetchone()[0] or 0)
            if following_token:
                right_count = int(agent.corpus.connection.execute(
                    "SELECT COALESCE(SUM(count),0) FROM bigrams WHERE w1=? AND w2=?",
                    (token, following_token),
                ).fetchone()[0] or 0)
            if previous_token and following_token:
                trigram_count = int(agent.corpus.connection.execute(
                    "SELECT COALESCE(SUM(count),0) FROM trigrams "
                    "WHERE w1=? AND w2=? AND w3=?",
                    (previous_token, token, following_token),
                ).fetchone()[0] or 0)
            left_log = math.log1p(left_count)
            right_log = math.log1p(right_count)
            trigram_log = math.log1p(trigram_count)
            if previous_token and following_token:
                # A frequent right bigram must not replace a phrase that joins
                # correctly on both sides (for example «به شرطی که»).
                value = (4.0 * trigram_log + 2.0 * min(left_log, right_log)
                         + 0.5 * max(left_log, right_log))
            else:
                value = left_log + right_log
            return value, {
                "left_bigram": left_count,
                "right_bigram": right_count,
                "trigram": trigram_count,
            }

        emissions = [float(row.get("emission_score") or 0.0) for _row_index, row in ordinary]
        low, high = min(emissions), max(emissions)
        spread = max(0.35, high - low)
        context_evidence = [corpus_context_evidence(str(row.get("candidate") or ""))
                            for _row_index, row in ordinary]
        context_values = [value for value, _counts in context_evidence]
        max_context_value = max(1.0, max(context_values))
        unigram_counts = [int(agent.corpus.connection.execute(
            "SELECT COALESCE(SUM(count),0) FROM unigrams WHERE w1=?",
            (str(row.get("candidate") or ""),),
        ).fetchone()[0] or 0) for _row_index, row in ordinary]
        unigram_logs = [math.log1p(value) for value in unigram_counts]
        max_unigram_log = max(1.0, max(unigram_logs))
        ranked: list[tuple[float, int, dict[str, Any]]] = []
        score_by_identity: dict[int, float] = {}
        for position, ((row_index, row), emission) in enumerate(zip(ordinary, emissions)):
            emission_norm = (emission - low) / spread
            local_similarity = float(local_similarities[row_index])
            global_similarity = float(global_analyses[row_index]["semantic_score"])
            context_norm = context_values[position] / max_context_value
            unigram_norm = unigram_logs[position] / max_unigram_log
            score = (0.18 * emission_norm + 0.28 * local_similarity
                     + 0.08 * global_similarity + 0.34 * context_norm
                     + 0.12 * unigram_norm)
            ranked.append((score, row_index, row))
            score_by_identity[id(row)] = score
        ranked.sort(key=lambda item: item[0], reverse=True)
        winner_score, winner_index, winner = ranked[0]
        if id(current) in score_by_identity:
            current_score = score_by_identity[id(current)]
        else:
            current_context = corpus_context_evidence(current_token)[0] / max_context_value
            current_unigram = int(agent.corpus.connection.execute(
                "SELECT COALESCE(SUM(count),0) FROM unigrams WHERE w1=?",
                (current_token,),
            ).fetchone()[0] or 0)
            current_score = (0.18 * ((current_emission - low) / spread)
                             + 0.28 * current_local + 0.08 * current_global
                             + 0.34 * current_context
                             + 0.12 * (math.log1p(current_unigram) / max_unigram_log))
        winner_local = float(local_similarities[winner_index])
        winner_token = str(winner.get("candidate") or "")
        current_unigram_count = int(agent.corpus.connection.execute(
            "SELECT COALESCE(SUM(count),0) FROM unigrams WHERE w1=?",
            (current_token,),
        ).fetchone()[0] or 0)
        winner_unigram_count = int(agent.corpus.connection.execute(
            "SELECT COALESCE(SUM(count),0) FROM unigrams WHERE w1=?",
            (winner_token,),
        ).fetchone()[0] or 0)
        semantic_improvement = winner_local >= current_local + 0.01
        stem_related = ((current_token.startswith(winner_token)
                         or winner_token.startswith(current_token))
                        and v4.token_similarity(current_token, winner_token) >= 0.80)
        safe_oov_repair = (current_unigram_count == 0 and winner_unigram_count > 0
                           and stem_related
                           and winner_local >= current_local - 0.005)
        current_context_value, current_context_counts = corpus_context_evidence(current_token)
        winner_context_value, winner_context_counts = corpus_context_evidence(winner_token)
        context_improvement = winner_context_value > current_context_value + 0.05
        close_inflection = (v4.token_similarity(current_token, winner_token) >= 0.80
                            and current_unigram_count > 0)
        inflection_has_strong_phrase_evidence = (
            winner_context_counts["trigram"] > current_context_counts["trigram"]
            and winner_local >= current_local + 0.08
        )
        if (winner is current or winner_token in CONTENT_STOPWORDS
                or winner_score < current_score + 0.035
                or not (semantic_improvement or safe_oov_repair)
                or not (context_improvement or safe_oov_repair)
                or (close_inflection and not inflection_has_strong_phrase_evidence)):
            continue
        before = float(winner.get("emission_score") or 0.0)
        prior = min(6.0, 3.5 + 10.0 * (winner_score - current_score))
        winner["emission_score_before_semantic_prior"] = round(before, 6)
        winner["emission_score"] = round(before + prior, 6)
        winner["local_semantic_prior"] = round(prior, 6)
        winner["local_semantic_prior_reason"] = "local-six-context-plus-semantic-retrieval"
        audits.append({
            "slot": index, "from": current_token,
            "favored_candidate": winner.get("candidate"),
            "reason": winner["local_semantic_prior_reason"],
            "prior": round(prior, 6),
            "combined_score_before": round(current_score, 6),
            "combined_score_favored": round(winner_score, 6),
            "local_similarity_before": round(current_local, 6),
            "local_similarity_favored": round(winner_local, 6),
            "corpus_context_before": corpus_context_evidence(current_token)[1],
            "corpus_context_favored": corpus_context_evidence(winner_token)[1],
        })
    return audits


def authorized_semantic_lattice(
        lattice: list[list[dict[str, Any]]], plain_path: list[int],
        local_semantic_priors: list[dict[str, Any]],
        resolver: v4.LexiconResolver,
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    """Limit neural reranking to audited ambiguity or a licensed medical rescue."""
    local_slots = {int(row["slot"]) for row in local_semantic_priors}
    constrained: list[list[dict[str, Any]]] = []
    audit: list[dict[str, Any]] = []
    for index, candidates in enumerate(lattice):
        plain = candidates[plain_path[index]]
        allowed = index in local_slots
        reason = "local-semantic-prior" if allowed else "plain-v6-locked"
        if not allowed:
            plain_token = str(plain.get("candidate") or "")
            plain_categories = set(
                plain.get("medical_categories") or resolver.medical_categories(plain_token))
            plain_lexical = bool(plain.get("general_lexicon") or plain.get("medical_lexicon")
                                 or plain.get("modern_spoken"))
            medical_rescues = []
            for row in candidates:
                token = str(row.get("candidate") or "")
                categories = set(
                    row.get("medical_categories") or resolver.medical_categories(token))
                rescue_categories = categories & {
                    "drug", "medication", "drug_class", "medical_device"
                }
                if (token and rescue_categories and v6.effective_family_count(row) >= 2
                        and (row.get("medical_lexicon") or row.get("general_lexicon"))
                        and row.get("origin") == "general-lexicon"
                        and (not plain_categories)
                        and (not plain_lexical or v6.effective_family_count(plain) < 2)):
                    medical_rescues.append(row)
            if medical_rescues:
                allowed = True
                reason = "two-family-licensed-medical-rescue"
        constrained.append(candidates if allowed else [plain])
        audit.append({
            "slot": index,
            "plain_candidate": plain.get("candidate"),
            "semantic_rerank_allowed": allowed,
            "reason": reason,
            "candidate_count": len(candidates if allowed else [plain]),
        })
    return constrained, audit


def decode_semantic_lattice(lattice: list[list[dict[str, Any]]],
                            ngrams: v5.DomainNgramEvidence,
                            phrase_evidence: v6.PhraseEvidence,
                            resolver: v4.LexiconResolver,
                            agent: SemanticRetrievalAgent,
                            beam_size: int = 72) -> tuple[list[int], list[dict[str, Any]], dict[str, Any]]:
    futures = [v6.future_anchor_tokens(lattice, index) for index in range(len(lattice))]
    right_cache: dict[tuple[int, int], tuple[float, list[dict[str, Any]]]] = {}
    for slot_index, candidates in enumerate(lattice):
        for candidate_index, row in enumerate(candidates):
            right_cache[(slot_index, candidate_index)] = v6.right_context_score(
                row, futures[slot_index], ngrams)

    beam = [{
        "score": 0.0, "semantic_bias": 0.0, "history": tuple(), "tokens": tuple(),
        "path": [], "transitions": [], "semantic_checkpoints": [],
    }]
    checkpoint_count = 0
    for slot_index, candidates in enumerate(lattice):
        expanded: list[dict[str, Any]] = []
        for state in beam:
            for candidate_index, row in enumerate(candidates):
                parts = tuple(row.get("candidate_tokens") or [])
                history = state["history"]
                forward_score = 0.0
                redundancy = 0.0
                forward_details = []
                next_history = history
                for part in parts:
                    value, detail = v6.calibrated_transition_score(
                        ngrams, next_history[-2:], part)
                    forward_score += max(-0.50, min(4.5, value))
                    penalty, reason = v4.redundancy_penalty(next_history[-2:], part)
                    redundancy += penalty
                    forward_details.append({"ngram": detail, "redundancy_reason": reason})
                    next_history = (*next_history[-(v6.MAX_PHRASE_ORDER - 2):], part)
                phrase_score, phrase_details = phrase_evidence.trailing_score(history, parts)
                applied_phrase_score = 0.34 * min(4.0, phrase_score)
                entity_score, entity_reasons = v6.entity_structure_score(
                    history, parts, futures[slot_index], row, resolver)
                right_score, right_details = right_cache[(slot_index, candidate_index)]
                acoustic = 0.52 * float(row.get("emission_score") or 0.0)
                family_bonus = 0.55 * v6.effective_family_count(row)
                source_bonus = 0.08 * min(6, int(row.get("exact_sources") or 0))
                drug_categories = set(row.get("medical_categories") or [])
                turbo_component = (3.00 if row.get("turbo_exact_anchor") and
                                   drug_categories & {"drug", "medication", "drug_class"}
                                   else 0.75 if row.get("turbo_exact_anchor")
                                   else 0.12 if float(row.get("turbo_anchor_similarity") or 0.0) >= 0.78
                                   else 0.0)
                lexical_valid = bool(row.get("general_lexicon") or row.get("medical_lexicon")
                                     or row.get("modern_spoken")
                                     or all(v6.is_number(part) for part in parts))
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
                    "semantic_bias": state["semantic_bias"],
                    "history": next_history[-(v6.MAX_PHRASE_ORDER - 1):],
                    "tokens": (*state["tokens"], *parts),
                    "path": [*state["path"], candidate_index],
                    "transitions": [*state["transitions"], transition],
                    "semantic_checkpoints": state["semantic_checkpoints"],
                })

        # Keep two paths for the same trailing history so semantic evidence can
        # still distinguish earlier clauses that a normal Markov beam collapses.
        grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for state in sorted(expanded, key=lambda item: item["score"] + item["semantic_bias"],
                            reverse=True):
            if len(grouped[state["history"]]) < 2:
                grouped[state["history"]].append(state)
        pool = [state for values in grouped.values() for state in values]
        pool.sort(key=lambda item: item["score"] + item["semantic_bias"], reverse=True)
        pool = pool[:beam_size]

        checkpoint = ((slot_index + 1) % CHECKPOINT_SLOTS == 0
                      or slot_index == len(lattice) - 1)
        if checkpoint and pool:
            checkpoint_count += 1
            texts = [text_from_parts(state["tokens"]) for state in pool]
            analyses = agent.score_texts(texts)
            for state, analysis in zip(pool, analyses):
                semantic_component = SEMANTIC_WEIGHT * float(analysis["semantic_score"])
                frame_component = FRAME_WEIGHT * float(analysis["frame_score"])
                state["semantic_bias"] = semantic_component + frame_component
                state["semantic_checkpoints"] = [*state["semantic_checkpoints"], {
                    "slot": slot_index,
                    "window_text": text_from_parts(state["tokens"]),
                    **analysis,
                    "semantic_rank_component": round(semantic_component, 6),
                    "frame_rank_component": round(frame_component, 6),
                }]
        beam = sorted(pool, key=lambda item: item["score"] + item["semantic_bias"],
                      reverse=True)[:beam_size]
    best = max(beam, key=lambda item: item["score"] + item["semantic_bias"])
    detail = {
        "base_decoder_score": round(float(best["score"]), 6),
        "semantic_bias": round(float(best["semantic_bias"]), 6),
        "combined_rank_score": round(float(best["score"] + best["semantic_bias"]), 6),
        "checkpoint_count": checkpoint_count,
        "checkpoints": best["semantic_checkpoints"],
        "semantic_weight": SEMANTIC_WEIGHT,
        "frame_weight": FRAME_WEIGHT,
        "beam_size": beam_size,
    }
    return best["path"], best["transitions"], detail


def write_outputs(run_dir: Path, selected: list[dict[str, Any]], final_text: str,
                  raw_selected_text: str, placeholders: list[dict[str, Any]],
                  protected_names: list[dict[str, Any]], validation: dict[str, Any],
                  corpus: v5.DomainCorpus, ngrams: v5.DomainNgramEvidence,
                  encoder: OnnxSentenceEncoder, agent: SemanticRetrievalAgent,
                  semantic_decode: dict[str, Any], elapsed: float,
                  dose_locks: list[dict[str, Any]], semantic_candidate_stats: dict[str, int],
                  numeric_canonicalizations: list[dict[str, Any]],
                  path_changes: list[dict[str, Any]],
                  local_semantic_priors: list[dict[str, Any]]) -> dict[str, Any]:
    out_dir = run_dir / OUTPUT_RELATIVE
    out_dir.mkdir(parents=True, exist_ok=True)
    review = [row for row in selected if row.get("status") == "REVIEW"]
    clips, _clip_limit = v4.make_review_clips(run_dir, out_dir, review)
    payload = {
        "algorithm": "v7 local semantic-retrieval agent over the V6 candidate lattice",
        "generative_llm_used": False,
        "pretrained_semantic_encoder_used": True,
        "encoder_generates_text": False,
        "external_api_used_at_runtime": False,
        "old_pipeline_modified": False,
        "output_replaces_v6": False,
        "candidate_policy": (
            "encoder and corpus only rerank candidates already licensed by the six-ASR V6 lattice"
        ),
        "runtime_seconds": round(elapsed, 3),
        "text": final_text,
        "raw_selected_text": raw_selected_text,
        "encoder": encoder.describe(),
        "semantic_agent": agent.describe(),
        "semantic_decode": semantic_decode,
        "local_semantic_priors": local_semantic_priors,
        "path_changes_from_plain_v6_decoder": path_changes,
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
    (out_dir / "final-v7.txt").write_text(final_text + "\n", encoding="utf-8")
    (out_dir / "final-v7.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "review-v7.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "review-clips-v7.json").write_text(
        json.dumps(clips, ensure_ascii=False, indent=2), encoding="utf-8")

    review_lines = [
        "# بازبینی V7 معنایی محلی", "",
        "Encoder فقط نامزدهای شش ASR را رتبه‌بندی کرده و هیچ واژه‌ای تولید نکرده است.", "",
        "| زمان | انتخاب | وضعیت | دلیل |", "|---:|---|---|---|",
    ]
    for row in review:
        review_lines.append(
            f"| {row['midpoint']:.2f} | {row.get('candidate') or 'ε'} | "
            f"{row['status']} | {row['reason']} |")
    (out_dir / "review-v7.md").write_text("\n".join(review_lines) + "\n", encoding="utf-8")

    previous = run_dir / v6.OUTPUT_RELATIVE / "final-v6.txt"
    comparison = ["# مقایسهٔ V6 و V7 معنایی", ""]
    if previous.is_file():
        comparison += ["## V6 — بدون تغییر", "", previous.read_text(encoding="utf-8").strip(), ""]
    comparison += ["## V7 — Encoder غیرمولد و بازیابی محلی", "", final_text, ""]
    comparison += ["## تغییر مسیر", "", f"تعداد اسلات تغییرکرده: {len(path_changes)}", ""]
    (out_dir / "comparison-v7.md").write_text("\n".join(comparison), encoding="utf-8")

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
        "local_semantic_prior_count": len(local_semantic_priors),
        "generative_llm_used": False,
        "pretrained_semantic_encoder_used": True,
        "encoder_generates_text": False,
        "old_pipeline_modified": False,
        "text": final_text,
    }
    (out_dir / "summary-v7.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output": str(out_dir), **summary}


def run(run_dir: Path, medical_index: Path, corpus_index: Path,
        encoder_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    hypotheses = load_hypotheses(run_dir)
    sequences = {key: words_of(hypotheses[key], key) for key in NETWORK_ORDER}
    medical_payload = json.loads(medical_index.read_text(encoding="utf-8"))
    resolver = v4.LexiconResolver(medical_payload)
    corpus = v5.DomainCorpus(corpus_index)
    try:
        ngrams = v5.DomainNgramEvidence(sequences, resolver, corpus)
        phrase_evidence = v6.PhraseEvidence(sequences)
        slots: list[dict[str, Any]] = []
        for hypothesis in NETWORK_ORDER:
            slots = v4.add_sequence_to_network(slots, sequences[hypothesis], hypothesis)
        lattice = [v4.build_slot_candidates(slot, resolver, ngrams) for slot in slots]
        v5.augment_adjacent_acoustic_support(slots, lattice)
        semantic_candidate_stats = v6.augment_semantic_candidates(
            slots, lattice, resolver, ngrams)
        semantic_candidate_stats.update(v6.augment_medication_frames(
            slots, lattice, resolver, ngrams))
        v6.mark_turbo_anchors(slots, lattice)
        lattice = v6.prune_lattice(lattice)

        # Compute the plain V6 path in memory.  V7 therefore works on a fresh
        # upload and never depends on a previously written V6 folder.
        plain_path, _plain_transitions = v6.decode_phrase_lattice(
            lattice, ngrams, phrase_evidence, resolver)
        previous_candidates = [str(lattice[index][choice].get("candidate") or "")
                               for index, choice in enumerate(plain_path)]
        encoder = OnnxSentenceEncoder(encoder_dir)
        agent = SemanticRetrievalAgent(
            encoder, hypotheses, sequences, resolver, corpus, ngrams.medical_mix, lattice)
        local_semantic_priors = apply_local_semantic_priors(
            slots, lattice, previous_candidates, agent, resolver)
        constrained_lattice, authorization_audit = authorized_semantic_lattice(
            lattice, plain_path, local_semantic_priors, resolver)
        constrained_path, transitions, semantic_decode = decode_semantic_lattice(
            constrained_lattice, ngrams, phrase_evidence, resolver, agent)
        path = []
        for index, choice in enumerate(constrained_path):
            chosen_row = constrained_lattice[index][choice]
            path.append(next(row_index for row_index, row in enumerate(lattice[index])
                             if row is chosen_row))
        semantic_decode["authorization_audit"] = authorization_audit
        semantic_decode["authorized_slot_count"] = sum(
            bool(row["semantic_rerank_allowed"]) for row in authorization_audit)
        selected = v6.classify(slots, lattice, path, transitions)
        entity_fragment_cleanups = v6.cleanup_entity_fragments(selected)
        semantic_candidate_stats["entity_fragment_cleanups"] = len(entity_fragment_cleanups)
        doses = cluster_doses(dose_occurrences(sequences))
        dose_locks = v4.apply_dose_locks(selected, doses)
        numeric_canonicalizations = v6.canonicalize_structured_numbers(selected, resolver)
        path_changes = []
        for index, row in enumerate(selected):
            before = previous_candidates[index] if index < len(previous_candidates) else ""
            after = str(row.get("candidate") or "")
            if before != after:
                path_changes.append({
                    "slot": index,
                    "plain_v6_candidate": before,
                    "semantic_v7_candidate": after,
                })
        raw_selected_text, _render_operations = v4.render(selected)
        validation = v6.validate_v6(slots, sequences, selected, raw_selected_text)
        placeholder_text, placeholders = v6.placeholder_render(selected, resolver)
        final_text, protected_names = v5.protect_honorific_names(placeholder_text)
        return write_outputs(
            run_dir, selected, final_text, raw_selected_text, placeholders,
            protected_names, validation, corpus, ngrams, encoder, agent,
            semantic_decode, time.perf_counter() - started, dose_locks,
            semantic_candidate_stats, numeric_canonicalizations, path_changes,
            local_semantic_priors)
    finally:
        corpus.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Independent six-ASR semantic retrieval agent using a local non-generative ONNX encoder."
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
