from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import consensus_v4 as v4
from consensus_v2 import BASE_KEY
from consensus_v3 import words_of


OUTPUT_RELATIVE = Path("final-delivery") / "02-after-algorithm-v5-domain-corpus"
SCORE_FILENAME = "ngram-turbo-quality-v3.json"
MEDICAL_CUES = {
    "پزشک", "دکتر", "بیمار", "بیماری", "دارو", "داروی", "قرص", "کپسول", "شربت",
    "آمپول", "دوز", "مصرف", "میلیگرم", "درمان", "آزمایش", "قند", "خون", "نسخه",
}


class NextWordPredictor:
    """Interpolated trigram/bigram probabilities from the frozen domain corpus."""

    def __init__(self, connection: sqlite3.Connection, medical_mix: float) -> None:
        self.connection = connection
        medical_weight = max(0.0, min(1.0, float(medical_mix)))
        raw_weights = {"medical": medical_weight, "daily": 1.0 - 0.45 * medical_weight}
        total = sum(raw_weights.values()) or 1.0
        self.domain_weights = {key: value / total for key, value in raw_weights.items()}

    @lru_cache(maxsize=100_000)
    def _rows(self, domain: str, order: int,
              history: tuple[str, ...]) -> tuple[int, tuple[tuple[str, int], ...]]:
        if order == 3 and len(history) >= 2:
            rows = self.connection.execute(
                "SELECT w3,count FROM trigrams WHERE domain=? AND w1=? AND w2=? "
                "ORDER BY count DESC LIMIT 24",
                (domain, history[-2], history[-1]),
            ).fetchall()
            total = self.connection.execute(
                "SELECT COALESCE(SUM(count),0) FROM trigrams "
                "WHERE domain=? AND w1=? AND w2=?",
                (domain, history[-2], history[-1]),
            ).fetchone()[0]
            return int(total or 0), tuple((str(word), int(count)) for word, count in rows)
        if order == 2 and history:
            rows = self.connection.execute(
                "SELECT w2,count FROM bigrams WHERE domain=? AND w1=? "
                "ORDER BY count DESC LIMIT 24",
                (domain, history[-1]),
            ).fetchall()
            total = self.connection.execute(
                "SELECT COALESCE(SUM(count),0) FROM bigrams WHERE domain=? AND w1=?",
                (domain, history[-1]),
            ).fetchone()[0]
            return int(total or 0), tuple((str(word), int(count)) for word, count in rows)
        if order == 1:
            rows = self.connection.execute(
                "SELECT w1,count FROM unigrams WHERE domain=? ORDER BY count DESC LIMIT 24",
                (domain,),
            ).fetchall()
            total = self.connection.execute(
                "SELECT COALESCE(SUM(count),0) FROM unigrams WHERE domain=?",
                (domain,),
            ).fetchone()[0]
            return int(total or 0), tuple((str(word), int(count)) for word, count in rows)
        return 0, tuple()

    @lru_cache(maxsize=200_000)
    def _observed_count(self, domain: str, order: int,
                        history: tuple[str, ...], word: str) -> int:
        if order == 3 and len(history) >= 2:
            row = self.connection.execute(
                "SELECT count FROM trigrams WHERE domain=? AND w1=? AND w2=? AND w3=?",
                (domain, history[-2], history[-1], word),
            ).fetchone()
            if row:
                return int(row[0])
        elif order == 2 and history:
            row = self.connection.execute(
                "SELECT count FROM bigrams WHERE domain=? AND w1=? AND w2=?",
                (domain, history[-1], word),
            ).fetchone()
            if row:
                return int(row[0])
        elif order == 1:
            row = self.connection.execute(
                "SELECT count FROM unigrams WHERE domain=? AND w1=?", (domain, word),
            ).fetchone()
            if row:
                return int(row[0])
        return 0

    @staticmethod
    def _interpolation_weights(trigram_total: int, bigram_total: int) -> dict[int, float]:
        """Use all orders; trust higher-order contexts only when they have evidence."""
        trigram_weight = trigram_total / (trigram_total + 6.0) if trigram_total else 0.0
        remainder = 1.0 - trigram_weight
        bigram_gate = bigram_total / (bigram_total + 18.0) if bigram_total else 0.0
        bigram_weight = remainder * bigram_gate
        return {3: trigram_weight, 2: bigram_weight, 1: remainder - bigram_weight}

    def predict(self, history: tuple[str, ...], observed: str) -> dict[str, Any]:
        history = tuple(v4.norm(token) for token in history[-2:] if v4.norm(token))
        observed = v4.norm(observed)
        probabilities: dict[str, float] = {}
        observed_probability = 0.0
        evidence = 0.0
        by_domain: dict[str, dict[str, Any]] = {}
        for domain, weight in self.domain_weights.items():
            order_data = {order: self._rows(domain, order, history) for order in (1, 2, 3)}
            order_maps = {order: dict(order_data[order][1]) for order in (1, 2, 3)}
            interpolation = self._interpolation_weights(
                order_data[3][0], order_data[2][0])
            observed_counts = {}
            observed_domain_probability = 0.0
            candidates = {word for _total, rows in order_data.values() for word, _count in rows}
            candidates.add(observed)
            for order in (1, 2, 3):
                total = order_data[order][0]
                count = self._observed_count(domain, order, history, observed) if total else 0
                observed_counts[order] = count
                order_maps[order][observed] = count
                observed_domain_probability += (
                    interpolation[order] * count / total if total else 0.0)
            observed_probability += weight * observed_domain_probability
            evidence += weight * (order_data[3][0] + order_data[2][0])
            for candidate in candidates:
                candidate_probability = 0.0
                for order in (1, 2, 3):
                    total = order_data[order][0]
                    if not total:
                        continue
                    count = order_maps[order].get(candidate, 0)
                    candidate_probability += interpolation[order] * count / total
                probabilities[candidate] = probabilities.get(candidate, 0.0) + weight * candidate_probability
            by_domain[domain] = {
                "prefix_counts": {str(order): order_data[order][0] for order in (1, 2, 3)},
                "observed_counts": {str(order): observed_counts[order] for order in (1, 2, 3)},
                "interpolation_weights": {
                    str(order): round(interpolation[order], 6) for order in (1, 2, 3)},
            }
        ranked = sorted(probabilities.items(), key=lambda item: (-item[1], item[0]))[:5]
        top_probability = ranked[0][1] if ranked else 0.0
        exact_relative = observed_probability / top_probability if top_probability else 0.0
        relative = exact_relative
        # Count close colloquial/phonetic continuations (گفتین/گفتید) without
        # admitting an unrelated word merely because it exists in a dictionary.
        if top_probability:
            for predicted_word, probability in ranked:
                similarity = v4.token_similarity(observed, predicted_word)
                if similarity < 0.68:
                    continue
                similarity_gate = (similarity - 0.68) / 0.32
                relative = max(relative, probability / top_probability * similarity_gate)
        return {
            "history": list(history),
            "heard": observed,
            "heard_probability": observed_probability,
            "top_probability": top_probability,
            "exact_relative_to_best": min(1.0, exact_relative),
            "relative_to_best": min(1.0, relative),
            "evidence": evidence,
            "predicted": [{"word": word, "probability": probability}
                          for word, probability in ranked],
            "by_domain": by_domain,
        }


def detect_medical_mix(tokens: list[str], resolver: v4.LexiconResolver) -> tuple[float, dict[str, int]]:
    """Detect domain from Turbo alone; the other five hypotheses are never inspected."""
    cue_hits = sum(token in MEDICAL_CUES for token in tokens)
    lexicon_hits = sum(resolver.is_medical(token) for token in tokens)
    if cue_hits >= 2 or lexicon_hits >= 4:
        medical_mix = 0.90
    elif cue_hits or lexicon_hits >= 2:
        medical_mix = 0.65
    elif lexicon_hits:
        medical_mix = 0.45
    else:
        medical_mix = 0.20
    return medical_mix, {"medical_cue_hits": cue_hits, "medical_lexicon_hits": lexicon_hits}


def lexical_validity(token: str, resolver: v4.LexiconResolver) -> float:
    token = v4.norm(token)
    if (resolver.is_medical(token) or v4.active_general_word(token)
            or token in v4.NUMBER_WORDS or token in v4.UNITS or token in v4.NEGATIONS):
        return 1.0
    frequency = v4.usage_frequency(token)
    return max(0.0, min(0.85, (frequency - 1.5) / 2.5))


def score_voice_text(text: str, resolver: v4.LexiconResolver,
                     connection: sqlite3.Connection,
                     turbo_words: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Score Turbo with interpolated n-grams, acoustics, lexicon, and bad streaks."""
    if turbo_words:
        tokens = [v4.norm(row.get("normalized") or row.get("word") or "")
                  for row in turbo_words]
        acoustic_probabilities = [max(0.0, min(1.0, float(row.get("probability") or 0.0)))
                                  for row in turbo_words]
    else:
        tokens = v4.phrase_tokens(text)
        acoustic_probabilities = [0.0] * len(tokens)
    valid_rows = [(token, acoustic_probabilities[index]) for index, token in enumerate(tokens) if token]
    tokens = [token for token, _probability in valid_rows]
    acoustic_probabilities = [probability for _token, probability in valid_rows]
    lexical_scores = [lexical_validity(token, resolver) for token in tokens]
    medical_mix, domain_detection = detect_medical_mix(tokens, resolver)
    predictor = NextWordPredictor(connection, medical_mix)
    transitions: list[dict[str, Any]] = []
    ngram_scores: list[float] = []
    context_covered = 0
    strong_surprises = 0
    longest_bad_streak = 0
    current_bad_streak = 0
    history: tuple[str, ...] = tuple()
    for index, word in enumerate(tokens):
        if not history:
            history = (word,)
            continue
        prediction = predictor.predict(history, word)
        context_supported = prediction["evidence"] > 0.0
        context_covered += int(context_supported)
        # A square root keeps a plausible but non-leading continuation from
        # being treated like an impossible one while preserving large gaps.
        corpus_plausibility = math.sqrt(prediction["relative_to_best"])
        transition_score = 100.0 * corpus_plausibility
        ngram_scores.append(transition_score)
        surprise = bool(
            context_supported
            and prediction["relative_to_best"] < 0.08
        )
        strong_surprises += int(surprise)
        acoustic_probability = acoustic_probabilities[index]
        lexical_score = lexical_scores[index]
        severe_mismatch = bool(transition_score < 12.0 and acoustic_probability < 0.72)
        if severe_mismatch:
            current_bad_streak += 1
            longest_bad_streak = max(longest_bad_streak, current_bad_streak)
        else:
            current_bad_streak = 0
        transitions.append({
            **prediction,
            "acoustic_probability": round(acoustic_probability, 6),
            "lexical_validity": round(lexical_score, 6),
            "transition_score": round(transition_score, 3),
            "strong_surprise": surprise,
            "severe_mismatch": severe_mismatch,
        })
        history = (*history[-1:], word)

    transition_count = len(ngram_scores)
    coverage = context_covered / transition_count if transition_count else 0.0
    mean_transition = sum(ngram_scores) / transition_count if transition_count else 0.0
    ordered_scores = sorted(ngram_scores)
    lower_quartile = (ordered_scores[math.floor((transition_count - 1) * 0.25)]
                      if transition_count else 0.0)
    ngram_quality = 0.70 * mean_transition + 0.30 * lower_quartile
    acoustic_quality = (100.0 * sum(acoustic_probabilities) / len(acoustic_probabilities)
                        if acoustic_probabilities else 0.0)
    lexical_quality = (100.0 * sum(lexical_scores) / len(lexical_scores)
                       if lexical_scores else 0.0)
    streak_quality = 100.0 * math.exp(-0.38 * max(0, longest_bad_streak - 1))
    low_transition_count = sum(score < 10.0 for score in ngram_scores)
    low_transition_rate = low_transition_count / transition_count if transition_count else 1.0
    final_score = max(0.0, min(100.0,
        0.50 * ngram_quality
        + 0.30 * acoustic_quality
        + 0.10 * lexical_quality
        + 0.10 * streak_quality))
    return {
        "score": round(final_score, 1),
        "scale": "0-100",
        "algorithm_version": 3,
        "method": "Turbo enhanced: interpolated trigram/bigram/unigram + acoustic + lexicon + bad-streak penalty",
        "weights": {"ngram": 0.50, "turbo_acoustic": 0.30,
                    "lexical_validity": 0.10, "bad_streak": 0.10},
        "scoring_source": BASE_KEY,
        "six_hypotheses_used_for_score": False,
        "meaning": "estimated Turbo n-gram coherence, not ground-truth accuracy",
        "text_was_changed": False,
        "fallback_used": False,
        "token_count": len(tokens),
        "transition_count": transition_count,
        "corpus_context_coverage": round(coverage, 6),
        "mean_transition_score": round(mean_transition, 3),
        "lower_quartile_transition_score": round(lower_quartile, 3),
        "component_scores": {
            "ngram": round(ngram_quality, 3),
            "turbo_acoustic": round(acoustic_quality, 3),
            "lexical_validity": round(lexical_quality, 3),
            "bad_streak": round(streak_quality, 3),
        },
        "low_transition_rate": round(low_transition_rate, 6),
        "longest_severe_mismatch_streak": longest_bad_streak,
        "strong_surprise_count": strong_surprises,
        "medical_mix": round(medical_mix, 4),
        "domain_detection": domain_detection,
        "transitions": transitions,
    }


def score_existing_run(run_dir: Path, medical_index: Path, corpus_index: Path) -> dict[str, Any]:
    # Local import avoids a circular dependency when consensus_v5 imports this
    # scoring function for future live runs.
    from consensus_v5 import DomainCorpus

    turbo_path = run_dir / "hypotheses" / BASE_KEY / f"{BASE_KEY}.json"
    if not turbo_path.is_file():
        raise FileNotFoundError(f"Turbo base hypothesis is missing: {turbo_path}")
    turbo_payload = json.loads(turbo_path.read_text(encoding="utf-8"))
    if f"{turbo_payload.get('model')}__{turbo_payload.get('source')}" != BASE_KEY:
        raise RuntimeError(f"Unexpected Turbo hypothesis identity: {turbo_path}")
    medical_payload = json.loads(medical_index.read_text(encoding="utf-8"))
    resolver = v4.LexiconResolver(medical_payload)
    corpus = DomainCorpus(corpus_index)
    try:
        turbo_text = str(turbo_payload.get("text") or "").strip()
        if not turbo_text:
            raise RuntimeError(f"Turbo base text is empty: {run_dir}")
        turbo_words = words_of(turbo_payload, BASE_KEY)
        result = score_voice_text(turbo_text, resolver, corpus.connection, turbo_words)
    finally:
        corpus.close()
    result.update({"run_dir": str(run_dir.resolve()), "scored_text": turbo_text})
    out_path = run_dir / OUTPUT_RELATIVE / SCORE_FILENAME
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="One n-gram quality score per voice; no text changes.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--medical-index", type=Path, required=True)
    parser.add_argument("--corpus-index", type=Path, required=True)
    args = parser.parse_args()
    result = score_existing_run(
        args.run_dir.resolve(), args.medical_index.resolve(), args.corpus_index.resolve())
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps({key: value for key, value in result.items() if key != "transitions"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
