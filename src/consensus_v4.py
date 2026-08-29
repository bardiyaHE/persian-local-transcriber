from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import mnk_persian_words as persian_words
from rapidfuzz import fuzz, process

from consensus_v2 import (
    MODERN_SPOKEN,
    USER_BLOCKLIST,
    general_contains,
    morphology_supported,
    norm,
    usage_frequency,
)
from consensus_v3 import (
    FAMILIES,
    NETWORK_ORDER,
    cluster_doses,
    dose_occurrences,
    load_hypotheses,
    words_of,
)


OUTPUT_RELATIVE = Path("final-delivery") / "02-after-algorithm-v4-ngram-lexicon"
TOKEN_RE = re.compile(r"[0-9۰-۹]+|[؀-ۿ]+")
NUMBER_WORDS = {
    "صفر", "یک", "دو", "سه", "چهار", "پنج", "شش", "هفت", "هشت", "نه", "ده",
    "یازده", "دوازده", "سی", "چهل", "پنجا", "پنجاه", "صد", "هزار",
}
NEGATIONS = {
    "نه", "نیست", "نبود", "نشد", "ندارد", "ندارم", "نداریم", "نخورید",
    "نکنید", "نمی", "نباید", "منفی",
}
DRUG_CONTEXT = {
    "دارو", "داروی", "قرص", "کپسول", "شربت", "آمپول", "دوز", "مصرف",
}
UNITS = {
    "میلیگرم", "میلی", "گرم", "میکروگرم", "واحد", "درصد", "لیتر", "میلیلیتر",
}
FUNCTION_WORDS = {
    "از", "به", "با", "برای", "در", "رو", "و", "یا", "که", "این", "اون", "هم", "یک",
    "می", "نمی", "تا", "ها", "های", "روزی", "بعد", "قبل", "فقط", "مثل", "مثلا",
}
PHONETIC_TRANSLATION = str.maketrans({
    "ص": "س", "ث": "س", "ذ": "ز", "ض": "ز", "ظ": "ز", "ط": "ت",
    "ح": "ه", "غ": "ق", "آ": "ا", "ؤ": "و", "ئ": "ی", "أ": "ا", "إ": "ا",
})


def phrase_tokens(text: str) -> list[str]:
    return [token for token in (norm(match.group(0)) for match in TOKEN_RE.finditer(str(text))) if token]


def phonetic_key(token: str) -> str:
    value = norm(token).translate(PHONETIC_TRANSLATION)
    return re.sub(r"(.)\1{2,}", r"\1\1", value)


@lru_cache(maxsize=100_000)
def productive_stems(token: str) -> frozenset[str]:
    """Return frequent stems behind short productive Persian suffixes."""
    token = norm(token)
    stems = {token}
    for suffix in ("هایی", "های", "ها", "ترین", "تر", "ام", "ات", "اش", "یم", "ین", "ید", "ند",
                   "م", "ت", "ش", "ه", "ی"):
        if not token.endswith(suffix) or len(token) <= len(suffix) + 2:
            continue
        stem = token[:-len(suffix)]
        if general_contains(stem) and usage_frequency(stem) >= 3.25:
            stems.add(stem)
    return frozenset(stems)


@lru_cache(maxsize=200_000)
def token_similarity(left: str, right: str) -> float:
    left, right = norm(left), norm(right)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    direct = fuzz.ratio(left, right) / 100.0
    phonetic = fuzz.ratio(phonetic_key(left), phonetic_key(right)) / 100.0
    # Whisper often hears a colloquial copula/enclitic differently across sizes
    # (گیجم/گیجه). Shared, frequent stems count as strong fuzzy acoustic support;
    # this does not invent a word and exact family votes remain distinct.
    morphological = 0.84 if (productive_stems(left) & productive_stems(right)) - {left, right} else 0.0
    return max(direct, phonetic, morphological)


@lru_cache(maxsize=100_000)
def active_general_word(token: str) -> bool:
    """Reject dictionary fossils unless contemporary frequency or morphology supports them."""
    token = norm(token)
    if not token:
        return False
    direct = bool(
        (general_contains(token) and usage_frequency(token) >= 3.25)
        or morphology_supported(token)
        or token in MODERN_SPOKEN or token in FUNCTION_WORDS
        or token in NUMBER_WORDS or token in UNITS
    )
    if direct:
        return True
    # Productive colloquial copula and second-person plural endings. This keeps
    # spoken forms without admitting arbitrary zero-frequency dictionary fossils.
    if token.endswith("ه") and len(token) >= 4:
        stem = token[:-1]
        if general_contains(stem) and usage_frequency(stem) >= 3.25:
            return True
    if token.endswith("ین") and len(token) >= 5:
        formal = token[:-2] + "ید"
        if general_contains(formal) and usage_frequency(formal) >= 3.25:
            return True
    # Productive Persian present/imperfect verbs are effectively unbounded and
    # cannot all exist as dictionary headwords: می‌خوابم, نمی‌خوریم, می‌گیرند, ...
    for prefix in ("نمی", "می"):
        if not token.startswith(prefix):
            continue
        for suffix in ("یم", "ید", "ند", "م", "ی", "د"):
            if not token.endswith(suffix) or len(token) <= len(prefix) + len(suffix) + 1:
                continue
            stem = token[len(prefix):-len(suffix)]
            if general_contains(stem) and usage_frequency(stem) >= 3.0:
                return True
    return False


def is_negative_token(token: str) -> bool:
    token = norm(token)
    return token in NEGATIONS or token.startswith(("نمی", "نشد", "ندار", "نبود", "نخور", "نکن"))


def midpoint(item: dict[str, Any]) -> float:
    return (item["start"] + item["end"]) / 2.0


def slot_midpoint(slot: dict[str, Any]) -> float:
    return float(statistics.median(midpoint(item) for item in slot["observations"].values()))


def slot_similarity(word: dict[str, Any], slot: dict[str, Any]) -> float:
    observations = list(slot["observations"].values())
    lexical = max(token_similarity(word["normalized"], item["normalized"]) for item in observations)
    temporal = max(0.0, 1.0 - abs(midpoint(word) - slot_midpoint(slot)) / 1.15)
    overlap = max(
        max(0.0, min(word["end"], item["end"]) - max(word["start"], item["start"]))
        / max(0.05, min(word["end"] - word["start"], item["end"] - item["start"]))
        for item in observations
    )
    return 0.47 * lexical + 0.36 * temporal + 0.17 * min(1.0, overlap)


def add_sequence_to_network(slots: list[dict[str, Any]], sequence: list[dict[str, Any]],
                            hypothesis: str) -> list[dict[str, Any]]:
    """Time-aware progressive WCN. It permits real split/join differences through epsilon paths."""
    if not slots:
        return [{"observations": {hypothesis: word}} for word in sequence]
    n, m, gap = len(sequence), len(slots), 0.68
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    trace = [[0] * (m + 1) for _ in range(n + 1)]
    for index in range(1, n + 1):
        dp[index][0], trace[index][0] = index * gap, 1
    for index in range(1, m + 1):
        dp[0][index], trace[0][index] = index * gap, 2
    for row in range(1, n + 1):
        for column in range(1, m + 1):
            values = (
                dp[row - 1][column - 1] + 1.0 - slot_similarity(sequence[row - 1], slots[column - 1]),
                dp[row - 1][column] + gap,
                dp[row][column - 1] + gap,
            )
            move = min(range(3), key=lambda item: values[item])
            dp[row][column], trace[row][column] = values[move], move
    operations: list[tuple[str, int | None, int | None]] = []
    row, column = n, m
    while row or column:
        move = trace[row][column]
        if row and column and move == 0:
            operations.append(("match", row - 1, column - 1))
            row, column = row - 1, column - 1
        elif row and (column == 0 or move == 1):
            operations.append(("insert", row - 1, None))
            row -= 1
        else:
            operations.append(("epsilon", None, column - 1))
            column -= 1
    merged = []
    for operation, sequence_index, slot_index in reversed(operations):
        if operation == "match":
            slot = slots[slot_index]
            slot["observations"][hypothesis] = sequence[sequence_index]
            merged.append(slot)
        elif operation == "insert":
            merged.append({"observations": {hypothesis: sequence[sequence_index]}})
        else:
            merged.append(slots[slot_index])
    return merged


class LexiconResolver:
    def __init__(self, medical_payload: dict[str, Any]) -> None:
        self.medical_exact: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.medical_phrase_ngrams: set[tuple[str, ...]] = set()
        self.medical_phonetic_ngrams: set[tuple[str, ...]] = set()
        self._medical_buckets: dict[tuple[str, int], list[str]] = defaultdict(list)
        self._medical_phonetic: dict[str, list[str]] = defaultdict(list)
        self._general_buckets: dict[tuple[str, int], list[str]] | None = None
        self._general_phonetic: dict[str, list[str]] | None = None
        self._general_cache: dict[str, list[tuple[str, float]]] = {}
        self._medical_cache: dict[str, list[tuple[str, float]]] = {}
        for row in medical_payload["terms"]:
            tokens = phrase_tokens(row.get("term") or row.get("normalized") or "")
            if not tokens:
                continue
            if len(tokens) == 1:
                token = tokens[0]
                self.medical_exact[token].append(row)
                key = phonetic_key(token)
                if key:
                    self._medical_buckets[(key[0], len(token))].append(token)
                    self._medical_phonetic[key].append(token)
            for size in (2, 3):
                for index in range(len(tokens) - size + 1):
                    gram = tuple(tokens[index:index + size])
                    self.medical_phrase_ngrams.add(gram)
                    self.medical_phonetic_ngrams.add(tuple(phonetic_key(token) for token in gram))
        for key, values in list(self._medical_buckets.items()):
            self._medical_buckets[key] = sorted(set(values))
        for key, values in list(self._medical_phonetic.items()):
            self._medical_phonetic[key] = sorted(set(values))

    def is_medical(self, token: str) -> bool:
        return norm(token) in self.medical_exact

    def medical_categories(self, token: str) -> set[str]:
        return {str(row.get("category") or "") for row in self.medical_exact.get(norm(token), [])}

    def medical_source_priority(self, token: str) -> int:
        """Prefer the clinician-reviewed base lexicon over supplemental lists."""
        rows = self.medical_exact.get(norm(token), [])
        if any(
            str(row.get("license") or "").strip().casefold() == "cc by 4.0"
            or "persianmedqa" in str(row.get("source") or "").casefold()
            for row in rows
        ):
            return 2
        return 1 if rows else 0

    def _ensure_general(self) -> None:
        if self._general_buckets is not None:
            return
        cache_path = Path(__file__).resolve().parent.parent / "offline-lexicon" / "general-fuzzy-index-v4.pkl"
        if cache_path.is_file():
            try:
                with cache_path.open("rb") as handle:
                    payload = pickle.load(handle)
                if payload.get("version") == 1 and payload.get("word_count") == persian_words.count_words():
                    self._general_buckets = payload["buckets"]
                    self._general_phonetic = payload["phonetic"]
                    return
            except (OSError, EOFError, pickle.UnpicklingError, AttributeError, KeyError):
                pass
        buckets: dict[tuple[str, int], list[str]] = defaultdict(list)
        phonetic: dict[str, list[str]] = defaultdict(list)
        for word in persian_words.iter_words(mode="clean", min_length=2, max_length=40, order="alpha"):
            word = norm(word)
            key = phonetic_key(word)
            if word and key and word not in USER_BLOCKLIST:
                buckets[(key[0], len(word))].append(word)
                phonetic[key].append(word)
        self._general_buckets = buckets
        self._general_phonetic = phonetic
        try:
            with cache_path.open("wb") as handle:
                pickle.dump({"version": 1, "word_count": persian_words.count_words(),
                             "buckets": dict(buckets), "phonetic": dict(phonetic)},
                            handle, protocol=pickle.HIGHEST_PROTOCOL)
        except OSError:
            pass

    @staticmethod
    def _pool(buckets: dict[tuple[str, int], list[str]], token: str, radius: int) -> list[str]:
        key = phonetic_key(token)
        if not key:
            return []
        result: list[str] = []
        for length in range(max(2, len(token) - radius), len(token) + radius + 1):
            result.extend(buckets.get((key[0], length), []))
        return result

    def near_general(self, token: str) -> list[tuple[str, float]]:
        token = norm(token)
        if token in self._general_cache:
            return self._general_cache[token]
        self._ensure_general()
        pool = self._pool(self._general_buckets or {}, token, 2)
        matches = process.extract(token, pool, scorer=fuzz.ratio, score_cutoff=74, limit=4)
        scored = {word: float(score) / 100.0 for word, score, _ in matches
                  if usage_frequency(word) >= 3.25 and word not in USER_BLOCKLIST}
        key = phonetic_key(token)
        phonetic_exact = [word for word in (self._general_phonetic or {}).get(key, [])
                          if usage_frequency(word) >= 3.25 and word not in USER_BLOCKLIST]
        for word in sorted(phonetic_exact, key=usage_frequency, reverse=True)[:4]:
            scored[word] = max(scored.get(word, 0.0), 0.96)
        result = sorted(scored.items(), key=lambda row: (row[1], usage_frequency(row[0])), reverse=True)[:6]
        self._general_cache[token] = result
        return result

    def near_medical(self, token: str) -> list[tuple[str, float]]:
        token = norm(token)
        if token in self._medical_cache:
            return self._medical_cache[token]
        pool = self._pool(self._medical_buckets, token, 2)
        matches = process.extract(token, pool, scorer=fuzz.ratio, score_cutoff=86, limit=5)
        scored = {word: float(score) / 100.0 for word, score, _ in matches}
        key = phonetic_key(token)
        for word in self._medical_phonetic.get(key, []):
            scored[word] = max(scored.get(word, 0.0), 0.96)
        result = sorted(scored.items(), key=lambda row: row[1], reverse=True)[:7]
        self._medical_cache[token] = result
        return result


class NgramEvidence:
    def __init__(self, sequences: dict[str, list[dict[str, Any]]], resolver: LexiconResolver) -> None:
        self.exact_families: dict[tuple[str, ...], set[str]] = defaultdict(set)
        self.phonetic_families: dict[tuple[str, ...], set[str]] = defaultdict(set)
        self.exact_counts: Counter[tuple[str, ...]] = Counter()
        self.phonetic_counts: Counter[tuple[str, ...]] = Counter()
        for hypothesis, words in sequences.items():
            family = hypothesis.split("__", 1)[0]
            tokens = [word["normalized"] for word in words]
            for size in (1, 2, 3):
                seen_exact: set[tuple[str, ...]] = set()
                seen_phonetic: set[tuple[str, ...]] = set()
                for index in range(len(tokens) - size + 1):
                    gram = tuple(tokens[index:index + size])
                    phonetic = tuple(phonetic_key(token) for token in gram)
                    self.exact_counts[gram] += 1
                    self.phonetic_counts[phonetic] += 1
                    seen_exact.add(gram)
                    seen_phonetic.add(phonetic)
                for gram in seen_exact:
                    self.exact_families[gram].add(family)
                for gram in seen_phonetic:
                    self.phonetic_families[gram].add(family)
        self.medical_exact = resolver.medical_phrase_ngrams
        self.medical_phonetic = resolver.medical_phonetic_ngrams

    def token_repetition(self, token: str) -> tuple[int, int]:
        exact = (norm(token),)
        phonetic = (phonetic_key(token),)
        families = max(len(self.exact_families.get(exact, set())),
                       len(self.phonetic_families.get(phonetic, set())))
        count = max(self.exact_counts.get(exact, 0), self.phonetic_counts.get(phonetic, 0))
        return families, count

    def repeated_split_candidates(self, observed: str) -> list[dict[str, Any]]:
        """Expand a fused ASR token only when the phrase repeats elsewhere in >=2 model families."""
        result = []
        for gram, families in self.exact_families.items():
            if len(gram) != 2 or len(families) < 2:
                continue
            compact = "".join(gram)
            similarity = token_similarity(observed, compact)
            if similarity < 0.80:
                continue
            # Do not prepend/append a neighbouring word to a complete token. The
            # intended case is a genuinely fused form such as a number+classifier.
            if max(token_similarity(observed, part) for part in gram) >= 0.92:
                continue
            number_classifier = gram[0] in NUMBER_WORDS and gram[1] == "تا"
            lexical_parts = all(active_general_word(part) for part in gram)
            if not (number_classifier or lexical_parts):
                continue
            result.append({
                "candidate": " ".join(gram), "candidate_tokens": gram,
                "origin": "repeated-text-bigram-split", "dictionary_similarity": similarity,
                "phrase_families": sorted(families), "phrase_count": self.exact_counts.get(gram, 0),
            })
        result.sort(key=lambda row: (len(row["phrase_families"]), row["phrase_count"],
                                     row["dictionary_similarity"]), reverse=True)
        return result[:3]

    def transition_score(self, history: tuple[str, ...], candidate: str) -> tuple[float, dict[str, Any]]:
        if not candidate:
            return 0.0, {"exact_family_support": 0, "phonetic_family_support": 0,
                         "medical_ngram": False}
        tokens = (*history[-2:], candidate)
        score = 0.0
        details: dict[str, Any] = {"grams": []}
        for size, weight in ((2, 0.72), (3, 1.03)):
            if len(tokens) < size:
                continue
            gram = tuple(tokens[-size:])
            phonetic = tuple(phonetic_key(token) for token in gram)
            exact_families = len(self.exact_families.get(gram, set()))
            phonetic_families = len(self.phonetic_families.get(phonetic, set()))
            exact_count = self.exact_counts.get(gram, 0)
            phonetic_count = self.phonetic_counts.get(phonetic, 0)
            family_support = max(exact_families, 0.72 * phonetic_families)
            occurrence_support = max(exact_count, int(0.72 * phonetic_count))
            medical = gram in self.medical_exact or phonetic in self.medical_phonetic
            contribution = weight * family_support + 0.10 * math.log1p(occurrence_support)
            if medical:
                contribution += 0.44 if size == 2 else 0.68
            if family_support == 0 and not medical:
                contribution -= 0.16 if size == 2 else 0.08
            score += contribution
            details["grams"].append({
                "gram": " ".join(gram), "exact_family_support": exact_families,
                "phonetic_family_support": phonetic_families,
                "occurrences": occurrence_support, "medical_ngram": medical,
                "score": round(contribution, 4),
            })
        return score, details


def family_similarity(candidate: str, observations: list[dict[str, Any]]) -> dict[str, float]:
    result = {}
    for family in FAMILIES:
        rows = [row for row in observations if row["family"] == family]
        result[family] = max((token_similarity(candidate, row["normalized"]) for row in rows), default=0.0)
    return result


def build_slot_candidates(slot: dict[str, Any], resolver: LexiconResolver,
                          ngrams: NgramEvidence) -> list[dict[str, Any]]:
    observations = list(slot["observations"].values())
    observed_tokens = sorted({row["normalized"] for row in observations})
    pool: dict[str, dict[str, Any]] = {
        token: {"candidate": token, "candidate_tokens": (token,),
                "origin": "observed", "dictionary_similarity": 1.0}
        for token in observed_tokens
    }
    for observed in observed_tokens:
        # Repair a commonly dropped /i/ in the Persian progressive marker:
        # مکنه -> میکنه. The repaired form is admitted only when it is an active
        # contemporary dictionary word and still has close acoustic similarity.
        if observed.startswith("م") and not observed.startswith("می") and len(observed) >= 4:
            repaired = "می" + observed[1:]
            similarity = token_similarity(observed, repaired)
            if similarity >= 0.82 and active_general_word(repaired):
                pool.setdefault(repaired, {
                    "candidate": repaired, "candidate_tokens": (repaired,),
                    "origin": "productive-prefix-repair",
                    "dictionary_similarity": similarity,
                })
        observed_is_active = active_general_word(observed) or resolver.is_medical(observed)
        if not observed_is_active:
            for candidate, similarity in resolver.near_general(observed):
                current = pool.setdefault(candidate, {"candidate": candidate, "candidate_tokens": (candidate,),
                                                       "origin": "general-lexicon",
                                                       "dictionary_similarity": similarity})
                current["dictionary_similarity"] = max(current.get("dictionary_similarity", 0.0), similarity)
            for candidate, similarity in resolver.near_medical(observed):
                current = pool.setdefault(candidate, {"candidate": candidate, "candidate_tokens": (candidate,),
                                                       "origin": "medical-lexicon",
                                                       "dictionary_similarity": similarity})
                current["dictionary_similarity"] = max(current.get("dictionary_similarity", 0.0), similarity)

            for phrase in ngrams.repeated_split_candidates(observed):
                pool.setdefault(phrase["candidate"], phrase)

    ranked: list[dict[str, Any]] = []
    for token, metadata in pool.items():
        similarities = family_similarity(token, observations)
        exact_families = {row["family"] for row in observations if row["normalized"] == token}
        strong_families = {family for family, score in similarities.items() if score >= 0.78}
        loose_families = {family for family, score in similarities.items() if score >= 0.66}
        observed = token in observed_tokens
        candidate_tokens = tuple(metadata.get("candidate_tokens") or (token,))
        frequency = usage_frequency(token) if len(candidate_tokens) == 1 else min(
            usage_frequency(part) for part in candidate_tokens)
        general_member = (general_contains(token) if len(candidate_tokens) == 1
                          else all(general_contains(part) for part in candidate_tokens))
        general = (active_general_word(token) if len(candidate_tokens) == 1
                   else all(active_general_word(part) for part in candidate_tokens))
        medical = (resolver.is_medical(token) if len(candidate_tokens) == 1
                   else candidate_tokens in resolver.medical_phrase_ngrams)
        medical_categories = (resolver.medical_categories(token) if len(candidate_tokens) == 1 else set())
        modern = (token in MODERN_SPOKEN if len(candidate_tokens) == 1
                  else all(part in MODERN_SPOKEN or active_general_word(part) for part in candidate_tokens))
        if token in USER_BLOCKLIST:
            continue
        # A dictionary is a candidate generator, never independent acoustic evidence.
        if not observed and len(strong_families) < 2:
            continue
        probability_by_family = []
        for family in strong_families:
            matching = [row for row in observations if row["family"] == family]
            if matching:
                probability_by_family.append(max(
                    row["probability"] * token_similarity(token, row["normalized"]) for row in matching))
        probability = sum(probability_by_family) / max(1, len(probability_by_family))
        exact_sources = sum(row["normalized"] == token for row in observations)
        if len(candidate_tokens) == 1:
            repetition_families, repetition_count = ngrams.token_repetition(token)
        else:
            repetition_families = len(ngrams.exact_families.get(candidate_tokens, set()))
            repetition_count = ngrams.exact_counts.get(candidate_tokens, 0)
        fuzzy_only = max(0, len(strong_families) - len(exact_families))
        medical_bonus = (0.78 if medical_categories & {"drug", "medication", "drug_class"}
                         else 0.38 if medical else 0.0)
        score = (
            2.20 * len(exact_families)
            + 1.45 * fuzzy_only
            + 0.26 * max(0, len(loose_families) - len(strong_families))
            + 0.16 * exact_sources
            + 0.42 * probability
            + 0.32 * repetition_families
            + 0.11 * math.log1p(repetition_count)
            + (0.50 if general else 0.0)
            + medical_bonus
            + (0.22 if general and medical else 0.0)
            + (0.30 if modern else 0.0)
            + 0.055 * min(6.0, frequency)
            + (0.10 if any(row["hypothesis"] == "large-v3-turbo__enhanced"
                           and row["normalized"] == token for row in observations) else 0.0)
            - (0.64 if not observed else 0.0)
            + (0.95 * repetition_families
               if metadata.get("origin") == "repeated-text-bigram-split" else 0.0)
        )
        if not (general or medical or modern) and len(strong_families) < 2:
            score -= 1.45
        ranked.append({
            **metadata, "candidate": token, "emission_score": round(score, 6),
            "candidate_tokens": list(candidate_tokens),
            "exact_families": sorted(exact_families), "strong_families": sorted(strong_families),
            "loose_families": sorted(loose_families), "family_similarity": similarities,
            "exact_sources": exact_sources, "repetition_families": repetition_families,
            "repetition_count": repetition_count, "general_dictionary_member": general_member,
            "general_lexicon": general,
            "medical_lexicon": medical, "modern_spoken": modern,
            "medical_categories": sorted(medical_categories),
            "acoustic_probability": round(probability, 6),
            "zipf_frequency_fa": round(frequency, 4), "observed": observed,
        })

    # A rare/old dictionary surface must not beat a close contemporary alternative merely
    # because the same ASR error repeated. Exact agreement is not a vocabulary licence:
    # independent Whisper sizes can share the same spelling error (خورداد, تلفونی, ...).
    # Productive spoken forms are retained by active_general_word above, while repeated OOV
    # medical names remain untouched whenever no close active rival exists.
    for row in ranked:
        if row["general_lexicon"] or row["medical_lexicon"] or row["modern_spoken"]:
            continue
        rivals = [other for other in ranked
                  if other is not row
                  and (other["general_lexicon"] or other["medical_lexicon"] or other["modern_spoken"])
                  and len(other["strong_families"]) >= 2
                  and (other["medical_lexicon"] or other["zipf_frequency_fa"] >= 4.0
                       or other.get("origin") == "repeated-text-bigram-split")
                  and token_similarity(row["candidate"], other["candidate"]) >= 0.78]
        if rivals:
            penalty = (2.20 + 0.80 * len(row["exact_families"])
                       + (1.50 if any(other.get("origin") == "repeated-text-bigram-split"
                                     for other in rivals) else 0.0))
            row["emission_score"] = round(row["emission_score"] - penalty, 6)
            row["rare_surface_penalty"] = penalty
            row["close_active_rivals"] = [other["candidate"] for other in rivals[:5]]
            row["disqualified_rare_surface"] = True

    # Repetition with consistently weak token probabilities is not equal to a reliable
    # repetition. Keep the higher-confidence rival when it still has >=2-family fuzzy support.
    for row in ranked:
        if len(row["exact_families"]) == len(FAMILIES):
            continue
        if len(row["exact_families"]) < 2 or row["acoustic_probability"] >= 0.58:
            continue
        reliable_rivals = [other for other in ranked if other is not row
                           and len(other["strong_families"]) >= 2
                           and other["acoustic_probability"] >= row["acoustic_probability"] + 0.12]
        if reliable_rivals:
            penalty = min(2.60, 1.20 + 4.0 * max(
                other["acoustic_probability"] - row["acoustic_probability"]
                for other in reliable_rivals))
            row["emission_score"] = round(row["emission_score"] - penalty, 6)
            row["low_confidence_repeat_penalty"] = round(penalty, 6)
            row["higher_confidence_rivals"] = [other["candidate"] for other in reliable_rivals[:5]]

    ranked = [row for row in ranked if not row.get("disqualified_rare_surface")]
    observed_families = {row["family"] for row in observations}
    epsilon_score = {3: 0.25, 2: 2.35, 1: 6.20}.get(len(observed_families), 6.5)
    ranked.append({
        "candidate": "", "candidate_tokens": [], "origin": "epsilon", "emission_score": epsilon_score,
        "exact_families": [], "strong_families": [], "loose_families": [],
        "family_similarity": {}, "exact_sources": 0, "repetition_families": 0,
        "repetition_count": 0, "general_dictionary_member": False,
        "general_lexicon": False, "medical_lexicon": False,
        "modern_spoken": False, "medical_categories": [], "zipf_frequency_fa": 0.0,
        "observed": False,
    })
    ranked.sort(key=lambda row: row["emission_score"], reverse=True)
    nonempty = [row for row in ranked if row["candidate"]][:8]
    epsilon = next(row for row in ranked if not row["candidate"])
    return nonempty + [epsilon]


def redundancy_penalty(history: tuple[str, ...], candidate: str) -> tuple[float, str | None]:
    if not history or not candidate:
        return 0.0, None
    previous = history[-1]
    if previous == candidate:
        return -6.0, "adjacent-exact-duplicate"
    if previous in {"می", "نمی"} and candidate.startswith(previous) and len(candidate) > len(previous) + 1:
        return -5.2, "split-prefix-duplicate"
    if candidate in {"تا", "ها", "های"} and previous.endswith(candidate) and len(previous) > len(candidate):
        return -5.0, "split-suffix-duplicate"
    return 0.0, None


def decode(candidate_lattice: list[list[dict[str, Any]]], ngrams: NgramEvidence,
           beam_size: int = 72) -> tuple[list[int], list[dict[str, Any]]]:
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
                for part in row.get("candidate_tokens") or []:
                    part_context_score, part_context = ngrams.transition_score(history, part)
                    part_penalty, part_penalty_reason = redundancy_penalty(history, part)
                    context_score += part_context_score
                    penalty += part_penalty
                    context_parts.append(part_context)
                    if part_penalty_reason:
                        penalty_reasons.append(part_penalty_reason)
                    history = (*history[-1:], part)
                if row.get("low_confidence_repeat_penalty") and any(
                        is_negative_token(part) for part in row.get("candidate_tokens") or []):
                    # A repeated low-confidence negation may not regain certainty solely
                    # through an n-gram made from the same ASR hypotheses.
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


def is_sensitive(index: int, selected: list[dict[str, Any]]) -> bool:
    token = selected[index]["candidate"]
    if token in NUMBER_WORDS or is_negative_token(token) or token in UNITS:
        return True
    if selected[index].get("medical_lexicon"):
        return True
    nearby = [row["candidate"] for row in selected[max(0, index - 3):index + 4]]
    return any(token in DRUG_CONTEXT or candidate in DRUG_CONTEXT for candidate in nearby)


def classify_choices(slots: list[dict[str, Any]], lattice: list[list[dict[str, Any]]],
                     path: list[int], transitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [lattice[index][choice] for index, choice in enumerate(path)]
    result = []
    for index, row in enumerate(selected):
        alternatives = sorted(lattice[index], key=lambda item: item["emission_score"], reverse=True)
        runner_up = next((item for item in alternatives if item["candidate"] != row["candidate"]), None)
        margin = row["emission_score"] - (runner_up["emission_score"] if runner_up else 0.0)
        sensitive = is_sensitive(index, selected)
        families = len(row["strong_families"])
        if not row["candidate"]:
            observed_families = {item["family"] for item in slots[index]["observations"].values()}
            status = "REVIEW" if len(observed_families) >= 2 else "OMIT"
            reason = "context-selected-epsilon"
        elif families >= 2 and (not sensitive or families == 3) and margin >= -0.25:
            status, reason = "ACCEPT", "family-plus-ngram-consensus"
        elif families >= 2:
            status, reason = "REVIEW", "sensitive-or-close-family-consensus"
        else:
            status, reason = "REVIEW", "single-family-context-choice"
        result.append({
            **row, "slot": index, "start": min(item["start"] for item in slots[index]["observations"].values()),
            "end": max(item["end"] for item in slots[index]["observations"].values()),
            "midpoint": slot_midpoint(slots[index]), "status": status, "reason": reason,
            "sensitive": sensitive, "local_margin": round(margin, 6),
            "transition": transitions[index],
            "alternatives": [{key: item.get(key) for key in (
                "candidate", "origin", "emission_score", "exact_families", "strong_families",
                "general_lexicon", "medical_lexicon", "zipf_frequency_fa")}
                for item in alternatives[:5]],
            "observations": [{key: item[key] for key in (
                "hypothesis", "family", "source", "word", "normalized", "start", "end", "probability")}
                for item in slots[index]["observations"].values()],
        })
    return result


def apply_dose_locks(selected: list[dict[str, Any]], doses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Canonicalize only doses whose numeric value is independently supported by all 3 families."""
    surfaces = {25: "بیست‌وپنج", 50: "پنجاه", 100: "صد"}
    locks = []
    occupied: set[int] = set()
    for dose in doses:
        if dose.get("status") != "ACCEPT" or len(dose.get("supporting_families") or []) != 3:
            continue
        overlapping = [index for index, row in enumerate(selected)
                       if dose["start"] - 0.05 <= row["midpoint"] <= dose["end"] + 0.05]
        overlapping = [index for index in overlapping if index not in occupied]
        if not overlapping or dose["value"] not in surfaces:
            continue
        first = overlapping[0]
        first_row = selected[first]
        first_row.update({
            "candidate": f"{surfaces[dose['value']]} میلی‌گرم",
            "candidate_tokens": [surfaces[dose["value"]], "میلی‌گرم"],
            "origin": "structured-dose-lock", "status": "ACCEPT",
            "reason": "three-family-structured-dose", "sensitive": True,
            "strong_families": list(dose["supporting_families"]),
            "exact_families": list(dose["supporting_families"]),
        })
        for index in overlapping[1:]:
            selected[index].update({
                "candidate": "", "candidate_tokens": [], "origin": "structured-dose-lock-continuation",
                "status": "OMIT", "reason": "consumed-by-three-family-structured-dose",
            })
        occupied.update(overlapping)
        locks.append({"value": dose["value"], "slots": overlapping,
                      "supporting_families": dose["supporting_families"]})
    return locks


def render(selected: list[dict[str, Any]]) -> tuple[str, list[str]]:
    output: list[str] = []
    operations: list[str] = []
    for row in selected:
        candidate_parts = row.get("candidate_tokens") or []
        for token in candidate_parts:
            if not token:
                continue
            if output and output[-1] == token:
                operations.append(f"drop-adjacent-duplicate:{token}")
                continue
            if output and output[-1] in {"می", "نمی"}:
                prefix = output[-1]
                if token.startswith(prefix) and len(token) > len(prefix) + 1:
                    output[-1] = token
                    operations.append(f"collapse-prefix:{prefix}+{token}")
                    continue
                if len(token) >= 2 and token not in FUNCTION_WORDS:
                    output[-1] = prefix + "‌" + token
                    operations.append(f"join-prefix:{prefix}+{token}")
                    continue
            if output and token in {"ها", "های"}:
                if output[-1].endswith(token):
                    operations.append(f"drop-split-suffix:{output[-1]}+{token}")
                    continue
                output[-1] = output[-1] + "‌" + token
                operations.append(f"join-suffix:{token}")
                continue
            if output and token == "تا" and output[-1].endswith("تا") and len(output[-1]) > 2:
                operations.append(f"drop-split-suffix:{output[-1]}+{token}")
                continue
            if token.endswith("ها") and len(token) > 3 and active_general_word(token[:-2]):
                output.append(token[:-2] + "‌ها")
                operations.append(f"normalize-plural-zwnj:{token}")
                continue
            output.append(token)
    return " ".join(output).strip(), operations


def validate(slots: list[dict[str, Any]], sequences: dict[str, list[dict[str, Any]]],
             selected: list[dict[str, Any]], final_text: str) -> dict[str, Any]:
    expected = {word["id"] for words in sequences.values() for word in words}
    observed = {item["id"] for slot in slots for item in slot["observations"].values()}
    unsupported = []
    sensitive_single_family = []
    for row in selected:
        token = row["candidate"]
        if not token:
            continue
        supported = (((row["observed"] or row["general_lexicon"] or row["medical_lexicon"]
                       or row["modern_spoken"]) and len(row["strong_families"]) >= 1)
                     or row.get("origin") == "structured-dose-lock")
        if not supported:
            unsupported.append({"slot": row["slot"], "candidate": token})
        if row["sensitive"] and len(row["strong_families"]) < 2:
            sensitive_single_family.append({"slot": row["slot"], "candidate": token})
    tokens = final_text.split()
    adjacent_duplicates = [tokens[index] for index in range(1, len(tokens)) if tokens[index] == tokens[index - 1]]
    checks = {
        "all_asr_observations_accounted_once": expected == observed,
        "no_unseen_unlicensed_candidate": not unsupported,
        "no_sensitive_single_family_accept": not any(
            row["status"] == "ACCEPT" and row["sensitive"] and len(row["strong_families"]) < 2
            for row in selected),
        "no_adjacent_exact_duplicate": not adjacent_duplicates,
        "no_blocklisted_output": not any(norm(token) in USER_BLOCKLIST for token in tokens),
    }
    return {
        "passed": all(checks.values()), "checks": checks,
        "missing_observations": sorted(expected - observed), "duplicated_or_unknown_observations": sorted(observed - expected),
        "unsupported_candidates": unsupported, "sensitive_single_family": sensitive_single_family,
        "adjacent_duplicates": adjacent_duplicates,
    }


def merge_review_intervals(review: list[dict[str, Any]], duration: float, limit: int = 8) -> tuple[list[dict[str, Any]], bool]:
    raw = []
    for row in review:
        raw.append({"start": max(0.0, row["start"] - 0.8), "end": min(duration, row["end"] + 0.8),
                    "slots": [row["slot"]], "tokens": [row["candidate"] or "ε"]})
    merged: list[dict[str, Any]] = []
    for interval in sorted(raw, key=lambda item: item["start"]):
        if merged and interval["start"] <= merged[-1]["end"] + 0.2:
            merged[-1]["end"] = max(merged[-1]["end"], interval["end"])
            merged[-1]["slots"].extend(interval["slots"])
            merged[-1]["tokens"].extend(interval["tokens"])
        else:
            merged.append(interval)
    limited = len(merged) > limit
    return merged[:limit], limited


def make_review_clips(run_dir: Path, out_dir: Path, review: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    audio = run_dir / "normalized_mono_48k.wav"
    if not audio.is_file() or not review:
        return [], False
    duration = max(row["end"] for row in review) + 1.0
    intervals, limited = merge_review_intervals(review, duration)
    clip_dir = out_dir / "review-clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = run_dir.parents[1] / "runtime" / "ffmpeg" / "ffmpeg.exe"
    if not ffmpeg.is_file():
        return intervals, limited
    for index, interval in enumerate(intervals, 1):
        clip = clip_dir / f"review-{index:03d}-{interval['start']:.2f}-{interval['end']:.2f}.wav"
        subprocess.run([
            str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{interval['start']:.3f}",
            "-to", f"{interval['end']:.3f}", "-i", str(audio), "-ac", "1", "-ar", "16000", str(clip),
        ], check=True)
        interval["clip"] = str(clip)
    return intervals, limited


def write_outputs(run_dir: Path, medical_payload: dict[str, Any], slots: list[dict[str, Any]],
                  lattice: list[list[dict[str, Any]]], selected: list[dict[str, Any]], final_text: str,
                  render_operations: list[str], doses: list[dict[str, Any]], dose_locks: list[dict[str, Any]],
                  validation: dict[str, Any], elapsed: float,
                  resolver: LexiconResolver) -> dict[str, Any]:
    out_dir = run_dir / OUTPUT_RELATIVE
    out_dir.mkdir(parents=True, exist_ok=True)
    review = [row for row in selected if row["status"] == "REVIEW"]
    clips, clip_limit = make_review_clips(run_dir, out_dir, review)
    payload = {
        "method": "three-family lexicon-aware word-confusion network with local bigram/trigram beam decoding",
        "llm_used": False, "sample_specific_phrases": False, "turbo_is_template": False,
        "runtime_seconds": round(elapsed, 3), "text": final_text,
        "family_vote_policy": "raw/enhanced collapse to one family; fuzzy variants cluster phonetically",
        "dictionary_policy": "dictionary candidates require acoustic support; repetition and n-gram context break ties",
        "general_lexicon": {"package": "mnk-persian-words", "clean_words": persian_words.count_words()},
        "medical_lexicon": {"source": medical_payload.get("source"), "license": medical_payload.get("license"),
                            "unique_terms": medical_payload.get("unique_terms")},
        "medical_phrase_bigram_trigram_count": len(resolver.medical_phrase_ngrams),
        "dose_entities": doses, "dose_locks": dose_locks,
        "hard_validation": validation, "render_operations": render_operations,
        "slots": selected,
    }
    (out_dir / "final-v4.txt").write_text(final_text + "\n", encoding="utf-8")
    (out_dir / "final-v4.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "review-v4.json").write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "review-clips-v4.json").write_text(json.dumps(clips, ensure_ascii=False, indent=2), encoding="utf-8")
    review_lines = [
        "# بازبینی نسخهٔ ۴ — بدون LLM", "",
        "واژه‌های REVIEW در متن به‌صورت مخفی تغییر نکرده‌اند؛ شواهد آن‌ها برای بازبینی ثبت شده است.", "",
        "| زمان | واژه | دلیل | خانواده‌های پشتیبان |", "|---:|---|---|---|",
    ]
    for row in review:
        review_lines.append(f"| {row['midpoint']:.2f} | {row['candidate'] or 'ε'} | {row['reason']} | {', '.join(row['strong_families']) or '—'} |")
    review_lines += ["", "## کلیپ‌ها", ""]
    for index, clip in enumerate(clips, 1):
        review_lines.append(f"- کلیپ {index}: {clip['start']:.2f} تا {clip['end']:.2f} ثانیه")
    if clip_limit:
        review_lines.append("- تعداد بازه‌ها بیش از سقف کلیپ بود؛ جزئیات کامل در JSON است.")
    (out_dir / "review-v4.md").write_text("\n".join(review_lines) + "\n", encoding="utf-8")

    score_lines = [
        "# امتیازنامهٔ واژه + n-gram — نسخهٔ ۴", "",
        "| جایگاه | زمان | انتخاب | خانواده | عمومی | پزشکی | n-gram | وضعیت |", "|---:|---:|---|---:|:---:|:---:|---:|---|",
    ]
    for row in selected:
        score_lines.append(
            f"| {row['slot']} | {row['midpoint']:.2f} | {row['candidate'] or 'ε'} | {len(row['strong_families'])} | "
            f"{'✓' if row['general_lexicon'] else ''} | {'✓' if row['medical_lexicon'] else ''} | "
            f"{row['transition']['context_score']:.3f} | {row['status']} |")
    score_lines += ["", "## خروجی", "", final_text, ""]
    (out_dir / "scorecard-v4.md").write_text("\n".join(score_lines), encoding="utf-8")
    previous = run_dir / "final-delivery" / "02-after-algorithm-v3-phrase-network" / "final-v3.txt"
    turbo = run_dir / "hypotheses" / "large-v3-turbo__enhanced" / "large-v3-turbo__enhanced.txt"
    comparison = ["# مقایسهٔ خروجی", ""]
    if turbo.is_file():
        comparison += ["## Turbo enhanced", "", turbo.read_text(encoding="utf-8").strip(), ""]
    if previous.is_file():
        comparison += ["## نسخهٔ ۳", "", previous.read_text(encoding="utf-8").strip(), ""]
    comparison += ["## نسخهٔ ۴", "", final_text, ""]
    (out_dir / "comparison-v4.md").write_text("\n".join(comparison), encoding="utf-8")
    summary = {
        "runtime_seconds": round(elapsed, 3), "slot_count": len(slots), "review_count": len(review),
        "review_clip_count": sum("clip" in row for row in clips), "review_clip_limit_reached": clip_limit,
        "hard_validation_passed": validation["passed"], "llm_used": False,
        "sample_specific_phrases": False, "accepted_dose_locks": len(dose_locks), "text": final_text,
    }
    (out_dir / "summary-v4.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output": str(out_dir), **summary}


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description="Deterministic lexicon + local n-gram consensus; no LLM.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--medical-index", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    hypotheses = load_hypotheses(run_dir)
    sequences = {key: words_of(hypotheses[key], key) for key in NETWORK_ORDER}
    medical_payload = json.loads(args.medical_index.read_text(encoding="utf-8"))
    resolver = LexiconResolver(medical_payload)
    ngrams = NgramEvidence(sequences, resolver)
    doses = cluster_doses(dose_occurrences(sequences))
    slots: list[dict[str, Any]] = []
    for hypothesis in NETWORK_ORDER:
        slots = add_sequence_to_network(slots, sequences[hypothesis], hypothesis)
    lattice = [build_slot_candidates(slot, resolver, ngrams) for slot in slots]
    path, transitions = decode(lattice, ngrams)
    selected = classify_choices(slots, lattice, path, transitions)
    dose_locks = apply_dose_locks(selected, doses)
    final_text, render_operations = render(selected)
    validation = validate(slots, sequences, selected, final_text)
    elapsed = time.perf_counter() - started
    result = write_outputs(run_dir, medical_payload, slots, lattice, selected, final_text,
                           render_operations, doses, dose_locks, validation, elapsed, resolver)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
