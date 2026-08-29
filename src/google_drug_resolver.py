from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable


GOOGLE_SUGGEST_URL = "https://suggestqueries.google.com/complete/search"
DRUG_CATEGORIES = {"drug", "drug_class", "medication"}
UNKNOWN_BRACKET_TERMS = {
    "نامفهوم", "نام دارو نامفهوم", "نام بیماری نامفهوم", "مقدار نامفهوم",
}
GENERIC_DRUG_TERMS = {
    "دارو", "داروی", "داروها", "داروهای", "قرص", "کپسول", "شربت", "آمپول",
    "دوز", "درمان", "داروی فعلی", "داروهای فعلی",
}
DRUG_CONTEXT_RE = re.compile(
    r"دارو|قرص|کپسول|شربت|آمپول|دوز|مصرف|قطع|ادامه|تجویز|میلی[\s\u200c-]*گرم")
BRACKET_RE = re.compile(r"\[([^\[\]]{1,100})\]")
PERSIAN_RE = re.compile(r"[\u0600-\u06ff]")


def normalize_name(value: str) -> str:
    value = str(value or "").lower()
    value = value.translate(str.maketrans({
        "ي": "ی", "ى": "ی", "ئ": "ی", "ك": "ک", "ة": "ه", "ۀ": "ه",
        "ؤ": "و", "أ": "ا", "إ": "ا", "ٱ": "ا",
    }))
    value = re.sub(r"[\u064b-\u065f\u0670]", "", value)
    value = re.sub(r"[^0-9a-z\u0600-\u06ff]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def compact_name(value: str) -> str:
    return normalize_name(value).replace(" ", "")


def name_similarity(left: str, right: str) -> float:
    a, b = compact_name(left), compact_name(right)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _group_key(row: dict[str, Any], alias: str) -> str:
    english = normalize_name(str(row.get("english") or ""))
    return "en:" + english if english else "fa:" + alias


@dataclass(frozen=True)
class DrugLexicon:
    aliases: tuple[str, ...]
    alias_groups: dict[str, frozenset[str]]
    group_aliases: dict[str, tuple[str, ...]]

    @classmethod
    def load(cls, path: Path) -> "DrugLexicon":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        alias_groups_mut: dict[str, set[str]] = {}
        group_aliases_mut: dict[str, set[str]] = {}
        for row in payload.get("terms") or []:
            if str(row.get("category") or "") not in DRUG_CATEGORIES:
                continue
            alias = normalize_name(str(row.get("normalized") or row.get("term") or ""))
            compact = compact_name(alias)
            if (not alias or alias in GENERIC_DRUG_TERMS or len(compact) < 4
                    or len(alias.split()) > 7 or not PERSIAN_RE.search(alias)):
                continue
            group = _group_key(row, alias)
            alias_groups_mut.setdefault(alias, set()).add(group)
            group_aliases_mut.setdefault(group, set()).add(alias)
        return cls(
            aliases=tuple(sorted(alias_groups_mut)),
            alias_groups={key: frozenset(value) for key, value in alias_groups_mut.items()},
            group_aliases={key: tuple(sorted(value)) for key, value in group_aliases_mut.items()},
        )

    def rank_groups(self, heard: str, limit: int = 4) -> list[dict[str, Any]]:
        best: dict[str, tuple[float, str]] = {}
        heard_length = len(compact_name(heard))
        for alias in self.aliases:
            alias_length = len(compact_name(alias))
            if alias_length < max(4, int(heard_length * 0.55)):
                continue
            if alias_length > max(8, int(heard_length * 1.65)):
                continue
            score = name_similarity(heard, alias)
            if score < 0.50:
                continue
            for group in self.alias_groups[alias]:
                current = best.get(group)
                if current is None or (score, -len(alias)) > (current[0], -len(current[1])):
                    best[group] = (score, alias)
        ranked = sorted(
            ({"group": group, "similarity": score, "seed": alias}
             for group, (score, alias) in best.items()),
            key=lambda row: (-row["similarity"], len(row["seed"]), row["seed"]),
        )
        return ranked[:limit]


def google_suggestions(query: str, timeout: float = 4.0) -> list[str]:
    url = GOOGLE_SUGGEST_URL + "?" + urllib.parse.urlencode({
        "client": "firefox", "hl": "fa", "q": query,
    })
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; PersianMedicalDrugResolver/1.0)",
            "Accept-Language": "fa,en;q=0.7",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
        raise ValueError("Google Suggest returned an unexpected schema")
    return [str(item) for item in payload[1] if str(item).strip()][:10]


class SuggestionCache:
    def __init__(self, path: Path | None, ttl_days: int = 30) -> None:
        self.path = path
        self.ttl_seconds = ttl_days * 86400
        self.rows: dict[str, dict[str, Any]] = {}
        self.dirty = False
        if path and path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
                if isinstance(payload, dict):
                    self.rows = dict(payload.get("queries") or {})
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                self.rows = {}

    def get(self, query: str) -> list[str] | None:
        row = self.rows.get(normalize_name(query))
        if not isinstance(row, dict):
            return None
        fetched_at = float(row.get("fetched_at") or 0.0)
        suggestions = row.get("suggestions")
        if time.time() - fetched_at > self.ttl_seconds or not isinstance(suggestions, list):
            return None
        return [str(item) for item in suggestions]

    def put(self, query: str, suggestions: list[str]) -> None:
        self.rows[normalize_name(query)] = {
            "fetched_at": time.time(), "suggestions": suggestions[:10],
        }
        self.dirty = True

    def save(self) -> None:
        if not self.path or not self.dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps({"version": 1, "queries": self.rows}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        temporary.replace(self.path)
        self.dirty = False


def bracketed_drug_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for match in BRACKET_RE.finditer(text):
        heard = normalize_name(match.group(1))
        if heard in UNKNOWN_BRACKET_TERMS or heard in GENERIC_DRUG_TERMS:
            continue
        sentence_start = max(
            text.rfind(".", 0, match.start()), text.rfind("؟", 0, match.start()),
            text.rfind("!", 0, match.start())) + 1
        sentence_ends = [index for marker in ".؟!" if (index := text.find(marker, match.end())) >= 0]
        sentence_end = min(sentence_ends) if sentence_ends else len(text)
        context = normalize_name(text[sentence_start:sentence_end])
        if not DRUG_CONTEXT_RE.search(context):
            continue
        if heard not in candidates:
            candidates.append(heard)
    return candidates


def _replace_bracketed_name(generated: dict[str, Any], heard: str, corrected: str) -> dict[str, Any]:
    output = {
        "summary": str(generated.get("summary") or ""),
        "confidence": str(generated.get("confidence") or "low"),
        "uncertain_items": [str(item) for item in generated.get("uncertain_items") or []],
    }
    def replace(match: re.Match[str]) -> str:
        return f"[{corrected}]" if normalize_name(match.group(1)) == heard else match.group(0)

    output["summary"] = BRACKET_RE.sub(replace, output["summary"])
    output["uncertain_items"] = [BRACKET_RE.sub(replace, item)
                                   for item in output["uncertain_items"]]
    return output


def resolve_one(
        heard: str, lexicon: DrugLexicon, fetcher: Callable[[str, float], list[str]],
        cache: SuggestionCache, timeout: float) -> tuple[str | None, dict[str, Any], int, int]:
    ranked = lexicon.rank_groups(heard)
    audit: dict[str, Any] = {
        "heard": heard,
        "status": "unresolved",
        "local_candidates": ranked,
    }
    if not ranked:
        audit["reason"] = "no-local-drug-candidate"
        return None, audit, 0, 0
    top = ranked[0]
    second_score = float(ranked[1]["similarity"]) if len(ranked) > 1 else 0.0
    margin = float(top["similarity"]) - second_score
    exact_groups = lexicon.alias_groups.get(normalize_name(heard), frozenset())
    exact_top_group = top["group"] in exact_groups
    audit["local_similarity"] = round(float(top["similarity"]), 4)
    audit["local_margin"] = round(margin, 4)
    if float(top["similarity"]) < 0.74:
        audit["reason"] = "local-similarity-below-threshold"
        return None, audit, 0, 0
    if not exact_top_group and margin < 0.055:
        audit["reason"] = "ambiguous-local-drug-candidates"
        return None, audit, 0, 0

    aliases = sorted(
        lexicon.group_aliases[top["group"]],
        key=lambda alias: (-name_similarity(heard, alias), len(alias), alias))
    queries: list[str] = []
    # One literal query and one dictionary seed keep worst-case latency bounded.
    for query in [heard, *aliases[:1]]:
        if query not in queries:
            queries.append(query)

    verified: list[tuple[int, int, int, str, str, list[str]]] = []
    network_requests = 0
    cache_hits = 0
    errors: list[str] = []
    for query_index, query in enumerate(queries):
        suggestions = cache.get(query)
        if suggestions is None:
            try:
                suggestions = fetcher(query, timeout)
                cache.put(query, suggestions)
                network_requests += 1
            except Exception as error:  # Network failure must never fail the medical pipeline.
                errors.append(f"{type(error).__name__}: {error}")
                continue
        else:
            cache_hits += 1
        for rank, suggestion in enumerate(suggestions):
            normalized = normalize_name(suggestion)
            if top["group"] not in lexicon.alias_groups.get(normalized, frozenset()):
                continue
            dosage_form_penalty = int(bool(re.search(r"^(?:قرص|کپسول|آمپول|شربت|داروی)\s", normalized)))
            verified.append(
                (rank, dosage_form_penalty, query_index, normalized, query, suggestions[:5]))
        if any(item[0] == 0 for item in verified):
            break

    if errors:
        audit["google_errors"] = errors
    if not verified:
        audit["reason"] = "google-did-not-confirm-local-candidate"
        audit["queries"] = queries
        return None, audit, network_requests, cache_hits
    verified.sort(key=lambda item: (item[0], item[1], item[2], len(item[3]), item[3]))
    rank, _, _, corrected, query, suggestions = verified[0]
    audit.update({
        "status": "confirmed" if corrected == normalize_name(heard) else "corrected",
        "corrected": corrected,
        "google_query": query,
        "google_rank": rank + 1,
        "google_suggestions_sample": suggestions,
        "reason": "local-phonetic-match-plus-google-suggest-plus-drug-dictionary",
    })
    return corrected, audit, network_requests, cache_hits


def resolve_summary_drug_names(
        generated: dict[str, Any], medical_index: Path, cache_path: Path | None = None,
        timeout: float = 4.0,
        fetcher: Callable[[str, float], list[str]] = google_suggestions) \
        -> tuple[dict[str, Any], dict[str, Any]]:
    """Correct bracketed heard drug forms without sending patient text to Google."""
    output = {
        "summary": str(generated.get("summary") or ""),
        "confidence": str(generated.get("confidence") or "low"),
        "uncertain_items": [str(item) for item in generated.get("uncertain_items") or []],
    }
    candidates: list[str] = []
    # Keep fields separate so a drug word in one uncertainty item cannot turn a
    # bracketed laser/device name in a neighboring item into a medication.
    for value in [output["summary"], *output["uncertain_items"]]:
        for candidate in bracketed_drug_candidates(value):
            if candidate not in candidates:
                candidates.append(candidate)
    audit: dict[str, Any] = {
        "enabled": True,
        "provider": "Google Suggest",
        "endpoint": GOOGLE_SUGGEST_URL,
        "privacy": "only isolated possible drug names are sent; no transcript or patient context",
        "candidate_count": len(candidates),
        "network_requests": 0,
        "cache_hits": 0,
        "corrections": [],
        "unresolved": [],
        "changed": False,
    }
    if not candidates:
        return output, audit
    lexicon = DrugLexicon.load(medical_index)
    cache = SuggestionCache(cache_path)
    try:
        for heard in candidates:
            corrected, item_audit, network_requests, cache_hits = resolve_one(
                heard, lexicon, fetcher, cache, timeout)
            audit["network_requests"] += network_requests
            audit["cache_hits"] += cache_hits
            if not corrected:
                audit["unresolved"].append(item_audit)
                continue
            if corrected != heard:
                output = _replace_bracketed_name(output, heard, corrected)
            audit["corrections"].append(item_audit)
    finally:
        cache.save()
    audit["changed"] = any(
        item.get("status") == "corrected" for item in audit["corrections"])
    return output, audit
