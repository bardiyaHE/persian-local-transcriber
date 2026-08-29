from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_lexicons import norm


SOURCE_NAME = "dadashzadeh/Collection-of-drug-names-in-Persian"
SOURCE_LICENSE = "MIT"
SOURCE_REVISION = "9ca2bcf9af0dce18e9e7d3ce5942c26a2f4be811"
EXCLUDED_SUPPLEMENTAL_GROUPS = {
    "herbal", "traditional medicines", "probiotics",
    "insecticides and insect repellents", "mineral solvents", "coloring agents",
    "suspending and thickening agents", "nonionic surfactants", "paraffins and similar bases",
    "soaps and other anionic surfactants", "cosmetics",
}


def source_descriptor(name: str, license_name: str, **extra: Any) -> dict[str, Any]:
    return {"name": name, "license": license_name, **extra}


def load_base(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = str(payload.get("source") or "unknown")
    license_name = str(payload.get("license") or "unknown")
    terms = []
    for original in payload.get("terms") or []:
        row = dict(original)
        row.setdefault("source", source)
        row.setdefault("license", license_name)
        terms.append(row)
    sources = payload.get("sources") or [
        source_descriptor(source, license_name, local_file=path.name)
    ]
    return terms, list(sources)


def load_persian_drug_names(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for source_row in csv.DictReader(handle):
            stats["source_rows"] += 1
            term = str(source_row.get("drugNameAFa") or "").strip()
            english = str(source_row.get("drugNameA") or "").strip()
            group_en = str(source_row.get("groupNameFA") or "").strip()
            group_fa = str(source_row.get("groupNameEN") or "").strip()
            normalized = norm(term)
            if not normalized or not english:
                stats["discarded_missing_name"] += 1
                continue
            if group_en.casefold() in EXCLUDED_SUPPLEMENTAL_GROUPS:
                stats["excluded_non_prescription_drug_group"] += 1
                continue
            # V9 currently resolves one acoustic lattice slot at a time. Keep
            # genuine single-token spellings only; joining a multiword drug
            # creates an alias that nobody actually says or writes.
            if " " in normalized:
                stats["deferred_multiword_names"] += 1
                continue
            if not all("\u0600" <= char <= "\u06ff" for char in normalized):
                stats["discarded_non_persian_single_token"] += 1
                continue
            rows.append({
                "term": term,
                "normalized": normalized,
                "category": "drug",
                "english": english,
                "source_file": path.name,
                "source": SOURCE_NAME,
                "license": SOURCE_LICENSE,
                "source_revision": SOURCE_REVISION,
                "alias_kind": "persian-generic-name",
                "drug_group_en": group_en or None,
                "drug_group_fa": group_fa or None,
                "supplemental_dictionary": True,
            })
            stats["usable_single_token_rows"] += 1
    return rows, dict(stats)


def merge_terms(base: list[dict[str, Any]], additions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    # Preserve the vetted PersianMedQA row when the same Persian spelling and
    # English identity occur in both sources. Distinct identities are retained
    # for audit, while the decoder applies its own conservative ambiguity gate.
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    duplicate_count = 0
    for row in [*base, *additions]:
        key = (
            norm(row.get("normalized") or row.get("term") or ""),
            str(row.get("english") or "").strip().casefold(),
            str(row.get("category") or "").strip().casefold(),
        )
        if not key[0] or key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        result.append(row)
    result.sort(key=lambda row: (
        norm(row.get("normalized") or row.get("term") or ""),
        str(row.get("english") or "").casefold(),
    ))
    return result, duplicate_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge licensed Persian medical and Persian drug-name dictionaries.")
    parser.add_argument("--base-index", type=Path, required=True)
    parser.add_argument("--persian-drug-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base, sources = load_base(args.base_index)
    additions, import_stats = load_persian_drug_names(args.persian_drug_csv)
    terms, duplicate_count = merge_terms(base, additions)
    sources.append(source_descriptor(
        SOURCE_NAME,
        SOURCE_LICENSE,
        revision=SOURCE_REVISION,
        local_file=args.persian_drug_csv.name,
        fields_used=["drugNameA", "drugNameAFa"],
    ))
    category_counts = Counter(str(row.get("category") or "") for row in terms)
    payload = {
        "source": "combined licensed Persian medical/drug dictionaries",
        "license": "mixed: CC BY 4.0 + MIT; see sources",
        "sources": sources,
        "unique_terms": len(terms),
        "categories": dict(sorted(category_counts.items())),
        "build": {
            "base_term_count": len(base),
            "imported_drug_row_count": len(additions),
            "duplicate_row_count": duplicate_count,
            **import_stats,
        },
        "terms": terms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "unique_terms": len(terms),
        "sources": sources,
        "build": payload["build"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
