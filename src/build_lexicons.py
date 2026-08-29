from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import openpyxl


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).replace("ي", "ی").replace("ك", "ک")
    text = text.replace("ۀ", "ه").replace("ة", "ه")
    text = re.sub(r"[\u064b-\u065f\u0670]", "", text)
    text = re.sub(r"[^\w\u0600-\u06ff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = {}
    categories = Counter()
    for path in sorted(args.source.glob("*.xlsx")):
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        sheet = workbook.active
        rows = sheet.iter_rows(values_only=True)
        header = [str(v) if v is not None else "" for v in next(rows)]
        try:
            fa_idx = header.index("term_fa")
        except ValueError:
            continue
        en_idx = header.index("term_en") if "term_en" in header else None
        type_idx = header.index("term_type") if "term_type" in header else None
        for row in rows:
            if fa_idx >= len(row) or not row[fa_idx]:
                continue
            term = str(row[fa_idx]).strip()
            normalized = norm(term)
            if not normalized:
                continue
            category = str(row[type_idx]).strip() if type_idx is not None and type_idx < len(row) and row[type_idx] else path.stem
            record = {"term": term, "normalized": normalized, "category": category,
                      "english": str(row[en_idx]).strip() if en_idx is not None and en_idx < len(row) and row[en_idx] else None,
                      "source_file": path.name}
            records.setdefault(normalized, record)
            categories[category] += 1
        workbook.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"source": "MohammadJRanjbar/PersianMedQA dictionary", "license": "CC BY 4.0",
               "unique_terms": len(records), "categories": dict(categories),
               "terms": sorted(records.values(), key=lambda r: r["normalized"])}
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ["source", "license", "unique_terms", "categories"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
