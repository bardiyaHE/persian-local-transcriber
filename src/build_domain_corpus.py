from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import json
import re
import sqlite3
import sys
import time
import urllib.parse
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from huggingface_hub import get_token, hf_hub_download

from consensus_v2 import norm


ROOT = Path(__file__).resolve().parent.parent
# Do not use the broad U+0600–U+06FF block here: it also contains Persian
# punctuation and digits, so forms such as «کنم؟» used to become a different
# unigram from «کنم».  Keep only actual Arabic/Persian letters (plus an
# internal ZWNJ); digits are handled by the first branch.
PERSIAN_LETTERS = "ءآأؤإئابتثجحخدذرزسشصضطظعغفقكلمنهويىةپچژگکیۀە"
TOKEN_RE = re.compile(
    rf"[0-9۰-۹]+|[{PERSIAN_LETTERS}]+(?:\u200c[{PERSIAN_LETTERS}]+)*"
)
SENTENCE_SPLIT_RE = re.compile(r"(?:[.!؟!?]+|\n+)\s*")
WIKIPEDIA_API = "https://fa.wikipedia.org/w/api.php"
WIKIPEDIA_TOPICS = [
    "پزشکی", "دارو", "دیابت", "دیابت نوع ۲", "فشار خون بالا", "قند خون",
    "انسولین", "بیماری قلبی-عروقی", "سکته قلبی", "نارسایی قلبی", "کلسترول",
    "کم‌خونی", "آسم", "آلرژی", "عفونت", "سرماخوردگی", "آنفلوانزا", "ذات‌الریه",
    "بیماری مزمن کلیه", "کبد چرب", "معده", "ریفلاکس معده", "میگرن", "افسردگی",
    "اضطراب", "تیروئید", "کم‌کاری تیروئید", "پرکاری تیروئید", "پوکی استخوان",
    "آرتروز", "بارداری", "واکسن", "آنتی‌بیوتیک", "پزشک", "نسخه پزشکی",
]
SOURCES = [
    {
        "id": "persianmedqa",
        "title": "PersianMedQA",
        "url": "https://huggingface.co/datasets/MohammadJRanjbar/PersianMedQA",
        "license": "CC BY 4.0; repository usage condition: non-commercial academic research",
        "domain": "medical",
        "weight": 3,
        "note": "Original Persian questions and answer choices only; English translations are excluded.",
    },
    {
        "id": "fa_wikipedia_medical",
        "title": "Persian Wikipedia medical pages",
        "url": "https://fa.wikipedia.org/",
        "license": "CC BY-SA 4.0 / GFDL (page histories retain attribution)",
        "domain": "medical",
        "weight": 2,
        "note": "Plain-text extracts of a fixed, general medical topic list.",
    },
    {
        "id": "common_voice_fa_clean",
        "title": "Persian Common Voice Clean 26.0",
        "url": "https://huggingface.co/datasets/pymmdrza/Common-Voice-Speech-26.0-Persian-Clean",
        "license": "CC0 1.0",
        "domain": "daily",
        "weight": 2,
        "note": "Text manifest only; no audio is downloaded.",
    },
    {
        "id": "persian_psydial_inputs",
        "title": "PersianPsyDial",
        "url": "https://huggingface.co/datasets/devsmehran/Persian_Psychosocial_Dialogues",
        "license": "CC BY 4.0",
        "domain": "daily",
        "weight": 1,
        "note": "User-side dialogue inputs only; repeated generated answers and instructions are excluded.",
    },
]


def tokens_of(text: str) -> list[str]:
    return [token for token in (norm(match.group(0)) for match in TOKEN_RE.finditer(str(text))) if token]


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).replace("\ufeff", " ")).strip()


def segments_of(text: str) -> Iterator[str]:
    for line in SENTENCE_SPLIT_RE.split(str(text)):
        line = clean_text(line)
        count = len(tokens_of(line))
        if 2 <= count <= 160:
            yield line


def hf_file(repo_id: str, filename: str, local_only: bool) -> Path:
    return Path(hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        token=get_token(),
        local_files_only=local_only,
    ))


def persianmedqa_rows(local_only: bool) -> Iterator[tuple[str, str, int]]:
    source = "persianmedqa"
    for filename in ("train.csv", "val.csv", "test.csv"):
        path = hf_file("MohammadJRanjbar/PersianMedQA", filename, local_only)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                question = clean_text(row.get("question") or "")
                if question:
                    yield source, question, 3
                raw_answers = row.get("answer") or ""
                try:
                    answers = ast.literal_eval(raw_answers)
                except (SyntaxError, ValueError):
                    answers = {}
                if isinstance(answers, dict):
                    for answer in answers.values():
                        answer = clean_text(answer)
                        if len(tokens_of(answer)) >= 2:
                            yield source, answer, 2


def common_voice_rows(local_only: bool) -> Iterator[tuple[str, str, int]]:
    path = hf_file(
        "pymmdrza/Common-Voice-Speech-26.0-Persian-Clean",
        "manifests/metadata.csv",
        local_only,
    )
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        header = next(handle, "")
        if "|text" not in header:
            raise RuntimeError("Unexpected Common Voice text-manifest schema.")
        for line in handle:
            _, separator, text = line.rstrip("\r\n").partition("|")
            if separator and clean_text(text):
                yield "common_voice_fa_clean", clean_text(text), 2


def psydial_rows(local_only: bool) -> Iterator[tuple[str, str, int]]:
    path = hf_file(
        "devsmehran/Persian_Psychosocial_Dialogues",
        "Persian Psychosocial Dialogues.zip",
        local_only,
    )
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist()
                 if name.endswith(".jsonl") and "/train_" in name]
        for name in sorted(names):
            with archive.open(name) as raw:
                for encoded in raw:
                    try:
                        row = json.loads(encoded.decode("utf-8-sig", errors="replace"))
                    except json.JSONDecodeError:
                        continue
                    text = clean_text(row.get("input") or "")
                    if text:
                        yield "persian_psydial_inputs", text, 1


def fetch_wikipedia(raw_path: Path, local_only: bool) -> list[dict[str, str]]:
    cached: list[dict[str, str]] = []
    if raw_path.is_file():
        cached = json.loads(raw_path.read_text(encoding="utf-8"))
        if local_only or len(cached) >= max(8, len(WIKIPEDIA_TOPICS) // 2):
            return cached
    if local_only:
        raise FileNotFoundError(f"Offline Wikipedia cache is missing: {raw_path}")
    pages_by_title = {row.get("title") or "": row for row in cached}
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    # MediaWiki lowers exlimit to one when a whole-article extract is requested,
    # so fetch the fixed topic list one page at a time rather than losing 14/15 pages.
    for title in WIKIPEDIA_TOPICS:
        if title in pages_by_title:
            continue
        params = {
            "action": "query", "prop": "extracts", "explaintext": "1",
            "exlimit": "1", "redirects": "1", "format": "json", "formatversion": "2",
            "titles": title,
        }
        request = urllib.request.Request(
            WIKIPEDIA_API + "?" + urllib.parse.urlencode(params),
            headers={"User-Agent": "whisper-persian-local-domain-corpus/1.0"},
        )
        payload = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    payload = json.load(response)
                break
            except urllib.error.HTTPError as exc:
                if exc.code != 429:
                    raise
                if attempt == 4:
                    print(
                        f"Wikipedia rate limit persisted for {title!r}; keeping "
                        f"{len(pages_by_title)} cached pages.",
                        file=sys.stderr,
                    )
                    break
                retry_after = float(exc.headers.get("Retry-After") or (2 ** attempt))
                time.sleep(min(12.0, max(1.0, retry_after)))
        if payload is None:
            continue
        for page in payload.get("query", {}).get("pages", []):
            if not page.get("missing") and page.get("extract"):
                page_title = page.get("title") or title
                pages_by_title[page_title] = {"title": page_title, "text": page["extract"]}
                raw_path.write_text(
                    json.dumps(list(pages_by_title.values()), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        time.sleep(0.35)
    pages = list(pages_by_title.values())
    raw_path.write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
    return pages


def wikipedia_rows(raw_path: Path, local_only: bool) -> Iterator[tuple[str, str, int]]:
    for page in fetch_wikipedia(raw_path, local_only):
        for sentence in segments_of(page.get("text") or ""):
            yield "fa_wikipedia_medical", sentence, 2


def all_rows(raw_path: Path, local_only: bool,
             public_only: bool) -> Iterable[tuple[str, str, str, int]]:
    if not public_only:
        for source, text, weight in persianmedqa_rows(local_only):
            yield "medical", source, text, weight
    for source, text, weight in wikipedia_rows(raw_path, local_only):
        yield "medical", source, text, weight
    for source, text, weight in common_voice_rows(local_only):
        yield "daily", source, text, weight
    if not public_only:
        for source, text, weight in psydial_rows(local_only):
            yield "daily", source, text, weight


def write_sources(path: Path, stats: dict[str, dict[str, int]], created_at: str) -> None:
    lines = [
        "# منابع مدل آماری دامنه‌ای", "",
        f"ساخته‌شده در: `{created_at}`", "",
        "در زمان رونویسی هیچ شبکه، API، مترجم یا LLM فراخوانی نمی‌شود. فقط نمایهٔ SQLite محلی خوانده می‌شود.", "",
        "| منبع | دامنه | مجوز | جملهٔ یکتا | توضیح |", "|---|---|---|---:|---|",
    ]
    for source in SOURCES:
        count = stats.get(source["id"], {}).get("sentences", 0)
        lines.append(
            f"| [{source['title']}]({source['url']}) | {source['domain']} | {source['license']} | "
            f"{count:,} | {source['note']} |"
        )
    lines += [
        "", "## سیاست", "",
        "- متن انگلیسی و ترجمهٔ ماشینی وارد n-gram فارسی نشده است.",
        "- داده‌ها در هر دامنه حذف تکرار شده‌اند تا یک عبارت قالبی رأی مصنوعی نگیرد.",
        "- این پیکره فقط tie-breaker است و اجازهٔ ساخت نامزد بدون پشتیبانی آوایی را ندارد.",
        "- عدد، دوز، نفی و هویت دارو با پیکره به‌تنهایی تغییر نمی‌کنند.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_wikipedia_attribution(raw_path: Path, output_path: Path) -> None:
    if not raw_path.is_file():
        return
    pages = json.loads(raw_path.read_text(encoding="utf-8"))
    lines = [
        "# صفحه‌های پزشکی ویکی‌پدیای فارسی", "",
        "این صفحه‌ها به‌صورت متن ساده و با حذف تکرار در ایندکس آماری استفاده شده‌اند. "
        "پیوند هر صفحه، تاریخچهٔ نویسندگان آن را نیز در دسترس می‌گذارد.", "",
    ]
    for page in sorted(pages, key=lambda row: row.get("title") or ""):
        title = page.get("title") or ""
        url = "https://fa.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
        lines.append(f"- [{title}]({url})")
    lines += [
        "", "مجوز متن: CC BY-SA 4.0 / GFDL؛ شرایط انتساب و بازنشر Wikimedia اعمال می‌شود.", "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def validate_existing(database: Path) -> dict[str, object]:
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    try:
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        counts = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("sentences", "unigrams", "bigrams", "trigrams")
        }
        if metadata.get("schema_version") != "1" or not all(counts.values()):
            raise RuntimeError("The existing domain-corpus index is incomplete.")
        return {"database": str(database), "metadata": metadata, "rows": counts, "reused": True}
    finally:
        connection.close()


def build(output_dir: Path, local_only: bool, rebuild: bool,
          public_only: bool) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    database = output_dir / "domain-ngrams-v1.sqlite3"
    if database.is_file() and not rebuild:
        result = validate_existing(database)
        metadata = result["metadata"]
        stats = json.loads(str(metadata.get("source_stats") or "{}"))
        write_sources(output_dir / "SOURCES.md", stats, str(metadata.get("created_at") or "unknown"))
        write_wikipedia_attribution(
            output_dir / "raw" / "wikipedia-medical.json",
            output_dir / "WIKIPEDIA_PAGES.md",
        )
        return result

    corpus_path = output_dir / "sentences-v1.jsonl.gz"
    counters: dict[str, dict[int, Counter[tuple[str, ...]]]] = {
        domain: {1: Counter(), 2: Counter(), 3: Counter()} for domain in ("medical", "daily")
    }
    stats: dict[str, dict[str, int]] = {}
    seen: set[bytes] = set()
    with gzip.open(corpus_path, "wt", encoding="utf-8") as corpus:
        for domain, source, text, weight in all_rows(
                output_dir / "raw" / "wikipedia-medical.json", local_only, public_only):
            tokens = tokens_of(text)
            if not (2 <= len(tokens) <= 160):
                continue
            normalized = " ".join(tokens)
            digest = hashlib.blake2b(
                f"{domain}\0{normalized}".encode("utf-8"), digest_size=16).digest()
            if digest in seen:
                continue
            seen.add(digest)
            source_stats = stats.setdefault(source, {"sentences": 0, "tokens": 0})
            source_stats["sentences"] += 1
            source_stats["tokens"] += len(tokens)
            row = {"domain": domain, "source": source, "text": text,
                   "normalized": normalized, "weight": weight}
            corpus.write(json.dumps(row, ensure_ascii=False) + "\n")
            for size in (1, 2, 3):
                for index in range(len(tokens) - size + 1):
                    counters[domain][size][tuple(tokens[index:index + size])] += weight

    required_sources = (
        ("fa_wikipedia_medical", "common_voice_fa_clean")
        if public_only else ("persianmedqa", "common_voice_fa_clean")
    )
    missing_sources = [source for source in required_sources if not stats.get(source)]
    if missing_sources:
        raise RuntimeError(
            "Required Persian corpus sources produced no usable text: "
            + ", ".join(missing_sources)
        )

    temporary = database.with_suffix(".building.sqlite3")
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript("""
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=MEMORY;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
            CREATE TABLE sentences (
                id INTEGER PRIMARY KEY,
                domain TEXT NOT NULL,
                source TEXT NOT NULL,
                text TEXT NOT NULL,
                normalized TEXT NOT NULL
            );
            CREATE TABLE unigrams (
                domain TEXT NOT NULL, w1 TEXT NOT NULL, count INTEGER NOT NULL,
                PRIMARY KEY (domain, w1)
            ) WITHOUT ROWID;
            CREATE TABLE bigrams (
                domain TEXT NOT NULL, w1 TEXT NOT NULL, w2 TEXT NOT NULL, count INTEGER NOT NULL,
                PRIMARY KEY (domain, w1, w2)
            ) WITHOUT ROWID;
            CREATE TABLE trigrams (
                domain TEXT NOT NULL, w1 TEXT NOT NULL, w2 TEXT NOT NULL, w3 TEXT NOT NULL,
                count INTEGER NOT NULL,
                PRIMARY KEY (domain, w1, w2, w3)
            ) WITHOUT ROWID;
            CREATE VIRTUAL TABLE sentence_fts USING fts5(normalized, tokenize='unicode61');
        """)
        created_at = datetime.now(timezone.utc).isoformat()
        metadata = {
            "schema_version": "1",
            "created_at": created_at,
            "algorithm": "weighted exact Persian unigram/bigram/trigram with FTS5 context retrieval",
            "translation_used": "false",
            "llm_used": "false",
            "public_only": str(public_only).lower(),
            "source_manifest": json.dumps(SOURCES, ensure_ascii=False),
            "source_stats": json.dumps(stats, ensure_ascii=False),
        }
        connection.executemany("INSERT INTO metadata(key,value) VALUES (?,?)", metadata.items())
        with gzip.open(corpus_path, "rt", encoding="utf-8") as corpus:
            sentence_batch = []
            fts_batch = []
            sentence_id = 0
            for line in corpus:
                row = json.loads(line)
                sentence_id += 1
                sentence_batch.append((sentence_id, row["domain"], row["source"], row["text"], row["normalized"]))
                fts_batch.append((sentence_id, row["normalized"]))
                if len(sentence_batch) >= 2000:
                    connection.executemany(
                        "INSERT INTO sentences(id,domain,source,text,normalized) VALUES (?,?,?,?,?)",
                        sentence_batch,
                    )
                    connection.executemany(
                        "INSERT INTO sentence_fts(rowid,normalized) VALUES (?,?)", fts_batch)
                    sentence_batch.clear()
                    fts_batch.clear()
            if sentence_batch:
                connection.executemany(
                    "INSERT INTO sentences(id,domain,source,text,normalized) VALUES (?,?,?,?,?)",
                    sentence_batch,
                )
                connection.executemany(
                    "INSERT INTO sentence_fts(rowid,normalized) VALUES (?,?)", fts_batch)

        for domain in ("medical", "daily"):
            connection.executemany(
                "INSERT INTO unigrams(domain,w1,count) VALUES (?,?,?)",
                ((domain, gram[0], count) for gram, count in counters[domain][1].items()),
            )
            connection.executemany(
                "INSERT INTO bigrams(domain,w1,w2,count) VALUES (?,?,?,?)",
                ((domain, gram[0], gram[1], count) for gram, count in counters[domain][2].items()),
            )
            connection.executemany(
                "INSERT INTO trigrams(domain,w1,w2,w3,count) VALUES (?,?,?,?,?)",
                ((domain, gram[0], gram[1], gram[2], count)
                 for gram, count in counters[domain][3].items()),
            )
        connection.commit()
        connection.execute("PRAGMA optimize")
    finally:
        connection.close()
    temporary.replace(database)
    write_sources(output_dir / "SOURCES.md", stats, created_at)
    write_wikipedia_attribution(
        output_dir / "raw" / "wikipedia-medical.json",
        output_dir / "WIKIPEDIA_PAGES.md",
    )
    result = validate_existing(database)
    result["reused"] = False
    result["source_stats"] = stats
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local Persian medical/daily n-gram index.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "offline-corpus")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--public-only", action="store_true",
        help="Build only from Persian Wikipedia and the public Common Voice text manifest.",
    )
    args = parser.parse_args()
    result = build(
        args.output_dir.resolve(), args.local_only, args.rebuild, args.public_only
    )
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
