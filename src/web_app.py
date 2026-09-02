from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parent.parent
FFMPEG_DIR = ROOT / "runtime" / "ffmpeg"
CUDA_BIN_DIRS = [path for path in (ROOT / "runtime" / "cuda-libs").glob("**/bin") if path.is_dir()]
os.environ["PATH"] = os.pathsep.join([str(FFMPEG_DIR), *(str(path) for path in CUDA_BIN_DIRS),
                                      os.environ.get("PATH", "")])

import gradio as gr


PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
GOOGLE_FALLBACK_ENABLED = os.environ.get(
    "PERSIAN_TRANSCRIBER_GOOGLE_FALLBACK", "0"
).strip().lower() in {"1", "true", "yes", "on"}
MEDICAL_INDEX = ROOT / "offline-lexicon" / "combined-medical-drug-index.json"
CORPUS_INDEX = ROOT / "offline-corpus" / "domain-ngrams-v1.sqlite3"
ENCODER_DIR = ROOT / "models" / "semantic-encoder-v1"
FINAL_RELATIVE = Path("final-delivery") / "02-after-algorithm-v2-turbo-lexicon"
BASE_RELATIVE = Path("final-delivery") / "02-after-algorithm"
SAFE_RELATIVE = Path("final-delivery") / "04-safe-no-llm"
V3_RELATIVE = Path("final-delivery") / "02-after-algorithm-v3-phrase-network"
V4_RELATIVE = Path("final-delivery") / "02-after-algorithm-v4-ngram-lexicon"
V5_RELATIVE = Path("final-delivery") / "02-after-algorithm-v5-domain-corpus"
V7_RELATIVE = Path("final-delivery") / "07-semantic-retrieval-agent"
V8_RELATIVE = Path("final-delivery") / "08-turbo-first-minilm"
V9_RELATIVE = Path("final-delivery") / "09-medical-drug-dictionary"
V10_RELATIVE = Path("final-delivery") / "10-local-qwen-reranker"
V11_RELATIVE = Path("final-delivery") / "11-local-qwen-summary"
GOOGLE_RECOGNITION_RELATIVE = Path("final-delivery") / "12-google-recognition-fallback"
FULL_HYPOTHESES = [
    "medium__raw", "medium__enhanced", "large-v3-turbo__raw",
    "large-v3-turbo__enhanced", "large-v3__raw", "large-v3__enhanced",
]
LITE_HYPOTHESES = ["large-v3-turbo__raw", "large-v3-turbo__enhanced"]


def installation_profile() -> str:
    manifest = ROOT / "runtime" / "install-profile.json"
    if not manifest.is_file():
        return "full"
    try:
        value = str(json.loads(manifest.read_text(encoding="utf-8-sig")).get("profile") or "")
    except (OSError, ValueError, TypeError):
        return "full"
    return value if value in {"lite", "full"} else "full"


def pipeline_status(run_dir: Path) -> str:
    profile = installation_profile()
    hypotheses = LITE_HYPOTHESES if profile == "lite" else FULL_HYPOTHESES
    if not run_dir.exists():
        return "در حال آماده‌سازی فایل…"
    if not (run_dir / "normalized_mono_48k.wav").exists():
        return "مرحله ۱ از ۱۴: تبدیل صوت با FFmpeg…"
    if not (run_dir / "enhanced_pyannote.wav").exists():
        return "مرحله ۲ از ۱۴: جداسازی موسیقی و گوینده با Demucs + pyannote…"
    completed = sum((run_dir / "hypotheses" / key / f"{key}.json").exists() for key in hypotheses)
    if completed < len(hypotheses):
        return (f"مراحل ۳ تا ۸ از ۱۴: Turbo و بازبینی انتخابی — "
                f"{completed} از {len(hypotheses)} فایل فرضیه آماده شده…")
    if profile == "lite":
        return "مرحلهٔ نهایی Lite: آماده‌سازی متن Turbo و فایل‌های خروجی…"
    if not (run_dir / V9_RELATIVE / "final-v9.txt").exists():
        return "مراحل ۹ تا ۱۱ از ۱۴: قفل Turbo، MiniLM و بانک گسترش‌یافتهٔ داروهای فارسی…"
    if not (run_dir / V10_RELATIVE / "final-v10.txt").exists():
        return "مراحل ۱۲ تا ۱۳ از ۱۴: انتخاب محدود واژه و عبارت با Qwen کاملاً محلی…"
    return "مرحله ۱۴ از ۱۴: ساخت خلاصهٔ جدا و اعتبارسنجی دارو، دوز، عدد و نفی…"


def execution_backend() -> tuple[str, str, str]:
    """Prefer the RTX/CUDA path and never silently mask an incomplete NVIDIA setup."""
    if shutil.which("nvidia-smi"):
        if not CUDA_BIN_DIRS:
            raise RuntimeError("کارت NVIDIA پیدا شد ولی کتابخانه‌های محلی CUDA/cuDNN موجود نیست؛ setup.ps1 را اجرا کنید.")
        return "cuda", "float16", "RTX/CUDA float16"
    return "cpu", "int8", "CPU int8"


def history_choices() -> list[tuple[str, str]]:
    choices = []
    for run_dir in sorted((path for path in (ROOT / "outputs").iterdir() if path.is_dir()),
                          key=lambda path: path.name, reverse=True):
        if ((run_dir / V10_RELATIVE / "final-v10.txt").is_file()
                or (run_dir / V9_RELATIVE / "final-v9.txt").is_file()
                or (run_dir / V8_RELATIVE / "final-v8.txt").is_file()
                or (run_dir / V7_RELATIVE / "final-v7.txt").is_file()
                or (run_dir / V5_RELATIVE / "final-v5.txt").is_file()
                or (run_dir / V4_RELATIVE / "final-v4.txt").is_file()
                or (run_dir / V3_RELATIVE / "final-v3.txt").is_file()
                or (run_dir / SAFE_RELATIVE / "safe-final.txt").is_file()
                or (run_dir / FINAL_RELATIVE / "final-v2.txt").is_file()
                or (run_dir / BASE_RELATIVE / "final.txt").is_file()):
            choices.append((run_dir.name.replace("ui-", ""), run_dir.name))
    return choices


def collect_result(run_dir: Path) -> tuple[str, str, str, list[str], dict]:
    v10_dir = run_dir / V10_RELATIVE
    v10_path = v10_dir / "final-v10.txt"
    if v10_path.is_file():
        final_text = v10_path.read_text(encoding="utf-8").strip()
        google_dir = run_dir / GOOGLE_RECOGNITION_RELATIVE
        google_payload_path = google_dir / "google-recognition.json"
        google_payload = (json.loads(google_payload_path.read_text(encoding="utf-8"))
                          if google_payload_path.is_file() else {})
        google_text_path = google_dir / "google-recognition.txt"
        if google_payload.get("selected") and google_text_path.is_file():
            final_text = google_text_path.read_text(encoding="utf-8").strip()
        v9_path = run_dir / V9_RELATIVE / "final-v9.txt"
        baseline = v9_path.read_text(encoding="utf-8").strip() if v9_path.is_file() else final_text
        review_path = v10_dir / "review-v10.md"
        review = (review_path.read_text(encoding="utf-8") if review_path.is_file()
                  else "مورد بازبینی ثبت نشده است.")
        google_review = google_dir / "review-google-recognition.md"
        if google_review.is_file():
            review = google_review.read_text(encoding="utf-8") + "\n" + review
        candidates = [
            v10_path, v10_dir / "final-v10.json", v10_dir / "summary-v10.json",
            v10_dir / "comparison-v10.md", review_path,
            google_text_path, google_payload_path, google_review,
            run_dir / V11_RELATIVE / "final-summary-v11.txt",
            run_dir / V11_RELATIVE / "final-summary-v11.json",
            run_dir / V11_RELATIVE / "summary-v11.json",
            run_dir / V11_RELATIVE / "review-v11.md",
            run_dir / V9_RELATIVE / "drug-audit-v9.json",
            run_dir / V9_RELATIVE / "review-clips-v9.json",
            run_dir / "final-delivery" / "03-denoised-audio" / "enhanced.wav",
            run_dir / f"{run_dir.name}-results.zip",
            ROOT / "models" / "qwen3.5-35b-a3b-gguf" / "MODEL_MANIFEST.json",
            ROOT / "offline-lexicon" / "LEXICON_SOURCES.md",
            ROOT / "offline-corpus" / "SOURCES.md",
        ]
        downloads = [str(path) for path in candidates if path.is_file()]
        summary_path = v10_dir / "summary-v10.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        v11_summary_path = run_dir / V11_RELATIVE / "summary-v11.json"
        v11_summary = (json.loads(v11_summary_path.read_text(encoding="utf-8"))
                       if v11_summary_path.is_file() else {})
        details = {
            "run_id": run_dir.name, "output_folder": str(run_dir),
            "backend": summary.get("model", {}).get("model"), "local_qwen_v10": summary,
            "local_qwen_summary_v11": v11_summary,
            "google_speech_fallback": google_payload,
        }
        return final_text, baseline, review, downloads, details

    v9_dir = run_dir / V9_RELATIVE
    v9_path = v9_dir / "final-v9.txt"
    if v9_path.is_file():
        final_text = v9_path.read_text(encoding="utf-8").strip()
        turbo_files = sorted((run_dir / "hypotheses" / "large-v3-turbo__enhanced").glob("*.txt"))
        baseline = turbo_files[0].read_text(encoding="utf-8").strip() if turbo_files else final_text
        review_path = v9_dir / "review-v9.md"
        review = (review_path.read_text(encoding="utf-8") if review_path.is_file()
                  else "مورد بازبینی ثبت نشده است.")
        candidates = [
            v9_path, v9_dir / "final-v9.json", v9_dir / "summary-v9.json",
            v9_dir / "comparison-v9.md", review_path, v9_dir / "review-v9.json",
            v9_dir / "review-clips-v9.json", v9_dir / "drug-audit-v9.json",
            v9_dir / "final-v9-user-filled.txt", run_dir / "adaptive-turbo-plan.json",
            ROOT / "models" / "semantic-encoder-v1" / "MODEL_MANIFEST.json",
            ROOT / "offline-lexicon" / "LEXICON_SOURCES.md",
            ROOT / "offline-corpus" / "SOURCES.md",
            ROOT / "offline-corpus" / "WIKIPEDIA_PAGES.md",
            run_dir / "final-delivery" / "03-denoised-audio" / "enhanced.wav",
            run_dir / f"{run_dir.name}-results.zip",
        ]
        clip_manifest = v9_dir / "review-clips-v9.json"
        if clip_manifest.is_file():
            candidates.extend(Path(row["clip"]) for row in json.loads(
                clip_manifest.read_text(encoding="utf-8")) if row.get("clip"))
        downloads = [str(path) for path in candidates if path.is_file()]
        summary_path = v9_dir / "summary-v9.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        details = {
            "run_id": run_dir.name, "output_folder": str(run_dir),
            "backend": summary.get("backend"), "medical_drug_v9": summary,
        }
        return final_text, baseline, review, downloads, details

    v8_dir = run_dir / V8_RELATIVE
    v8_path = v8_dir / "final-v8.txt"
    if v8_path.is_file():
        final_text = v8_path.read_text(encoding="utf-8").strip()
        turbo_files = sorted((run_dir / "hypotheses" / "large-v3-turbo__enhanced").glob("*.txt"))
        baseline = turbo_files[0].read_text(encoding="utf-8").strip() if turbo_files else final_text
        review_path = v8_dir / "review-v8.md"
        review = (review_path.read_text(encoding="utf-8") if review_path.is_file()
                  else "مورد بازبینی ثبت نشده است.")
        candidates = [
            v8_path, v8_dir / "final-v8.json", v8_dir / "summary-v8.json",
            v8_dir / "comparison-v8.md", review_path, v8_dir / "review-v8.json",
            v8_dir / "review-clips-v8.json", v8_dir / "final-v8-user-filled.txt",
            run_dir / "adaptive-turbo-plan.json",
            ROOT / "models" / "semantic-encoder-v1" / "MODEL_MANIFEST.json",
            ROOT / "offline-corpus" / "SOURCES.md",
            ROOT / "offline-corpus" / "WIKIPEDIA_PAGES.md",
            run_dir / "final-delivery" / "03-denoised-audio" / "enhanced.wav",
            run_dir / f"{run_dir.name}-results.zip",
        ]
        clip_manifest = v8_dir / "review-clips-v8.json"
        if clip_manifest.is_file():
            candidates.extend(Path(row["clip"]) for row in json.loads(
                clip_manifest.read_text(encoding="utf-8")) if row.get("clip"))
        downloads = [str(path) for path in candidates if path.is_file()]
        summary_path = v8_dir / "summary-v8.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        details = {
            "run_id": run_dir.name, "output_folder": str(run_dir),
            "backend": summary.get("backend"), "turbo_first_minilm_v8": summary,
        }
        return final_text, baseline, review, downloads, details

    v7_dir = run_dir / V7_RELATIVE
    v7_path = v7_dir / "final-v7.txt"
    if v7_path.is_file():
        final_text = v7_path.read_text(encoding="utf-8").strip()
        turbo_files = sorted((run_dir / "hypotheses" / "large-v3-turbo__enhanced").glob("*.txt"))
        baseline = turbo_files[0].read_text(encoding="utf-8").strip() if turbo_files else final_text
        review_path = v7_dir / "review-v7.md"
        review = (review_path.read_text(encoding="utf-8") if review_path.is_file()
                  else "مورد بازبینی ثبت نشده است.")
        candidates = [
            v7_path, v7_dir / "final-v7.json", v7_dir / "summary-v7.json",
            v7_dir / "comparison-v7.md", review_path, v7_dir / "review-v7.json",
            v7_dir / "review-clips-v7.json", v7_dir / "final-v7-user-filled.txt",
            ROOT / "models" / "semantic-encoder-v1" / "MODEL_MANIFEST.json",
            ROOT / "offline-corpus" / "SOURCES.md",
            ROOT / "offline-corpus" / "WIKIPEDIA_PAGES.md",
            run_dir / "final-delivery" / "03-denoised-audio" / "enhanced.wav",
            run_dir / f"{run_dir.name}-results.zip",
        ]
        clip_manifest = v7_dir / "review-clips-v7.json"
        if clip_manifest.is_file():
            candidates.extend(Path(row["clip"]) for row in json.loads(
                clip_manifest.read_text(encoding="utf-8")) if row.get("clip"))
        downloads = [str(path) for path in candidates if path.is_file()]
        summary_path = v7_dir / "summary-v7.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        details = {
            "run_id": run_dir.name, "output_folder": str(run_dir),
            "backend": summary.get("backend"), "semantic_agent_v7": summary,
        }
        return final_text, baseline, review, downloads, details

    v5_dir = run_dir / V5_RELATIVE
    v5_path = v5_dir / "final-v5.txt"
    if v5_path.is_file():
        final_text = v5_path.read_text(encoding="utf-8").strip()
        turbo_files = sorted((run_dir / "hypotheses" / "large-v3-turbo__enhanced").glob("*.txt"))
        baseline = turbo_files[0].read_text(encoding="utf-8").strip() if turbo_files else final_text
        review_path = v5_dir / "review-v5.md"
        review = review_path.read_text(encoding="utf-8") if review_path.is_file() else "مورد بازبینی ثبت نشده است."
        targeted_path = v5_dir / "targeted-review.md"
        if targeted_path.is_file():
            review += "\n\n" + targeted_path.read_text(encoding="utf-8")
        candidates = [
            v5_path, v5_dir / "final-v5.json", v5_dir / "scorecard-v5.md",
            v5_dir / "ngram-turbo-quality-v3.json",
            v5_dir / "comparison-v5.md", review_path, v5_dir / "review-v5.json",
            v5_dir / "name-slots-v5.json", v5_dir / "final-v5-user-filled.txt",
            targeted_path, v5_dir / "targeted-review.json", v5_dir / "review-clips-v5.json",
            ROOT / "offline-corpus" / "SOURCES.md",
            ROOT / "offline-corpus" / "WIKIPEDIA_PAGES.md",
            ROOT / "runtime" / "live-readiness-report.json",
            run_dir / "final-delivery" / "03-denoised-audio" / "enhanced.wav",
            run_dir / f"{run_dir.name}-results.zip",
        ]
        clip_manifest = v5_dir / "review-clips-v5.json"
        if clip_manifest.is_file():
            candidates.extend(Path(row["clip"]) for row in json.loads(
                clip_manifest.read_text(encoding="utf-8")) if row.get("clip"))
        downloads = [str(path) for path in candidates if path.is_file()]
        summary_path = v5_dir / "summary-v5.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        score_path = v5_dir / "ngram-turbo-quality-v3.json"
        if score_path.is_file():
            score_payload = json.loads(score_path.read_text(encoding="utf-8"))
            summary["ngram_voice_score"] = score_payload.get("score")
            summary["ngram_voice_score_scale"] = "0-100"
        details = {"run_id": run_dir.name, "output_folder": str(run_dir),
                   "backend": summary.get("backend"), "domain_corpus_v5": summary}
        return final_text, baseline, review, downloads, details

    v4_dir = run_dir / V4_RELATIVE
    v4_path = v4_dir / "final-v4.txt"
    if v4_path.is_file():
        final_text = v4_path.read_text(encoding="utf-8").strip()
        turbo_files = sorted((run_dir / "hypotheses" / "large-v3-turbo__enhanced").glob("*.txt"))
        baseline = turbo_files[0].read_text(encoding="utf-8").strip() if turbo_files else final_text
        review_path = v4_dir / "review-v4.md"
        review = review_path.read_text(encoding="utf-8") if review_path.is_file() else "مورد بازبینی ثبت نشده است."
        targeted_path = v4_dir / "targeted-review.md"
        if targeted_path.is_file():
            review += "\n\n" + targeted_path.read_text(encoding="utf-8")
        candidates = [
            v4_path, v4_dir / "final-v4.json", v4_dir / "scorecard-v4.md",
            v4_dir / "comparison-v4.md", review_path, v4_dir / "review-v4.json",
            targeted_path, v4_dir / "targeted-review.json", v4_dir / "review-clips-v4.json",
            run_dir / "final-delivery" / "03-denoised-audio" / "enhanced.wav",
            run_dir / f"{run_dir.name}-results.zip",
        ]
        clip_manifest = v4_dir / "review-clips-v4.json"
        if clip_manifest.is_file():
            candidates.extend(Path(row["clip"]) for row in json.loads(
                clip_manifest.read_text(encoding="utf-8")) if row.get("clip"))
        downloads = [str(path) for path in candidates if path.is_file()]
        summary_path = v4_dir / "summary-v4.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        details = {"run_id": run_dir.name, "output_folder": str(run_dir),
                   "backend": summary.get("backend"), "ngram_lexicon_v4": summary}
        return final_text, baseline, review, downloads, details

    v3_dir = run_dir / V3_RELATIVE
    v3_path = v3_dir / "final-v3.txt"
    if v3_path.is_file():
        final_text = v3_path.read_text(encoding="utf-8").strip()
        turbo_files = sorted((run_dir / "hypotheses" / "large-v3-turbo__enhanced").glob("*.txt"))
        baseline = turbo_files[0].read_text(encoding="utf-8").strip() if turbo_files else final_text
        review_path = v3_dir / "review-v3.md"
        review = review_path.read_text(encoding="utf-8") if review_path.is_file() else "مورد بازبینی ثبت نشده است."
        targeted_path = v3_dir / "targeted-review.md"
        if targeted_path.is_file():
            review += "\n\n" + targeted_path.read_text(encoding="utf-8")
        candidates = [
            v3_path, v3_dir / "final-v3.json", v3_dir / "scorecard-v3.md",
            v3_dir / "comparison-v3.md",
            review_path, v3_dir / "review-v3.json", targeted_path,
            v3_dir / "targeted-review.json", v3_dir / "review-clips-v3.json",
            run_dir / "final-delivery" / "03-denoised-audio" / "enhanced.wav",
            run_dir / f"{run_dir.name}-results.zip",
        ]
        clip_manifest = v3_dir / "review-clips-v3.json"
        if clip_manifest.is_file():
            candidates.extend(Path(row["clip"]) for row in json.loads(
                clip_manifest.read_text(encoding="utf-8")) if row.get("clip"))
        downloads = [str(path) for path in candidates if path.is_file()]
        summary_path = v3_dir / "summary-v3.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        details = {"run_id": run_dir.name, "output_folder": str(run_dir),
                   "backend": summary.get("backend"), "phrase_network_v3": summary}
        return final_text, baseline, review, downloads, details

    base_path = run_dir / BASE_RELATIVE / "final.txt"
    if base_path.is_file():
        final_text = base_path.read_text(encoding="utf-8").strip()
        report_path = run_dir / "report.md"
        review = (report_path.read_text(encoding="utf-8") if report_path.is_file()
                  else "حالت Lite بدون بازبینی معنایی و خلاصه‌ساز اجرا شده است.")
        candidates = [
            base_path,
            run_dir / BASE_RELATIVE / "final.json",
            report_path,
            run_dir / "runtime-benchmark.json",
            run_dir / "final-delivery" / "03-denoised-audio" / "enhanced.wav",
            run_dir / f"{run_dir.name}-results.zip",
        ]
        details = {
            "run_id": run_dir.name,
            "output_folder": str(run_dir),
            "backend": "whisper-large-v3-turbo",
            "profile": "lite",
        }
        return final_text, final_text, review, [str(path) for path in candidates if path.is_file()], details

    safe_dir, final_dir = run_dir / SAFE_RELATIVE, run_dir / FINAL_RELATIVE
    safe_path = safe_dir / "safe-final.txt"
    if not safe_path.is_file():
        safe_path = final_dir / "final-v2.txt"
    safe_text = safe_path.read_text(encoding="utf-8").strip()
    suggested_path = safe_dir / "suggested-for-review.txt"
    suggested = (suggested_path.read_text(encoding="utf-8").strip()
                 if suggested_path.is_file() else safe_text)
    review_path = safe_dir / "review-items.md"
    review = (review_path.read_text(encoding="utf-8") if review_path.is_file()
              else "برای این اجرای قدیمی گزارش محافظ بدون LLM موجود نیست.")
    candidates = [
        safe_path, suggested_path, review_path, safe_dir / "verdicts.json",
        safe_dir / "sensitive-phrase-audit.json", final_dir / "scorecard-v2.md",
        final_dir / "scorecard-v2.json", final_dir / "report-v2.md",
        run_dir / "final-delivery" / "03-denoised-audio" / "enhanced.wav",
        run_dir / f"{run_dir.name}-results.zip",
    ]
    candidates.extend(sorted((safe_dir / "review-clips").glob("*.wav")))
    downloads = [str(path) for path in candidates if path.is_file()]
    safe_summary_path = safe_dir / "summary.json"
    safe_summary = (json.loads(safe_summary_path.read_text(encoding="utf-8"))
                    if safe_summary_path.is_file() else {})
    details = {"run_id": run_dir.name, "output_folder": str(run_dir),
               "backend": safe_summary.get("backend"), "safety_gate": safe_summary}
    return safe_text, suggested, review, downloads, details


def refresh_history():
    choices = history_choices()
    return gr.update(choices=choices, value=(choices[0][1] if choices else None))


def collect_generated_summary(run_dir: Path) -> str:
    summary_path = run_dir / V11_RELATIVE / "final-summary-v11.txt"
    return summary_path.read_text(encoding="utf-8").strip() if summary_path.is_file() else ""


def load_history(run_id: str | None):
    if not run_id:
        return "اجرایی انتخاب نشده است.", "", "", "", "", [], {}
    run_dir = ROOT / "outputs" / run_id
    if not run_dir.is_dir() or run_dir.parent != (ROOT / "outputs"):
        return "اجرای انتخاب‌شده پیدا نشد.", "", "", "", "", [], {}
    safe, suggested, review, downloads, details = collect_result(run_dir)
    generated_summary = collect_generated_summary(run_dir)
    score = (details.get("domain_corpus_v5") or {}).get("ngram_voice_score")
    score_note = f" امتیاز کیفیت Turbo+n-gram: {float(score):.1f} از ۱۰۰." if score is not None else ""
    if details.get("medical_drug_v9"):
        semantic = details["medical_drug_v9"]
        score_note = (
            f" V9 دارویی فعال بود؛ {int(semantic.get('drug_dictionary_repair_count') or 0)} "
            f"نام دارو با واژه‌نامه اصلاح و {float(semantic.get('turbo_retention_ratio') or 0.0) * 100:.1f}٪ "
            "مسیر Turbo حفظ شد."
        )
    if details.get("local_qwen_v10"):
        local = details["local_qwen_v10"]
        score_note = (
            f" V10 محلی فعال بود؛ {int(local.get('applied_slot_count') or 0)} تغییر واژه‌ای و "
            f"{int(local.get('applied_region_count') or 0)} تغییر عبارتی اعمال شد؛ "
            "هیچ متن آزاد مدل وارد رونویسی نشده است."
        )
    if details.get("local_qwen_summary_v11"):
        summary_v11 = details["local_qwen_summary_v11"]
        score_note += (
            " خلاصهٔ جداگانهٔ V11 "
            + ("پذیرفته شد." if summary_v11.get("accepted") else "به‌علت قفل اعتبارسنجی fallback شد.")
        )
    elif details.get("turbo_first_minilm_v8"):
        semantic = details["turbo_first_minilm_v8"]
        score_note = (
            f" V8 Turbo-first فعال بود؛ {int(semantic.get('path_change_count') or 0)} "
            f"تغییر مجاز و {float(semantic.get('turbo_retention_ratio') or 0.0) * 100:.1f}٪ "
            "حفظ Turbo ثبت شد."
        )
    elif details.get("semantic_agent_v7"):
        semantic = details["semantic_agent_v7"]
        score_note = (
            f" Agent معنایی محلی فعال بود؛ {int(semantic.get('path_change_count') or 0)} "
            "تغییر مسیر ثبت شد."
        )
    return (f"✅ نتیجهٔ ذخیره‌شده بارگذاری شد.{score_note}", safe, generated_summary,
            suggested, review, downloads, details)


def fill_name_slots(template: str, slots: list[dict], values: list[str]) -> tuple[str, int]:
    """Fill protected blanks in order; user spelling is preserved byte-for-byte after trimming."""
    updated = template
    used = min(len(values), len(slots))
    for index in range(used):
        placeholder = str(slots[index].get("placeholder") or "________")
        updated = updated.replace(placeholder, values[index], 1)
    return updated, used


def apply_name_values(name_values: str | None, details: dict | None):
    """Insert user-supplied names into protected blanks without normalizing or scoring them."""
    values = [line.strip() for line in (name_values or "").splitlines() if line.strip()]
    if not values:
        return "نام صحیح را در کادر وارد کنید.", gr.skip(), gr.skip()
    run_id = str((details or {}).get("run_id") or "")
    outputs_root = (ROOT / "outputs").resolve()
    run_dir = (outputs_root / run_id).resolve()
    if not run_id or run_dir.parent != outputs_root or not run_dir.is_dir():
        return "❌ اجرای مربوط به این متن پیدا نشد.", gr.skip(), gr.skip()

    v10_dir = run_dir / V10_RELATIVE
    v9_dir = run_dir / V9_RELATIVE
    v10_template = v10_dir / "final-v10.txt"
    v9_payload = v9_dir / "final-v9.json"
    google_dir = run_dir / GOOGLE_RECOGNITION_RELATIVE
    google_payload_path = google_dir / "google-recognition.json"
    google_payload = (json.loads(google_payload_path.read_text(encoding="utf-8"))
                      if google_payload_path.is_file() else {})
    google_template = google_dir / "google-recognition.txt"
    if google_payload.get("selected") and google_template.is_file():
        template_path = google_template
        slots = google_payload.get("protected_name_slots") or []
        filled_path = google_dir / "google-recognition-user-filled.txt"
    elif v10_template.is_file() and v9_payload.is_file():
        template_path = v10_template
        payload = json.loads(v9_payload.read_text(encoding="utf-8"))
        slots = payload.get("protected_name_slots") or []
        filled_path = v10_dir / "final-v10-user-filled.txt"
    elif (v9_dir / "final-v9.txt").is_file() and v9_payload.is_file():
        template_path = v9_dir / "final-v9.txt"
        payload = json.loads(v9_payload.read_text(encoding="utf-8"))
        slots = payload.get("protected_name_slots") or []
        filled_path = v9_dir / "final-v9-user-filled.txt"
    else:
        v8_dir = run_dir / V8_RELATIVE
        template_path = v8_dir / "final-v8.txt"
        payload_path = v8_dir / "final-v8.json"
        if template_path.is_file() and payload_path.is_file():
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            slots = payload.get("protected_name_slots") or []
            filled_path = v8_dir / "final-v8-user-filled.txt"
        else:
            v7_dir = run_dir / V7_RELATIVE
            template_path = v7_dir / "final-v7.txt"
            payload_path = v7_dir / "final-v7.json"
            if template_path.is_file() and payload_path.is_file():
                payload = json.loads(payload_path.read_text(encoding="utf-8"))
                slots = payload.get("protected_name_slots") or []
                filled_path = v7_dir / "final-v7-user-filled.txt"
            else:
                v5_dir = run_dir / V5_RELATIVE
                template_path = v5_dir / "final-v5.txt"
                slots_path = v5_dir / "name-slots-v5.json"
                if not template_path.is_file() or not slots_path.is_file():
                    return "این خروجی جای خالیِ نام ندارد.", gr.skip(), gr.skip()
                slots = json.loads(slots_path.read_text(encoding="utf-8"))
                filled_path = v5_dir / "final-v5-user-filled.txt"
    if not slots:
        return "در این متن نامی بعد از خانم/آقا برای تکمیل پیدا نشده است.", gr.skip(), gr.skip()

    updated, used = fill_name_slots(
        template_path.read_text(encoding="utf-8").strip(), slots, values)
    filled_path.write_text(updated + "\n", encoding="utf-8")
    shutil.make_archive(
        str(run_dir / f"{run_id}-results"), "zip", root_dir=run_dir / "final-delivery")
    downloads = collect_result(run_dir)[3]
    remaining = len(slots) - used
    note = (f"✅ {used} نام عیناً و بدون تصحیح خودکار جای‌گذاری و ذخیره شد."
            if remaining == 0 else
            f"✅ {used} نام جای‌گذاری شد؛ {remaining} جای خالی هنوز باقی است. هر نام را در یک خط بنویسید.")
    return note, updated, downloads


def run_command(command: list[str], log_path: Path, run_dir: Path) -> Iterator[str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   env=env, creationflags=subprocess.CREATE_NO_WINDOW)
        while process.poll() is None:
            yield pipeline_status(run_dir)
            time.sleep(2)
        if process.returncode:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
            raise RuntimeError(f"پردازش با کد {process.returncode} متوقف شد.\n\n{tail}")


def process_audio(audio_path: str | None):
    empty_files: list[str] = []
    if not audio_path:
        yield "ابتدا یک فایل صوتی انتخاب یا ضبط کنید.", "", "", "", "", empty_files, {}
        return
    source = Path(audio_path).resolve()
    if not source.is_file():
        yield "فایل صوتی پیدا نشد.", "", "", "", "", empty_files, {}
        return

    run_id = "ui-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = ROOT / "outputs" / run_id
    ui_log = ROOT / "outputs" / f"{run_id}-ui.log"
    threads = os.cpu_count() or 4
    profile = installation_profile()
    try:
        device, compute_type, backend_label = execution_backend()
    except Exception as exc:
        yield f"❌ خطای آماده‌سازی شتاب‌دهنده: {exc}", "", "", "", "", empty_files, {"run_id": run_id}
        return
    wall_started = time.perf_counter()
    pipeline_command = [
        str(PYTHON), str(ROOT / "src" / "pipeline.py"),
        "--audio", str(source), "--root", str(ROOT),
        "--device", device, "--compute-type", compute_type, "--threads", str(threads),
        "--run-id", run_id, "--adaptive-turbo", "--profile", profile,
    ]
    try:
        yield f"شروع پردازش محلی با {backend_label}…", "", "", "", "", empty_files, {"run_id": run_id}
        for status in run_command(pipeline_command, ui_log, run_dir):
            yield status, "", "", "", "", empty_files, {"run_id": run_id, "backend": backend_label}

        if profile == "lite":
            elapsed = time.perf_counter() - wall_started
            run_summary = json.loads((run_dir / "run-summary.json").read_text(encoding="utf-8"))
            audio_duration = float(
                run_summary.get("raw_metadata", {}).get("format", {}).get("duration") or 0.0
            )
            benchmark = {
                "run_id": run_id,
                "profile": "lite",
                "recorded_at_utc": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
                "audio_duration_seconds": audio_duration,
                "end_to_end_seconds": round(elapsed, 3),
                "backend": backend_label,
                "local_summary_available": False,
                "external_api_used": False,
            }
            (run_dir / "runtime-benchmark.json").write_text(
                json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            archive_base = run_dir / f"{run_id}-results"
            shutil.make_archive(str(archive_base), "zip", root_dir=run_dir / "final-delivery")
            lite_text, suggested, review, downloads, details = collect_result(run_dir)
            details.update({"backend": backend_label, "profile": "lite"})
            yield (
                f"✅ پردازش Lite در {elapsed:.1f} ثانیه کامل شد؛ متن Turbo آماده است. "
                "این پروفایل خلاصه‌ساز و بازبینی معنایی Full را نصب نمی‌کند."
            ), lite_text, "خلاصه در پروفایل Lite موجود نیست.", suggested, review, downloads, details
            return

        consensus_command = [
            str(PYTHON), str(ROOT / "src" / "consensus_v9_medical_drugs.py"),
            "--run-dir", str(run_dir), "--medical-index", str(MEDICAL_INDEX),
            "--corpus-index", str(CORPUS_INDEX), "--encoder-dir", str(ENCODER_DIR),
        ]
        for _ in run_command(consensus_command, ui_log, run_dir):
            yield (
                "مراحل ۹ تا ۱۱ از ۱۴: قفل Turbo، MiniLM و بانک گسترش‌یافتهٔ داروهای فارسی…"
            ), "", "", "", "", empty_files, {"run_id": run_id, "backend": backend_label}

        v10_command = [
            "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
            str(ROOT / "run_local_qwen_v10.ps1"), "-RunDir", str(run_dir),
        ]
        for _ in run_command(v10_command, ui_log, run_dir):
            yield (
                "مراحل ۱۲ تا ۱۳ از ۱۴: Qwen محلی فقط بین نامزدهای مجاز واژه و عبارت انتخاب می‌کند…"
            ), "", "", "", "", empty_files, {"run_id": run_id, "backend": backend_label}

        google_payload_path = run_dir / GOOGLE_RECOGNITION_RELATIVE / "google-recognition.json"
        if GOOGLE_FALLBACK_ENABLED:
            google_command = [
                str(PYTHON), str(ROOT / "src" / "google_speech_fallback.py"),
                "--run-dir", str(run_dir), "--ffmpeg", str(FFMPEG_DIR / "ffmpeg.exe"),
                "--language", "fa-IR", "--timeout", "12", "--chunk-seconds", "45",
            ]
            for _ in run_command(google_command, ui_log, run_dir):
                yield (
                    "بررسی کیفیت متن محلی؛ fallback آنلاینِ فعال‌شده در صورت نیاز امتحان می‌شود…"
                ), "", "", "", "", empty_files, {"run_id": run_id, "backend": backend_label}
            google_payload = json.loads(google_payload_path.read_text(encoding="utf-8"))
        else:
            google_payload = {
                "provider": "Google Speech Recognition",
                "requested": False,
                "selected": False,
                "disabled_by_default": True,
                "external_audio_sent": False,
            }
            google_payload_path.parent.mkdir(parents=True, exist_ok=True)
            google_payload_path.write_text(
                json.dumps(google_payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        v11_command = [
            "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
            str(ROOT / "run_local_qwen_summary_v11.ps1"), "-RunDir", str(run_dir),
        ]
        if google_payload.get("selected"):
            v11_command.extend([
                "-SourceTranscript",
                str(run_dir / GOOGLE_RECOGNITION_RELATIVE / "google-recognition.txt"),
            ])
        for _ in run_command(v11_command, ui_log, run_dir):
            yield (
                "مرحله ۱۴ از ۱۴: خلاصهٔ محلی ساخته و با متن، داروها، دوزها و نفی تطبیق داده می‌شود…"
            ), "", "", "", "", empty_files, {"run_id": run_id, "backend": backend_label}

        run_summary = json.loads((run_dir / "run-summary.json").read_text(encoding="utf-8"))
        audio_duration = float(run_summary.get("raw_metadata", {}).get("format", {}).get("duration") or 0.0)
        targeted_status = "v9-adaptive-turbo-medical-drug-review-created"

        elapsed = time.perf_counter() - wall_started
        safe_summary_path = run_dir / V10_RELATIVE / "summary-v10.json"
        safe_summary = json.loads(safe_summary_path.read_text(encoding="utf-8"))
        safe_summary.update({"backend": backend_label, "end_to_end_seconds": round(elapsed, 3),
                             "rtx_60_second_target_met": (elapsed <= 60.0 if device == "cuda" else None),
                             "targeted_review_status": targeted_status,
                             "audio_duration_seconds": audio_duration})
        safe_summary_path.write_text(json.dumps(safe_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        v11_summary_path = run_dir / V11_RELATIVE / "summary-v11.json"
        v11_summary = json.loads(v11_summary_path.read_text(encoding="utf-8"))
        benchmark = {
            "run_id": run_id,
            "recorded_at_utc": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "audio_duration_seconds": audio_duration,
            "end_to_end_seconds": round(elapsed, 3),
            "under_60_seconds": elapsed < 60.0,
            "under_60_seconds_is_rtx_result": device == "cuda",
            "backend": backend_label,
            "qwen_runtime_seconds": safe_summary.get("runtime_seconds"),
            "qwen_model_latency_seconds": (safe_summary.get("model") or {}).get("latency_seconds"),
            "qwen_call_count": (safe_summary.get("model") or {}).get("call_count"),
            "summary_runtime_seconds": v11_summary.get("runtime_seconds"),
            "summary_model_latency_seconds": (v11_summary.get("model") or {}).get("latency_seconds"),
            "summary_accepted": v11_summary.get("accepted"),
            "external_api_used": bool(
                safe_summary.get("external_api_used_at_runtime")
                or google_payload.get("external_audio_sent")),
            "google_recognition_requested": google_payload.get("requested"),
            "google_recognition_selected": google_payload.get("selected"),
            "google_recognition_runtime_seconds": google_payload.get("runtime_seconds"),
            "free_text_generation_enters_output": safe_summary.get("free_text_generation_enters_output"),
            "generated_summary_enters_transcript": v11_summary.get("generated_summary_enters_transcript"),
        }
        (run_dir / "runtime-benchmark.json").write_text(
            json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
        archive_base = run_dir / f"{run_id}-results"
        shutil.make_archive(str(archive_base), "zip", root_dir=run_dir / "final-delivery")
        safe_text, suggested, review, downloads, summary = collect_result(run_dir)
        generated_summary = collect_generated_summary(run_dir)
        summary["backend"] = backend_label
        semantic_summary = summary.get("local_qwen_v10") or {}
        score_note = (
            f"؛ V10 تعداد {int(semantic_summary.get('applied_slot_count') or 0)} تغییر واژه‌ای و "
            f"{int(semantic_summary.get('applied_region_count') or 0)} تغییر عبارتی ثبت کرد"
        )
        target_note = ("؛ هدف ۶۰ ثانیهٔ RTX رعایت شد"
                       if device == "cuda" and elapsed <= 60 else
                       "؛ هشدار: زمان RTX از هدف ۶۰ ثانیه بیشتر شد"
                       if device == "cuda" else "")
        summary_note = ("؛ خلاصهٔ محلی پذیرفته شد" if v11_summary.get("accepted")
                        else "؛ خلاصه به‌علت ابهام با پیام محافظ جایگزین شد")
        google_note = ("؛ متن Google Recognition به‌عنوان fallback انتخاب شد"
                       if google_payload.get("selected") else
                       "؛ Google Recognition لازم نشد"
                       if not google_payload.get("requested") else
                       "؛ Google Recognition نتیجهٔ قابل‌استفاده نداد و مسیر محلی حفظ شد")
        yield (f"✅ پردازش در {elapsed:.1f} ثانیه کامل شد{target_note}{score_note}{summary_note}؛ "
               f"{google_note}؛ متن، خلاصه و بازبینی آماده‌اند."), safe_text, generated_summary, suggested, review, downloads, summary
    except Exception as exc:
        yield f"❌ خطا: {exc}", "", "", "", "", ([str(ui_log)] if ui_log.exists() else []), {"run_id": run_id}


CSS = """
body, .gradio-container { direction: rtl; font-family: Tahoma, Segoe UI, sans-serif; }
.gradio-container { max-width: 1120px !important; margin: auto !important; }
#hero { background: linear-gradient(135deg,#102a43,#0b7285); color:white; padding:22px; border-radius:18px; }
#status { border-right: 5px solid #0b7285; padding: 10px; }
textarea { direction: rtl !important; text-align: right !important; line-height: 2 !important; }
"""


def build_demo() -> gr.Blocks:
    profile = installation_profile()
    if profile == "lite":
        intro = (
            "# رونویسی فارسی محلی — Lite\n"
            "فایل صوتی با FFmpeg نرمال می‌شود؛ Demucs موسیقی را جدا می‌کند و pyannote گویندهٔ غالب را نگه می‌دارد. "
            "Whisper Large V3 Turbo روی نسخهٔ خام و نویزگیری‌شده اجرا می‌شود و متن Turbo تحویل می‌گردد. "
            "این پروفایل برای دانلود و اجرای سبک‌تر است و Qwen، MiniLM، n-gram و خلاصه‌ساز Full را اجرا نمی‌کند."
        )
        run_label = "شروع پردازش Lite"
        transcript_label = "متن نهایی Lite (Whisper Large V3 Turbo)"
        summary_label = "خلاصه (فقط در پروفایل Full موجود است)"
    else:
        intro = (
            "# رونویسی فارسی محلی — Full\n"
            "Turbo خام و نویزگیری‌شده ابتدا متن پایه را می‌سازند؛ مدل‌های دیگر فقط بازه‌های "
            "نامطمئن را بررسی می‌کنند و MiniLM و Qwen محلی همان نقاط را با واژه‌نامه و "
            "پیکرهٔ فارسی رتبه‌بندی می‌کنند. Qwen فقط نامزدهای مجاز را برای رونویسی انتخاب "
            "می‌کند و سپس یک خلاصهٔ جدا می‌سازد. fallback آنلاین پیش‌فرض خاموش است و فقط "
            "با متغیر محیطی مستندشده فعال می‌شود. هیچ فایل یا خروجی‌ای به پیکره اضافه نمی‌شود."
        )
        run_label = "شروع پردازش کامل"
        transcript_label = "متن نهایی Full"
        summary_label = "خلاصهٔ خودکار Full (جدا از متن و دارای قفل عدم‌حدس)"
    with gr.Blocks(title="رونویسی محلی گفتار فارسی") as demo:
        gr.Markdown(intro, elem_id="hero", rtl=True)
        with gr.Row():
            with gr.Column(scale=1):
                audio = gr.File(type="filepath", file_count="single", file_types=None,
                                label="فایل دارای صدا با هر پسوند یا قالب")
                run_button = gr.Button(run_label, variant="primary", size="lg")
                gr.Markdown(
                    "پسوند فایل ملاک نیست؛ MP3، MPGA، WAV، M4A، FLAC، OGG، صوتِ داخل ویدئو و هر "
                    "قالبی که FFmpeg محلی بتواند باز کند، از روی محتوا تشخیص و خودکار به WAV تبدیل می‌شود. "
                    "پردازش روی این سیستم و به‌صورت صف یک‌تایی انجام می‌شود؛ صفحه را نبندید.", rtl=True)
            with gr.Column(scale=2):
                status = gr.Markdown("آماده دریافت فایل", elem_id="status", rtl=True)
                text_output = gr.Textbox(label=transcript_label,
                                         lines=10, max_lines=24, interactive=True, rtl=True,
                                         buttons=["copy"])
                summary_output = gr.Textbox(
                    label=summary_label,
                    lines=5, max_lines=12, interactive=True, rtl=True, buttons=["copy"])
                with gr.Row():
                    name_input = gr.Textbox(
                        label="نام صحیح بعد از خانم/آقا",
                        placeholder="نام را وارد کنید؛ برای چند نام، هر کدام را در یک خط بنویسید",
                        lines=2, max_lines=5, rtl=True,
                    )
                    apply_name_button = gr.Button("جای‌گذاری نام", variant="secondary")
                gr.Markdown(
                    "اگر پس از «خانم/خانوم/آقا/آقای» نام شخص تشخیص داده شود، در متن به‌صورت "
                    "`________` خالی می‌ماند. مقدار این کادر عیناً جای‌گذاری می‌شود و وارد لغت‌نامه، "
                    "n-gram یا یادگیری سیستم نمی‌شود.", rtl=True)
                suggested_output = gr.Textbox(label="متن Turbo پیش از ادغام (برای مقایسه)",
                                               lines=7, max_lines=20, interactive=True, rtl=True,
                                               buttons=["copy"])
                review_output = gr.Markdown(
                    "موارد نام دارو، عدد/دوز و نفی در اینجا با کلیپ صوتی نمایش داده می‌شوند.", rtl=True)
        with gr.Row():
            downloads = gr.File(label="دانلود متن، امتیازها، گزارش، صوت پالایش‌شده و ZIP",
                                file_count="multiple", interactive=False)
            details = gr.JSON(label="خلاصه اجرا")
        run_button.click(process_audio, inputs=audio,
                         outputs=[status, text_output, summary_output, suggested_output, review_output,
                                  downloads, details],
                         concurrency_limit=1, scroll_to_output=True)
        apply_name_button.click(
            apply_name_values, inputs=[name_input, details],
            outputs=[status, text_output, downloads], scroll_to_output=True,
        )
        with gr.Accordion("تاریخچهٔ خروجی‌ها", open=False):
            choices = history_choices()
            with gr.Row():
                history = gr.Dropdown(choices=choices,
                                      value=(choices[0][1] if choices else None),
                                      label="اجرای ذخیره‌شده")
                refresh_button = gr.Button("تازه‌سازی فهرست")
                load_button = gr.Button("نمایش این خروجی", variant="secondary")
            refresh_button.click(refresh_history, outputs=history)
            load_button.click(load_history, inputs=history,
                              outputs=[status, text_output, summary_output, suggested_output, review_output,
                                       downloads, details], scroll_to_output=True)
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    demo = build_demo()
    if args.smoke_test:
        print(json.dumps({"status": "ok", "gradio": gr.__version__, "root": str(ROOT)}))
        return
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=args.port, share=False,
        allowed_paths=[str(ROOT / "outputs")], show_error=True, css=CSS,
    )


if __name__ == "__main__":
    main()
