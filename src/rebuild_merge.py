from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline import merge, norm


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild only deterministic merge artifacts from six saved hypotheses.")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    hypotheses = {}
    for path in sorted((run_dir / "hypotheses").glob("*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        hypotheses[f"{payload['model']}__{payload['source']}"] = payload
    expected = {f"{m}__{s}" for m in ["medium", "large-v3-turbo", "large-v3"] for s in ["raw", "enhanced"]}
    if set(hypotheses) != expected:
        raise RuntimeError(f"Expected six complete hypotheses; found {sorted(hypotheses)}")
    final_text, decisions = merge(hypotheses)
    algo_dir = run_dir / "final-delivery" / "02-after-algorithm"
    final_json = {"base": "large-v3__enhanced", "method": "timestamp consensus without LLM",
                  "text": final_text, "decisions": decisions}
    (algo_dir / "final.txt").write_text(final_text + "\n", encoding="utf-8")
    (algo_dir / "final.json").write_text(json.dumps(final_json, ensure_ascii=False, indent=2), encoding="utf-8")
    (algo_dir / "decisions.json").write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path = run_dir / "run-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["final_text"] = final_text
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    system_info = json.loads((run_dir / "system-info.json").read_text(encoding="utf-8"))
    sensitive = [d for d in decisions if d["locked"] or norm(d["base"]) != norm(d["chosen"])]
    enhancement_lines = ([
        "- زنجیرهٔ enhanced: HTDemucs + pyannote Community-1",
        "- سیاست گوینده: نگه‌داشتن گویندهٔ غالب در صورت تشخیص چند گوینده",
    ] if system_info.get("audio_enhancement") else [
        f"- DeepFilterNet: {system_info.get('deepfilternet', 'legacy')}",
        f"- فرمان دقیق DeepFilterNet: `{system_info.get('deepfilternet_command', '')}`",
    ])
    report = [
        "# گزارش اجرای محلی", "",
        f"- شناسه اجرا: `{summary['run_id']}`",
        f"- دستگاه: `{summary['device']}` / `{summary['compute_type']}` / {system_info['used_cpu_threads']} CPU threads",
        *enhancement_lines,
        "", "## مشخصات صدای خام", "", "```json",
        json.dumps(summary["raw_metadata"], ensure_ascii=False, indent=2), "```",
        "", "## مشخصات صدای نویزگیری‌شده", "", "```json",
        json.dumps(summary["enhanced_metadata"], ensure_ascii=False, indent=2), "```",
        "", "## زمان‌ها", "", "| فرضیه | بارگذاری (ثانیه) | تبدیل (ثانیه) |", "|---|---:|---:|",
    ]
    report += [f"| {r['stage']} | {r['load_seconds']:.2f} | {r['processing_seconds']:.2f} |" for r in summary["timings"]]
    report += ["", "## شش فرضیه", ""]
    for key, text in summary["hypotheses"].items():
        report += [f"### {key}", "", text, ""]
    report += ["## متن پایه large-v3 (نسخه نویزگیری‌شده)", "", hypotheses["large-v3__enhanced"]["text"], "",
               "## متن نهایی الگوریتمی بدون LLM", "", final_text, "",
               "## اختلاف‌های حساس نام/عدد/دوز/دارو/اصطلاح پزشکی", "", "```json",
               json.dumps(sensitive, ensure_ascii=False, indent=2), "```", "",
               "> هشدار: نویزگیری نمی‌تواند صدایی را که ثبت نشده، به‌شدت clipping شده، یا زیر صدای گویندهٔ دیگری پوشیده شده بازیابی کند.", "",
               "> برای هرگونه بررسی پزشکی، صدای خام مرجع اصلی و نهایی است؛ این رونویسی صحت پزشکی را تضمین نمی‌کند."]
    (run_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(final_text)


if __name__ == "__main__":
    main()
