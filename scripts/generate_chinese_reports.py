from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(r"D:\OneDrive\Desktop\PaddleOCR-VL\output-pyfi301-ernie45-text-full-20260422")


CAP_MAP = {
    "Calculation_analysis": "\u8ba1\u7b97\u5206\u6790",
    "Data_extraction": "\u6570\u636e\u62bd\u53d6",
    "Decision_support": "\u51b3\u7b56\u652f\u6301",
    "Logical_reasoning": "\u903b\u8f91\u63a8\u7406",
    "Pattern_recognition": "\u6a21\u5f0f\u8bc6\u522b",
    "Perception": "\u611f\u77e5\u7406\u89e3",
}

STATUS_MAP = {
    "completed": "\u5b8c\u6210",
    "failed": "\u5931\u8d25",
    "missing": "\u7f3a\u5931",
    "skipped_existing": "\u8df3\u8fc7\u5df2\u6709\u7ed3\u679c",
}


def load_json(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def write_text(name: str, content: str) -> None:
    (ROOT / name).write_text(content, encoding="utf-8")


def build_analysis_summary() -> str:
    run_summary = load_json("run_summary.json")
    score_report = load_json("score_report.json")
    detailed_report = load_json("detailed_report.json")

    lines: list[str] = []
    lines.append("# 301\u6761\u6837\u672c\u8fd0\u884c\u4e2d\u6587\u603b\u7ed3")
    lines.append("")
    lines.append("## \u603b\u4f53\u8bc4\u5206")
    lines.append("")
    lines.append(f"- \u603b\u6837\u672c\u6570\uff1a`{score_report['total_samples']}`")
    lines.append(f"- \u5df2\u5b8c\u6210\uff1a`{run_summary['completed']}`")
    lines.append(f"- \u5931\u8d25\uff1a`{run_summary['failed']}`")
    lines.append(f"- \u5df2\u8bc4\u5206\u6837\u672c\uff1a`{score_report['scored_samples']}`")
    lines.append(f"- \u6b63\u786e\u6837\u672c\uff1a`{score_report['correct_samples']}`")
    lines.append(f"- \u51c6\u786e\u7387\uff1a`{score_report['accuracy']}`")
    lines.append(f"- \u7f3a\u5931\u9884\u6d4b\uff1a`{score_report['missing_predictions']}`")
    lines.append("")
    lines.append("## \u8017\u65f6\u60c5\u51b5")
    lines.append("")
    lines.append(
        f"- \u8fd0\u884c\u533a\u95f4\uff1a`{run_summary['started_at']}` \u5230 `{run_summary['finished_at']}`"
    )
    lines.append(
        f"- \u5e73\u5747\u603b\u8017\u65f6\uff1a`{detailed_report['summary']['avg_total_elapsed_seconds']}s/\u6761`"
    )
    lines.append(
        f"- \u5e73\u5747 PaddleOCR \u8017\u65f6\uff1a`{detailed_report['summary']['avg_paddleocr_elapsed_seconds']}s/\u6761`"
    )
    lines.append(
        f"- \u5e73\u5747 ERNIE \u8017\u65f6\uff1a`{detailed_report['summary']['avg_ernie_elapsed_seconds']}s/\u6761`"
    )
    lines.append("")
    lines.append(
        "\u8fd9\u6b21\u8dd1\u6279\u7684\u8017\u65f6\u51e0\u4e4e\u5168\u7531 ERNIE \u51b3\u5b9a\uff0cPaddleOCR \u8017\u65f6\u5f88\u7a33\u5b9a\uff0c\u5927\u90e8\u5206\u8017\u65f6\u6ce2\u52a8\u90fd\u6765\u81ea ERNIE \u751f\u6210\u9636\u6bb5\u3002"
    )
    lines.append("")
    lines.append("## \u5404\u80fd\u529b\u9898\u578b\u51c6\u786e\u7387")
    lines.append("")
    lines.append("| \u9898\u578b | \u5df2\u8bc4\u5206 | \u6b63\u786e | \u51c6\u786e\u7387 |")
    lines.append("|---|---:|---:|---:|")
    for key, value in score_report["by_stratum"].items():
        lines.append(
            f"| {CAP_MAP.get(key, key)} | {value['scored']} | {value['correct']} | {value['accuracy']} |"
        )
    lines.append("")
    lines.append("## \u7ed3\u679c\u89e3\u8bfb")
    lines.append("")
    lines.append(
        "- `\u51b3\u7b56\u652f\u6301`\u548c`\u6a21\u5f0f\u8bc6\u522b`\u8868\u73b0\u6700\u597d\uff0c\u8bf4\u660e\u5f53\u524d OCR \u6587\u672c\u8bc1\u636e\u52a0\u91d1\u878d\u9886\u57df\u63d0\u793a\u8bcd\uff0c\u5bf9\u57fa\u4e8e\u8bc1\u636e\u7684\u7ed3\u8bba\u9898\u548c\u8d8b\u52bf\u9898\u66f4\u6709\u6548\u3002"
    )
    lines.append(
        "- `\u611f\u77e5\u7406\u89e3`\u548c`\u8ba1\u7b97\u5206\u6790`\u6700\u5f31\u3002\u8fd9\u4e0e\u8fd9\u6b21\u8fd0\u884c\u65b9\u5f0f\u76f4\u63a5\u76f8\u5173\uff1a`ernie-4.5-21b-a3b` \u8fd9\u8f6e\u53ea\u5403 OCR \u6587\u672c\uff0c\u4e0d\u76f4\u63a5\u5403\u56fe\u50cf\u50cf\u7d20\uff0c\u56e0\u6b64\u989c\u8272\u3001\u7a7a\u95f4\u5e03\u5c40\u3001\u7eaf\u89c6\u89c9\u7ec6\u8282\u9898\u76ee\u4f1a\u5403\u4e8f\uff1b\u8ba1\u7b97\u9898\u5219\u5bb9\u6613\u88ab OCR \u6807\u7b7e\u7f3a\u5931\u3001\u5750\u6807\u8f74\u4e0d\u5b8c\u6574\u6216\u56fe\u8868\u8fd8\u539f\u8bef\u5dee\u62d6\u7d2f\u3002"
    )
    lines.append(
        "- \u82e5\u8981\u63d0\u901f\uff0c\u4f18\u5148\u5e94\u8be5\u4f18\u5316 ERNIE \u4fa7\uff0c\u800c\u4e0d\u662f\u7ee7\u7eed\u6298\u817e PaddleOCR\u3002"
    )
    lines.append("")
    lines.append("## \u6700\u6162\u768410\u6761\u6837\u672c")
    lines.append("")
    lines.append("| sample | \u9898\u578b | \u603b\u8017\u65f6(s) | OCR(s) | ERNIE(s) | \u662f\u5426\u6b63\u786e | \u9898\u76ee |")
    lines.append("|---|---|---:|---:|---:|---|---|")
    slow_entries = sorted(
        [e for e in detailed_report["entries"] if e.get("status") == "completed"],
        key=lambda item: item.get("total_elapsed_seconds") or -1,
        reverse=True,
    )[:10]
    for entry in slow_entries:
        correct = "\u662f" if entry["score"].get("correct") else "\u5426"
        lines.append(
            f"| `{entry['sample_id']}` | {CAP_MAP.get(entry['stratum'], entry['stratum'])} | "
            f"{entry.get('total_elapsed_seconds')} | {entry['paddleocr'].get('elapsed_seconds')} | "
            f"{entry['ernie'].get('elapsed_seconds')} | {correct} | {entry.get('question')} |"
        )
    lines.append("")
    lines.append("## \u8be6\u7ec6\u68c0\u6d4b\u6587\u6863")
    lines.append("")
    lines.append(
        "- \u82f1\u6587\u539f\u7248\u76d1\u63a7\u6458\u8981\uff1a[monitor_summary.md](/D:/OneDrive/Desktop/PaddleOCR-VL/output-pyfi301-ernie45-text-full-20260422/monitor_summary.md)"
    )
    lines.append(
        "- \u82f1\u6587\u539f\u7248\u603b\u7ed3\uff1a[analysis_summary.md](/D:/OneDrive/Desktop/PaddleOCR-VL/output-pyfi301-ernie45-text-full-20260422/analysis_summary.md)"
    )
    lines.append(
        "- \u7ed3\u6784\u5316\u9010\u6761\u660e\u7ec6\uff1a[detailed_report.json](/D:/OneDrive/Desktop/PaddleOCR-VL/output-pyfi301-ernie45-text-full-20260422/detailed_report.json)"
    )
    lines.append(
        "- \u7ed3\u6784\u5316\u8bc4\u5206\u7ed3\u679c\uff1a[score_report.json](/D:/OneDrive/Desktop/PaddleOCR-VL/output-pyfi301-ernie45-text-full-20260422/score_report.json)"
    )
    lines.append("")
    lines.append("## \u5355\u6761\u6837\u672c\u600e\u4e48\u770b")
    lines.append("")
    lines.append("1. \u5148\u6253\u5f00 `monitor_summary_zh.md`\uff0c\u770b\u8be5\u6837\u672c\u7684\u4eba\u8bfb\u7248\u6458\u8981\u3002")
    lines.append(
        "2. \u518d\u8fdb\u5165\u5bf9\u5e94 `sample_xxxx` \u76ee\u5f55\uff0c\u4f9d\u6b21\u67e5\u770b `sample_meta.json`\u3001`run_meta.json`\u3001`doc_0.md`\u3001`prompt.md`\u3001`analysis_finance.md`\u3001`analysis_finance.json`\u3001`analysis_finance.parsed.json`\u3002"
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def build_monitor_summary() -> str:
    detailed_report = load_json("detailed_report.json")

    lines: list[str] = []
    lines.append("# 301\u6761\u6837\u672c\u9010\u6761\u76d1\u63a7\u6458\u8981\uff08\u4e2d\u6587\u7248\uff09")
    lines.append("")
    lines.append(
        "\u8bf4\u660e\uff1a\u672c\u6587\u4ef6\u6309\u539f\u7248 `monitor_summary.md` \u7684\u7ed3\u6784\uff0c\u4e3a\u6bcf\u4e2a sample \u63d0\u4f9b PaddleOCR \u9636\u6bb5\u3001ERNIE \u9636\u6bb5\u3001\u5404\u9636\u6bb5\u8017\u65f6\u3001\u603b\u8017\u65f6\u3001\u5f97\u5206\u4e0e\u6b63\u786e\u6027\u3002"
    )
    lines.append("")
    lines.append("## \u805a\u5408\u6307\u6807")
    lines.append("")
    summary = detailed_report["summary"]
    status_counts = ", ".join(f"{STATUS_MAP.get(k, k)}={v}" for k, v in summary["status_counts"].items())
    lines.append(
        f"- \u8fd0\u884c\u8f93\u51fa\u76ee\u5f55\uff1a`{detailed_report['run_output_dir']}`"
    )
    lines.append(f"- \u603b\u6837\u672c\u6570\uff1a`{summary['total_samples']}`")
    lines.append(f"- \u72b6\u6001\u8ba1\u6570\uff1a`{status_counts}`")
    lines.append(f"- \u51c6\u786e\u7387\uff1a`{summary['accuracy']}`")
    lines.append(f"- \u5df2\u8bc4\u5206\u6837\u672c\uff1a`{summary['scored_samples']}`")
    lines.append(f"- \u6b63\u786e\u6837\u672c\uff1a`{summary['correct_samples']}`")
    lines.append(f"- \u7f3a\u5931\u9884\u6d4b\uff1a`{summary['missing_predictions']}`")
    lines.append(f"- \u5e73\u5747\u603b\u8017\u65f6\uff1a`{summary['avg_total_elapsed_seconds']}s`")
    lines.append(f"- \u5e73\u5747 PaddleOCR \u8017\u65f6\uff1a`{summary['avg_paddleocr_elapsed_seconds']}s`")
    lines.append(f"- \u5e73\u5747 ERNIE \u8017\u65f6\uff1a`{summary['avg_ernie_elapsed_seconds']}s`")
    lines.append("")
    lines.append("## \u9010\u6761\u6837\u672c\u8bf4\u660e")
    lines.append("")

    for entry in detailed_report["entries"]:
        sample_id = entry["sample_id"]
        ocr = entry["paddleocr"]
        ernie = entry["ernie"]
        score = entry["score"]
        artifacts = ocr.get("artifacts") or {}
        lines.append(f"### {sample_id}")
        lines.append(
            f"- PaddleOCR\uff1a\u5df2\u5c06\u6e90\u56fe\u50cf\u89e3\u6790\u4e3a\u5e26\u7248\u9762\u7ed3\u6784\u7684 markdown/\u6587\u672c\u4ea7\u7269\uff08doc_markdown_files={artifacts.get('doc_markdown_files')}, markdown_image_files={artifacts.get('markdown_image_files')}, output_image_files={artifacts.get('output_image_files')}\uff09\u3002"
        )
        lines.append(
            "- ERNIE\uff1a\u57fa\u4e8e OCR \u6587\u672c\u751f\u6210\u4e86\u7b54\u6848\u4ea7\u7269\u3002"
        )
        lines.append(
            f"- \u8017\u65f6\uff1aPaddleOCR `{ocr.get('elapsed_seconds')}s`\uff0cERNIE `{ernie.get('elapsed_seconds')}s`\uff0c\u603b\u8017\u65f6 `{entry.get('total_elapsed_seconds')}s`\u3002"
        )
        lines.append(
            f"- \u8bc4\u5206\uff1a\u72b6\u6001 `{STATUS_MAP.get(entry.get('status'), entry.get('status'))}`\uff0c"
            f"\u662f\u5426\u5df2\u8bc4\u5206 `{score.get('scored')}`\uff0c"
            f"\u662f\u5426\u6b63\u786e `{score.get('correct')}`\uff0c"
            f"\u9884\u6d4b `{score.get('predicted_choice')}`\uff0c"
            f"\u6807\u51c6\u7b54\u6848 `{score.get('gold_choice')}`\u3002"
        )
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    write_text("analysis_summary_zh.md", build_analysis_summary())
    write_text("monitor_summary_zh.md", build_monitor_summary())


if __name__ == "__main__":
    main()
