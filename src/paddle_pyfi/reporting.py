from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .manifest_runner import load_manifest
from .scoring import score_manifest_run


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _find_first(sample_dir: Path, pattern: str) -> Path | None:
    matches = sorted(sample_dir.glob(pattern))
    return matches[0] if matches else None


def _load_run_results(run_root: Path) -> dict[str, dict[str, Any]]:
    results_path = run_root / "run_results.jsonl"
    by_sample: dict[str, dict[str, Any]] = {}
    if not results_path.exists():
        return by_sample
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        sample_id = record.get("sample_id")
        if sample_id:
            by_sample[sample_id] = record
    return by_sample


def _count_by_status(records: dict[str, dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records.values():
        status = str(record.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def build_manifest_report(
    *,
    manifest_path: str | Path,
    run_output_dir: str | Path,
    domain: str,
    output_path: str | Path | None = None,
) -> Path:
    manifest = load_manifest(manifest_path)
    run_root = Path(run_output_dir)
    score_path = run_root / "score_report.json"
    if not score_path.exists():
        score_path = score_manifest_run(
            manifest_path=manifest_path,
            run_output_dir=run_output_dir,
            domain=domain,
            output_path=score_path,
        )

    score_report = json.loads(score_path.read_text(encoding="utf-8"))
    score_by_sample = {detail.get("sample_id"): detail for detail in score_report.get("sample_details", [])}
    run_results = _load_run_results(run_root)
    run_summary = _load_json(run_root / "run_summary.json") or {}

    entries: list[dict[str, Any]] = []
    for index, sample in enumerate(manifest.get("rows", [])):
        sample_id = sample.get("sample_id", f"sample_{index + 1:04d}")
        sample_dir = run_root / sample_id
        run_meta_path = _find_first(sample_dir, "*/run_meta.json")
        analysis_path = _find_first(sample_dir, f"*/analysis_{domain}.json")
        parsed_path = _find_first(sample_dir, f"*/analysis_{domain}.parsed.json")
        answer_path = _find_first(sample_dir, f"*/analysis_{domain}.md")
        prompt_path = _find_first(sample_dir, "*/prompt.md")

        sample_meta = _load_json(sample_dir / "sample_meta.json") or {}
        run_meta = _load_json(run_meta_path) or {}
        analysis = _load_json(analysis_path) or {}
        score_detail = score_by_sample.get(sample_id, {})
        result = run_results.get(sample_id, {})

        paddleocr = run_meta.get("paddleocr", {})
        ernie = analysis.get("ernie") or run_meta.get("ernie", {})
        input_payload = sample.get("input", {})

        entries.append(
            {
                "sample_id": sample_id,
                "index": result.get("index", index),
                "stratum": sample.get("stratum", input_payload.get("capability")),
                "status": result.get("status", "missing"),
                "attempt": result.get("attempt"),
                "question": input_payload.get("question"),
                "resolved_image_path": sample_meta.get("resolved_image_path"),
                "total_elapsed_seconds": result.get("elapsed_seconds"),
                "paddleocr": {
                    "started_at": paddleocr.get("started_at"),
                    "finished_at": paddleocr.get("finished_at"),
                    "elapsed_seconds": paddleocr.get("elapsed_seconds"),
                    "ocr_preset": paddleocr.get("ocr_preset"),
                    "layout_options": paddleocr.get("layout_options"),
                    "artifacts": paddleocr.get("artifacts") or run_meta.get("artifacts"),
                },
                "ernie": {
                    "started_at": ernie.get("started_at"),
                    "finished_at": ernie.get("finished_at"),
                    "elapsed_seconds": ernie.get("elapsed_seconds"),
                    "model": ernie.get("model") or analysis.get("model"),
                    "web_search": ernie.get("web_search", analysis.get("web_search")),
                    "max_completion_tokens": ernie.get("max_completion_tokens", analysis.get("max_completion_tokens")),
                    "reasoning_chars": ernie.get("reasoning_chars"),
                    "content_chars": ernie.get("content_chars"),
                    "structured_json_present": ernie.get("structured_json_present", analysis.get("structured_json") is not None),
                },
                "score": {
                    "scored": score_detail.get("scored"),
                    "correct": score_detail.get("correct"),
                    "gold_choice": score_detail.get("gold_choice"),
                    "predicted_choice": score_detail.get("predicted_choice"),
                    "predicted_answer": score_detail.get("predicted_answer"),
                    "gold_option_text": score_detail.get("gold_option_text"),
                },
                "paths": {
                    "sample_dir": str(sample_dir),
                    "run_meta": str(run_meta_path) if run_meta_path else None,
                    "prompt": str(prompt_path) if prompt_path else None,
                    "analysis_markdown": str(answer_path) if answer_path else None,
                    "analysis_json": str(analysis_path) if analysis_path else None,
                    "parsed_json": str(parsed_path) if parsed_path else None,
                },
            }
        )

    completed = [entry for entry in entries if entry.get("status") == "completed"]
    paddle_times = [entry["paddleocr"]["elapsed_seconds"] for entry in completed if entry["paddleocr"].get("elapsed_seconds") is not None]
    ernie_times = [entry["ernie"]["elapsed_seconds"] for entry in completed if entry["ernie"].get("elapsed_seconds") is not None]
    total_times = [entry.get("total_elapsed_seconds") for entry in completed if entry.get("total_elapsed_seconds") is not None]

    report = {
        "manifest_path": str(manifest_path),
        "run_output_dir": str(run_output_dir),
        "domain": domain,
        "summary": {
            "total_samples": len(entries),
            "status_counts": _count_by_status(run_results),
            "accuracy": score_report.get("accuracy"),
            "scored_samples": score_report.get("scored_samples"),
            "correct_samples": score_report.get("correct_samples"),
            "missing_predictions": score_report.get("missing_predictions"),
            "avg_total_elapsed_seconds": round(sum(total_times) / len(total_times), 3) if total_times else None,
            "avg_paddleocr_elapsed_seconds": round(sum(paddle_times) / len(paddle_times), 3) if paddle_times else None,
            "avg_ernie_elapsed_seconds": round(sum(ernie_times) / len(ernie_times), 3) if ernie_times else None,
        },
        "run_summary": run_summary,
        "score_report_path": str(score_path),
        "entries": entries,
    }

    destination = Path(output_path) if output_path else run_root / "detailed_report.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination
