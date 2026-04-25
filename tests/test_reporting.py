import json
from pathlib import Path

from paddle_pyfi.reporting import build_manifest_report


def test_build_manifest_report_merges_timing_and_score(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    run_root = tmp_path / "run"
    sample_dir = run_root / "sample_0001" / "img-abc123"
    sample_dir.mkdir(parents=True)

    manifest = {
        "dataset_root": "data/pyfi-600k",
        "rows": [
            {
                "sample_id": "sample_0001",
                "stratum": "Logical_reasoning",
                "input": {
                    "question": "Who guarantees the loan?",
                    "capability": "Logical_reasoning",
                    "options": "{'A': 'Ministry of Finance', 'B': 'DCCE'}",
                },
                "heldout": {
                    "actions": "[{'answer': 'A'}]",
                },
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    (run_root / "run_results.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "sample_0001",
                "index": 0,
                "status": "completed",
                "attempt": 1,
                "elapsed_seconds": 12.5,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_root / "run_summary.json").write_text(
        json.dumps({"requested": 1, "completed": 1, "failed": 0, "skipped": 0}, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_root / "sample_0001" / "sample_meta.json").write_text(
        json.dumps({"resolved_image_path": "data/pyfi-600k/images/000001.jpg"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (sample_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "paddleocr": {
                    "elapsed_seconds": 4.2,
                    "ocr_preset": "auto",
                    "layout_options": {"use_chart_recognition": False},
                    "artifacts": {"doc_markdown_files": 1, "markdown_image_files": 0, "output_image_files": 1},
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (sample_dir / "analysis_finance.json").write_text(
        json.dumps(
            {
                "model": "ernie-4.5-21b-a3b",
                "web_search": False,
                "max_completion_tokens": 8192,
                "ernie": {
                    "elapsed_seconds": 7.8,
                    "structured_json_present": True,
                    "reasoning_chars": 123,
                    "content_chars": 456,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (sample_dir / "analysis_finance.parsed.json").write_text(
        json.dumps({"choice": "A", "answer": "Ministry of Finance"}, ensure_ascii=False),
        encoding="utf-8",
    )

    report_path = build_manifest_report(
        manifest_path=manifest_path,
        run_output_dir=run_root,
        domain="finance",
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["summary"]["accuracy"] == 1.0
    assert report["summary"]["avg_paddleocr_elapsed_seconds"] == 4.2
    assert report["summary"]["avg_ernie_elapsed_seconds"] == 7.8
    entry = report["entries"][0]
    assert entry["status"] == "completed"
    assert entry["score"]["correct"] is True
    assert entry["ernie"]["model"] == "ernie-4.5-21b-a3b"
