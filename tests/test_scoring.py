import json
from pathlib import Path

from paddle_pyfi.scoring import score_manifest_run


def test_score_manifest_run_matches_choice_and_answer(tmp_path: Path) -> None:
    manifest = {
        "dataset_root": str(tmp_path),
        "rows": [
            {
                "sample_id": "sample_0001",
                "stratum": "Logical_reasoning",
                "input": {"options": "{'A': 'Alpha', 'B': 'Beta'}"},
                "heldout": {"actions": "[{'answer': 'B'}]"},
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    sample_dir = tmp_path / "run" / "sample_0001" / "hash"
    sample_dir.mkdir(parents=True)
    (sample_dir / "analysis_finance.parsed.json").write_text(
        json.dumps({"answer": "Beta", "choice": "B"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = score_manifest_run(
        manifest_path=manifest_path,
        run_output_dir=tmp_path / "run",
        domain="finance",
    )
    payload = json.loads(report.read_text(encoding="utf-8"))

    assert payload["accuracy"] == 1.0
    assert payload["by_stratum"]["Logical_reasoning"]["correct"] == 1
