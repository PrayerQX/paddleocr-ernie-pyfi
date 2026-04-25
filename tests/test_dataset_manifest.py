import csv
import json
from pathlib import Path

from paddle_pyfi.dataset import create_stratified_manifest_from_csv


def test_create_stratified_manifest_separates_heldout_columns(tmp_path: Path) -> None:
    image_dir = tmp_path / "images" / "g1"
    image_dir.mkdir(parents=True)
    for index in range(1, 5):
        (image_dir / f"{index:06d}.jpg").write_bytes(b"img")

    csv_path = tmp_path / "dataset.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "question_node_no",
                "options",
                "complexity",
                "fq_no",
                "capability",
                "victory_count",
                "image_background",
                "parent_node_no",
                "actions",
                "question",
                "image_path",
            ],
        )
        writer.writeheader()
        for index, capability in enumerate(["Perception", "Perception", "Reasoning", "Reasoning"], start=1):
            writer.writerow(
                {
                    "question_node_no": str(index),
                    "options": "{'A': 'x'}",
                    "complexity": "1",
                    "fq_no": "1",
                    "capability": capability,
                    "victory_count": "1",
                    "image_background": "background",
                    "parent_node_no": "0",
                    "actions": "[{'answer': 'A'}]",
                    "question": "question?",
                    "image_path": f"./images/g1/{index:06d}.jpg",
                }
            )

    output = tmp_path / "manifest.json"
    create_stratified_manifest_from_csv(csv_path, output, sample_size=2, seed=7)
    manifest = json.loads(output.read_text(encoding="utf-8"))

    assert manifest["sampling"]["allocation"] == {"Perception": 1, "Reasoning": 1}
    assert len(manifest["rows"]) == 2
    row = manifest["rows"][0]
    assert "question" in row["input"]
    assert "actions" not in row["input"]
    assert "actions" in row["heldout"]
