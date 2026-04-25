from __future__ import annotations

import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .manifest_runner import load_manifest
from .question_router import parse_options


def _normalize_text(value: str | None) -> str:
    text = (value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .,:;!?\"'")
    return text


def _parse_actions(raw_actions: str | None) -> list[dict[str, Any]]:
    if not raw_actions:
        return []
    try:
        data = ast.literal_eval(raw_actions)
    except (SyntaxError, ValueError):
        return []
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def _load_prediction(sample_dir: Path, domain: str) -> dict[str, Any] | None:
    matches = list(sample_dir.glob(f"*/analysis_{domain}.parsed.json"))
    if not matches:
        return None
    return json.loads(matches[0].read_text(encoding="utf-8"))


def _is_correct(sample: dict[str, Any], prediction: dict[str, Any] | None) -> tuple[bool | None, dict[str, Any]]:
    heldout = sample.get("heldout", {})
    safe_input = sample.get("input", {})
    actions = _parse_actions(heldout.get("actions"))
    if not actions:
        return None, {"reason": "missing_gold_actions"}

    gold = actions[0]
    gold_choice = _normalize_text(str(gold.get("answer", "")))
    options = parse_options(safe_input.get("options"))
    prediction = prediction or {}
    predicted_choice = _normalize_text(prediction.get("choice"))
    predicted_answer = _normalize_text(prediction.get("answer"))

    option_value = _normalize_text(options.get(gold_choice.upper()) if gold_choice else "")
    choice_match = bool(predicted_choice and predicted_choice == gold_choice)
    answer_match = bool(predicted_answer and option_value and predicted_answer == option_value)
    answer_letter_match = bool(predicted_answer and predicted_answer == gold_choice)
    correct = choice_match or answer_match or answer_letter_match
    return correct, {
        "gold_choice": gold_choice or None,
        "predicted_choice": predicted_choice or None,
        "predicted_answer": predicted_answer or None,
        "gold_option_text": option_value or None,
    }


def score_manifest_run(
    *,
    manifest_path: str | Path,
    run_output_dir: str | Path,
    domain: str,
    output_path: str | Path | None = None,
) -> Path:
    manifest = load_manifest(manifest_path)
    rows = manifest.get("rows", [])
    run_root = Path(run_output_dir)
    sample_details: list[dict[str, Any]] = []
    correct_counter = Counter()
    total_counter = Counter()
    missing_predictions = 0

    for sample in rows:
        sample_id = sample.get("sample_id")
        stratum = sample.get("stratum", "UNKNOWN")
        prediction = _load_prediction(run_root / sample_id, domain) if sample_id else None
        if prediction is None:
            missing_predictions += 1
        is_correct, detail = _is_correct(sample, prediction)
        sample_details.append(
            {
                "sample_id": sample_id,
                "stratum": stratum,
                "scored": is_correct is not None and prediction is not None,
                "correct": is_correct,
                **detail,
            }
        )
        if is_correct is not None and prediction is not None:
            total_counter[stratum] += 1
            if is_correct:
                correct_counter[stratum] += 1

    by_stratum: dict[str, Any] = {}
    total_correct = 0
    total_scored = 0
    for stratum in sorted(total_counter):
        scored = total_counter[stratum]
        correct = correct_counter[stratum]
        accuracy = correct / scored if scored else 0.0
        by_stratum[stratum] = {
            "scored": scored,
            "correct": correct,
            "accuracy": round(accuracy, 6),
        }
        total_correct += correct
        total_scored += scored

    payload = {
        "manifest_path": str(manifest_path),
        "run_output_dir": str(run_output_dir),
        "domain": domain,
        "total_samples": len(rows),
        "scored_samples": total_scored,
        "correct_samples": total_correct,
        "accuracy": round((total_correct / total_scored) if total_scored else 0.0, 6),
        "missing_predictions": missing_predictions,
        "by_stratum": by_stratum,
        "sample_details": sample_details,
    }

    destination = Path(output_path) if output_path else Path(run_output_dir) / "score_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination
