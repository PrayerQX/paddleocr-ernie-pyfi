from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from .config import Settings
from .pipeline import analyze_document


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_manifest_image(dataset_root: str | Path, image_path: str) -> Path:
    cleaned = image_path.replace("\\", "/").removeprefix("./")
    return Path(dataset_root) / cleaned


def build_context(sample: dict[str, Any]) -> str:
    safe_input = sample.get("input", {})
    fields = [
        ("capability", safe_input.get("capability")),
        ("complexity", safe_input.get("complexity")),
        ("fq_no", safe_input.get("fq_no")),
        ("options", safe_input.get("options")),
        ("image_background", safe_input.get("image_background")),
    ]
    lines = [
        "PyFi stratified evaluation sample.",
        "Use only the question, options, image background, OCR evidence, and image evidence.",
        "Do not use hidden labels, actions, victory counts, or any held-out answer fields.",
    ]
    for key, value in fields:
        if value:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _already_completed(sample_dir: Path, domain: str) -> Path | None:
    matches = list(sample_dir.glob(f"*/analysis_{domain}.json"))
    return matches[0] if matches else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_single_sample(
    *,
    sample: dict[str, Any],
    index: int,
    dataset_root: Path,
    output_root: Path,
    settings: Settings,
    domain: str,
    requested_profile: str,
    requested_ocr_preset: str,
    include_image: bool,
    web_search: bool,
    max_completion_tokens: int,
    layout_overrides: dict[str, Any] | None,
    retries: int,
    retry_delay_seconds: float,
) -> dict[str, Any]:
    safe_input = sample.get("input", {})
    sample_id = sample.get("sample_id", f"sample_{index + 1:04d}")
    stratum = sample.get("stratum", safe_input.get("capability", "UNKNOWN"))
    sample_dir = output_root / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    image_path = resolve_manifest_image(dataset_root, safe_input.get("image_path", ""))
    question = safe_input.get("question") or "Analyze this document."
    context = build_context(sample)
    sample_meta_path = sample_dir / "sample_meta.json"
    _write_json(
        sample_meta_path,
        {
            "sample": sample,
            "resolved_image_path": str(image_path),
            "runner_started_at": _utc_now(),
        },
    )

    attempts = retries + 1
    last_error: str | None = None
    for attempt in range(1, attempts + 1):
        t0 = time.monotonic()
        try:
            answer_path = analyze_document(
                input_path=image_path,
                output_root=sample_dir,
                settings=settings,
                domain_name=domain,
                question=question,
                context=context,
                capability=safe_input.get("capability"),
                raw_options=safe_input.get("options"),
                requested_profile=requested_profile,
                requested_ocr_preset=requested_ocr_preset,
                web_search=web_search,
                include_image=include_image,
                max_completion_tokens=max_completion_tokens,
                layout_overrides=layout_overrides,
            )
            return {
                "time": _utc_now(),
                "sample_id": sample_id,
                "index": index,
                "stratum": stratum,
                "status": "completed",
                "image_path": str(image_path),
                "question": question,
                "profile": requested_profile,
                "ocr_preset": requested_ocr_preset,
                "attempt": attempt,
                "answer_path": str(answer_path),
                "elapsed_seconds": round(time.monotonic() - t0, 3),
            }
        except Exception as exc:
            last_error = str(exc)
            if attempt < attempts:
                time.sleep(retry_delay_seconds)
                continue
            return {
                "time": _utc_now(),
                "sample_id": sample_id,
                "index": index,
                "stratum": stratum,
                "status": "failed",
                "image_path": str(image_path),
                "question": question,
                "profile": requested_profile,
                "ocr_preset": requested_ocr_preset,
                "attempt": attempt,
                "error": last_error,
                "elapsed_seconds": round(time.monotonic() - t0, 3),
            }

    raise RuntimeError("unreachable")


def run_manifest(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    settings: Settings,
    domain: str,
    layout_overrides: dict[str, Any] | None = None,
    requested_profile: str = "auto",
    requested_ocr_preset: str = "auto",
    limit: int | None = None,
    start: int = 0,
    resume: bool = True,
    include_image: bool = True,
    web_search: bool = False,
    max_completion_tokens: int = 8192,
    workers: int = 2,
    retries: int = 2,
    retry_delay_seconds: float = 5.0,
) -> Path:
    manifest = load_manifest(manifest_path)
    dataset_root = Path(manifest["dataset_root"])
    rows = manifest.get("rows", [])
    selected = rows[start:]
    if limit is not None:
        selected = selected[:limit]

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    results_path = output_root / "run_results.jsonl"
    summary_path = output_root / "run_summary.json"

    completed = 0
    failed = 0
    skipped = 0
    started_at = _utc_now()
    lock = Lock()
    pending: list[tuple[int, dict[str, Any]]] = []

    with results_path.open("a", encoding="utf-8") as results_file:
        for local_index, sample in enumerate(selected, start=start):
            safe_input = sample.get("input", {})
            sample_id = sample.get("sample_id", f"sample_{local_index + 1:04d}")
            sample_dir = output_root / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            existing = _already_completed(sample_dir, domain)
            if resume and existing:
                record = {
                    "time": _utc_now(),
                    "sample_id": sample_id,
                    "index": local_index,
                    "stratum": sample.get("stratum", safe_input.get("capability", "UNKNOWN")),
                    "status": "skipped_existing",
                    "answer_path": str(existing),
                }
                results_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                results_file.flush()
                skipped += 1
            else:
                pending.append((local_index, sample))

        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {
                executor.submit(
                    _run_single_sample,
                    sample=sample,
                    index=index,
                    dataset_root=dataset_root,
                    output_root=output_root,
                    settings=settings,
                    domain=domain,
                    requested_profile=requested_profile,
                    requested_ocr_preset=requested_ocr_preset,
                    include_image=include_image,
                    web_search=web_search,
                    max_completion_tokens=max_completion_tokens,
                    layout_overrides=layout_overrides,
                    retries=retries,
                    retry_delay_seconds=retry_delay_seconds,
                ): (index, sample)
                for index, sample in pending
            }

            for future in as_completed(futures):
                try:
                    record = future.result()
                except Exception as exc:  # pragma: no cover - executor wrapper
                    index, sample = futures[future]
                    safe_input = sample.get("input", {})
                    record = {
                        "time": _utc_now(),
                        "sample_id": sample.get("sample_id", f"sample_{index + 1:04d}"),
                        "index": index,
                        "stratum": sample.get("stratum", safe_input.get("capability", "UNKNOWN")),
                        "status": "failed",
                        "error": str(exc),
                    }

                with lock:
                    results_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    results_file.flush()
                    if record["status"] == "completed":
                        completed += 1
                    elif record["status"] == "failed":
                        failed += 1

    summary = {
        "manifest_path": str(manifest_path),
        "output_dir": str(output_root),
        "domain": domain,
        "profile": requested_profile,
        "ocr_preset": requested_ocr_preset,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "requested": len(selected),
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
        "web_search": web_search,
        "max_completion_tokens": max_completion_tokens,
        "workers": workers,
        "retries": retries,
        "retry_delay_seconds": retry_delay_seconds,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path
