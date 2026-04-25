from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests


PYFI_DATASET_ID = "AgenticFinLab/PyFi-600K"
HF_API_URL = f"https://huggingface.co/api/datasets/{PYFI_DATASET_ID}"
HF_RESOLVE_BASE = f"https://huggingface.co/datasets/{PYFI_DATASET_ID}/resolve/main"

CORE_FILES = [
    "README.md",
    "PyFi-600K-dataset.csv",
    "images.zip",
]

FULL_FILES = [
    "README.md",
    "PyFi-600K-dataset.csv",
    "PyFi-600K-dataset.json",
    "PyFi-600K-chain-dataset.json",
    "PyFi-600K-chain-CoT-dataset.json",
    "images.zip",
    "models/qwen_3B_sft_with_cot_5w/test.txt",
]

INPUT_COLUMNS = {
    "question_node_no",
    "options",
    "complexity",
    "fq_no",
    "capability",
    "image_background",
    "question",
    "image_path",
}

HELDOUT_COLUMNS = {
    "actions",
    "victory_count",
    "visit_count",
    "parent_node_no",
}


def fetch_pyfi_dataset_info(timeout: int = 60) -> dict[str, Any]:
    response = requests.get(HF_API_URL, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    files: list[dict[str, Any]] = []
    for sibling in data.get("siblings", []):
        name = sibling.get("rfilename")
        if not name:
            continue
        files.append({"name": name, "url": f"{HF_RESOLVE_BASE}/{name}"})
    return {
        "id": data.get("id", PYFI_DATASET_ID),
        "sha": data.get("sha"),
        "last_modified": data.get("lastModified"),
        "license": (data.get("cardData") or {}).get("license"),
        "tags": data.get("tags", []),
        "used_storage": data.get("usedStorage"),
        "files": files,
    }


def create_manifest_from_csv(csv_path: str | Path, output_path: str | Path, limit: int = 100) -> Path:
    source = Path(csv_path)
    destination = Path(output_path)
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            if index >= limit:
                break
            rows.append(dict(row))

    payload = {
        "source": str(source),
        "limit": limit,
        "rows": rows,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def _resolve_dataset_image_path(dataset_root: Path, image_path: str) -> Path:
    cleaned = image_path.replace("\\", "/").removeprefix("./")
    return dataset_root / cleaned


def _split_row(row: dict[str, str]) -> dict[str, Any]:
    safe_input = {key: row.get(key, "") for key in INPUT_COLUMNS if key in row}
    heldout = {key: row.get(key, "") for key in HELDOUT_COLUMNS if key in row}
    extra = {
        key: value
        for key, value in row.items()
        if key not in INPUT_COLUMNS and key not in HELDOUT_COLUMNS
    }
    return {"input": safe_input, "heldout": heldout, "extra": extra}


def _allocate_stratified(counts: dict[str, int], sample_size: int) -> dict[str, int]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    total = sum(counts.values())
    if total == 0:
        raise ValueError("No rows available for stratified sampling")
    if sample_size > total:
        raise ValueError(f"sample_size {sample_size} exceeds available rows {total}")

    raw = {key: sample_size * value / total for key, value in counts.items()}
    allocation = {key: min(counts[key], int(raw[key])) for key in counts}

    # Keep every non-empty stratum represented when possible.
    if sample_size >= len(counts):
        for key, count in counts.items():
            if count > 0 and allocation[key] == 0:
                allocation[key] = 1

    while sum(allocation.values()) > sample_size:
        candidates = [key for key, value in allocation.items() if value > 1]
        key = min(candidates, key=lambda item: raw[item] - int(raw[item]))
        allocation[key] -= 1

    remainders = sorted(
        counts,
        key=lambda key: (raw[key] - int(raw[key]), counts[key]),
        reverse=True,
    )
    index = 0
    while sum(allocation.values()) < sample_size:
        key = remainders[index % len(remainders)]
        if allocation[key] < counts[key]:
            allocation[key] += 1
        index += 1

    return dict(sorted(allocation.items()))


def create_stratified_manifest_from_csv(
    csv_path: str | Path,
    output_path: str | Path,
    *,
    sample_size: int = 301,
    stratify_column: str = "capability",
    seed: int = 20260419,
    require_existing_image: bool = True,
    exclude_strata: list[str] | None = None,
) -> Path:
    source = Path(csv_path)
    dataset_root = source.parent
    destination = Path(output_path)
    excluded = set(exclude_strata or [])

    counts: dict[str, int] = {}
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or stratify_column not in reader.fieldnames:
            raise ValueError(f"CSV does not contain stratify column '{stratify_column}'")
        for row in reader:
            image_path = row.get("image_path", "")
            if require_existing_image and image_path:
                if not _resolve_dataset_image_path(dataset_root, image_path).exists():
                    continue
            stratum = row.get(stratify_column, "") or "UNKNOWN"
            if stratum in excluded:
                continue
            counts[stratum] = counts.get(stratum, 0) + 1

    allocation = _allocate_stratified(counts, sample_size)
    rng = random.Random(seed)
    seen = {key: 0 for key in allocation}
    reservoirs: dict[str, list[dict[str, Any]]] = {key: [] for key in allocation}

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            image_path = row.get("image_path", "")
            if require_existing_image and image_path:
                if not _resolve_dataset_image_path(dataset_root, image_path).exists():
                    continue
            stratum = row.get(stratify_column, "") or "UNKNOWN"
            if stratum in excluded:
                continue
            target = allocation.get(stratum, 0)
            if target <= 0:
                continue

            seen[stratum] += 1
            item = {
                "row_number": row_number,
                "stratum": stratum,
                **_split_row(row),
            }
            bucket = reservoirs[stratum]
            if len(bucket) < target:
                bucket.append(item)
            else:
                replacement_index = rng.randint(0, seen[stratum] - 1)
                if replacement_index < target:
                    bucket[replacement_index] = item

    rows: list[dict[str, Any]] = []
    for stratum in sorted(reservoirs):
        rows.extend(reservoirs[stratum])
    rng.shuffle(rows)
    for index, row in enumerate(rows, start=1):
        row["sample_id"] = f"sample_{index:04d}"

    payload = {
        "source": str(source),
        "dataset_root": str(dataset_root),
        "sampling": {
            "method": "stratified_reservoir",
            "stratify_column": stratify_column,
            "sample_size": sample_size,
            "seed": seed,
            "require_existing_image": require_existing_image,
            "exclude_strata": sorted(excluded),
            "counts": dict(sorted(counts.items())),
            "allocation": allocation,
        },
        "rows": rows,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def resolve_file_list(selection: str, explicit_files: list[str] | None = None) -> list[str]:
    if explicit_files:
        return explicit_files
    if selection == "core":
        return CORE_FILES
    if selection == "full":
        return FULL_FILES
    raise ValueError("selection must be 'core' or 'full', or pass explicit files")


def download_file(
    file_name: str,
    output_dir: str | Path,
    timeout: int = 300,
    overwrite: bool = False,
    retries: int = 3,
) -> Path:
    output_root = Path(output_dir)
    destination = output_root / file_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return destination

    encoded_name = quote(file_name, safe="/")
    url = f"{HF_RESOLVE_BASE}/{encoded_name}"
    temp_path = destination.with_suffix(destination.suffix + ".part")

    if overwrite and temp_path.exists():
        temp_path.unlink()

    last_error: Exception | None = None
    for _attempt in range(retries):
        resume_from = temp_path.stat().st_size if temp_path.exists() else 0
        headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
        try:
            with requests.get(url, stream=True, timeout=timeout, headers=headers) as response:
                response.raise_for_status()
                if resume_from and response.status_code != 206:
                    temp_path.unlink(missing_ok=True)
                    resume_from = 0
                mode = "ab" if resume_from else "wb"
                with temp_path.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            temp_path.replace(destination)
            return destination
        except Exception as exc:  # pragma: no cover - network resilience branch
            last_error = exc

    if last_error:
        raise last_error
    return destination


def download_pyfi_files(
    output_dir: str | Path,
    *,
    selection: str = "core",
    explicit_files: list[str] | None = None,
    timeout: int = 300,
    overwrite: bool = False,
) -> list[Path]:
    files = resolve_file_list(selection, explicit_files)
    return [
        download_file(file_name, output_dir=output_dir, timeout=timeout, overwrite=overwrite)
        for file_name in files
    ]
