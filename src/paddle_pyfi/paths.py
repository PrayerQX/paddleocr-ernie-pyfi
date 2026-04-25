from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slugify(value: str, fallback: str = "document") -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    return slug or fallback


def make_run_dir(input_path: str | Path, output_root: str | Path) -> Path:
    path = Path(input_path)
    digest = file_sha256(path)[:12]
    run_dir = Path(output_root) / f"{slugify(path.stem)}-{digest}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def safe_child_path(root: str | Path, relative_path: str, fallback_name: str = "asset") -> Path:
    root_path = Path(root).resolve()
    raw = str(relative_path).replace("\\", "/")
    pure = PurePosixPath(raw)
    parts = [
        slugify(part, fallback=fallback_name)
        for part in pure.parts
        if part not in ("", ".", "..", "/") and ":" not in part
    ]
    if not parts:
        parts = [fallback_name]
    candidate = (root_path / Path(*parts)).resolve()
    if root_path != candidate and root_path not in candidate.parents:
        raise ValueError(f"Refusing to write outside output directory: {relative_path}")
    return candidate
