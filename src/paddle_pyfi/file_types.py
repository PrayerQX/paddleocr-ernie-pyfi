from __future__ import annotations

from pathlib import Path


PDF_FILE_TYPE = 0
IMAGE_FILE_TYPE = 1

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def infer_paddle_file_type(path: str | Path) -> int:
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return PDF_FILE_TYPE
    if suffix in IMAGE_EXTENSIONS:
        return IMAGE_FILE_TYPE
    supported = ", ".join(sorted([".pdf", *IMAGE_EXTENSIONS]))
    raise ValueError(f"Unsupported file extension '{suffix}'. Supported extensions: {supported}")


def is_supported_document(path: str | Path) -> bool:
    try:
        infer_paddle_file_type(path)
    except ValueError:
        return False
    return True
