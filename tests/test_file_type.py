from pathlib import Path

import pytest

from paddle_pyfi.file_types import IMAGE_FILE_TYPE, PDF_FILE_TYPE, infer_paddle_file_type


def test_infer_pdf_file_type() -> None:
    assert infer_paddle_file_type(Path("report.PDF")) == PDF_FILE_TYPE


@pytest.mark.parametrize("name", ["a.png", "b.JPG", "c.jpeg", "d.webp", "e.tiff"])
def test_infer_image_file_type(name: str) -> None:
    assert infer_paddle_file_type(name) == IMAGE_FILE_TYPE


def test_unknown_file_type_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported file extension"):
        infer_paddle_file_type("notes.txt")
