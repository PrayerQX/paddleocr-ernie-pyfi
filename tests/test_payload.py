import base64
from pathlib import Path

from paddle_pyfi.paddleocr_client import LayoutOptions, build_layout_payload


def test_build_layout_payload(tmp_path: Path) -> None:
    document = tmp_path / "sample.pdf"
    document.write_bytes(b"abc")

    payload = build_layout_payload(
        document,
        options=LayoutOptions(
            use_doc_orientation_classify=True,
            use_doc_unwarping=False,
            use_chart_recognition=True,
        ),
    )

    assert payload["file"] == base64.b64encode(b"abc").decode("ascii")
    assert payload["fileType"] == 0
    assert payload["useDocOrientationClassify"] is True
    assert payload["useDocUnwarping"] is False
    assert payload["useChartRecognition"] is True
