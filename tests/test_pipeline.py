import json
from pathlib import Path

from paddle_pyfi.config import Settings
from paddle_pyfi.ernie_client import ErnieResponse
from paddle_pyfi.pipeline import analyze_run_dir


class _FakeErnieClient:
    calls: list[dict[str, object]] = []

    def __init__(self, api_key: str, base_url: str, model: str, timeout: int = 300) -> None:
        self.model = model

    def supports_image_input(self) -> bool:
        return False

    def complete(
        self,
        prompt: str,
        *,
        image_paths: list[str | Path] | None = None,
        web_search: bool = True,
        max_completion_tokens: int = 65536,
        stream: bool = True,
    ) -> ErnieResponse:
        self.__class__.calls.append(
            {
                "prompt": prompt,
                "image_paths": [str(path) for path in image_paths or []],
                "web_search": web_search,
                "max_completion_tokens": max_completion_tokens,
                "stream": stream,
            }
        )
        return ErnieResponse(
            content='{"answer":"A","choice":"A","confidence":"high","document_summary":"ok","extracted_metrics":[],"calculations":[],"chart_consistency":{"status":"not_applicable","checked_items":[],"issues":[]},"evidence":[],"uncertainties":[],"research_conclusion":"ok"}',
            reasoning_content="",
            model=self.model,
        )


def test_analyze_run_dir_drops_image_for_text_only_model(tmp_path: Path, monkeypatch) -> None:
    (_FakeErnieClient.calls).clear()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "doc_0.md").write_text("test ocr", encoding="utf-8")
    (run_dir / "run_meta.json").write_text("{}", encoding="utf-8")
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"img")

    monkeypatch.setattr("paddle_pyfi.pipeline.ErnieClient", _FakeErnieClient)

    settings = Settings(
        paddleocr_api_url="https://example.com/ocr",
        paddleocr_api_token="ocr-token",
        ernie_api_key="ernie-token",
        ernie_base_url="https://aistudio.baidu.com/llm/lmapi/v3",
        ernie_model="ernie-4.5-21b-a3b",
        request_timeout=60,
    )

    answer_path = analyze_run_dir(
        run_dir=run_dir,
        domain_name="finance",
        settings=settings,
        question="Which option is correct?",
        image_paths=[image_path],
        web_search=False,
        max_completion_tokens=8000,
    )

    assert answer_path.exists()
    assert _FakeErnieClient.calls[0]["image_paths"] == []
    prompt_text = (run_dir / "prompt.md").read_text(encoding="utf-8")
    assert "No original image evidence is attached." in prompt_text
    analysis = json.loads((run_dir / "analysis_finance.json").read_text(encoding="utf-8"))
    assert analysis["requested_image_paths"] == [str(image_path)]
    assert analysis["image_paths"] == []
    assert analysis["ernie"]["image_input_supported"] is False
