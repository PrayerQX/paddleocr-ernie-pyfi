from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from .file_types import infer_paddle_file_type
from .paths import safe_child_path


@dataclass(frozen=True)
class LayoutOptions:
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False
    use_layout_detection: bool = True
    use_chart_recognition: bool = False
    use_seal_recognition: bool = True
    use_ocr_for_image_block: bool = True
    merge_tables: bool = True
    relevel_titles: bool = True
    layout_shape_mode: str = "auto"
    prompt_label: str = "ocr"
    repetition_penalty: float = 1
    temperature: float = 0
    top_p: float = 1
    min_pixels: int = 147384
    max_pixels: int = 2822400
    layout_nms: bool = True
    restructure_pages: bool = True
    markdown_ignore_labels: tuple[str, ...] = (
        "header",
        "header_image",
        "footer",
        "footer_image",
        "number",
        "footnote",
        "aside_text",
    )

    def as_payload(self) -> dict[str, object]:
        return {
            "markdownIgnoreLabels": list(self.markdown_ignore_labels),
            "useDocOrientationClassify": self.use_doc_orientation_classify,
            "useDocUnwarping": self.use_doc_unwarping,
            "useLayoutDetection": self.use_layout_detection,
            "useChartRecognition": self.use_chart_recognition,
            "useSealRecognition": self.use_seal_recognition,
            "useOcrForImageBlock": self.use_ocr_for_image_block,
            "mergeTables": self.merge_tables,
            "relevelTitles": self.relevel_titles,
            "layoutShapeMode": self.layout_shape_mode,
            "promptLabel": self.prompt_label,
            "repetitionPenalty": self.repetition_penalty,
            "temperature": self.temperature,
            "topP": self.top_p,
            "minPixels": self.min_pixels,
            "maxPixels": self.max_pixels,
            "layoutNms": self.layout_nms,
            "restructurePages": self.restructure_pages,
        }


def build_layout_payload(
    file_path: str | Path,
    file_type: int | None = None,
    options: LayoutOptions | None = None,
) -> dict[str, Any]:
    path = Path(file_path)
    inferred_type = infer_paddle_file_type(path) if file_type is None else file_type
    with path.open("rb") as file:
        file_data = base64.b64encode(file.read()).decode("ascii")
    return {
        "file": file_data,
        "fileType": inferred_type,
        **(options or LayoutOptions()).as_payload(),
    }


class PaddleOCRClient:
    def __init__(
        self,
        api_url: str,
        token: str,
        timeout: int = 300,
        session: requests.Session | None = None,
    ) -> None:
        self.api_url = api_url
        self.token = token
        self.timeout = timeout
        self.session = session or requests.Session()

    def parse(self, file_path: str | Path, options: LayoutOptions | None = None) -> dict[str, Any]:
        payload = build_layout_payload(file_path, options=options)
        headers = {
            "Authorization": f"token {self.token}",
            "Content-Type": "application/json",
        }
        response = self.session.post(
            self.api_url,
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        if "result" not in body:
            raise ValueError("PaddleOCR response does not contain 'result'.")
        return body["result"]


def download_url(url: str, destination: Path, session: requests.Session | None = None, timeout: int = 300) -> None:
    http = session or requests.Session()
    response = http.get(url, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download {url}: HTTP {response.status_code}")
    content_type = response.headers.get("Content-Type", "")
    if content_type and not (
        content_type.startswith("image/")
        or content_type.startswith("application/octet-stream")
        or content_type.startswith("binary/")
    ):
        raise RuntimeError(f"Refusing unexpected content type for {url}: {content_type}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)


def save_layout_result(
    result: dict[str, Any],
    output_dir: str | Path,
    session: requests.Session | None = None,
    timeout: int = 300,
) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    raw_path = out / "ocr_result.json"
    raw_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    written.append(raw_path)

    layout_results = result.get("layoutParsingResults", [])
    for index, item in enumerate(layout_results):
        markdown = item.get("markdown", {})
        markdown_text = markdown.get("text", "")

        md_path = out / f"doc_{index}.md"
        md_path.write_text(markdown_text, encoding="utf-8")
        written.append(md_path)

        markdown_images_dir = out / "markdown_images"
        for image_path, image_url in markdown.get("images", {}).items():
            destination = safe_child_path(markdown_images_dir, image_path, fallback_name=f"image_{index}.jpg")
            download_url(image_url, destination, session=session, timeout=timeout)
            written.append(destination)

        output_images_dir = out / "output_images"
        for image_name, image_url in item.get("outputImages", {}).items():
            destination = safe_child_path(output_images_dir, f"{image_name}_{index}.jpg")
            download_url(image_url, destination, session=session, timeout=timeout)
            written.append(destination)

    return written
