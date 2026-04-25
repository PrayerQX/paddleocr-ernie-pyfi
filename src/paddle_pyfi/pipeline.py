from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Settings
from .domains import load_domain
from .ernie_client import ErnieClient
from .file_types import IMAGE_FILE_TYPE, infer_paddle_file_type
from .ocr_presets import build_layout_options
from .paddleocr_client import LayoutOptions, PaddleOCRClient, save_layout_result
from .paths import file_sha256, make_run_dir
from .question_router import AnalysisProfile, parse_options, route_profile
from .prompts import build_analysis_prompt, read_markdown_documents
from .response_parser import extract_structured_json


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file())


def _summarize_run_artifacts(run_dir: Path) -> dict[str, int]:
    return {
        "doc_markdown_files": len(list(run_dir.glob("doc_*.md"))),
        "markdown_image_files": _count_files(run_dir / "markdown_images"),
        "output_image_files": _count_files(run_dir / "output_images"),
    }


def parse_document(
    *,
    input_path: str | Path,
    output_root: str | Path,
    settings: Settings,
    options: LayoutOptions | None = None,
    ocr_preset: str | None = None,
) -> Path:
    if not settings.paddleocr_api_token:
        raise RuntimeError("PADDLEOCR_API_TOKEN is required for remote PaddleOCR parsing.")

    source = Path(input_path)
    run_dir = make_run_dir(source, output_root)
    started_at = _utc_now()
    t0 = time.monotonic()
    client = PaddleOCRClient(
        api_url=settings.paddleocr_api_url,
        token=settings.paddleocr_api_token,
        timeout=settings.request_timeout,
    )
    result = client.parse(source, options=options)
    save_layout_result(result, run_dir, session=client.session, timeout=settings.request_timeout)
    finished_at = _utc_now()

    meta = {
        "created_at": _utc_now(),
        "input_path": str(source),
        "input_sha256": file_sha256(source),
        "paddleocr_api_url": settings.paddleocr_api_url,
        "ocr_preset": ocr_preset,
        "layout_options": asdict(options or LayoutOptions()),
        "paddleocr": {
            "api_url": settings.paddleocr_api_url,
            "ocr_preset": ocr_preset,
            "layout_options": asdict(options or LayoutOptions()),
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_seconds": round(time.monotonic() - t0, 3),
            "artifacts": _summarize_run_artifacts(run_dir),
        },
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir


def analyze_run_dir(
    *,
    run_dir: str | Path,
    domain_name: str,
    settings: Settings,
    question: str | None = None,
    context: str | None = None,
    capability: str | None = None,
    raw_options: str | None = None,
    requested_profile: str = "auto",
    routed_profile: AnalysisProfile | None = None,
    ocr_preset: str | None = None,
    image_paths: list[str | Path] | None = None,
    web_search: bool = True,
    max_completion_tokens: int = 8192,
) -> Path:
    if not settings.ernie_api_key:
        raise RuntimeError("ERNIE_API_KEY is required for ERNIE analysis.")

    output_dir = Path(run_dir)
    domain = load_domain(domain_name)
    options = parse_options(raw_options)
    profile = routed_profile or route_profile(
        capability=capability,
        question=question,
        options=options,
        requested_profile=requested_profile,
    )
    ocr_markdown = read_markdown_documents(output_dir)
    if not ocr_markdown.strip():
        raise RuntimeError(f"No OCR markdown files found in {output_dir}. Run parse first.")

    ernie = ErnieClient(
        api_key=settings.ernie_api_key,
        base_url=settings.ernie_base_url,
        model=settings.ernie_model,
        timeout=settings.request_timeout,
    )
    requested_image_paths = [str(path) for path in image_paths or []]
    effective_image_paths = list(image_paths or []) if ernie.supports_image_input() else []

    prompt = build_analysis_prompt(
        domain=domain,
        profile=profile,
        ocr_markdown=ocr_markdown,
        question=question,
        context=context,
        has_image_evidence=bool(effective_image_paths),
        options=options,
    )
    (output_dir / "prompt.md").write_text(prompt, encoding="utf-8")

    ernie_started_at = _utc_now()
    t0 = time.monotonic()
    response = ernie.complete(
        prompt,
        image_paths=effective_image_paths,
        web_search=web_search,
        max_completion_tokens=max_completion_tokens,
    )
    ernie_finished_at = _utc_now()

    answer_path = output_dir / f"analysis_{domain.name}.md"
    answer_path.write_text(response.content, encoding="utf-8")
    parsed_json = extract_structured_json(response.content)
    ernie_trace = {
        "model": response.model,
        "web_search": web_search,
        "max_completion_tokens": max_completion_tokens,
        "image_input_supported": ernie.supports_image_input(),
        "requested_image_paths": requested_image_paths,
        "image_paths": [str(path) for path in effective_image_paths],
        "started_at": ernie_started_at,
        "finished_at": ernie_finished_at,
        "elapsed_seconds": round(time.monotonic() - t0, 3),
        "reasoning_chars": len(response.reasoning_content),
        "content_chars": len(response.content),
        "structured_json_present": parsed_json is not None,
    }
    response_payload: dict[str, Any] = {
        "created_at": _utc_now(),
        "domain": domain.name,
        "profile": profile.name,
        "ocr_preset": ocr_preset,
        "model": response.model,
        "web_search": web_search,
        "max_completion_tokens": max_completion_tokens,
        "requested_image_paths": requested_image_paths,
        "image_paths": [str(path) for path in effective_image_paths],
        "ernie": ernie_trace,
        "content": response.content,
        "reasoning_content": response.reasoning_content,
        "structured_json": parsed_json,
    }
    (output_dir / f"analysis_{domain.name}.json").write_text(
        json.dumps(response_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if parsed_json is not None:
        (output_dir / f"analysis_{domain.name}.parsed.json").write_text(
            json.dumps(parsed_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    run_meta_path = output_dir / "run_meta.json"
    if run_meta_path.exists():
        run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
        run_meta["artifacts"] = _summarize_run_artifacts(output_dir)
        run_meta["ernie"] = {
            **ernie_trace,
            "prompt_path": str(output_dir / "prompt.md"),
            "answer_path": str(answer_path),
            "analysis_json_path": str(output_dir / f"analysis_{domain.name}.json"),
            "parsed_json_path": str(output_dir / f"analysis_{domain.name}.parsed.json") if parsed_json is not None else None,
        }
        run_meta_path.write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return answer_path


def analyze_document(
    *,
    input_path: str | Path,
    output_root: str | Path,
    settings: Settings,
    domain_name: str,
    question: str | None = None,
    context: str | None = None,
    capability: str | None = None,
    raw_options: str | None = None,
    requested_profile: str = "auto",
    requested_ocr_preset: str = "auto",
    web_search: bool = True,
    include_image: bool = True,
    max_completion_tokens: int = 8192,
    layout_overrides: dict[str, Any] | None = None,
) -> Path:
    parsed_options = parse_options(raw_options)
    profile = route_profile(
        capability=capability,
        question=question,
        options=parsed_options,
        requested_profile=requested_profile,
    )
    ocr_preset, effective_options = build_layout_options(
        profile_name=profile.name,
        requested_preset=requested_ocr_preset,
        explicit_overrides=layout_overrides,
        default_preset="medium",
    )
    run_dir = parse_document(
        input_path=input_path,
        output_root=output_root,
        settings=settings,
        options=effective_options,
        ocr_preset=ocr_preset,
    )
    image_paths: list[Path] = []
    source = Path(input_path)
    if include_image and infer_paddle_file_type(source) == IMAGE_FILE_TYPE:
        image_paths.append(source)

    return analyze_run_dir(
        run_dir=run_dir,
        domain_name=domain_name,
        settings=settings,
        question=question,
        context=context,
        capability=capability,
        raw_options=raw_options,
        requested_profile=requested_profile,
        routed_profile=profile,
        ocr_preset=ocr_preset,
        image_paths=image_paths,
        web_search=web_search,
        max_completion_tokens=max_completion_tokens,
    )
