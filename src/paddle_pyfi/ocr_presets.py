from __future__ import annotations

from dataclasses import replace
from typing import Any

from .paddleocr_client import LayoutOptions


OCR_PRESET_BY_PROFILE = {
    "perception_visual": "light",
    "data_extraction_precise": "medium",
    "calculation_formula": "heavy",
    "pattern_visual_consistency": "heavy",
    "logical_options": "medium",
    "decision_support_evidence": "medium",
}


PRESET_OVERRIDES: dict[str, dict[str, Any]] = {
    "light": {
        "use_layout_detection": True,
        "use_chart_recognition": False,
        "use_seal_recognition": False,
        "use_ocr_for_image_block": True,
        "merge_tables": False,
        "relevel_titles": False,
        "prompt_label": "ocr",
        "restructure_pages": False,
    },
    "medium": {
        "use_layout_detection": True,
        "use_chart_recognition": True,
        "use_seal_recognition": False,
        "use_ocr_for_image_block": True,
        "merge_tables": False,
        "relevel_titles": False,
        "prompt_label": "chart",
        "restructure_pages": False,
    },
    "heavy": {
        "use_layout_detection": True,
        "use_chart_recognition": True,
        "use_seal_recognition": True,
        "use_ocr_for_image_block": True,
        "merge_tables": True,
        "relevel_titles": True,
        "prompt_label": "chart",
        "restructure_pages": True,
    },
    "baidu_sample": {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_layout_detection": True,
        "use_chart_recognition": True,
        "use_seal_recognition": True,
        "use_ocr_for_image_block": True,
        "merge_tables": True,
        "relevel_titles": True,
        "prompt_label": "ocr",
        "repetition_penalty": 1,
        "temperature": 0,
        "top_p": 1,
        "min_pixels": 147384,
        "max_pixels": 2822400,
        "layout_nms": True,
        "restructure_pages": True,
    },
}


def resolve_ocr_preset(profile_name: str | None, requested_preset: str = "auto", default_preset: str = "medium") -> str:
    if requested_preset != "auto":
        if requested_preset not in PRESET_OVERRIDES:
            known = ", ".join(sorted(["auto", *PRESET_OVERRIDES]))
            raise ValueError(f"Unknown OCR preset '{requested_preset}'. Available presets: {known}")
        return requested_preset
    if profile_name:
        return OCR_PRESET_BY_PROFILE.get(profile_name, default_preset)
    return default_preset


def build_layout_options(
    *,
    profile_name: str | None,
    requested_preset: str = "auto",
    explicit_overrides: dict[str, Any] | None = None,
    default_preset: str = "medium",
) -> tuple[str, LayoutOptions]:
    preset = resolve_ocr_preset(profile_name, requested_preset=requested_preset, default_preset=default_preset)
    options = replace(LayoutOptions(), **PRESET_OVERRIDES[preset])
    if explicit_overrides:
        options = replace(options, **explicit_overrides)
    return preset, options
