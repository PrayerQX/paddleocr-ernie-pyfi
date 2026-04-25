from __future__ import annotations

import ast
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisProfile:
    name: str
    objective: str
    primary_evidence: str
    answer_style: str
    extra_rules: tuple[str, ...]


PROFILE_LIBRARY = {
    "perception_visual": AnalysisProfile(
        name="perception_visual",
        objective="Answer visual questions about color, position, legend, shape, or line identity.",
        primary_evidence="Prioritize original image evidence. Use OCR only for labels and supporting structure.",
        answer_style="Return a short direct answer. If the task is multiple-choice, return the best choice label.",
        extra_rules=(
            "Do not infer colors or line styles from OCR text unless the OCR explicitly contains them.",
            "For legend questions, identify the legend entry and the matching color or line style.",
        ),
    ),
    "data_extraction_precise": AnalysisProfile(
        name="data_extraction_precise",
        objective="Extract exact values, names, labels, dates, and categories.",
        primary_evidence="Prioritize OCR tables and explicit OCR text. Use the image to validate ambiguous values.",
        answer_style="Return the smallest answer needed for scoring, then include evidence.",
        extra_rules=(
            "Prefer exact values over descriptive summaries.",
            "If OCR and image disagree on a value, mark the answer uncertain and explain the conflict.",
        ),
    ),
    "calculation_formula": AnalysisProfile(
        name="calculation_formula",
        objective="Compute differences, growth rates, ratios, minima, maxima, or rankings from extracted values.",
        primary_evidence="Use OCR-extracted values when they are consistent with the image. Validate against visual evidence.",
        answer_style="Show the formula, substituted values, and the final answer concisely.",
        extra_rules=(
            "Do not do hidden arithmetic. Show the calculation.",
            "If required values are missing or inconsistent, do not force a result.",
        ),
    ),
    "pattern_visual_consistency": AnalysisProfile(
        name="pattern_visual_consistency",
        objective="Identify trends, peaks, turning points, anomalies, and chart patterns.",
        primary_evidence="Prioritize the original chart trend and use OCR chart-to-table only as reconstructed evidence.",
        answer_style="Keep the trend summary concise and explicitly report chart consistency.",
        extra_rules=(
            "Do not rely on OCR chart-to-table if the visible chart shows a different trend.",
            "Mark needs_human_review when OCR reconstruction conflicts with the visual chart.",
        ),
    ),
    "logical_options": AnalysisProfile(
        name="logical_options",
        objective="Answer reasoning questions, especially multiple-choice questions, by eliminating wrong options.",
        primary_evidence="Use OCR evidence, image evidence, and provided background/context together.",
        answer_style="Return the final choice first, then briefly justify why the other options are less supported.",
        extra_rules=(
            "If options are present, compare them explicitly.",
            "Do not introduce answer keys or hidden labels.",
        ),
    ),
    "decision_support_evidence": AnalysisProfile(
        name="decision_support_evidence",
        objective="Provide cautious research-oriented conclusions from chart, table, and context evidence.",
        primary_evidence="Use OCR evidence, original image evidence, and user context. Stay within evidence boundaries.",
        answer_style="Return a concise research conclusion with uncertainty and risk notes.",
        extra_rules=(
            "Do not give deterministic investment advice.",
            "Prefer conservative conclusions when evidence is incomplete or conflicting.",
        ),
    ),
}


CAPABILITY_TO_PROFILE = {
    "perception": "perception_visual",
    "data_extraction": "data_extraction_precise",
    "calculation_analysis": "calculation_formula",
    "pattern_recognition": "pattern_visual_consistency",
    "logical_reasoning": "logical_options",
    "decision_support": "decision_support_evidence",
    "task_support": "decision_support_evidence",
}


def normalize_capability(value: str | None) -> str:
    text = (value or "").strip().lower()
    text = re.sub(r"[\s/-]+", "_", text)
    return text


def parse_options(raw_options: str | None) -> dict[str, str]:
    if not raw_options:
        return {}
    try:
        data = ast.literal_eval(raw_options)
    except (SyntaxError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    parsed: dict[str, str] = {}
    for key, value in data.items():
        parsed[str(key)] = str(value)
    return parsed


def route_profile(
    *,
    capability: str | None,
    question: str | None,
    options: dict[str, str] | None = None,
    requested_profile: str = "auto",
) -> AnalysisProfile:
    if requested_profile != "auto":
        if requested_profile not in PROFILE_LIBRARY:
            known = ", ".join(sorted(["auto", *PROFILE_LIBRARY]))
            raise ValueError(f"Unknown profile '{requested_profile}'. Available profiles: {known}")
        return PROFILE_LIBRARY[requested_profile]

    question_text = (question or "").lower()
    options = options or {}
    normalized_capability = normalize_capability(capability)

    if any(keyword in question_text for keyword in ("which color", "what color", "legend", "line style", "bar color")):
        return PROFILE_LIBRARY["perception_visual"]
    if any(keyword in question_text for keyword in ("growth rate", "difference", "ratio", "increase", "decrease by", "percentage point")):
        return PROFILE_LIBRARY["calculation_formula"]
    if any(keyword in question_text for keyword in ("trend", "pattern", "peak", "decline", "fluctuation", "above the average")):
        return PROFILE_LIBRARY["pattern_visual_consistency"]
    if options:
        if normalized_capability == "data_extraction":
            return PROFILE_LIBRARY["data_extraction_precise"]
        return PROFILE_LIBRARY["logical_options"]

    profile_name = CAPABILITY_TO_PROFILE.get(normalized_capability)
    if profile_name:
        return PROFILE_LIBRARY[profile_name]
    return PROFILE_LIBRARY["data_extraction_precise"]
