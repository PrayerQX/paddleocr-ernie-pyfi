from __future__ import annotations

import json
from pathlib import Path

from .domains import DomainAdapter
from .question_router import AnalysisProfile


def read_markdown_documents(run_dir: str | Path) -> str:
    paths = sorted(Path(run_dir).glob("doc_*.md"))
    sections: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        sections.append(f"## {path.name}\n\n{text}")
    return "\n\n".join(sections)


def build_analysis_prompt(
    *,
    domain: DomainAdapter,
    profile: AnalysisProfile,
    ocr_markdown: str,
    question: str | None = None,
    context: str | None = None,
    has_image_evidence: bool = False,
    options: dict[str, str] | None = None,
) -> str:
    schema = json.dumps(domain.output_schema, ensure_ascii=False, indent=2)
    tasks = "\n".join(f"- {task}" for task in domain.tasks)
    forbidden = "\n".join(f"- {item}" for item in domain.forbidden)
    user_question = question or "Analyze this document from the OCR evidence and produce a structured, evidence-grounded answer."
    extra_context = context or "No extra context."
    options_block = (
        "Answer options:\n" + "\n".join(f"- {key}: {value}" for key, value in options.items())
        if options
        else "Answer options:\n- No explicit multiple-choice options were provided."
    )
    image_evidence_note = (
        "Original image evidence is attached to this message. Use it only as visual evidence and compare it against the OCR output."
        if has_image_evidence
        else "No original image evidence is attached. Do not infer visual-only details such as colors or legend swatches from OCR text alone."
    )
    chart_consistency_rule = (
        """
Chart consistency check:
- If the document contains a chart, compare OCR/chart-to-table evidence against the original image evidence.
- Treat PaddleOCR chart-to-table output as a model reconstruction, not as unquestionable ground truth.
- Check legend colors, line/bar/area colors, axis labels, visible trend direction, key values, and table values separately.
- If OCR table values conflict with the visible chart trend or legend, mark the result as "needs_human_review" and explain the conflict.
- For visual questions about color, shape, line style, or layout, prioritize original image evidence and cite OCR only as supporting structure.
- For numeric questions, use OCR table values only when they are consistent with the visual chart; otherwise report uncertainty.
"""
        if has_image_evidence
        else """
Chart consistency check:
- Original image evidence is unavailable, so visual consistency cannot be checked.
- If OCR contains a chart-to-table conversion, explicitly label it as reconstructed OCR evidence.
- Do not answer color, line-style, or legend-swatch questions unless the OCR text explicitly contains that mapping.
"""
    )

    return f"""You are an evidence-constrained document understanding agent.

Domain: {domain.name}

Analysis profile: {profile.name}
Profile objective: {profile.objective}
Primary evidence rule: {profile.primary_evidence}
Answer style: {profile.answer_style}
Profile-specific rules:
{chr(10).join(f"- {rule}" for rule in profile.extra_rules)}

Domain instructions:
{domain.system_instructions}

Allowed tasks:
{tasks}

Forbidden:
{forbidden}

Output schema reference:
```json
{schema}
```

User question:
{user_question}

{options_block}

Extra context:
{extra_context}

Image evidence status:
{image_evidence_note}

{chart_consistency_rule}

OCR evidence:
```markdown
{ocr_markdown}
```

Output format:

1. Final answer
2. Evidence summary
3. Calculations or verification process
4. Chart consistency check
5. Uncertainty and missing information
6. Structured JSON

Requirements:
- Every important conclusion must be grounded in OCR evidence, image evidence, or user-provided context.
- Missing information must be marked as missing. Do not fabricate data.
- If OCR evidence and image evidence conflict, say so directly and do not force a confident answer.
- Keep the prose concise. The structured JSON is the authoritative output.
- If this is a multiple-choice question, include a single best `choice` when supported.
- Include a `chart_consistency` object in the structured JSON with:
  - `status`: one of `consistent`, `inconsistent`, `uncertain`, `not_applicable`, or `needs_human_review`
  - `checked_items`: list of checked items such as legend colors, visual trend, table values, axes
  - `issues`: list of conflicts or limitations
- Include these top-level fields in the structured JSON:
  - `answer`: final short answer suitable for scoring
  - `choice`: choice label such as A/B/C/D when applicable, else null
  - `confidence`: one of `low`, `medium`, `high`
  - `document_summary`
  - `extracted_metrics`
  - `calculations`
  - `chart_consistency`
  - `evidence`
  - `uncertainties`
  - `research_conclusion`
- The structured JSON should match the domain schema as much as possible and may include the extra `chart_consistency` object.
"""
