from paddle_pyfi.domains import load_domain
from paddle_pyfi.question_router import PROFILE_LIBRARY
from paddle_pyfi.prompts import build_analysis_prompt


def test_prompt_includes_chart_consistency_when_image_is_attached() -> None:
    prompt = build_analysis_prompt(
        domain=load_domain("finance"),
        profile=PROFILE_LIBRARY["pattern_visual_consistency"],
        ocr_markdown="| Date | Value |\n| --- | --- |\n| Jan | 10% |",
        question="Summarize the chart.",
        has_image_evidence=True,
    )

    assert "Chart consistency check" in prompt
    assert "original image evidence" in prompt
    assert "needs_human_review" in prompt
    assert "chart_consistency" in prompt
    assert "answer" in prompt
    assert "pattern_visual_consistency" in prompt


def test_prompt_warns_when_image_is_not_attached() -> None:
    prompt = build_analysis_prompt(
        domain=load_domain("finance"),
        profile=PROFILE_LIBRARY["perception_visual"],
        ocr_markdown="<table></table>",
        question="What color is the line?",
        has_image_evidence=False,
    )

    assert "Original image evidence is unavailable" in prompt
    assert "Do not answer color" in prompt
