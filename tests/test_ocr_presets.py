from paddle_pyfi.ocr_presets import build_layout_options, resolve_ocr_preset


def test_resolve_auto_preset_from_profile() -> None:
    assert resolve_ocr_preset("perception_visual", requested_preset="auto") == "light"
    assert resolve_ocr_preset("calculation_formula", requested_preset="auto") == "heavy"


def test_build_layout_options_applies_preset_and_explicit_override() -> None:
    preset, options = build_layout_options(
        profile_name="perception_visual",
        requested_preset="auto",
        explicit_overrides={"use_chart_recognition": True, "prompt_label": "chart"},
    )

    assert preset == "light"
    assert options.use_chart_recognition is True
    assert options.prompt_label == "chart"
    assert options.merge_tables is False


def test_baidu_sample_matches_requested_layout_defaults() -> None:
    preset, options = build_layout_options(
        profile_name=None,
        requested_preset="baidu_sample",
    )

    assert preset == "baidu_sample"
    assert options.use_doc_orientation_classify is False
    assert options.use_doc_unwarping is False
    assert options.use_layout_detection is True
    assert options.use_chart_recognition is True
    assert options.use_seal_recognition is True
    assert options.use_ocr_for_image_block is True
    assert options.merge_tables is True
    assert options.relevel_titles is True
    assert options.prompt_label == "ocr"
    assert options.repetition_penalty == 1
    assert options.temperature == 0
    assert options.top_p == 1
    assert options.min_pixels == 147384
    assert options.max_pixels == 2822400
    assert options.layout_nms is True
    assert options.restructure_pages is True
