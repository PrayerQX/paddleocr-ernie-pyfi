from paddle_pyfi.question_router import parse_options, route_profile


def test_route_profile_from_capability() -> None:
    profile = route_profile(
        capability="Calculation_analysis",
        question="What is the percentage point change?",
        options={},
    )
    assert profile.name == "calculation_formula"


def test_route_profile_prefers_visual_for_color_questions() -> None:
    profile = route_profile(
        capability="Data_extraction",
        question="Which color represents the no debt category?",
        options={},
    )
    assert profile.name == "perception_visual"


def test_parse_options_dict_string() -> None:
    options = parse_options("{'A': 'Micro', 'B': 'Small'}")
    assert options == {"A": "Micro", "B": "Small"}
