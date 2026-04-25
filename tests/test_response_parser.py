from paddle_pyfi.response_parser import extract_structured_json


def test_extract_structured_json_from_fenced_block() -> None:
    text = "Final answer\n```json\n{\"answer\": \"A\", \"confidence\": \"high\"}\n```"
    parsed = extract_structured_json(text)
    assert parsed == {"answer": "A", "confidence": "high"}
