from collections.abc import Iterator

from paddle_pyfi.config import DEFAULT_ERNIE_MODEL
from paddle_pyfi.ernie_client import ErnieClient


class _FakeChunk:
    def __init__(self, reasoning: str | None = None, content: str | None = None) -> None:
        delta = type("Delta", (), {"reasoning_content": reasoning, "content": content})()
        choice = type("Choice", (), {"delta": delta})()
        self.choices = [choice]


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> Iterator[_FakeChunk]:
        self.calls.append(kwargs)
        return iter(
            [
                _FakeChunk(reasoning="step 1"),
                _FakeChunk(reasoning="step 2"),
                _FakeChunk(content="final "),
                _FakeChunk(content="answer"),
            ]
        )


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeOpenAI:
    def __init__(self, **_: object) -> None:
        self.chat = _FakeChat()


def test_default_ernie_model_targets_thinking_preview() -> None:
    assert DEFAULT_ERNIE_MODEL == "ernie-4.5-21b-a3b"


def test_supports_image_input_detects_text_only_model(monkeypatch) -> None:
    monkeypatch.setattr("paddle_pyfi.ernie_client.OpenAI", _FakeOpenAI)

    text_client = ErnieClient(
        api_key="test-key",
        base_url="https://aistudio.baidu.com/llm/lmapi/v3",
        model="ernie-4.5-21b-a3b",
    )
    vl_client = ErnieClient(
        api_key="test-key",
        base_url="https://aistudio.baidu.com/llm/lmapi/v3",
        model="ernie-4.5-turbo-vl",
    )

    assert text_client.supports_image_input() is False
    assert vl_client.supports_image_input() is True


def test_complete_uses_openai_compatible_baidu_payload(monkeypatch) -> None:
    monkeypatch.setattr("paddle_pyfi.ernie_client.OpenAI", _FakeOpenAI)

    client = ErnieClient(
        api_key="test-key",
        base_url="https://aistudio.baidu.com/llm/lmapi/v3",
        model=DEFAULT_ERNIE_MODEL,
    )

    response = client.complete("test prompt", web_search=True, max_completion_tokens=65536, stream=True)

    call = client.client.chat.completions.calls[0]
    assert call["model"] == "ernie-4.5-21b-a3b"
    assert call["messages"] == [{"role": "user", "content": "test prompt"}]
    assert call["stream"] is True
    assert call["extra_body"] == {"penalty_score": 1}
    assert call["max_completion_tokens"] == 65536
    assert call["temperature"] == 0.8
    assert call["top_p"] == 0.8
    assert call["frequency_penalty"] == 0
    assert call["presence_penalty"] == 0
    assert response.reasoning_content == "step 1step 2"
    assert response.content == "final answer"
