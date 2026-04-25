from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI


IMAGE_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}
TEXT_ONLY_MODELS = {
    "ernie-4.5-21b-a3b",
}


@dataclass(frozen=True)
class ErnieResponse:
    content: str
    reasoning_content: str
    model: str


class ErnieClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: int = 300,
    ) -> None:
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def supports_image_input(self) -> bool:
        return self.model not in TEXT_ONLY_MODELS

    def complete(
        self,
        prompt: str,
        *,
        image_paths: list[str | Path] | None = None,
        web_search: bool = True,
        max_completion_tokens: int = 65536,
        stream: bool = True,
    ) -> ErnieResponse:
        _ = web_search
        content: str | list[dict[str, Any]]
        if image_paths:
            content = [{"type": "text", "text": prompt}]
            for image_path in image_paths:
                path = Path(image_path)
                mime_type = IMAGE_MIME_TYPES.get(path.suffix.lower())
                if not mime_type:
                    continue
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                    }
                )
        else:
            content = prompt

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            stream=stream,
            extra_body={"penalty_score": 1},
            max_completion_tokens=max_completion_tokens,
            temperature=0.8,
            top_p=0.8,
            frequency_penalty=0,
            presence_penalty=0,
        )

        if not stream:
            choice = completion.choices[0]
            content = choice.message.content or ""
            reasoning = getattr(choice.message, "reasoning_content", "") or ""
            return ErnieResponse(content=content, reasoning_content=reasoning, model=self.model)

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        for chunk in completion:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None)
            content = getattr(delta, "content", None)
            if reasoning:
                reasoning_parts.append(reasoning)
            if content:
                content_parts.append(content)

        return ErnieResponse(
            content="".join(content_parts),
            reasoning_content="".join(reasoning_parts),
            model=self.model,
        )
