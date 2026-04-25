from __future__ import annotations

import json
import re
from typing import Any


def extract_structured_json(text: str) -> dict[str, Any] | None:
    candidates = re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if not candidates:
        candidates = re.findall(r"(\{(?:.|\n)*\})", text, flags=re.DOTALL)

    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None
