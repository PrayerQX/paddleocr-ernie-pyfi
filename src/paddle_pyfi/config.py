from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_PADDLEOCR_API_URL = "https://i0u1edb895ael4d6.aistudio-app.com/layout-parsing"
DEFAULT_ERNIE_BASE_URL = "https://aistudio.baidu.com/llm/lmapi/v3"
DEFAULT_ERNIE_MODEL = "ernie-4.5-21b-a3b"


@dataclass(frozen=True)
class Settings:
    paddleocr_api_url: str
    paddleocr_api_token: str | None
    ernie_api_key: str | None
    ernie_base_url: str
    ernie_model: str
    request_timeout: int


def load_settings(env_file: str | Path | None = None) -> Settings:
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()

    return Settings(
        paddleocr_api_url=os.getenv("PADDLEOCR_API_URL", DEFAULT_PADDLEOCR_API_URL),
        paddleocr_api_token=os.getenv("PADDLEOCR_API_TOKEN"),
        ernie_api_key=os.getenv("ERNIE_API_KEY"),
        ernie_base_url=os.getenv("ERNIE_BASE_URL", DEFAULT_ERNIE_BASE_URL),
        ernie_model=os.getenv("ERNIE_MODEL", DEFAULT_ERNIE_MODEL),
        request_timeout=int(os.getenv("REQUEST_TIMEOUT", "300")),
    )
