import os
from dataclasses import dataclass, field

from dotenv import find_dotenv, load_dotenv


_DOTENV_PATH = find_dotenv(filename=".env", usecwd=True)
load_dotenv(dotenv_path=_DOTENV_PATH, override=False)


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))


def get_settings() -> Settings:
    return Settings()
