"""Application configuration loaded from environment variables."""

import logging
import sys
from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Configuration for the document extraction service.

    All values can be overridden via environment variables or a `.env` file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    openai_api_key: str
    openai_model: str = "gpt-4.1-mini"
    openai_api_url: str = "https://api.openai.com/v1/responses"
    openai_timeout: float = 30.0
    log_level: str = "INFO"

    @field_validator("openai_api_key")
    @classmethod
    def api_key_must_not_be_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("OPENAI_API_KEY must not be empty")
        return value.strip()


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (reads env once)."""
    return Settings()  # type: ignore[call-arg]


def configure_logging(level: str = "INFO") -> None:
    """Set up structured logging for the application."""
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
