"""App configuration, loaded from .env. See docs/CONVENTIONS.md — never hardcode secrets."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    sarvam_api_key: str = ""
    groq_api_key: str = ""
    port: int = 8000


settings = Settings()
