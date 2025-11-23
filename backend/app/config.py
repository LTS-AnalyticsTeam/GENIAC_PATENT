from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration."""

    model_config = SettingsConfigDict(env_prefix="PATENT_ALPHA_", case_sensitive=False)

    app_title: str = "Patent Alpha Analyzer"
    app_version: str = "0.1.0"

    cors_allow_origins: List[str] = Field(
        default_factory=lambda: ["http://127.0.0.1:5173", "http://localhost:5173"]
    )
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["*"]
    cors_allow_headers: List[str] = ["*"]


@lru_cache()
def get_settings() -> Settings:
    return Settings()
