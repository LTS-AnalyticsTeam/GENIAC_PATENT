from __future__ import annotations

from functools import lru_cache

from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings


class ApiSettings(BaseSettings):
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    allow_origins: str = "*"

    redis_url: str = "redis://redis:6379/0"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_prefix = ""

    def cors_origins(self) -> list[AnyHttpUrl] | list[str]:
        if self.allow_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.allow_origins.split(",")]


@lru_cache
def get_settings() -> ApiSettings:
    return ApiSettings()
