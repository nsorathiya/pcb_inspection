from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.runtime_paths import default_runtime_root, resolve_runtime_root


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    PLAIN = "plain"


class Settings(BaseSettings):
    application_name: str = "pcb-aoi-api"
    application_version: str = "0.1.0"
    environment: str = "development"
    api_prefix: str = "/api/v1"
    debug: bool = False
    log_level: LogLevel = LogLevel.INFO
    log_format: LogFormat = LogFormat.PLAIN
    runtime_root: Path = Field(default_factory=default_runtime_root)

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @field_validator("log_format", mode="before")
    @classmethod
    def normalize_log_format(cls, value: object) -> object:
        return value.lower() if isinstance(value, str) else value

    @field_validator("runtime_root", mode="before")
    @classmethod
    def normalize_runtime_root(cls, value: object) -> Path:
        return resolve_runtime_root(Path(value))

    model_config = SettingsConfigDict(
        env_prefix="PCB_AOI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
