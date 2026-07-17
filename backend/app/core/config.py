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
    database_filename: str = "pcb_aoi.sqlite3"
    sqlite_busy_timeout_ms: int = Field(default=5000, gt=0, le=60000)
    database_echo: bool = False
    max_rgb_bytes: int = Field(default=50 * 1024 * 1024, gt=0)
    max_height_bytes: int = Field(default=256 * 1024 * 1024, gt=0)
    max_mask_bytes: int = Field(default=64 * 1024 * 1024, gt=0)
    max_calibration_bytes: int = Field(default=5 * 1024 * 1024, gt=0)
    max_generated_artifact_bytes: int = Field(default=50 * 1024 * 1024, gt=0)

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

    @field_validator("database_filename", mode="before")
    @classmethod
    def validate_database_filename(cls, value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("database filename must be a non-empty filename")
        filename = value.strip()
        path = Path(filename)
        if (
            path.is_absolute()
            or path.name != filename
            or filename in {".", ".."}
            or "/" in filename
            or "\\" in filename
            or ":" in filename
        ):
            raise ValueError(
                "database filename must not contain a path or escape runtime root"
            )
        return filename

    model_config = SettingsConfigDict(
        env_prefix="PCB_AOI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
