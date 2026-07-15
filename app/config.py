"""Application configuration loading."""

from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigError(Exception):
    """Raised when application settings cannot be loaded."""


class Settings(BaseSettings):
    """Typed application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="REPOPILOT_", extra="ignore")

    llm_provider: Literal["openai_compatible", "anthropic"] = "openai_compatible"
    model: str = "deepseek-v4-pro"
    openai_base_url: str = Field(
        default="https://api.deepseek.com/v1",
        validation_alias="OPENAI_BASE_URL",
    )
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")

    max_steps: int = Field(default=20, ge=1)
    tool_timeout_s: int = Field(default=60, ge=1)
    max_replans: int = Field(default=3, ge=1)
    max_fix_cycles: int = Field(default=2, ge=1)
    test_command: str = "pytest -q"
    test_timeout_s: int = Field(default=120, ge=1)

    db_path: Path = Path("data/repopilot.sqlite3")
    trace_dir: Path = Path("data/traces")
    workspace_dir: Path = Path("data/repos")

    @field_validator("openai_api_key", "anthropic_api_key", mode="before")
    @classmethod
    def empty_api_key_to_none(cls, value: str | None) -> str | None:
        """Treat empty API key values as missing."""
        if value == "":
            return None
        return value

    @field_validator("test_command")
    @classmethod
    def require_test_command(cls, value: str) -> str:
        """Reject an empty operator-configured test command."""
        if not value.strip():
            raise ValueError("test command must not be empty")
        return value

    @model_validator(mode="after")
    def require_active_provider_key(self) -> Self:
        """Require the API key for the selected LLM provider."""
        if self.llm_provider == "openai_compatible" and self.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY is required; see .env.example")
        if self.llm_provider == "anthropic" and self.anthropic_api_key is None:
            raise ValueError("ANTHROPIC_API_KEY is required; see .env.example")
        return self


_ENV_NAMES_BY_FIELD: dict[str, str] = {
    "llm_provider": "REPOPILOT_LLM_PROVIDER",
    "model": "REPOPILOT_MODEL",
    "openai_base_url": "OPENAI_BASE_URL",
    "openai_api_key": "OPENAI_API_KEY",
    "anthropic_api_key": "ANTHROPIC_API_KEY",
    "max_steps": "REPOPILOT_MAX_STEPS",
    "tool_timeout_s": "REPOPILOT_TOOL_TIMEOUT_S",
    "max_replans": "REPOPILOT_MAX_REPLANS",
    "max_fix_cycles": "REPOPILOT_MAX_FIX_CYCLES",
    "test_command": "REPOPILOT_TEST_COMMAND",
    "test_timeout_s": "REPOPILOT_TEST_TIMEOUT_S",
    "db_path": "REPOPILOT_DB_PATH",
    "trace_dir": "REPOPILOT_TRACE_DIR",
    "workspace_dir": "REPOPILOT_WORKSPACE_DIR",
}


def load_settings() -> Settings:
    """Load application settings and convert validation failures into ConfigError."""
    try:
        return Settings()
    except ValidationError as exc:
        error_lines = [_format_validation_error(error) for error in exc.errors()]
        raise ConfigError("Invalid configuration:\n" + "\n".join(error_lines)) from exc


def _format_validation_error(error: Mapping[str, object]) -> str:
    location = error.get("loc", ())
    message = error.get("msg", "invalid value")
    if isinstance(location, tuple) and location:
        env_name = _ENV_NAMES_BY_FIELD.get(str(location[0]), str(location[0]))
        return f"- {env_name}: {message}"
    return f"- {message}"
