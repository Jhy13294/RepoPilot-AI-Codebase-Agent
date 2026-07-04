from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import ConfigError, Settings, load_settings

CONFIG_ENV_VARS = (
    "REPOPILOT_LLM_PROVIDER",
    "REPOPILOT_MODEL",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "REPOPILOT_MAX_STEPS",
    "REPOPILOT_TOOL_TIMEOUT_S",
    "REPOPILOT_MAX_REPLANS",
    "REPOPILOT_MAX_FIX_CYCLES",
    "REPOPILOT_DB_PATH",
    "REPOPILOT_TRACE_DIR",
    "REPOPILOT_WORKSPACE_DIR",
)


@pytest.fixture(autouse=True)
def clean_config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for env_var in CONFIG_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)


def test_settings__loads_defaults_with_fake_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")

    settings = Settings(_env_file=None)

    assert settings.llm_provider == "openai_compatible"
    assert settings.model == "deepseek-v4-pro"
    assert settings.openai_base_url == "https://api.deepseek.com/v1"
    assert settings.openai_api_key == "sk-test-openai"
    assert settings.anthropic_api_key is None
    assert settings.max_steps == 20
    assert settings.tool_timeout_s == 60
    assert settings.max_replans == 3
    assert settings.max_fix_cycles == 2
    assert settings.db_path == Path("data/repopilot.sqlite3")
    assert settings.trace_dir == Path("data/traces")
    assert settings.workspace_dir == Path("data/repos")


def test_settings__environment_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    monkeypatch.setenv("REPOPILOT_MODEL", "test-model")
    monkeypatch.setenv("REPOPILOT_MAX_STEPS", "42")
    monkeypatch.setenv("REPOPILOT_DB_PATH", "tmp/test.sqlite3")

    settings = Settings(_env_file=None)

    assert settings.model == "test-model"
    assert settings.max_steps == 42
    assert settings.db_path == Path("tmp/test.sqlite3")


def test_load_settings__missing_default_provider_key_raises_config_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ConfigError) as exc_info:
        load_settings()

    message = str(exc_info.value)
    assert "OPENAI_API_KEY" in message
    assert ".env.example" in message


def test_load_settings__missing_anthropic_key_raises_config_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("REPOPILOT_LLM_PROVIDER", "anthropic")

    with pytest.raises(ConfigError) as exc_info:
        load_settings()

    message = str(exc_info.value)
    assert "ANTHROPIC_API_KEY" in message
    assert ".env.example" in message


def test_load_settings__anthropic_provider_accepts_fake_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("REPOPILOT_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-anthropic")

    settings = load_settings()

    assert settings.llm_provider == "anthropic"
    assert settings.anthropic_api_key == "sk-test-anthropic"
    assert settings.openai_api_key is None


def test_settings__rejects_invalid_llm_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REPOPILOT_LLM_PROVIDER", "invalid-provider")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
