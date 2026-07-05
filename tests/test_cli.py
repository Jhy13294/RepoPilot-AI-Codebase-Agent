import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from click.testing import Result
from pydantic import JsonValue
from typer.testing import CliRunner

import app.cli as cli
from app.config import Settings
from app.schemas.llm_io import LLMMessage, LLMResponse, Role, StopReason, ToolCall, Usage
from app.services.llm_client import ToolSchema

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


@dataclass(frozen=True, slots=True)
class _ClientCall:
    messages: list[LLMMessage]
    tools: list[ToolSchema] | None


class _ScriptedClient:
    def __init__(self, responses: Sequence[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[_ClientCall] = []

    def complete(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[ToolSchema] | None = None,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls.append(
            _ClientCall(
                messages=list(messages),
                tools=list(tools) if tools is not None else None,
            )
        )
        if system is not None or temperature is not None or max_tokens is not None:
            raise AssertionError("CLI should leave loop-owned completion options unset.")
        if not self._responses:
            raise AssertionError("No scripted LLM response remains.")
        return self._responses.pop(0)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def clean_cli_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for env_var in CONFIG_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.chdir(tmp_path)


def test_ask__happy_path_renders_answer_summary_and_citation(
    monkeypatch: pytest.MonkeyPatch,
    runner: CliRunner,
    mini_repo: Path,
) -> None:
    _set_fake_openai_env(monkeypatch)
    client = _ScriptedClient(
        [
            _response(
                StopReason.tool_use,
                tool_calls=[
                    _tool_call(
                        "call-search",
                        "search_code",
                        {"query": "def parse_date", "glob": "**/*.py", "context_lines": 0},
                    )
                ],
                tokens_in=11,
                tokens_out=3,
                cost_usd=0.01,
            ),
            _response(
                StopReason.end_turn,
                content="parse_date is defined in src/sample_pkg/dates.py:6.",
                tokens_in=17,
                tokens_out=7,
                cost_usd=0.02,
            ),
        ]
    )
    _patch_client(monkeypatch, client)

    result = runner.invoke(
        cli.app,
        ["ask", "Where is parse_date defined?", "--repo", str(mini_repo)],
    )

    output = _combined_output(result)
    assert result.exit_code == 0, output
    assert "parse_date is defined in src/sample_pkg/dates.py:6." in output
    assert "status=answered" in output
    assert "steps=2" in output
    assert "tool_calls=1" in output
    assert "tokens_in=28" in output
    assert "tokens_out=10" in output
    assert "cost=$0.030000" in output
    assert "Traceback" not in output
    assert len(client.calls) == 2

    tool_message = client.calls[1].messages[-1]
    assert tool_message.role is Role.tool
    payload = json.loads(tool_message.content)
    assert payload["ok"] is True
    assert payload["data"]["matches"][0]["path"] == "src/sample_pkg/dates.py"
    assert payload["data"]["matches"][0]["line"] == 6


def test_ask__config_error_prints_friendly_message(
    runner: CliRunner,
    mini_repo: Path,
) -> None:
    result = runner.invoke(
        cli.app,
        ["ask", "Where is parse_date defined?", "--repo", str(mini_repo)],
    )

    output = _combined_output(result)
    assert result.exit_code != 0
    assert "Configuration error:" in output
    assert "OPENAI_API_KEY" in output
    assert "Traceback" not in output


@pytest.mark.parametrize("repo_kind", ["missing", "file"])
def test_ask__bad_repo_prints_friendly_message(
    monkeypatch: pytest.MonkeyPatch,
    runner: CliRunner,
    tmp_path: Path,
    repo_kind: str,
) -> None:
    _set_fake_openai_env(monkeypatch)
    repo_path = tmp_path / "missing"
    if repo_kind == "file":
        repo_path = tmp_path / "not-a-repo.txt"
        repo_path.write_text("not a directory", encoding="utf-8")

    result = runner.invoke(
        cli.app,
        ["ask", "Where is parse_date defined?", "--repo", str(repo_path)],
    )

    output = _combined_output(result)
    assert result.exit_code != 0
    assert "Repository error:" in output
    assert "directory" in output
    assert "Traceback" not in output


def test_ask__refusal_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    runner: CliRunner,
    mini_repo: Path,
) -> None:
    _set_fake_openai_env(monkeypatch)
    client = _ScriptedClient(
        [
            _response(
                StopReason.refusal,
                content="I cannot answer that request.",
                tokens_in=5,
                tokens_out=4,
                cost_usd=None,
            )
        ]
    )
    _patch_client(monkeypatch, client)

    result = runner.invoke(
        cli.app,
        ["ask", "Where is parse_date defined?", "--repo", str(mini_repo)],
    )

    output = _combined_output(result)
    assert result.exit_code == 1
    assert "I cannot answer that request." in output
    assert "status=refused" in output
    assert "tool_calls=0" in output
    assert "cost=n/a" in output
    assert "Traceback" not in output


def _set_fake_openai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-cli")


def _patch_client(monkeypatch: pytest.MonkeyPatch, client: _ScriptedClient) -> None:
    def build_fake_client(_settings: Settings) -> _ScriptedClient:
        return client

    monkeypatch.setattr(cli, "build_llm_client", build_fake_client)


def _response(
    stop_reason: StopReason,
    *,
    content: str = "",
    tool_calls: list[ToolCall] | None = None,
    tokens_in: int = 10,
    tokens_out: int = 5,
    cost_usd: float | None = 0.01,
) -> LLMResponse:
    return LLMResponse(
        message=LLMMessage(
            role=Role.assistant,
            content=content,
            tool_calls=tool_calls or [],
        ),
        stop_reason=stop_reason,
        usage=Usage(tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=cost_usd),
        model="fake-model",
        raw_finish_reason=stop_reason.value,
    )


def _tool_call(call_id: str, name: str, arguments: dict[str, JsonValue]) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


def _combined_output(result: Result) -> str:
    try:
        return f"{result.output}{result.stderr}"
    except ValueError:
        return result.output
