import json
import subprocess
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
from app.schemas.trace import TraceEventKind
from app.services.llm_client import ToolSchema
from app.storage.trace_store import TraceStore

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
    def __init__(
        self,
        responses: Sequence[LLMResponse],
        *,
        allow_loop_options: bool = False,
    ) -> None:
        self._responses = list(responses)
        self._allow_loop_options = allow_loop_options
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
        if not self._allow_loop_options and (
            system is not None or temperature is not None or max_tokens is not None
        ):
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


def test_run__happy_path_traces_tool_calls_and_renders_run_id(
    monkeypatch: pytest.MonkeyPatch,
    runner: CliRunner,
    tmp_path: Path,
    mini_repo: Path,
) -> None:
    _set_fake_openai_env(monkeypatch)
    trace_dir, _db_path = _set_storage_env(monkeypatch, tmp_path)
    client = _ScriptedClient(
        [
            _response(
                StopReason.end_turn,
                content=json.dumps(
                    {
                        "steps": [
                            {
                                "intent": "Locate the parse_date definition.",
                                "suggested_tools": ["search_code"],
                                "success_check": (
                                    "The parse_date definition path and line are known."
                                ),
                            }
                        ]
                    }
                ),
                tokens_in=10,
                tokens_out=2,
                cost_usd=0.01,
            ),
            _response(
                StopReason.tool_use,
                tool_calls=[
                    _tool_call(
                        "call-search",
                        "search_code",
                        {"query": "def parse_date", "glob": "**/*.py", "context_lines": 0},
                    )
                ],
                tokens_in=12,
                tokens_out=2,
                cost_usd=0.02,
            ),
            _response(
                StopReason.end_turn,
                content=json.dumps(
                    {
                        "findings": "parse_date is defined in src/sample_pkg/dates.py:6.",
                        "evidence": ["search_code: src/sample_pkg/dates.py:6"],
                    }
                ),
                tokens_in=8,
                tokens_out=3,
                cost_usd=0.03,
            ),
            _response(
                StopReason.end_turn,
                content=json.dumps(
                    {
                        "decision": "proceed",
                        "reason": "The raw search evidence proves the cited definition.",
                        "hint": "",
                    }
                ),
                tokens_in=7,
                tokens_out=2,
                cost_usd=0.04,
            ),
        ],
        allow_loop_options=True,
    )
    _patch_client(monkeypatch, client)

    result = runner.invoke(
        cli.app,
        ["run", "Where is parse_date defined?", "--repo", str(mini_repo)],
    )

    output = _combined_output(result)
    assert result.exit_code == 0, output
    assert "status=DONE" in output
    assert "steps=1" in output
    assert "replans=0" in output
    assert "fix_cycles=0" in output
    assert "tokens_in=37" in output
    assert "tokens_out=9" in output
    assert "cost=$0.100000" in output
    assert "run_id=" in output
    assert "repopilot replay" in output
    assert "Traceback" not in output

    run_id = _extract_run_id(output)
    assert (trace_dir / f"{run_id}.jsonl").exists()
    events = TraceStore(trace_dir).read(run_id)
    assert [event.kind for event in events] == [
        TraceEventKind.plan,
        TraceEventKind.tool_call,
        TraceEventKind.tool_result,
        TraceEventKind.critic_verdict,
        TraceEventKind.report,
    ]
    assert events[1].payload["tool_name"] == "search_code"
    assert events[1].payload["ok"] is True


def test_run__planner_refusal_exits_nonzero_and_keeps_report_trace(
    monkeypatch: pytest.MonkeyPatch,
    runner: CliRunner,
    tmp_path: Path,
    mini_repo: Path,
) -> None:
    _set_fake_openai_env(monkeypatch)
    trace_dir, _db_path = _set_storage_env(monkeypatch, tmp_path)
    client = _ScriptedClient(
        [
            _response(
                StopReason.refusal,
                content="I cannot plan that request.",
                tokens_in=5,
                tokens_out=4,
                cost_usd=None,
            )
        ],
        allow_loop_options=True,
    )
    _patch_client(monkeypatch, client)

    result = runner.invoke(
        cli.app,
        ["run", "Where is parse_date defined?", "--repo", str(mini_repo)],
    )

    output = _combined_output(result)
    assert result.exit_code == 1, output
    assert "status=FAILED" in output
    assert "run_id=" in output
    assert "I cannot plan that request." in output
    assert "Traceback" not in output

    run_id = _extract_run_id(output)
    events = TraceStore(trace_dir).read(run_id)
    assert any(event.kind is TraceEventKind.report for event in events)


def test_replay__prints_jsonl_timeline(
    monkeypatch: pytest.MonkeyPatch,
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    _set_fake_openai_env(monkeypatch)
    trace_dir, _db_path = _set_storage_env(monkeypatch, tmp_path)
    store = TraceStore(trace_dir)
    run_id = "known-run"
    store.append(run_id, TraceEventKind.plan, {"summary": "Planned 1 step."})
    store.append(
        run_id,
        TraceEventKind.tool_call,
        {
            "tool_name": "search_code",
            "args": {"query": "parse_date"},
            "ok": True,
            "error_type": None,
            "truncated": False,
        },
    )
    store.append(run_id, TraceEventKind.report, {"summary": "Run known-run succeeded."})

    result = runner.invoke(cli.app, ["replay", run_id])

    output = _combined_output(result)
    assert result.exit_code == 0, output
    assert "plan - Planned 1 step." in output
    assert "tool_call - search_code ok" in output
    assert "report - Run known-run succeeded." in output
    assert "Traceback" not in output


def test_replay__unknown_run_id_prints_friendly_message(
    monkeypatch: pytest.MonkeyPatch,
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    _set_fake_openai_env(monkeypatch)
    _set_storage_env(monkeypatch, tmp_path)

    result = runner.invoke(cli.app, ["replay", "missing-run"])

    output = _combined_output(result)
    assert result.exit_code == 1
    assert "no trace found for missing-run" in output
    assert "Traceback" not in output


def test_console_script_help_lists_cli_commands() -> None:
    result = subprocess.run(
        ["uv", "run", "repopilot", "--help"],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )

    output = f"{result.stdout}{result.stderr}"
    assert result.returncode == 0, output
    assert "ask" in output
    assert "run" in output
    assert "replay" in output


def _set_fake_openai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-cli")


def _set_storage_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    trace_dir = tmp_path / "traces"
    db_path = tmp_path / "repopilot.sqlite3"
    monkeypatch.setenv("REPOPILOT_TRACE_DIR", str(trace_dir))
    monkeypatch.setenv("REPOPILOT_DB_PATH", str(db_path))
    return trace_dir, db_path


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


def _extract_run_id(output: str) -> str:
    for line in output.splitlines():
        if line.startswith("run_id="):
            return line.removeprefix("run_id=").strip()
    raise AssertionError(f"run_id line missing from output:\n{output}")


def _combined_output(result: Result) -> str:
    try:
        return f"{result.output}{result.stderr}"
    except ValueError:
        return result.output
