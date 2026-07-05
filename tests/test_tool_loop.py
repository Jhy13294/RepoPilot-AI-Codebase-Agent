import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import JsonValue

from app.agent.tool_loop import ASK_SYSTEM_PROMPT, run_tool_loop
from app.safety.path_jail import PathJail
from app.schemas.agent_io import AskStatus
from app.schemas.llm_io import LLMMessage, LLMResponse, Role, StopReason, ToolCall, Usage
from app.schemas.tool_io import ErrorType
from app.services.llm_client import ToolSchema
from app.tools.get_file_tree import register as register_get_file_tree
from app.tools.read_file import register as register_read_file
from app.tools.registry import ToolRegistry, ToolTraceRecord
from app.tools.search_code import register as register_search_code


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
            raise AssertionError("run_tool_loop should pass system prompt as a message only.")
        if not self._responses:
            raise AssertionError("No scripted LLM response remains.")
        return self._responses.pop(0)


class _TraceSink:
    def __init__(self) -> None:
        self.records: list[ToolTraceRecord] = []

    def append(self, record: ToolTraceRecord) -> None:
        self.records.append(record)


def _registry(trace_sink: _TraceSink | None = None) -> ToolRegistry:
    registry = ToolRegistry(trace_sink=trace_sink)
    register_get_file_tree(registry)
    register_read_file(registry)
    register_search_code(registry)
    return registry


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


def _run(
    client: _ScriptedClient,
    mini_repo: Path,
    *,
    registry: ToolRegistry | None = None,
    max_steps: int = 10,
) -> object:
    return run_tool_loop(
        "Where is parse_date defined?",
        client=client,
        registry=registry or _registry(),
        jail=PathJail(mini_repo),
        max_steps=max_steps,
    )


def _tool_payload(message: LLMMessage) -> dict[str, object]:
    parsed = json.loads(message.content)
    assert isinstance(parsed, dict)
    return parsed


def test_tool_loop__happy_path_dispatches_two_tools_and_answers(mini_repo: Path) -> None:
    trace_sink = _TraceSink()
    registry = _registry(trace_sink)
    client = _ScriptedClient(
        [
            _response(
                StopReason.tool_use,
                tool_calls=[
                    _tool_call(
                        "call-search",
                        "search_code",
                        {"query": "def parse_date", "glob": "**/*.py"},
                    )
                ],
                tokens_in=11,
                tokens_out=3,
                cost_usd=0.11,
            ),
            _response(
                StopReason.tool_use,
                tool_calls=[
                    _tool_call(
                        "call-read",
                        "read_file",
                        {"path": "src/sample_pkg/dates.py", "start_line": 1, "end_line": 12},
                    )
                ],
                tokens_in=13,
                tokens_out=5,
                cost_usd=0.13,
            ),
            _response(
                StopReason.end_turn,
                content="parse_date is defined in src/sample_pkg/dates.py:6.",
                tokens_in=17,
                tokens_out=7,
                cost_usd=0.17,
            ),
        ]
    )

    result = run_tool_loop(
        "Where is parse_date defined?",
        client=client,
        registry=registry,
        jail=PathJail(mini_repo),
        max_steps=10,
    )

    assert result.status is AskStatus.answered
    assert result.answer == "parse_date is defined in src/sample_pkg/dates.py:6."
    assert result.steps == 3
    assert [record.tool_name for record in trace_sink.records] == ["search_code", "read_file"]
    assert [invocation.tool for invocation in result.tool_invocations] == [
        "search_code",
        "read_file",
    ]
    assert all(invocation.ok for invocation in result.tool_invocations)
    assert result.usage.tokens_in == 41
    assert result.usage.tokens_out == 15
    assert result.usage.cost_usd == pytest.approx(0.41)
    assert client.calls[0].messages == [
        LLMMessage(role=Role.system, content=ASK_SYSTEM_PROMPT),
        LLMMessage(role=Role.user, content="Where is parse_date defined?"),
    ]
    assert all(call.tools for call in client.calls)


def test_tool_loop__all_dispatches_share_one_run_id(mini_repo: Path) -> None:
    trace_sink = _TraceSink()
    registry = _registry(trace_sink)
    client = _ScriptedClient(
        [
            _response(
                StopReason.tool_use,
                tool_calls=[
                    _tool_call(
                        "call-search",
                        "search_code",
                        {"query": "def parse_date", "glob": "**/*.py"},
                    )
                ],
            ),
            _response(
                StopReason.tool_use,
                tool_calls=[
                    _tool_call(
                        "call-read",
                        "read_file",
                        {"path": "src/sample_pkg/dates.py", "start_line": 1, "end_line": 12},
                    )
                ],
            ),
            _response(
                StopReason.end_turn,
                content="parse_date is defined in src/sample_pkg/dates.py:6.",
            ),
        ]
    )

    result = run_tool_loop(
        "Where is parse_date defined?",
        client=client,
        registry=registry,
        jail=PathJail(mini_repo),
        max_steps=10,
    )

    assert result.status is AskStatus.answered
    assert [record.tool_name for record in trace_sink.records] == ["search_code", "read_file"]
    assert len({record.run_id for record in trace_sink.records}) == 1


def test_tool_loop__feeds_invalid_args_error_back_then_repairs(mini_repo: Path) -> None:
    client = _ScriptedClient(
        [
            _response(
                StopReason.tool_use,
                tool_calls=[_tool_call("call-bad", "read_file", {})],
            ),
            _response(
                StopReason.tool_use,
                tool_calls=[
                    _tool_call(
                        "call-good",
                        "read_file",
                        {"path": "README.md", "start_line": 1, "end_line": 2},
                    )
                ],
            ),
            _response(
                StopReason.end_turn,
                content="README evidence is available at README.md:1.",
            ),
        ]
    )

    result = _run(client, mini_repo)

    assert result.status is AskStatus.answered
    assert [invocation.ok for invocation in result.tool_invocations] == [False, True]
    assert result.tool_invocations[0].error_type is ErrorType.InvalidArgsError
    repair_observation = client.calls[1].messages[-1]
    assert repair_observation.role is Role.tool
    assert repair_observation.tool_call_id == "call-bad"
    payload = _tool_payload(repair_observation)
    assert payload["ok"] is False
    assert payload["error"]["type"] == ErrorType.InvalidArgsError.value


def test_tool_loop__stops_after_third_consecutive_invalid_args(mini_repo: Path) -> None:
    client = _ScriptedClient(
        [
            _response(
                StopReason.tool_use,
                tool_calls=[_tool_call("call-bad-1", "read_file", {})],
            ),
            _response(
                StopReason.tool_use,
                tool_calls=[_tool_call("call-bad-2", "read_file", {})],
            ),
            _response(
                StopReason.tool_use,
                tool_calls=[_tool_call("call-bad-3", "read_file", {})],
            ),
            _response(StopReason.end_turn, content="This response must not be used."),
        ]
    )

    result = _run(client, mini_repo)

    assert result.status is AskStatus.error
    assert result.steps == 3
    assert len(client.calls) == 3
    assert len(result.tool_invocations) == 3
    assert all(
        invocation.error_type is ErrorType.InvalidArgsError
        for invocation in result.tool_invocations
    )
    tool_messages_before_third_attempt = [
        message for message in client.calls[2].messages if message.role is Role.tool
    ]
    assert len(tool_messages_before_third_attempt) == 2


def test_tool_loop__refusal_is_terminal_without_retry(mini_repo: Path) -> None:
    client = _ScriptedClient(
        [
            _response(StopReason.refusal, content="I cannot help with that."),
            _response(StopReason.end_turn, content="This response must not be used."),
        ]
    )

    result = _run(client, mini_repo)

    assert result.status is AskStatus.refused
    assert result.answer == "I cannot help with that."
    assert result.steps == 1
    assert result.tool_invocations == []
    assert len(client.calls) == 1


def test_tool_loop__max_steps_exhausts_budget_for_endless_tool_use(mini_repo: Path) -> None:
    client = _ScriptedClient(
        [
            _response(
                StopReason.tool_use,
                content="Searching.",
                tool_calls=[_tool_call("call-1", "search_code", {"query": "parse_date"})],
            ),
            _response(
                StopReason.tool_use,
                content="Still searching.",
                tool_calls=[_tool_call("call-2", "search_code", {"query": "parse_date"})],
            ),
            _response(
                StopReason.tool_use,
                tool_calls=[_tool_call("call-3", "search_code", {"query": "parse_date"})],
            ),
        ]
    )

    result = _run(client, mini_repo, max_steps=2)

    assert result.status is AskStatus.budget_exhausted
    assert result.answer == "Still searching."
    assert result.steps == 2
    assert len(client.calls) == 2
    assert len(result.tool_invocations) == 2


def test_tool_loop__max_tokens_gets_one_continuation_to_end_turn(mini_repo: Path) -> None:
    client = _ScriptedClient(
        [
            _response(StopReason.max_tokens, content="Partial "),
            _response(StopReason.end_turn, content="answer at README.md:1."),
        ]
    )

    result = _run(client, mini_repo)

    assert result.status is AskStatus.answered
    assert result.answer == "Partial answer at README.md:1."
    assert result.steps == 2
    assert result.tool_invocations == []
    assert client.calls[1].messages[-1] == LLMMessage(
        role=Role.assistant,
        content="Partial ",
    )


def test_tool_loop__pause_turn_resumes_and_accumulates_usage(mini_repo: Path) -> None:
    client = _ScriptedClient(
        [
            _response(
                StopReason.pause_turn,
                content="Need one more turn.",
                tokens_in=2,
                tokens_out=3,
                cost_usd=0.2,
            ),
            _response(
                StopReason.end_turn,
                content="Done with evidence at README.md:1.",
                tokens_in=5,
                tokens_out=7,
                cost_usd=0.5,
            ),
        ]
    )

    result = _run(client, mini_repo)

    assert result.status is AskStatus.answered
    assert result.steps == 2
    assert result.usage.tokens_in == 7
    assert result.usage.tokens_out == 10
    assert result.usage.cost_usd == pytest.approx(0.7)
    assert client.calls[1].messages[-1] == LLMMessage(
        role=Role.assistant,
        content="Need one more turn.",
    )


def test_tool_loop__feeds_tool_result_as_json_with_matching_call_id(mini_repo: Path) -> None:
    client = _ScriptedClient(
        [
            _response(
                StopReason.tool_use,
                tool_calls=[_tool_call("call-json", "search_code", {"query": "parse_date"})],
            ),
            _response(StopReason.end_turn, content="Found evidence at src/sample_pkg/dates.py:6."),
        ]
    )

    result = _run(client, mini_repo)

    assert result.status is AskStatus.answered
    tool_message = client.calls[1].messages[-1]
    assert tool_message.role is Role.tool
    assert tool_message.tool_call_id == "call-json"
    payload = _tool_payload(tool_message)
    assert payload["ok"] is True
    assert payload["error"] is None
    assert payload["meta"]["tool_name"] == "search_code"
    assert payload["data"]["total_found"] >= 1


def test_tool_loop__max_tokens_non_end_turn_exhausts_without_dispatch(mini_repo: Path) -> None:
    client = _ScriptedClient(
        [
            _response(StopReason.max_tokens, content="Partial "),
            _response(
                StopReason.tool_use,
                content="need a tool",
                tool_calls=[_tool_call("call-after-length", "search_code", {"query": "x"})],
            ),
        ]
    )

    result = _run(client, mini_repo)

    assert result.status is AskStatus.budget_exhausted
    assert result.answer == "Partial need a tool"
    assert result.steps == 2
    assert result.tool_invocations == []
