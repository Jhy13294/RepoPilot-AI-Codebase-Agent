import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import BaseModel

from app.agent.executor import Executor, ExecutorError, ExecutorErrorReason
from app.agent.state import PlanStep
from app.safety.path_jail import PathJail
from app.schemas.agent_io import StepOutcome
from app.schemas.llm_io import LLMMessage, LLMResponse, Role, StopReason, ToolCall, Usage
from app.schemas.trace import TraceEventKind
from app.services.llm_client import ToolSchema
from app.storage.trace_store import RegistryTraceSink, TraceStore
from app.tools.apply_patch import register as register_apply_patch
from app.tools.base import ToolContext
from app.tools.get_file_tree import register as register_get_file_tree
from app.tools.read_file import register as register_read_file
from app.tools.registry import ApprovalOutcome, ToolRegistry, ToolSpec
from app.tools.search_code import register as register_search_code


@dataclass(frozen=True, slots=True)
class _ClientCall:
    messages: list[LLMMessage]
    tools: list[ToolSchema] | None
    system: str | None
    temperature: float | None
    max_tokens: int | None


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
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
        if not self._responses:
            raise AssertionError("No scripted LLM response remains.")
        return self._responses.pop(0)


class _DenyingGate:
    def __init__(self, reason: str) -> None:
        self.reason = reason
        self.calls: list[tuple[ToolSpec, BaseModel, ToolContext]] = []

    def check(
        self,
        spec: ToolSpec,
        args: BaseModel,
        context: ToolContext,
    ) -> ApprovalOutcome:
        self.calls.append((spec, args, context))
        return ApprovalOutcome(approved=False, reason=self.reason)


def _registry(store: TraceStore) -> ToolRegistry:
    registry = ToolRegistry(trace_sink=RegistryTraceSink(store))
    register_get_file_tree(registry)
    register_read_file(registry)
    register_search_code(registry)
    return registry


def _step() -> PlanStep:
    return PlanStep(
        index=2,
        intent="Locate the parse_date implementation.",
        suggested_tools=["search_code", "read_file"],
        success_check="The implementation file and line are cited.",
    )


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


def _tool_call(call_id: str, name: str, arguments: dict[str, object]) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


def _result_json(
    findings: str = "parse_date is implemented in src/sample_pkg/dates.py:6.",
) -> str:
    return json.dumps(
        {
            "findings": findings,
            "evidence": ["src/sample_pkg/dates.py:6"],
        }
    )


def _executor(
    client: _ScriptedClient,
    mini_repo: Path,
    store: TraceStore,
    *,
    max_tool_calls: int = 8,
    max_arg_repairs: int = 2,
    max_output_repairs: int = 2,
) -> Executor:
    return Executor(
        client,
        _registry(store),
        PathJail(mini_repo),
        store,
        max_tool_calls=max_tool_calls,
        max_arg_repairs=max_arg_repairs,
        max_output_repairs=max_output_repairs,
    )


def test_executor__dispatches_real_registry_and_writes_single_tool_result(
    mini_repo: Path,
    tmp_path: Path,
) -> None:
    store = TraceStore(tmp_path)
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
                tokens_in=3,
                tokens_out=4,
                cost_usd=0.03,
            ),
            _response(
                StopReason.tool_use,
                tool_calls=[
                    _tool_call(
                        "call-read",
                        "read_file",
                        {
                            "path": "src/sample_pkg/dates.py",
                            "start_line": 1,
                            "end_line": 12,
                        },
                    )
                ],
                tokens_in=5,
                tokens_out=6,
                cost_usd=0.05,
            ),
            _response(
                StopReason.end_turn,
                content=f"```json\n{_result_json()}\n```",
                tokens_in=7,
                tokens_out=8,
                cost_usd=0.07,
            ),
        ]
    )

    result = _executor(client, mini_repo, store).execute_step("run-executor", _step())

    assert result.step_index == 2
    assert result.status is StepOutcome.completed
    assert result.tool_calls == 2
    assert result.findings == "parse_date is implemented in src/sample_pkg/dates.py:6."
    assert result.evidence == ["src/sample_pkg/dates.py:6"]
    assert result.usage.tokens_in == 15
    assert result.usage.tokens_out == 18
    assert result.usage.cost_usd == pytest.approx(0.15)

    events = store.read("run-executor")
    assert [event.seq for event in events] == [0, 1, 2]
    assert [event.kind for event in events] == [
        TraceEventKind.tool_call,
        TraceEventKind.tool_call,
        TraceEventKind.tool_result,
    ]
    assert [event.payload["tool_name"] for event in events[:2]] == ["search_code", "read_file"]
    terminal = events[-1]
    assert terminal.payload["status"] == "completed"
    assert terminal.payload["step_index"] == 2
    assert terminal.payload["tool_calls"] == 2
    assert terminal.tokens_in == 15
    assert terminal.tokens_out == 18
    assert terminal.cost_usd == pytest.approx(0.15)

    assert all(call.tools for call in client.calls)
    assert all(call.temperature is None for call in client.calls)


def test_executor__max_tool_calls_returns_incomplete_with_tool_result_event(
    mini_repo: Path,
    tmp_path: Path,
) -> None:
    store = TraceStore(tmp_path)
    client = _ScriptedClient(
        [
            _response(
                StopReason.tool_use,
                tool_calls=[_tool_call("call-search", "search_code", {"query": "parse_date"})],
            ),
            _response(StopReason.end_turn, content=_result_json("This response is unused.")),
        ]
    )

    result = _executor(client, mini_repo, store, max_tool_calls=1).execute_step(
        "run-budget",
        _step(),
    )

    assert result.status is StepOutcome.incomplete
    assert result.tool_calls == 1
    assert len(client.calls) == 1

    events = store.read("run-budget")
    assert [event.kind for event in events] == [
        TraceEventKind.tool_call,
        TraceEventKind.tool_result,
    ]
    assert events[-1].payload["status"] == "incomplete"
    assert events[-1].payload["reason"] == "tool_budget_exhausted"


def test_executor__invalid_arg_repair_exhaustion_is_soft_failure(
    mini_repo: Path,
    tmp_path: Path,
) -> None:
    store = TraceStore(tmp_path)
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
        ]
    )

    result = _executor(client, mini_repo, store, max_arg_repairs=1).execute_step(
        "run-args",
        _step(),
    )

    assert result.status is StepOutcome.incomplete
    assert result.tool_calls == 2
    events = store.read("run-args")
    assert [event.kind for event in events] == [
        TraceEventKind.tool_call,
        TraceEventKind.tool_call,
        TraceEventKind.tool_result,
    ]
    assert events[-1].payload["reason"] == "arg_repair_exhausted"


def test_executor__approval_denial_terminates_step_before_resubmission(
    mini_repo: Path,
    tmp_path: Path,
) -> None:
    store = TraceStore(tmp_path)
    denial_reason = "The proposed change needs a different approach."
    gate = _DenyingGate(denial_reason)
    registry = ToolRegistry(approval_gate=gate, trace_sink=RegistryTraceSink(store))
    register_apply_patch(registry)
    client = _ScriptedClient(
        [
            _response(
                StopReason.tool_use,
                tool_calls=[
                    _tool_call(
                        "call-denied",
                        "apply_patch",
                        {
                            "diff": "--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-old\n+first\n",
                            "rationale": "Apply the first proposal.",
                        },
                    )
                ],
            ),
            _response(
                StopReason.tool_use,
                tool_calls=[
                    _tool_call(
                        "call-resubmit",
                        "apply_patch",
                        {
                            "diff": "--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-old\n+first\n",
                            "rationale": "Resubmit the same proposal.",
                        },
                    )
                ],
            ),
        ]
    )

    result = Executor(client, registry, PathJail(mini_repo), store).execute_step(
        "run-denial",
        _step(),
    )

    assert result.status is StepOutcome.incomplete
    assert result.findings == denial_reason
    assert result.tool_calls == 1
    assert len(client.calls) == 1
    assert [spec.name for spec, _args, _context in gate.calls] == ["apply_patch"]
    events = store.read("run-denial")
    assert [event.kind for event in events] == [
        TraceEventKind.approval_decision,
        TraceEventKind.tool_call,
        TraceEventKind.tool_result,
    ]
    assert events[0].payload == {
        "tool_name": "apply_patch",
        "risk_level": "high",
        "decision": "denied",
        "actor": "human",
        "reason": denial_reason,
    }
    assert events[1].payload["error_type"] == "ApprovalDeniedError"
    assert events[-1].payload["reason"] == "approval_denied"


def test_executor__refusal_traces_error_and_raises_without_dispatch(
    mini_repo: Path,
    tmp_path: Path,
) -> None:
    store = TraceStore(tmp_path)
    client = _ScriptedClient(
        [
            _response(
                StopReason.refusal,
                content="I cannot help with that.",
                tokens_in=9,
                tokens_out=2,
                cost_usd=0.09,
            ),
            _response(StopReason.end_turn, content=_result_json("This response is unused.")),
        ]
    )

    with pytest.raises(ExecutorError) as exc_info:
        _executor(client, mini_repo, store).execute_step("run-refusal", _step())

    assert exc_info.value.reason is ExecutorErrorReason.refusal
    assert len(client.calls) == 1
    [event] = store.read("run-refusal")
    assert event.kind is TraceEventKind.error
    assert event.payload["reason"] == "refusal"
    assert event.payload["tool_calls"] == 0
    assert event.tokens_in == 9
    assert event.tokens_out == 2
    assert event.cost_usd == 0.09


def test_executor__synthesis_repair_exhaustion_traces_error_and_raises(
    mini_repo: Path,
    tmp_path: Path,
) -> None:
    store = TraceStore(tmp_path)
    client = _ScriptedClient(
        [
            _response(StopReason.end_turn, content="not json", tokens_in=1, tokens_out=2),
            _response(StopReason.end_turn, content='{"findings": 123}', tokens_in=3, tokens_out=4),
        ]
    )

    with pytest.raises(ExecutorError) as exc_info:
        _executor(client, mini_repo, store, max_output_repairs=1).execute_step(
            "run-synthesis",
            _step(),
        )

    assert exc_info.value.reason is ExecutorErrorReason.synthesis_repair_exhausted
    assert len(client.calls) == 2
    repair_prompt = client.calls[1].messages[-1]
    assert repair_prompt.role is Role.user
    assert "Validation error:" in repair_prompt.content

    [event] = store.read("run-synthesis")
    assert event.kind is TraceEventKind.error
    assert event.payload["reason"] == "synthesis_repair_exhausted"
    assert event.payload["attempts"] == 2
    assert event.tokens_in == 4
    assert event.tokens_out == 6


def test_executor__stalled_pause_turns_terminate_as_incomplete(
    mini_repo: Path,
    tmp_path: Path,
) -> None:
    # A degenerate model that never dispatches a tool or synthesizes must still terminate: the
    # loop is bounded in code, not by trusting the model's stop_reason. Four pause_turns cross
    # the _MAX_STALLED_COMPLETIONS=3 ceiling.
    store = TraceStore(tmp_path)
    client = _ScriptedClient([_response(StopReason.pause_turn) for _ in range(4)])

    result = _executor(client, mini_repo, store).execute_step("run-stall", _step())

    assert result.status is StepOutcome.incomplete
    assert result.tool_calls == 0
    assert len(client.calls) == 4
    events = store.read("run-stall")
    assert [event.kind for event in events] == [TraceEventKind.tool_result]
    assert events[-1].payload["reason"] == "stalled_without_progress"


def test_executor__step_context_and_scratchpad_reach_prompt(
    mini_repo: Path,
    tmp_path: Path,
) -> None:
    store = TraceStore(tmp_path)
    client = _ScriptedClient([_response(StopReason.end_turn, content=_result_json())])
    scratchpad = "Earlier evidence mentions src/sample_pkg/core.py."
    step = _step()

    _executor(client, mini_repo, store).execute_step(
        "run-prompt",
        step,
        scratchpad=scratchpad,
        repo_overview="Small Python package.",
    )

    [call] = client.calls
    assert call.system is not None
    assert "Role & mission" in call.system
    assert call.tools
    prompt = "\n\n".join(message.content for message in call.messages)
    assert step.intent in prompt
    assert step.success_check in prompt
    assert scratchpad in prompt
