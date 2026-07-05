import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from app.agent.planner import Planner, PlannerError, PlannerErrorReason
from app.agent.state import PlanStepStatus, TaskSpec
from app.schemas.llm_io import LLMMessage, LLMResponse, Role, StopReason, Usage
from app.schemas.trace import TraceEventKind
from app.services.llm_client import ToolSchema
from app.storage.trace_store import TraceStore, render_timeline


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


def _task() -> TaskSpec:
    return TaskSpec(
        task_type="issue",
        prompt="Fix the failing parser test.",
        repo=".",
    )


def _draft_json(*, steps: int = 1) -> str:
    return json.dumps(
        {
            "steps": [
                {
                    "intent": f"Inspect evidence for part {index}.",
                    "suggested_tools": ["get_file_tree", "search_code"],
                    "success_check": f"Evidence for part {index} is cited.",
                }
                for index in range(steps)
            ]
        }
    )


def _response(
    content: str,
    *,
    stop_reason: StopReason = StopReason.end_turn,
    tokens_in: int = 10,
    tokens_out: int = 5,
    cost_usd: float | None = 0.01,
) -> LLMResponse:
    return LLMResponse(
        message=LLMMessage(role=Role.assistant, content=content),
        stop_reason=stop_reason,
        usage=Usage(tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=cost_usd),
        model="fake-model",
        raw_finish_reason=stop_reason.value,
    )


def test_planner__writes_plan_event_to_real_trace_store(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    client = _ScriptedClient(
        [
            _response(
                f"```json\n{_draft_json(steps=2)}\n```",
                tokens_in=17,
                tokens_out=11,
                cost_usd=0.23,
            )
        ]
    )

    result = Planner(client, store, tools_doc="search_code: low risk search.").plan(
        "run-plan",
        _task(),
    )

    assert result.attempts == 1
    assert result.replanned is False
    assert result.usage == Usage(tokens_in=17, tokens_out=11, cost_usd=0.23)
    assert [step.index for step in result.plan] == [0, 1]
    assert [step.status for step in result.plan] == [
        PlanStepStatus.pending,
        PlanStepStatus.pending,
    ]

    events = store.read("run-plan")
    assert len(events) == 1
    [event] = events
    assert event.kind is TraceEventKind.plan
    assert event.payload["summary"] == "Planned 2 step(s)."
    assert event.payload["attempts"] == 1
    assert event.payload["steps"][0]["index"] == 0
    assert event.payload["steps"][0]["status"] == "pending"
    assert event.tokens_in == 17
    assert event.tokens_out == 11
    assert event.cost_usd == 0.23
    assert event.latency_ms is not None

    [call] = client.calls
    assert call.tools is None
    assert call.temperature is None
    assert call.max_tokens is None
    assert call.system is not None
    assert "Role & mission" in call.system
    assert "Hard rules" in call.system
    assert "Tool documentation" in call.system
    assert "search_code: low risk search." in call.system
    assert "Output contract" in call.system


def test_planner__repair_message_reaches_next_prompt_and_accumulates_usage(
    tmp_path: Path,
) -> None:
    store = TraceStore(tmp_path)
    invalid_reply = '{"steps":[]}'
    client = _ScriptedClient(
        [
            _response(invalid_reply, tokens_in=3, tokens_out=4, cost_usd=0.03),
            _response(_draft_json(), tokens_in=5, tokens_out=6, cost_usd=0.05),
        ]
    )

    result = Planner(client, store).plan("run-repair", _task())

    assert result.attempts == 2
    assert result.usage == Usage(tokens_in=8, tokens_out=10, cost_usd=0.08)
    assert len(store.read("run-repair")) == 1
    assert len(client.calls) == 2
    second_prompt_messages = client.calls[1].messages
    assert second_prompt_messages[-2] == LLMMessage(role=Role.assistant, content=invalid_reply)
    correction = second_prompt_messages[-1]
    assert correction.role is Role.user
    assert invalid_reply in correction.content
    assert "Validation error:" in correction.content
    assert "steps" in correction.content


def test_planner__failure_summary_and_repo_overview_reach_replan_prompt(
    tmp_path: Path,
) -> None:
    store = TraceStore(tmp_path)
    client = _ScriptedClient([_response(_draft_json())])
    repo_overview = "Python package under app with pytest tests."
    failure_summary = "The previous step found NotFoundError for app/missing.py."

    result = Planner(client, store).plan(
        "run-replan",
        _task(),
        repo_overview=repo_overview,
        failure_summary=failure_summary,
    )

    assert result.replanned is True
    [event] = store.read("run-replan")
    assert event.kind is TraceEventKind.replan
    assert event.payload["summary"] == "Replanned 1 step(s)."

    prompt = client.calls[0].messages[0].content
    assert repo_overview in prompt
    assert failure_summary in prompt


def test_planner__plan_event_composes_with_real_render_timeline(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    client = _ScriptedClient([_response(_draft_json())])

    Planner(client, store).plan("run-timeline", _task())

    timeline = render_timeline(store.read("run-timeline"))
    assert "plan - Planned 1 step(s)." in timeline


def test_planner__repair_exhaustion_emits_error_event_and_raises(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    client = _ScriptedClient(
        [
            _response("not json", tokens_in=1, tokens_out=2, cost_usd=0.01),
            _response("still not json", tokens_in=3, tokens_out=4, cost_usd=0.03),
            _response("final bad json", tokens_in=5, tokens_out=6, cost_usd=0.05),
        ]
    )

    with pytest.raises(PlannerError) as exc_info:
        Planner(client, store, max_repairs=2).plan("run-exhausted", _task())

    assert exc_info.value.reason is PlannerErrorReason.repair_exhausted
    assert len(client.calls) == 3
    [event] = store.read("run-exhausted")
    assert event.kind is TraceEventKind.error
    assert event.payload["reason"] == "repair_exhausted"
    assert event.payload["attempts"] == 3
    assert event.tokens_in == 9
    assert event.tokens_out == 12
    assert event.cost_usd == 0.09


def test_planner__empty_plan_exhaustion_has_empty_plan_reason(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    client = _ScriptedClient([_response('{"steps":[]}')])

    with pytest.raises(PlannerError) as exc_info:
        Planner(client, store, max_repairs=0).plan("run-empty", _task())

    assert exc_info.value.reason is PlannerErrorReason.empty_plan
    [event] = store.read("run-empty")
    assert event.kind is TraceEventKind.error
    assert event.payload["reason"] == "empty_plan"


def test_planner__refusal_is_traced_as_error_without_retry(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    client = _ScriptedClient(
        [
            _response("I cannot help with that.", stop_reason=StopReason.refusal),
            _response(_draft_json()),
        ]
    )

    with pytest.raises(PlannerError) as exc_info:
        Planner(client, store).plan("run-refusal", _task())

    assert exc_info.value.reason is PlannerErrorReason.refusal
    # No retry: the second scripted response is never consumed.
    assert len(client.calls) == 1
    # Trace-first: a terminal refusal is still recorded as an error event.
    [event] = store.read("run-refusal")
    assert event.kind is TraceEventKind.error
    assert event.payload["reason"] == "refusal"
