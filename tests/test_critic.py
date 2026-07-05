import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from app.agent.critic import Critic, CriticError, CriticErrorReason
from app.agent.state import PlanStep
from app.schemas.agent_io import StepOutcome, StepResult, VerdictDecision
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


def _step() -> PlanStep:
    return PlanStep(
        index=2,
        intent="Verify the parse_date implementation location.",
        suggested_tools=["search_code", "read_file"],
        success_check="The implementation file and line are cited from raw tool output.",
    )


def _step_result(
    *,
    status: StepOutcome = StepOutcome.completed,
    findings: str = "parse_date is implemented in src/sample_pkg/dates.py:6.",
) -> StepResult:
    return StepResult(
        step_index=2,
        status=status,
        findings=findings,
        evidence=["src/sample_pkg/dates.py:6"],
        tool_calls=2,
        usage=Usage(tokens_in=11, tokens_out=7, cost_usd=0.04),
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


def _verdict_json(
    decision: str = "proceed",
    *,
    reason: str = "The cited raw evidence proves the success check.",
    hint: str = "",
) -> str:
    return json.dumps({"decision": decision, "reason": reason, "hint": hint})


def test_critic__writes_proceed_verdict_to_real_trace_store(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    client = _ScriptedClient(
        [
            _response(
                f"```json\n{_verdict_json()}\n```",
                tokens_in=17,
                tokens_out=11,
                cost_usd=0.23,
            )
        ]
    )

    verdict = Critic(client, store).critique("run-critic", _step(), _step_result())

    assert verdict.step_index == 2
    assert verdict.decision is VerdictDecision.proceed
    assert verdict.reason == "The cited raw evidence proves the success check."
    assert verdict.hint == ""
    assert verdict.usage == Usage(tokens_in=17, tokens_out=11, cost_usd=0.23)

    events = store.read("run-critic")
    assert len(events) == 1
    [event] = events
    assert event.kind is TraceEventKind.critic_verdict
    assert event.payload["summary"] == "Critic verdict for step 2: proceed."
    assert event.payload["step_index"] == 2
    assert event.payload["decision"] == "proceed"
    assert event.payload["reason"] == verdict.reason
    assert event.payload["hint"] == ""
    assert event.payload["attempts"] == 1
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
    assert "Output contract" in call.system


@pytest.mark.parametrize(
    ("decision", "hint"),
    [
        ("retry", "Read the cited file again and include the exact line."),
        ("replan", "The planned file does not exist; choose a different lookup path."),
    ],
)
def test_critic__retry_and_replan_verdicts_keep_hint_in_payload(
    tmp_path: Path,
    decision: str,
    hint: str,
) -> None:
    store = TraceStore(tmp_path)
    client = _ScriptedClient(
        [_response(_verdict_json(decision, reason="More verification is needed.", hint=hint))]
    )

    verdict = Critic(client, store).critique(f"run-{decision}", _step(), _step_result())

    assert verdict.decision is VerdictDecision(decision)
    assert verdict.hint == hint
    [event] = store.read(f"run-{decision}")
    assert event.kind is TraceEventKind.critic_verdict
    assert event.payload["decision"] == decision
    assert event.payload["hint"] == hint


@pytest.mark.parametrize(
    "invalid_reply",
    [
        "not json",
        _verdict_json("abort", reason="Invalid route."),
    ],
)
def test_critic__repair_message_reaches_next_prompt_and_accumulates_usage(
    tmp_path: Path,
    invalid_reply: str,
) -> None:
    store = TraceStore(tmp_path)
    client = _ScriptedClient(
        [
            _response(invalid_reply, tokens_in=3, tokens_out=4, cost_usd=0.03),
            _response(
                _verdict_json("retry", hint="Gather direct raw evidence."),
                tokens_in=5,
                tokens_out=6,
                cost_usd=0.05,
            ),
        ]
    )

    verdict = Critic(client, store).critique("run-repair", _step(), _step_result())

    assert verdict.decision is VerdictDecision.retry
    assert verdict.usage.tokens_in == 8
    assert verdict.usage.tokens_out == 10
    assert verdict.usage.cost_usd == pytest.approx(0.08)
    assert len(client.calls) == 2
    second_prompt_messages = client.calls[1].messages
    assert second_prompt_messages[-2] == LLMMessage(role=Role.assistant, content=invalid_reply)
    correction = second_prompt_messages[-1]
    assert correction.role is Role.user
    assert invalid_reply in correction.content
    assert "Validation error:" in correction.content

    [event] = store.read("run-repair")
    assert event.kind is TraceEventKind.critic_verdict
    assert event.payload["decision"] == "retry"
    assert event.tokens_in == 8
    assert event.tokens_out == 10
    assert event.cost_usd == pytest.approx(0.08)


def test_critic__refusal_traces_error_and_raises_without_retry(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    client = _ScriptedClient(
        [
            _response(
                "I cannot judge that.",
                stop_reason=StopReason.refusal,
                tokens_in=9,
                tokens_out=2,
                cost_usd=0.09,
            ),
            _response(_verdict_json()),
        ]
    )

    with pytest.raises(CriticError) as exc_info:
        Critic(client, store).critique("run-refusal", _step(), _step_result())

    assert exc_info.value.reason is CriticErrorReason.refusal
    assert len(client.calls) == 1
    [event] = store.read("run-refusal")
    assert event.kind is TraceEventKind.error
    assert event.payload["reason"] == "refusal"
    assert event.payload["step_index"] == 2
    assert event.payload["attempts"] == 1
    assert event.tokens_in == 9
    assert event.tokens_out == 2
    assert event.cost_usd == 0.09


def test_critic__repair_exhaustion_traces_error_and_raises(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    client = _ScriptedClient(
        [
            _response("not json", tokens_in=1, tokens_out=2, cost_usd=0.01),
            _response('{"decision":"abort"}', tokens_in=3, tokens_out=4, cost_usd=0.03),
            _response('{"decision":"continue"}', tokens_in=5, tokens_out=6, cost_usd=0.05),
        ]
    )

    with pytest.raises(CriticError) as exc_info:
        Critic(client, store, max_repairs=2).critique(
            "run-exhausted",
            _step(),
            _step_result(),
        )

    assert exc_info.value.reason is CriticErrorReason.repair_exhausted
    assert len(client.calls) == 3
    [event] = store.read("run-exhausted")
    assert event.kind is TraceEventKind.error
    assert event.payload["reason"] == "repair_exhausted"
    assert event.payload["attempts"] == 3
    assert event.tokens_in == 9
    assert event.tokens_out == 12
    assert event.cost_usd == pytest.approx(0.09)


def test_critic__raw_evidence_success_check_and_findings_reach_prompt(
    tmp_path: Path,
) -> None:
    store = TraceStore(tmp_path)
    client = _ScriptedClient([_response(_verdict_json())])
    step = _step()
    result = _step_result(
        status=StepOutcome.incomplete,
        findings="The executor could not prove the target line.",
    )
    raw_evidence = [
        "search_code returned src/sample_pkg/dates.py:6:def parse_date(value):",
        "read_file returned lines 1-12 from src/sample_pkg/dates.py.",
    ]

    Critic(client, store).critique(
        "run-prompt",
        step,
        result,
        raw_evidence=raw_evidence,
        repo_overview="Small Python package.",
    )

    [call] = client.calls
    assert call.system is not None
    assert "Role & mission" in call.system
    assert call.tools is None
    assert call.temperature is None
    prompt = "\n\n".join(message.content for message in call.messages)
    assert step.success_check in prompt
    assert result.findings in prompt
    assert "incomplete" in prompt
    assert "Small Python package." in prompt
    for evidence in raw_evidence:
        assert evidence in prompt


def test_critic__shares_trace_store_sequence_and_timeline(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    store.append("run-seq", TraceEventKind.plan, {"summary": "Planned 1 step."})
    client = _ScriptedClient([_response(_verdict_json("proceed"))])

    Critic(client, store).critique("run-seq", _step(), _step_result())

    events = store.read("run-seq")
    assert [event.seq for event in events] == [0, 1]
    assert [event.kind for event in events] == [
        TraceEventKind.plan,
        TraceEventKind.critic_verdict,
    ]
    timeline = render_timeline(events)
    assert "critic_verdict - Critic verdict for step 2: proceed." in timeline
