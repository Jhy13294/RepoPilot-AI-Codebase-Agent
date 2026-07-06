import inspect
import json
from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from app.agent.reporter import Reporter, ReporterError, ReporterErrorReason
from app.agent.state import TaskSpec
from app.schemas.agent_io import ReportConfidence
from app.schemas.llm_io import LLMMessage, LLMResponse, Role, StopReason, Usage
from app.services.llm_client import ToolSchema


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
        prompt="Explain why parse_date is failing.",
        repo=".",
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


def _report_json(
    *,
    headline: str = "parse_date failure is explained",
    analysis: str = "The run found the failing behavior and cited the relevant file.",
    confidence: str = "high",
    open_questions: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "headline": headline,
            "analysis": analysis,
            "confidence": confidence,
            "open_questions": open_questions or [],
        }
    )


def test_reporter__happy_path_returns_analysis_report_and_grounded_prompt() -> None:
    final_findings = "Step 0 findings: parse_date rejects ISO dates."
    timeline_digest = "#0 plan - Planned 1 step.\n#1 tool_result - Executor step 0 completed."
    failure_summary = "No failure."
    content = _report_json(open_questions=["Should timezone formats be allowed?"])
    client = _ScriptedClient(
        [
            _response(
                f"```json\n{content}\n```",
                tokens_in=17,
                tokens_out=11,
                cost_usd=0.23,
            )
        ]
    )

    report = Reporter(client).report(
        _task(),
        outcome="succeeded",
        final_findings=final_findings,
        timeline_digest=timeline_digest,
        failure_summary=failure_summary,
    )

    assert report.headline == "parse_date failure is explained"
    assert report.confidence is ReportConfidence.high
    assert report.open_questions == ["Should timezone formats be allowed?"]
    assert report.usage == Usage(tokens_in=17, tokens_out=11, cost_usd=0.23)

    [call] = client.calls
    assert call.tools is None
    assert call.temperature is None
    assert call.max_tokens is None
    assert call.system is not None
    assert "Role & mission" in call.system
    assert "Evidence boundary" in call.system
    assert "Report policy" in call.system
    assert "Output contract" in call.system
    prompt = call.messages[0].content
    assert _task().prompt in prompt
    assert final_findings in prompt
    assert timeline_digest in prompt
    assert failure_summary in prompt
    assert "succeeded" in prompt


def test_reporter__repair_message_reaches_next_prompt_and_accumulates_usage() -> None:
    invalid_reply = "not json"
    client = _ScriptedClient(
        [
            _response(invalid_reply, tokens_in=3, tokens_out=4, cost_usd=0.03),
            _response(_report_json(confidence="medium"), tokens_in=5, tokens_out=6, cost_usd=0.05),
        ]
    )

    report = Reporter(client).report(
        _task(),
        outcome="succeeded",
        final_findings="The run found evidence.",
        timeline_digest="#0 plan - Planned 1 step.",
    )

    assert report.confidence is ReportConfidence.medium
    assert report.usage == Usage(tokens_in=8, tokens_out=10, cost_usd=0.08)
    assert len(client.calls) == 2
    second_prompt_messages = client.calls[1].messages
    assert second_prompt_messages[-2] == LLMMessage(role=Role.assistant, content=invalid_reply)
    correction = second_prompt_messages[-1]
    assert correction.role is Role.user
    assert invalid_reply in correction.content
    assert "Validation error:" in correction.content
    assert "reporter schema" in correction.content


def test_reporter__repair_exhaustion_raises_with_usage() -> None:
    client = _ScriptedClient(
        [
            _response("not json", tokens_in=1, tokens_out=2, cost_usd=0.01),
            _response("still not json", tokens_in=3, tokens_out=4, cost_usd=0.03),
            _response("final bad json", tokens_in=5, tokens_out=6, cost_usd=0.05),
        ]
    )

    with pytest.raises(ReporterError) as exc_info:
        Reporter(client, max_repairs=2).report(
            _task(),
            outcome="failed",
            final_findings="",
            timeline_digest="#0 error - Planner failed.",
            failure_summary="Planning failed.",
        )

    assert exc_info.value.reason is ReporterErrorReason.repair_exhausted
    assert exc_info.value.usage == Usage(tokens_in=9, tokens_out=12, cost_usd=0.09)
    assert len(client.calls) == 3


def test_reporter__refusal_raises_without_retry_and_carries_usage() -> None:
    client = _ScriptedClient(
        [
            _response(
                "I cannot report on that.",
                stop_reason=StopReason.refusal,
                tokens_in=9,
                tokens_out=2,
                cost_usd=0.09,
            ),
            _response(_report_json()),
        ]
    )

    with pytest.raises(ReporterError) as exc_info:
        Reporter(client).report(
            _task(),
            outcome="failed",
            final_findings="",
            timeline_digest="#0 error - Planner failed.",
            failure_summary="Planning failed.",
        )

    assert exc_info.value.reason is ReporterErrorReason.refusal
    assert exc_info.value.usage == Usage(tokens_in=9, tokens_out=2, cost_usd=0.09)
    assert len(client.calls) == 1


def test_reporter__invalid_confidence_repairs_to_valid_report() -> None:
    invalid_reply = _report_json(confidence="certain")
    client = _ScriptedClient(
        [
            _response(invalid_reply, tokens_in=2, tokens_out=3, cost_usd=0.02),
            _response(_report_json(confidence="low"), tokens_in=4, tokens_out=5, cost_usd=0.04),
        ]
    )

    report = Reporter(client).report(
        _task(),
        outcome="succeeded",
        final_findings="The run found partial evidence.",
        timeline_digest="#0 plan - Planned 1 step.",
    )

    assert report.confidence is ReportConfidence.low
    assert report.usage == Usage(tokens_in=6, tokens_out=8, cost_usd=0.06)
    assert len(client.calls) == 2
    assert "confidence" in client.calls[1].messages[-1].content


def test_reporter__public_signatures_do_not_take_stateful_dependencies() -> None:
    init_params = set(inspect.signature(Reporter).parameters)
    report_params = set(inspect.signature(Reporter.report).parameters)

    assert init_params == {"client", "max_repairs"}
    assert "store" not in report_params
    assert "registry" not in report_params
    assert "jail" not in report_params
    assert "run_id" not in report_params
