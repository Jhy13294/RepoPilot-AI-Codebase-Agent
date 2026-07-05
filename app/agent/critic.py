"""Critic role with constrained JSON output and trace emission."""

import json
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from app.agent.state import PlanStep
from app.agent.usage import UsageAccumulator
from app.schemas.agent_io import StepResult, Verdict, VerdictDecision
from app.schemas.llm_io import LLMMessage, Role, StopReason, Usage
from app.schemas.trace import TraceEventKind
from app.services.llm_client import LLMClient
from app.storage.trace_store import TraceStore

_JSON_FENCE_RE = re.compile(r"\A```json\s*(?P<body>.*?)\s*```\Z", flags=re.DOTALL)


class CriticErrorReason(StrEnum):
    """Terminal critic failure categories."""

    refusal = "refusal"
    repair_exhausted = "repair_exhausted"


class CriticError(Exception):
    """Raised when the critic cannot produce a valid verdict."""

    def __init__(self, reason: CriticErrorReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class _VerdictDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: VerdictDecision
    reason: str
    hint: str = ""


@dataclass(frozen=True, slots=True)
class _ParseFailure:
    reason: CriticErrorReason
    message: str


class Critic:
    """Verify one executor result without dispatching any tools."""

    def __init__(
        self,
        client: LLMClient,
        store: TraceStore,
        *,
        max_repairs: int = 2,
    ) -> None:
        if max_repairs < 0:
            raise ValueError("max_repairs must be non-negative.")

        self._client = client
        self._store = store
        self._system_prompt = _critic_system_prompt()
        self._max_repairs = max_repairs

    def critique(
        self,
        run_id: str,
        step: PlanStep,
        step_result: StepResult,
        *,
        raw_evidence: Sequence[str] = (),
        repo_overview: str | None = None,
    ) -> Verdict:
        """Produce a validated verdict and persist one critic trace event."""
        messages = [
            LLMMessage(
                role=Role.user,
                content=_critic_user_prompt(
                    step,
                    step_result,
                    raw_evidence=raw_evidence,
                    repo_overview=repo_overview,
                ),
            )
        ]
        usage = UsageAccumulator()
        attempts = 0
        max_attempts = self._max_repairs + 1
        last_failure: _ParseFailure | None = None
        start = time.perf_counter()

        while attempts < max_attempts:
            response = self._client.complete(
                messages,
                tools=None,
                system=self._system_prompt,
                temperature=None,
            )
            attempts += 1
            usage.add(response.usage)

            if response.stop_reason is StopReason.refusal:
                refusal_failure = _ParseFailure(
                    reason=CriticErrorReason.refusal,
                    message=_refusal_message(response.message.content),
                )
                self._append_error_event(
                    run_id=run_id,
                    step=step,
                    failure=refusal_failure,
                    attempts=attempts,
                    usage=usage.snapshot(),
                    latency_ms=_elapsed_ms(start),
                )
                raise CriticError(refusal_failure.reason, refusal_failure.message)

            content = response.message.content
            draft, failure = _parse_verdict_draft(content)
            if draft is not None:
                result_usage = usage.snapshot()
                verdict = Verdict(
                    step_index=step.index,
                    decision=draft.decision,
                    reason=draft.reason,
                    hint=draft.hint,
                    usage=result_usage,
                )
                self._append_verdict_event(
                    run_id=run_id,
                    verdict=verdict,
                    attempts=attempts,
                    latency_ms=_elapsed_ms(start),
                )
                return verdict

            last_failure = failure
            if attempts < max_attempts:
                messages.append(LLMMessage(role=Role.assistant, content=content))
                messages.append(_repair_message(content, failure))

        assert last_failure is not None
        result_usage = usage.snapshot()
        self._append_error_event(
            run_id=run_id,
            step=step,
            failure=last_failure,
            attempts=attempts,
            usage=result_usage,
            latency_ms=_elapsed_ms(start),
        )
        raise CriticError(
            last_failure.reason,
            f"Critic failed after {attempts} attempt(s): {last_failure.message}",
        )

    def _append_verdict_event(
        self,
        *,
        run_id: str,
        verdict: Verdict,
        attempts: int,
        latency_ms: int,
    ) -> None:
        payload: dict[str, JsonValue] = {
            "summary": f"Critic verdict for step {verdict.step_index}: {verdict.decision.value}.",
            "step_index": verdict.step_index,
            "decision": verdict.decision.value,
            "reason": verdict.reason,
            "hint": verdict.hint,
            "attempts": attempts,
        }
        self._store.append(
            run_id,
            TraceEventKind.critic_verdict,
            payload,
            latency_ms=latency_ms,
            tokens_in=verdict.usage.tokens_in,
            tokens_out=verdict.usage.tokens_out,
            cost_usd=verdict.usage.cost_usd,
        )

    def _append_error_event(
        self,
        *,
        run_id: str,
        step: PlanStep,
        failure: _ParseFailure,
        attempts: int,
        usage: Usage,
        latency_ms: int,
    ) -> None:
        payload: dict[str, JsonValue] = {
            "summary": f"Critic failed: {failure.reason.value}.",
            "message": failure.message,
            "reason": failure.reason.value,
            "step_index": step.index,
            "attempts": attempts,
        }
        self._store.append(
            run_id,
            TraceEventKind.error,
            payload,
            latency_ms=latency_ms,
            tokens_in=usage.tokens_in,
            tokens_out=usage.tokens_out,
            cost_usd=usage.cost_usd,
        )


def _critic_system_prompt() -> str:
    return "\n\n".join(
        [
            (
                "Role & mission\n"
                "You are RepoPilot, a task-oriented codebase agent. As the Critic, independently "
                "verify whether one executed step satisfies its success_check."
            ),
            (
                "Evidence standard\n"
                "Grade evidence, not effort. Compare the Executor findings against the raw "
                "evidence and the success_check. Do not trust the Executor's self-report. An "
                "incomplete step_result status is a strong signal against proceed."
            ),
            (
                "Decision policy\n"
                "Use proceed only when the raw evidence proves the success_check. Use retry when "
                "the direction is plausible but evidence is missing or ordinary tool errors need "
                "another attempt. Use replan when the step is wrong, impossible, or no longer "
                "useful. For retry, hint must guide the next Executor attempt. For replan, hint "
                "must summarize the failure for the Planner. Proceed may use an empty hint."
            ),
            (
                "Output contract\n"
                "Return only a JSON object shaped exactly as "
                '{"decision":"proceed|retry|replan","reason":str,"hint":str}. Do not include '
                "markdown, prose, comments, step_index, usage, or extra fields."
            ),
        ]
    )


def _critic_user_prompt(
    step: PlanStep,
    step_result: StepResult,
    *,
    raw_evidence: Sequence[str],
    repo_overview: str | None,
) -> str:
    return "\n\n".join(
        [
            f"Plan step JSON:\n{step.model_dump_json(indent=2)}",
            f"Success check:\n{step.success_check}",
            f"Executor step result JSON:\n{step_result.model_dump_json(indent=2)}",
            f"Executor findings:\n{step_result.findings}",
            f"Raw evidence:\n{_format_raw_evidence(raw_evidence)}",
            f"Repo overview:\n{repo_overview if repo_overview is not None else 'Not supplied.'}",
        ]
    )


def _format_raw_evidence(raw_evidence: Sequence[str]) -> str:
    if not raw_evidence:
        return "None supplied."
    return "\n".join(
        f"[{index}] {evidence}" for index, evidence in enumerate(raw_evidence, start=1)
    )


def _parse_verdict_draft(content: str) -> tuple[_VerdictDraft | None, _ParseFailure]:
    candidate = _strip_json_fence(content.strip())
    try:
        parsed: object = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, _ParseFailure(
            reason=CriticErrorReason.repair_exhausted,
            message=f"json.loads error: {exc}",
        )

    try:
        return _VerdictDraft.model_validate(parsed), _ParseFailure(
            reason=CriticErrorReason.repair_exhausted,
            message="",
        )
    except ValidationError as exc:
        return None, _ParseFailure(
            reason=CriticErrorReason.repair_exhausted,
            message=str(exc),
        )


def _strip_json_fence(content: str) -> str:
    match = _JSON_FENCE_RE.fullmatch(content)
    if match is None:
        return content
    return match.group("body").strip()


def _repair_message(content: str, failure: _ParseFailure) -> LLMMessage:
    return LLMMessage(
        role=Role.user,
        content=(
            "Your previous critic verdict was invalid.\n"
            "Bad reply:\n"
            f"{content}\n\n"
            "Validation error:\n"
            f"{failure.message}\n\n"
            "Return only corrected JSON with the exact critic schema: "
            '{"decision":"proceed|retry|replan","reason":str,"hint":str}.'
        ),
    )


def _elapsed_ms(start: float) -> int:
    return max(0, round((time.perf_counter() - start) * 1000))


def _refusal_message(content: str) -> str:
    if content:
        return f"Critic refused: {content}"
    return "Critic refused to produce a verdict."
