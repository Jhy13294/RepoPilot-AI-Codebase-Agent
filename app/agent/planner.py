"""Planner role with constrained JSON output and trace emission."""

import json
import re
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from app.agent.state import PlanStep, TaskSpec
from app.agent.usage import UsageAccumulator
from app.schemas.llm_io import LLMMessage, Role, StopReason, Usage
from app.schemas.trace import TraceEventKind
from app.services.llm_client import LLMClient
from app.storage.trace_store import TraceStore

Plan = list[PlanStep]

_JSON_FENCE_RE = re.compile(r"\A```json\s*(?P<body>.*?)\s*```\Z", flags=re.DOTALL)


class PlannerErrorReason(StrEnum):
    """Terminal planner failure categories."""

    refusal = "refusal"
    repair_exhausted = "repair_exhausted"
    empty_plan = "empty_plan"


class PlannerError(Exception):
    """Raised when the planner cannot produce a valid plan."""

    def __init__(self, reason: PlannerErrorReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class PlanResult(BaseModel):
    """Validated planner output plus accounting metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan: Plan = Field(min_length=1)
    attempts: int = Field(ge=1)
    usage: Usage
    replanned: bool


class _PlanStepDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    intent: str
    suggested_tools: list[str]
    success_check: str


class _PlanDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    steps: list[_PlanStepDraft] = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class _ParseFailure:
    reason: PlannerErrorReason
    message: str


class Planner:
    """Generate an execution plan without dispatching any tools."""

    def __init__(
        self,
        client: LLMClient,
        store: TraceStore,
        *,
        tools_doc: str = "",
        max_repairs: int = 2,
    ) -> None:
        if max_repairs < 0:
            raise ValueError("max_repairs must be non-negative.")

        self._client = client
        self._store = store
        self._system_prompt = _planner_system_prompt(tools_doc)
        self._max_repairs = max_repairs

    def plan(
        self,
        run_id: str,
        task: TaskSpec,
        *,
        repo_overview: str | None = None,
        failure_summary: str | None = None,
    ) -> PlanResult:
        """Produce a validated plan and persist one plan or replan trace event."""
        messages = [
            LLMMessage(
                role=Role.user,
                content=_planner_user_prompt(
                    task,
                    repo_overview=repo_overview,
                    failure_summary=failure_summary,
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
                    reason=PlannerErrorReason.refusal,
                    message=_refusal_message(response.message.content),
                )
                self._append_error_event(
                    run_id=run_id,
                    failure=refusal_failure,
                    attempts=attempts,
                    usage=usage.snapshot(),
                    latency_ms=_elapsed_ms(start),
                )
                raise PlannerError(refusal_failure.reason, refusal_failure.message)

            content = response.message.content
            draft, failure = _parse_plan_draft(content)
            if draft is not None:
                plan = _materialize_plan(draft)
                result_usage = usage.snapshot()
                replanned = failure_summary is not None
                self._append_plan_event(
                    run_id=run_id,
                    plan=plan,
                    attempts=attempts,
                    usage=result_usage,
                    latency_ms=_elapsed_ms(start),
                    replanned=replanned,
                )
                return PlanResult(
                    plan=plan,
                    attempts=attempts,
                    usage=result_usage,
                    replanned=replanned,
                )

            last_failure = failure
            if attempts < max_attempts:
                messages.append(LLMMessage(role=Role.assistant, content=content))
                messages.append(_repair_message(content, failure))

        assert last_failure is not None
        result_usage = usage.snapshot()
        self._append_error_event(
            run_id=run_id,
            failure=last_failure,
            attempts=attempts,
            usage=result_usage,
            latency_ms=_elapsed_ms(start),
        )
        raise PlannerError(
            last_failure.reason,
            f"Planner failed after {attempts} attempt(s): {last_failure.message}",
        )

    def _append_plan_event(
        self,
        *,
        run_id: str,
        plan: Plan,
        attempts: int,
        usage: Usage,
        latency_ms: int,
        replanned: bool,
    ) -> None:
        label = "Replanned" if replanned else "Planned"
        steps_payload = [step.model_dump(mode="json") for step in plan]
        payload: dict[str, JsonValue] = {
            "summary": f"{label} {len(plan)} step(s).",
            "steps": cast(JsonValue, steps_payload),
            "attempts": attempts,
        }
        self._store.append(
            run_id,
            TraceEventKind.replan if replanned else TraceEventKind.plan,
            payload,
            latency_ms=latency_ms,
            tokens_in=usage.tokens_in,
            tokens_out=usage.tokens_out,
            cost_usd=usage.cost_usd,
        )

    def _append_error_event(
        self,
        *,
        run_id: str,
        failure: _ParseFailure,
        attempts: int,
        usage: Usage,
        latency_ms: int,
    ) -> None:
        payload: dict[str, JsonValue] = {
            "summary": f"Planner failed: {failure.reason.value}.",
            "message": failure.message,
            "reason": failure.reason.value,
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


def _planner_system_prompt(tools_doc: str) -> str:
    tool_section = tools_doc.strip() or "No tool documentation was supplied."
    return "\n\n".join(
        [
            (
                "Role & mission\n"
                "You are RepoPilot, a codebase task agent. You are not a chatbot. "
                "As the Planner, turn the task into a short, executable plan for later roles."
            ),
            (
                "Hard rules\n"
                "Do not call tools or ask for tool results. Do not invent file paths, line "
                "numbers, APIs, or repository behavior. Prefer steps that let the Executor read "
                "before writing. High-risk tools may pause for human approval, so plan around "
                "approval boundaries. For fix tasks, create the work branch first, read before "
                "writing, propose the patch before applying it, and land the patch only on the "
                "work branch. After applying a patch, run the tests for verification and finish "
                "only when the test report has zero failures. Stop the plan when each step's "
                "success_check can prove the task is met."
            ),
            f"Tool documentation\n{tool_section}",
            (
                "Output contract\n"
                "Return only a JSON object shaped exactly as "
                '{"steps":[{"intent":str,"suggested_tools":[str],"success_check":str}]}. '
                "Do not include markdown, prose, index, status, comments, or extra fields. "
                "The steps list must contain at least one step."
            ),
        ]
    )


def _planner_user_prompt(
    task: TaskSpec,
    *,
    repo_overview: str | None,
    failure_summary: str | None,
) -> str:
    sections = [
        f"Task JSON:\n{task.model_dump_json(indent=2)}",
        f"Repo overview:\n{repo_overview if repo_overview is not None else 'Not supplied.'}",
    ]
    if failure_summary is None:
        sections.append("Failure summary:\nNone. Produce an initial plan.")
    else:
        sections.append(f"Failure summary for replan:\n{failure_summary}")
    return "\n\n".join(sections)


def _parse_plan_draft(content: str) -> tuple[_PlanDraft | None, _ParseFailure]:
    candidate = _strip_json_fence(content.strip())
    try:
        parsed: object = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, _ParseFailure(
            reason=PlannerErrorReason.repair_exhausted,
            message=f"json.loads error: {exc}",
        )

    try:
        return _PlanDraft.model_validate(parsed), _ParseFailure(
            reason=PlannerErrorReason.repair_exhausted,
            message="",
        )
    except ValidationError as exc:
        return None, _ParseFailure(
            reason=_validation_failure_reason(parsed, exc),
            message=str(exc),
        )


def _strip_json_fence(content: str) -> str:
    match = _JSON_FENCE_RE.fullmatch(content)
    if match is None:
        return content
    return match.group("body").strip()


def _validation_failure_reason(
    parsed: object,
    validation_error: ValidationError,
) -> PlannerErrorReason:
    if isinstance(parsed, dict) and parsed.get("steps") == []:
        return PlannerErrorReason.empty_plan

    for error in validation_error.errors():
        if tuple(error.get("loc", ())) == ("steps",) and error.get("type") == "too_short":
            return PlannerErrorReason.empty_plan

    return PlannerErrorReason.repair_exhausted


def _repair_message(content: str, failure: _ParseFailure) -> LLMMessage:
    return LLMMessage(
        role=Role.user,
        content=(
            "Your previous planner output was invalid.\n"
            "Bad reply:\n"
            f"{content}\n\n"
            "Validation error:\n"
            f"{failure.message}\n\n"
            "Return only corrected JSON with the exact planner schema."
        ),
    )


def _materialize_plan(draft: _PlanDraft) -> Plan:
    return [
        PlanStep(
            index=index,
            intent=step.intent,
            suggested_tools=step.suggested_tools,
            success_check=step.success_check,
        )
        for index, step in enumerate(draft.steps)
    ]


def _elapsed_ms(start: float) -> int:
    return max(0, round((time.perf_counter() - start) * 1000))


def _refusal_message(content: str) -> str:
    if content:
        return f"Planner refused: {content}"
    return "Planner refused to produce a plan."
