"""Executor role for one bounded plan-step tool loop."""

import json
import re
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from pydantic import JsonValue, ValidationError

from app.agent.state import PlanStep
from app.agent.usage import UsageAccumulator
from app.safety.path_jail import PathJail
from app.schemas.agent_io import StepOutcome, StepResult
from app.schemas.llm_io import LLMMessage, LLMResponse, Role, StopReason, Usage
from app.schemas.tool_io import ErrorType, ToolResult
from app.schemas.trace import TraceEventKind
from app.services.llm_client import LLMClient, ToolSchema
from app.storage.trace_store import TraceStore
from app.tools.base import ToolContext
from app.tools.registry import ToolRegistry

_JSON_FENCE_RE = re.compile(r"\A```json\s*(?P<body>.*?)\s*```\Z", flags=re.DOTALL)

# Hard ceiling on consecutive completions that make no progress -- neither dispatching a tool
# nor attempting a synthesis (a pause_turn, or a tool_use turn with no tool calls). It guarantees
# termination without trusting the model to stop: RepoPilot enforces budgets in code, never by
# prompt. Any progress (a dispatch or a synthesis attempt) resets the counter.
_MAX_STALLED_COMPLETIONS = 3


class ExecutorErrorReason(StrEnum):
    """Terminal executor failure categories."""

    refusal = "refusal"
    synthesis_repair_exhausted = "synthesis_repair_exhausted"


class ExecutorError(Exception):
    """Raised when the executor hits a hard failure."""

    def __init__(self, reason: ExecutorErrorReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class _SynthesisFailure:
    reason: ExecutorErrorReason
    message: str


class Executor:
    """Execute one plan step through a bounded ReAct micro-loop."""

    def __init__(
        self,
        client: LLMClient,
        registry: ToolRegistry,
        jail: PathJail,
        store: TraceStore,
        *,
        max_tool_calls: int = 8,
        max_arg_repairs: int = 2,
        max_output_repairs: int = 2,
    ) -> None:
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls must be at least one.")
        if max_arg_repairs < 0:
            raise ValueError("max_arg_repairs must be non-negative.")
        if max_output_repairs < 0:
            raise ValueError("max_output_repairs must be non-negative.")

        self._client = client
        self._registry = registry
        self._jail = jail
        self._store = store
        self._max_tool_calls = max_tool_calls
        self._max_arg_repairs = max_arg_repairs
        self._max_output_repairs = max_output_repairs
        self._system_prompt = _executor_system_prompt()

    def execute_step(
        self,
        run_id: str,
        step: PlanStep,
        *,
        scratchpad: str = "",
        repo_overview: str | None = None,
    ) -> StepResult:
        """Execute one plan step and persist exactly one terminal executor event."""
        messages = [
            LLMMessage(
                role=Role.user,
                content=_executor_user_prompt(
                    step,
                    scratchpad=scratchpad,
                    repo_overview=repo_overview,
                ),
            )
        ]
        context = ToolContext(run_id=run_id, jail=self._jail)
        usage = UsageAccumulator()
        tool_calls = 0
        completion_calls = 0
        invalid_arg_errors = 0
        output_attempts = 0
        stalled = 0
        last_synthesis_failure: _SynthesisFailure | None = None
        start = time.perf_counter()

        while True:
            response = self._complete(messages)
            completion_calls += 1
            usage.add(response.usage)

            match response.stop_reason:
                case StopReason.end_turn:
                    output_attempts += 1
                    result, failure = _parse_step_result(
                        response.message.content,
                        step_index=step.index,
                        tool_calls=tool_calls,
                        usage=usage.snapshot(),
                    )
                    if result is not None:
                        self._append_tool_result_event(
                            run_id=run_id,
                            result=result,
                            latency_ms=_elapsed_ms(start),
                            reason=None,
                        )
                        return result

                    last_synthesis_failure = failure
                    if output_attempts <= self._max_output_repairs:
                        messages.append(response.message)
                        messages.append(_repair_message(response.message.content, failure))
                        continue

                    assert last_synthesis_failure is not None
                    self._raise_synthesis_error(
                        run_id=run_id,
                        step=step,
                        failure=last_synthesis_failure,
                        attempts=output_attempts,
                        completion_calls=completion_calls,
                        tool_calls=tool_calls,
                        usage=usage.snapshot(),
                        latency_ms=_elapsed_ms(start),
                    )

                case StopReason.refusal:
                    failure = _SynthesisFailure(
                        reason=ExecutorErrorReason.refusal,
                        message=_refusal_message(response.message.content),
                    )
                    self._append_error_event(
                        run_id=run_id,
                        step=step,
                        failure=failure,
                        attempts=completion_calls,
                        completion_calls=completion_calls,
                        tool_calls=tool_calls,
                        usage=usage.snapshot(),
                        latency_ms=_elapsed_ms(start),
                    )
                    raise ExecutorError(failure.reason, failure.message)

                case StopReason.max_tokens:
                    messages.append(response.message)
                    continuation = self._complete(messages)
                    completion_calls += 1
                    usage.add(continuation.usage)
                    combined_content = _combine_content(
                        response.message.content,
                        continuation.message.content,
                    )

                    if continuation.stop_reason is StopReason.end_turn:
                        output_attempts += 1
                        result, failure = _parse_step_result(
                            combined_content,
                            step_index=step.index,
                            tool_calls=tool_calls,
                            usage=usage.snapshot(),
                        )
                        if result is not None:
                            self._append_tool_result_event(
                                run_id=run_id,
                                result=result,
                                latency_ms=_elapsed_ms(start),
                                reason=None,
                            )
                            return result

                        last_synthesis_failure = failure
                        if output_attempts <= self._max_output_repairs:
                            messages.append(
                                LLMMessage(role=Role.assistant, content=combined_content)
                            )
                            messages.append(_repair_message(combined_content, failure))
                            continue

                        assert last_synthesis_failure is not None
                        self._raise_synthesis_error(
                            run_id=run_id,
                            step=step,
                            failure=last_synthesis_failure,
                            attempts=output_attempts,
                            completion_calls=completion_calls,
                            tool_calls=tool_calls,
                            usage=usage.snapshot(),
                            latency_ms=_elapsed_ms(start),
                        )

                    if continuation.stop_reason is StopReason.refusal:
                        failure = _SynthesisFailure(
                            reason=ExecutorErrorReason.refusal,
                            message=_refusal_message(continuation.message.content),
                        )
                        self._append_error_event(
                            run_id=run_id,
                            step=step,
                            failure=failure,
                            attempts=completion_calls,
                            completion_calls=completion_calls,
                            tool_calls=tool_calls,
                            usage=usage.snapshot(),
                            latency_ms=_elapsed_ms(start),
                        )
                        raise ExecutorError(failure.reason, failure.message)

                    return self._incomplete_result(
                        run_id=run_id,
                        step=step,
                        reason="max_tokens_continuation_incomplete",
                        findings=(
                            combined_content
                            or (
                                "Evidence is insufficient because synthesis did not finish "
                                "after max_tokens."
                            )
                        ),
                        tool_calls=tool_calls,
                        usage=usage.snapshot(),
                        latency_ms=_elapsed_ms(start),
                    )

                case StopReason.pause_turn:
                    messages.append(response.message)
                    stalled += 1
                    if stalled > _MAX_STALLED_COMPLETIONS:
                        return self._incomplete_result(
                            run_id=run_id,
                            step=step,
                            reason="stalled_without_progress",
                            findings=(
                                "Evidence is insufficient because the model stopped making "
                                "progress before synthesizing a result."
                            ),
                            tool_calls=tool_calls,
                            usage=usage.snapshot(),
                            latency_ms=_elapsed_ms(start),
                        )

                case StopReason.tool_use:
                    messages.append(response.message)
                    if not response.message.tool_calls:
                        stalled += 1
                        if stalled > _MAX_STALLED_COMPLETIONS:
                            return self._incomplete_result(
                                run_id=run_id,
                                step=step,
                                reason="stalled_without_progress",
                                findings=(
                                    "Evidence is insufficient because the model stopped making "
                                    "progress before synthesizing a result."
                                ),
                                tool_calls=tool_calls,
                                usage=usage.snapshot(),
                                latency_ms=_elapsed_ms(start),
                            )
                        continue
                    stalled = 0
                    for tool_call in response.message.tool_calls:
                        tool_result = self._registry.dispatch(
                            tool_call.name,
                            tool_call.arguments,
                            context,
                        )
                        tool_calls += 1

                        if _is_invalid_args(tool_result):
                            invalid_arg_errors += 1
                            if invalid_arg_errors > self._max_arg_repairs:
                                return self._incomplete_result(
                                    run_id=run_id,
                                    step=step,
                                    reason="arg_repair_exhausted",
                                    findings=(
                                        "Evidence is insufficient because tool argument repair "
                                        "was exhausted."
                                    ),
                                    tool_calls=tool_calls,
                                    usage=usage.snapshot(),
                                    latency_ms=_elapsed_ms(start),
                                )
                        else:
                            invalid_arg_errors = 0

                        messages.append(_tool_message(tool_call.id, tool_result))

                        if tool_calls >= self._max_tool_calls:
                            return self._incomplete_result(
                                run_id=run_id,
                                step=step,
                                reason="tool_budget_exhausted",
                                findings=(
                                    "Evidence is insufficient because the step reached the tool "
                                    "call budget."
                                ),
                                tool_calls=tool_calls,
                                usage=usage.snapshot(),
                                latency_ms=_elapsed_ms(start),
                            )

    def _complete(self, messages: list[LLMMessage]) -> LLMResponse:
        return self._client.complete(
            messages,
            tools=cast(list[ToolSchema], self._registry.to_llm_schema()),
            system=self._system_prompt,
            temperature=None,
        )

    def _incomplete_result(
        self,
        *,
        run_id: str,
        step: PlanStep,
        reason: str,
        findings: str,
        tool_calls: int,
        usage: Usage,
        latency_ms: int,
    ) -> StepResult:
        result = StepResult(
            step_index=step.index,
            status=StepOutcome.incomplete,
            findings=findings,
            evidence=[],
            tool_calls=tool_calls,
            usage=usage,
        )
        self._append_tool_result_event(
            run_id=run_id,
            result=result,
            latency_ms=latency_ms,
            reason=reason,
        )
        return result

    def _raise_synthesis_error(
        self,
        *,
        run_id: str,
        step: PlanStep,
        failure: _SynthesisFailure,
        attempts: int,
        completion_calls: int,
        tool_calls: int,
        usage: Usage,
        latency_ms: int,
    ) -> None:
        self._append_error_event(
            run_id=run_id,
            step=step,
            failure=failure,
            attempts=attempts,
            completion_calls=completion_calls,
            tool_calls=tool_calls,
            usage=usage,
            latency_ms=latency_ms,
        )
        raise ExecutorError(
            failure.reason,
            f"Executor synthesis failed after {attempts} attempt(s): {failure.message}",
        )

    def _append_tool_result_event(
        self,
        *,
        run_id: str,
        result: StepResult,
        latency_ms: int,
        reason: str | None,
    ) -> None:
        payload: dict[str, JsonValue] = {
            "summary": f"Executor step {result.step_index} {result.status.value}.",
            "step_index": result.step_index,
            "status": result.status.value,
            "findings": result.findings,
            "evidence": cast(JsonValue, result.evidence),
            "tool_calls": result.tool_calls,
        }
        if reason is not None:
            payload["reason"] = reason

        self._store.append(
            run_id,
            TraceEventKind.tool_result,
            payload,
            latency_ms=latency_ms,
            tokens_in=result.usage.tokens_in,
            tokens_out=result.usage.tokens_out,
            cost_usd=result.usage.cost_usd,
        )

    def _append_error_event(
        self,
        *,
        run_id: str,
        step: PlanStep,
        failure: _SynthesisFailure,
        attempts: int,
        completion_calls: int,
        tool_calls: int,
        usage: Usage,
        latency_ms: int,
    ) -> None:
        payload: dict[str, JsonValue] = {
            "summary": f"Executor failed: {failure.reason.value}.",
            "message": failure.message,
            "reason": failure.reason.value,
            "step_index": step.index,
            "attempts": attempts,
            "completion_calls": completion_calls,
            "tool_calls": tool_calls,
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


def _executor_system_prompt() -> str:
    return "\n\n".join(
        [
            (
                "Role & mission\n"
                "You are RepoPilot, a task-oriented codebase agent. As the Executor, complete "
                "exactly one planned step by using repository tools and synthesizing grounded "
                "findings for the Critic."
            ),
            (
                "Hard rules\n"
                "Use tools for repository facts. Do not invent file paths, line numbers, APIs, or "
                "behavior. Treat denied approvals and tool failures as ordinary tool observations. "
                "Do not attempt to bypass safety checks or mutate files unless a registered tool "
                "does so through the approval gate."
            ),
            (
                "Tool loop\n"
                "Call only the supplied tools. Prefer the step's suggested tools when useful, but "
                "choose the smallest evidence needed to satisfy the success check. If evidence is "
                "insufficient, say so in the final findings."
            ),
            (
                "Output contract\n"
                "When done, return only JSON shaped exactly as "
                '{"findings":str,"evidence":[str]}. Evidence entries should be concise citations '
                "or tool-observation references. Do not include markdown, prose, comments, or "
                "bookkeeping fields such as step_index, status, tool_calls, or usage."
            ),
        ]
    )


def _executor_user_prompt(
    step: PlanStep,
    *,
    scratchpad: str,
    repo_overview: str | None,
) -> str:
    return "\n\n".join(
        [
            f"Plan step JSON:\n{step.model_dump_json(indent=2)}",
            f"Step intent:\n{step.intent}",
            f"Success check:\n{step.success_check}",
            (
                "Suggested tools:\n"
                f"{', '.join(step.suggested_tools) if step.suggested_tools else 'None.'}"
            ),
            f"Scratchpad:\n{scratchpad if scratchpad else 'Not supplied.'}",
            f"Repo overview:\n{repo_overview if repo_overview is not None else 'Not supplied.'}",
        ]
    )


def _parse_step_result(
    content: str,
    *,
    step_index: int,
    tool_calls: int,
    usage: Usage,
) -> tuple[StepResult | None, _SynthesisFailure]:
    candidate = _strip_json_fence(content.strip())
    try:
        parsed: object = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, _SynthesisFailure(
            reason=ExecutorErrorReason.synthesis_repair_exhausted,
            message=f"json.loads error: {exc}",
        )

    if not isinstance(parsed, dict):
        return None, _SynthesisFailure(
            reason=ExecutorErrorReason.synthesis_repair_exhausted,
            message="Executor output must be a JSON object.",
        )

    payload = dict(parsed)
    payload["step_index"] = step_index
    payload["status"] = StepOutcome.completed.value
    payload["tool_calls"] = tool_calls
    payload["usage"] = usage.model_dump(mode="json")

    try:
        return StepResult.model_validate(payload), _SynthesisFailure(
            reason=ExecutorErrorReason.synthesis_repair_exhausted,
            message="",
        )
    except ValidationError as exc:
        return None, _SynthesisFailure(
            reason=ExecutorErrorReason.synthesis_repair_exhausted,
            message=str(exc),
        )


def _strip_json_fence(content: str) -> str:
    match = _JSON_FENCE_RE.fullmatch(content)
    if match is None:
        return content
    return match.group("body").strip()


def _repair_message(content: str, failure: _SynthesisFailure) -> LLMMessage:
    return LLMMessage(
        role=Role.user,
        content=(
            "Your previous executor synthesis was invalid.\n"
            "Bad reply:\n"
            f"{content}\n\n"
            "Validation error:\n"
            f"{failure.message}\n\n"
            "Return only corrected JSON with the exact executor schema: "
            '{"findings":str,"evidence":[str]}.'
        ),
    )


def _tool_message(tool_call_id: str, result: ToolResult) -> LLMMessage:
    return LLMMessage(
        role=Role.tool,
        tool_call_id=tool_call_id,
        content=result.model_dump_json(),
    )


def _is_invalid_args(result: ToolResult) -> bool:
    return (
        not result.ok
        and result.error is not None
        and result.error.type is ErrorType.InvalidArgsError
    )


def _combine_content(prefix: str, suffix: str) -> str:
    if not prefix:
        return suffix
    if not suffix:
        return prefix
    return f"{prefix}{suffix}"


def _elapsed_ms(start: float) -> int:
    return max(0, round((time.perf_counter() - start) * 1000))


def _refusal_message(content: str) -> str:
    if content:
        return f"Executor refused: {content}"
    return "Executor refused to execute the step."
