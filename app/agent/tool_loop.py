"""Single ReAct-style tool-calling loop for repository questions."""

from collections.abc import Sequence
from uuid import uuid4

from pydantic import JsonValue

from app.agent.usage import UsageAccumulator
from app.safety.path_jail import PathJail
from app.schemas.agent_io import AskResult, AskStatus, ToolInvocation
from app.schemas.llm_io import LLMMessage, LLMResponse, Role, StopReason
from app.schemas.tool_io import ErrorType, ToolResult
from app.services.llm_client import LLMClient, ToolSchema
from app.tools.base import ToolContext
from app.tools.registry import ToolRegistry

MAX_ARG_REPAIRS = 2

ASK_SYSTEM_PROMPT = """\
You are RepoPilot, a task-oriented codebase agent. You are not a general chatbot.
Answer only after checking the repository with tools. Ground every factual claim in tool evidence
using file:line citations. If the tools cannot verify an answer, say that the evidence was not
found. Never invent file paths, line numbers, APIs, or behavior.
"""


def run_tool_loop(
    question: str,
    *,
    client: LLMClient,
    registry: ToolRegistry,
    jail: PathJail,
    max_steps: int,
    system_prompt: str = ASK_SYSTEM_PROMPT,
) -> AskResult:
    """Run a bounded single-turn tool loop and return the final answer."""
    messages = [
        LLMMessage(role=Role.system, content=system_prompt),
        LLMMessage(role=Role.user, content=question),
    ]
    usage = UsageAccumulator()
    tool_invocations: list[ToolInvocation] = []
    best_content = ""
    invalid_arg_errors = 0
    steps = 0

    # One run identity per ask: all tool dispatches share it so the run stays correlatable in
    # the trace (architecture "trace-first"; ToolContext.run_id is the per-run key).
    context = ToolContext(run_id=str(uuid4()), jail=jail)

    while steps < max_steps:
        try:
            response = _complete(client, messages, registry.to_llm_schema())
        except Exception as exc:
            return _result(
                answer=f"LLM completion failed: {exc}",
                status=AskStatus.error,
                steps=steps,
                tool_invocations=tool_invocations,
                usage=usage,
            )
        steps += 1
        usage.add(response.usage)
        best_content = _best_content(best_content, response.message.content)

        match response.stop_reason:
            case StopReason.end_turn:
                return _result(
                    answer=response.message.content,
                    status=AskStatus.answered,
                    steps=steps,
                    tool_invocations=tool_invocations,
                    usage=usage,
                )
            case StopReason.refusal:
                return _result(
                    answer=response.message.content,
                    status=AskStatus.refused,
                    steps=steps,
                    tool_invocations=tool_invocations,
                    usage=usage,
                )
            case StopReason.max_tokens:
                return _continue_after_max_tokens(
                    client=client,
                    registry=registry,
                    messages=messages,
                    partial_message=response.message,
                    best_content=best_content,
                    steps=steps,
                    max_steps=max_steps,
                    tool_invocations=tool_invocations,
                    usage=usage,
                )
            case StopReason.pause_turn:
                messages.append(response.message)
            case StopReason.tool_use:
                messages.append(response.message)
                for tool_call in response.message.tool_calls:
                    tool_result = registry.dispatch(tool_call.name, tool_call.arguments, context)
                    tool_invocations.append(
                        _tool_invocation(tool_call.name, tool_call.arguments, tool_result)
                    )

                    if _is_invalid_args(tool_result):
                        invalid_arg_errors += 1
                        if invalid_arg_errors > MAX_ARG_REPAIRS:
                            return _result(
                                answer=_error_answer(tool_result),
                                status=AskStatus.error,
                                steps=steps,
                                tool_invocations=tool_invocations,
                                usage=usage,
                            )
                    else:
                        invalid_arg_errors = 0

                    messages.append(_tool_message(tool_call.id, tool_result))

    return _result(
        answer=best_content,
        status=AskStatus.budget_exhausted,
        steps=steps,
        tool_invocations=tool_invocations,
        usage=usage,
    )


def _complete(
    client: LLMClient,
    messages: Sequence[LLMMessage],
    tools: Sequence[ToolSchema],
) -> LLMResponse:
    return client.complete(messages, tools=tools)


def _continue_after_max_tokens(
    *,
    client: LLMClient,
    registry: ToolRegistry,
    messages: list[LLMMessage],
    partial_message: LLMMessage,
    best_content: str,
    steps: int,
    max_steps: int,
    tool_invocations: list[ToolInvocation],
    usage: UsageAccumulator,
) -> AskResult:
    messages.append(partial_message)
    if steps >= max_steps:
        return _result(
            answer=best_content,
            status=AskStatus.budget_exhausted,
            steps=steps,
            tool_invocations=tool_invocations,
            usage=usage,
        )

    try:
        response = _complete(client, messages, registry.to_llm_schema())
    except Exception as exc:
        return _result(
            answer=f"LLM completion failed: {exc}",
            status=AskStatus.error,
            steps=steps,
            tool_invocations=tool_invocations,
            usage=usage,
        )
    steps += 1
    usage.add(response.usage)
    combined_content = _combine_content(partial_message.content, response.message.content)
    best_content = _best_content(best_content, combined_content)

    if response.stop_reason is StopReason.end_turn:
        return _result(
            answer=combined_content,
            status=AskStatus.answered,
            steps=steps,
            tool_invocations=tool_invocations,
            usage=usage,
        )

    return _result(
        answer=best_content,
        status=AskStatus.budget_exhausted,
        steps=steps,
        tool_invocations=tool_invocations,
        usage=usage,
    )


def _tool_invocation(
    name: str,
    args: dict[str, JsonValue],
    result: ToolResult,
) -> ToolInvocation:
    return ToolInvocation(
        tool=name,
        args=args,
        ok=result.ok,
        error_type=result.error.type if result.error is not None else None,
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


def _error_answer(result: ToolResult) -> str:
    if result.error is None:
        return "Tool call argument repair budget exhausted."
    return f"Tool call argument repair budget exhausted: {result.error.message}"


def _best_content(current: str, candidate: str) -> str:
    if candidate:
        return candidate
    return current


def _combine_content(prefix: str, suffix: str) -> str:
    if not prefix:
        return suffix
    if not suffix:
        return prefix
    return f"{prefix}{suffix}"


def _result(
    *,
    answer: str,
    status: AskStatus,
    steps: int,
    tool_invocations: list[ToolInvocation],
    usage: UsageAccumulator,
) -> AskResult:
    return AskResult(
        answer=answer,
        status=status,
        steps=steps,
        tool_invocations=list(tool_invocations),
        usage=usage.snapshot(),
    )
