"""Provider-agnostic LLM client and OpenAI-compatible adapter."""

import json
import re
from collections.abc import Sequence
from typing import Any, Protocol, cast

from openai import OpenAI
from pydantic import JsonValue, ValidationError

from app.config import Settings
from app.schemas.llm_io import LLMMessage, LLMResponse, Role, StopReason, ToolCall, Usage

ToolSchema = dict[str, JsonValue]

_DEEPSEEK_V4_PRO_PRICE_PER_MILLION = (0.435, 0.87)
_PRICE_TABLE: dict[str, tuple[float, float]] = {
    "deepseek-v4-pro": _DEEPSEEK_V4_PRO_PRICE_PER_MILLION,
}


class LLMError(Exception):
    """Raised when an LLM transport or response parsing step fails."""


class LLMClient(Protocol):
    """Provider-neutral synchronous completion client."""

    def complete(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[ToolSchema] | None = None,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Complete one chat turn and return a normalized response."""


class OpenAICompatibleClient:
    """Adapter for OpenAI-compatible chat completion APIs."""

    def __init__(self, model: str, client: Any) -> None:
        self._model = model
        self._client = client

    def complete(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[ToolSchema] | None = None,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Complete one chat turn with an OpenAI-compatible client."""
        request_messages = _serialize_messages(messages, system)
        kwargs: dict[str, object] = {
            "model": self._model,
            "messages": request_messages,
        }
        if tools is not None:
            kwargs["tools"] = tools
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise LLMError(f"LLM completion failed: {exc}") from exc

        return _normalize_openai_response(
            response,
            self._model,
            tools_offered=tools is not None,
        )


def build_llm_client(settings: Settings) -> LLMClient:
    """Build the configured LLM client from application settings."""
    if settings.llm_provider == "openai_compatible":
        if settings.openai_api_key is None:
            raise LLMError("OPENAI_API_KEY is required for openai_compatible provider.")
        client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
        return OpenAICompatibleClient(model=settings.model, client=client)

    if settings.llm_provider == "anthropic":
        raise NotImplementedError("Anthropic adapter lands in RP-P2-FEAT-002")

    raise LLMError(f"Unsupported LLM provider: {settings.llm_provider}")


def _serialize_messages(
    messages: Sequence[LLMMessage],
    system: str | None,
) -> list[dict[str, object]]:
    serialized: list[dict[str, object]] = []
    if system is not None:
        serialized.append({"role": Role.system.value, "content": system})

    serialized.extend(_serialize_message(message) for message in messages)
    return serialized


def _serialize_message(message: LLMMessage) -> dict[str, object]:
    serialized: dict[str, object] = {
        "role": message.role.value,
        "content": message.content,
    }
    if message.role is Role.tool:
        serialized["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        serialized["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(tool_call.arguments),
                },
            }
            for tool_call in message.tool_calls
        ]
    return serialized


def _normalize_openai_response(
    response: Any,
    fallback_model: str,
    *,
    tools_offered: bool,
) -> LLMResponse:
    try:
        choice = _first_choice(response)
        raw_message = _get_required(choice, "message")
        raw_finish_reason = _string_or_empty(_get_optional(choice, "finish_reason"))
        content = _optional_string(raw_message, "content")
        tool_calls = _parse_tool_calls(_get_optional(raw_message, "tool_calls"))
        model = _optional_string(response, "model") or fallback_model
        usage = _parse_usage(_get_optional(response, "usage"), model)
        recovered_tool_calls = False
        stop_reason = _map_finish_reason(raw_finish_reason)

        if tools_offered and not tool_calls:
            salvaged_tool_calls = _salvage_plain_text_tool_calls(content)
            if salvaged_tool_calls:
                tool_calls = salvaged_tool_calls
                content = ""
                recovered_tool_calls = True
                stop_reason = StopReason.tool_use

        message = LLMMessage(role=Role.assistant, content=content, tool_calls=tool_calls)
        return LLMResponse(
            message=message,
            stop_reason=stop_reason,
            usage=usage,
            model=model,
            raw_finish_reason=raw_finish_reason,
            recovered_tool_calls=recovered_tool_calls,
        )
    except LLMError:
        raise
    except (AttributeError, IndexError, TypeError, ValidationError, ValueError) as exc:
        raise LLMError(f"Invalid LLM response: {exc}") from exc


def _first_choice(response: Any) -> Any:
    choices = _get_required(response, "choices")
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
        raise LLMError("Invalid LLM response: choices must be a non-empty sequence.")
    if not choices:
        raise LLMError("Invalid LLM response: choices must be non-empty.")
    return choices[0]


def _parse_tool_calls(raw_tool_calls: Any) -> list[ToolCall]:
    if raw_tool_calls is None:
        return []
    if not isinstance(raw_tool_calls, Sequence) or isinstance(raw_tool_calls, (str, bytes)):
        raise LLMError("Invalid LLM response: tool_calls must be a sequence.")

    tool_calls: list[ToolCall] = []
    for raw_tool_call in raw_tool_calls:
        function = _get_required(raw_tool_call, "function")
        raw_arguments = _required_string(function, "arguments")
        try:
            parsed_arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise LLMError("Invalid tool call arguments: expected JSON object.") from exc
        if not isinstance(parsed_arguments, dict):
            raise LLMError("Invalid tool call arguments: expected JSON object.")

        tool_calls.append(
            ToolCall(
                id=_required_string(raw_tool_call, "id"),
                name=_required_string(function, "name"),
                arguments=cast(dict[str, JsonValue], parsed_arguments),
            )
        )
    return tool_calls


def _salvage_plain_text_tool_calls(content: str) -> list[ToolCall]:
    stripped = content.strip()
    if not stripped:
        return []

    candidate = _strip_plain_text_tool_call_wrapper(stripped)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return []

    raw_calls = parsed if isinstance(parsed, list) else [parsed]
    if not raw_calls:
        return []

    tool_calls: list[ToolCall] = []
    for index, raw_call in enumerate(raw_calls):
        tool_call = _salvage_one_plain_text_tool_call(raw_call, index)
        if tool_call is None:
            return []
        tool_calls.append(tool_call)
    return tool_calls


def _strip_plain_text_tool_call_wrapper(content: str) -> str:
    fenced_match = re.fullmatch(r"```json\s*(?P<body>.*?)\s*```", content, flags=re.DOTALL)
    if fenced_match is not None:
        return fenced_match.group("body").strip()

    tagged_match = re.fullmatch(
        r"<tool_call>\s*(?P<body>.*?)\s*</tool_call>",
        content,
        flags=re.DOTALL,
    )
    if tagged_match is not None:
        return tagged_match.group("body").strip()

    return content


def _salvage_one_plain_text_tool_call(raw_call: Any, index: int) -> ToolCall | None:
    if not isinstance(raw_call, dict):
        return None
    if set(raw_call) != {"name", "arguments"}:
        return None

    name = raw_call.get("name")
    arguments = raw_call.get("arguments")
    if not isinstance(name, str) or not name.strip():
        return None
    if not isinstance(arguments, dict):
        return None

    return ToolCall(
        id=f"call-salvaged-{index}",
        name=name,
        arguments=cast(dict[str, JsonValue], arguments),
    )


def _parse_usage(raw_usage: Any, model: str) -> Usage:
    tokens_in = _optional_int(raw_usage, "prompt_tokens")
    tokens_out = _optional_int(raw_usage, "completion_tokens")
    return Usage(
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=_estimate_cost(model, tokens_in, tokens_out),
    )


def _estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float | None:
    price = _PRICE_TABLE.get(model)
    if price is None:
        return None
    input_price, output_price = price
    return ((tokens_in * input_price) + (tokens_out * output_price)) / 1_000_000


def _map_finish_reason(finish_reason: str) -> StopReason:
    match finish_reason:
        case "tool_calls" | "function_call":
            return StopReason.tool_use
        case "length":
            return StopReason.max_tokens
        case "content_filter" | "refusal":
            return StopReason.refusal
        case "pause_turn":
            return StopReason.pause_turn
        case _:
            return StopReason.end_turn


def _get_required(source: Any, key: str) -> Any:
    value = _get_optional(source, key)
    if value is None:
        raise LLMError(f"Invalid LLM response: missing {key}.")
    return value


def _get_optional(source: Any, key: str) -> Any:
    if source is None:
        return None
    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)


def _required_string(source: Any, key: str) -> str:
    value = _get_required(source, key)
    if not isinstance(value, str):
        raise LLMError(f"Invalid LLM response: {key} must be a string.")
    return value


def _optional_string(source: Any, key: str) -> str:
    value = _get_optional(source, key)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise LLMError(f"Invalid LLM response: {key} must be a string.")
    return value


def _string_or_empty(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise LLMError("Invalid LLM response: finish_reason must be a string.")
    return value


def _optional_int(source: Any, key: str) -> int:
    value = _get_optional(source, key)
    if value is None:
        return 0
    if not isinstance(value, int) or isinstance(value, bool):
        raise LLMError(f"Invalid LLM response: {key} must be an integer.")
    return value
