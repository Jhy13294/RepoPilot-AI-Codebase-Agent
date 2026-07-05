"""Provider-neutral LLM input/output schemas."""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class Role(StrEnum):
    """Supported chat message roles."""

    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class StopReason(StrEnum):
    """Provider-neutral completion stop reasons."""

    tool_use = "tool_use"
    end_turn = "end_turn"
    max_tokens = "max_tokens"
    refusal = "refusal"
    pause_turn = "pause_turn"


class ToolCall(BaseModel):
    """Normalized assistant tool call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    name: str
    arguments: dict[str, JsonValue] = Field(default_factory=dict)


class LLMMessage(BaseModel):
    """Normalized chat message."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def validate_role_fields(self) -> Self:
        """Validate role-specific message fields."""
        if self.tool_calls and self.role is not Role.assistant:
            raise ValueError("tool_calls are only allowed on assistant messages")
        if self.tool_call_id is not None and self.role is not Role.tool:
            raise ValueError("tool_call_id is only allowed on tool messages")
        return self


class Usage(BaseModel):
    """Normalized token usage and estimated cost."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    cost_usd: float | None = None


class LLMResponse(BaseModel):
    """Normalized completion response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    message: LLMMessage
    stop_reason: StopReason
    usage: Usage
    model: str
    raw_finish_reason: str

    @model_validator(mode="after")
    def validate_assistant_message(self) -> Self:
        """Require completion responses to carry assistant messages."""
        if self.message.role is not Role.assistant:
            raise ValueError("LLMResponse.message must be an assistant message")
        return self
