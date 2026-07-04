"""Shared tool input/output envelope schemas."""

from enum import StrEnum
from typing import Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_serializer,
    field_validator,
    model_validator,
)


class ErrorType(StrEnum):
    """Shared tool error taxonomy."""

    InvalidArgsError = "InvalidArgsError"
    PathJailError = "PathJailError"
    NotFoundError = "NotFoundError"
    BinaryFileError = "BinaryFileError"
    ToolTimeoutError = "ToolTimeoutError"
    PatchApplyError = "PatchApplyError"
    TestExecutionError = "TestExecutionError"
    ApprovalDeniedError = "ApprovalDeniedError"
    InternalToolError = "InternalToolError"


class ToolMeta(BaseModel):
    """Execution metadata attached to every tool result."""

    model_config = ConfigDict(frozen=True)

    tool_name: str
    latency_ms: int = Field(ge=0)
    truncated: bool = False


class ToolError(BaseModel):
    """Structured error returned inside a failed tool result."""

    model_config = ConfigDict(frozen=True)

    type: ErrorType
    message: str
    details: dict[str, JsonValue] | None = None


class _JsonPayload(BaseModel):
    """JSON payload holder used when decoding an envelope without a concrete schema."""

    model_config = ConfigDict(extra="allow", frozen=True)


class ToolResult(BaseModel):
    """Uniform envelope returned by every tool execution."""

    model_config = ConfigDict(frozen=True)

    ok: bool
    data: BaseModel | None = None
    error: ToolError | None = None
    meta: ToolMeta

    @field_serializer("data")
    def serialize_payload(self, value: BaseModel | None) -> dict[str, JsonValue] | None:
        """Serialize concrete payload models without losing their fields."""
        if value is None:
            return None
        return cast(dict[str, JsonValue], value.model_dump(mode="json"))

    @field_validator("data", mode="before")
    @classmethod
    def decode_json_payload(cls, value: object) -> object:
        """Preserve JSON payload fields when validating an envelope from JSON."""
        if isinstance(value, dict):
            return _JsonPayload.model_validate(value)
        return value

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        """Validate the envelope success/failure invariants."""
        if self.ok and self.error is not None:
            raise ValueError("successful tool results must not include an error")
        if not self.ok and self.error is None:
            raise ValueError("failed tool results must include an error")
        if not self.ok and self.data is not None:
            raise ValueError("failed tool results must not include data")
        return self

    @classmethod
    def success(cls, data: BaseModel | None, meta: ToolMeta) -> Self:
        """Build a successful tool result."""
        return cls(ok=True, data=data, meta=meta)

    @classmethod
    def failure(cls, error: ToolError, meta: ToolMeta) -> Self:
        """Build a failed tool result."""
        return cls(ok=False, error=error, meta=meta)
