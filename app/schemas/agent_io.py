"""Agent loop result schemas."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from app.schemas.llm_io import Usage
from app.schemas.tool_io import ErrorType


class AskStatus(StrEnum):
    """Terminal status for a single agent ask run."""

    answered = "answered"
    budget_exhausted = "budget_exhausted"
    refused = "refused"
    error = "error"


class ToolInvocation(BaseModel):
    """Summary of one tool dispatch performed by the loop."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool: str
    args: dict[str, JsonValue]
    ok: bool
    error_type: ErrorType | None = None


class AskResult(BaseModel):
    """Final in-memory result returned by the single tool-calling loop."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    answer: str
    status: AskStatus
    steps: int = Field(ge=0)
    tool_invocations: list[ToolInvocation]
    usage: Usage
