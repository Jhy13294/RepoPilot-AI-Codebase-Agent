"""Agent loop result schemas."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from app.agent.state import RunStatus
from app.schemas.llm_io import Usage
from app.schemas.tool_io import ErrorType


class AskStatus(StrEnum):
    """Terminal status for a single agent ask run."""

    answered = "answered"
    budget_exhausted = "budget_exhausted"
    refused = "refused"
    error = "error"


class StepOutcome(StrEnum):
    """Terminal outcome for one executor step."""

    completed = "completed"
    incomplete = "incomplete"


class VerdictDecision(StrEnum):
    """Critic routing decision for one executed plan step."""

    proceed = "proceed"
    retry = "retry"
    replan = "replan"


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


class StepResult(BaseModel):
    """Validated result for one executed plan step."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step_index: int = Field(ge=0)
    status: StepOutcome
    findings: str
    evidence: list[str]
    tool_calls: int = Field(ge=0)
    usage: Usage


class Verdict(BaseModel):
    """Validated critic decision for one executed plan step."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step_index: int = Field(ge=0)
    decision: VerdictDecision
    reason: str
    hint: str = ""
    usage: Usage


class RunResult(BaseModel):
    """Final result returned by the full Planner-Executor-Critic run loop."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    status: RunStatus
    summary: str
    steps_used: int = Field(ge=0)
    replans_used: int = Field(ge=0)
    fix_cycles_used: int = Field(ge=0)
    usage: Usage
