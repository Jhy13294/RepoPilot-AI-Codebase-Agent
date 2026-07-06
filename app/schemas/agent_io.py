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


class ReportConfidence(StrEnum):
    """Confidence level for a model-authored run report."""

    high = "high"
    medium = "medium"
    low = "low"


class SuspectFile(BaseModel):
    """Ranked file suspected in an issue analysis report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    reason: str


class CitationGrounding(BaseModel):
    """Deterministic validation result for one model-authored citation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    citation: str
    status: str
    grounded: bool
    detail: str


class GroundingReport(BaseModel):
    """Aggregate deterministic validation for model-authored citations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    checks: list[CitationGrounding] = Field(default_factory=list)

    @property
    def grounded_count(self) -> int:
        return sum(check.grounded for check in self.checks)

    @property
    def all_grounded(self) -> bool:
        return all(check.grounded for check in self.checks)

    @property
    def ungrounded(self) -> tuple[CitationGrounding, ...]:
        return tuple(check for check in self.checks if not check.grounded)


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


class AnalysisReport(BaseModel):
    """Model-authored analysis report plus accounting metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    headline: str
    analysis: str
    confidence: ReportConfidence
    open_questions: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    suspects: list[SuspectFile] = Field(default_factory=list)
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
    report: AnalysisReport | None = None
    grounding: GroundingReport | None = None
