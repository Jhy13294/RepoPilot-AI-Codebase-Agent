"""Pydantic request and response contracts for the HTTP run API."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.agent.state import RunStatus


class CreateRunRequest(BaseModel):
    """Request for one read-only agent run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_type: Literal["question", "issue"]
    prompt: str
    repo: str
    max_steps: int | None = Field(default=None, ge=1)


class CreateRunResponse(BaseModel):
    """Acknowledgement returned after a run is durably created."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    status: RunStatus


class RunSummaryView(BaseModel):
    """Lightweight persisted run projection used by collection reads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    task_type: str
    prompt: str
    repo: str
    status: RunStatus
    step_count: int = Field(ge=0)
    steps_used: int = Field(ge=0)
    replans_used: int = Field(ge=0)
    fix_cycles_used: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime


class ToolCallEventView(BaseModel):
    """Safe public projection of one live tool-call trace event."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seq: int = Field(ge=0)
    ts: datetime
    tool_name: str
    ok: bool
    error_type: str | None
    latency_ms: int | None = Field(default=None, ge=0)


class RunView(RunSummaryView):
    """Detailed persisted run projection with its live tool-call timeline."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: str | None
    tool_calls: list[ToolCallEventView]


class RunListResponse(BaseModel):
    """Collection response for persisted runs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    runs: list[RunSummaryView]
