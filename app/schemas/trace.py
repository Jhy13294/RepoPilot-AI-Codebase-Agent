"""Trace event schemas for append-only run logs."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class TraceEventKind(StrEnum):
    """Kinds of events persisted in a run trace."""

    plan = "plan"
    tool_call = "tool_call"
    tool_result = "tool_result"
    approval_request = "approval_request"
    approval_decision = "approval_decision"
    critic_verdict = "critic_verdict"
    replan = "replan"
    report = "report"
    error = "error"


class TraceEvent(BaseModel):
    """One append-only event in a run trace."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    seq: int = Field(ge=0)
    ts: datetime
    kind: TraceEventKind
    payload: dict[str, JsonValue]
    latency_ms: int | None = Field(default=None, ge=0)
    tokens_in: int | None = Field(default=None, ge=0)
    tokens_out: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
