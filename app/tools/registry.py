"""Tool registry and dispatch chokepoint."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import UTC, datetime
from time import perf_counter
from typing import Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from app.safety.loop_guard import LoopGuard
from app.safety.path_jail import PathJailViolation
from app.schemas.tool_io import ErrorType, ToolError, ToolMeta, ToolResult
from app.tools.base import ToolContext, ToolFailure

ToolHandler = Callable[[BaseModel, ToolContext], BaseModel]


class ToolExample(BaseModel):
    """Example tool call for LLM guidance and generated documentation."""

    model_config = ConfigDict(frozen=True)

    description: str
    args: dict[str, JsonValue]


class ToolSpec(BaseModel):
    """Complete registration metadata for one tool."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    args_schema: type[BaseModel]
    returns_schema: type[BaseModel]
    risk_level: Literal["low", "medium", "high"]
    timeout_s: int = Field(default=60, ge=1)
    examples: list[ToolExample] = Field(default_factory=list)


class ApprovalOutcome(BaseModel):
    """Approval decision returned by an approval gate."""

    model_config = ConfigDict(frozen=True)

    approved: bool
    reason: str | None = None
    actor: str = "human"


class ApprovalGate(Protocol):
    """Policy gate for high-risk tool calls."""

    def check(self, spec: ToolSpec, args: BaseModel, context: ToolContext) -> ApprovalOutcome:
        """Return whether the validated high-risk call may execute."""


class ApprovalTraceRecord(BaseModel):
    """Trace record emitted once for every high-risk approval decision."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    tool_name: str
    risk_level: Literal["low", "medium", "high"]
    decision: Literal["approved", "denied"]
    actor: str
    reason: str | None
    ts: datetime


class ToolTraceRecord(BaseModel):
    """Trace record emitted once for every dispatch attempt."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    tool_name: str
    args: dict[str, JsonValue]
    ok: bool
    error_type: ErrorType | None
    latency_ms: int = Field(ge=0)
    truncated: bool
    ts: datetime
    outcome: dict[str, JsonValue] | None = None


class TraceSink(Protocol):
    """Append-only sink for dispatch trace records."""

    def append(self, record: ToolTraceRecord) -> None:
        """Persist or collect one trace record."""


class _DispatchFailure(Exception):
    """Internal control-flow exception for envelope failures."""

    def __init__(
        self,
        error_type: ErrorType,
        message: str,
        details: dict[str, JsonValue] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.details = details


class ToolRegistry:
    """Register tools and dispatch all calls through one audited path."""

    def __init__(
        self,
        approval_gate: ApprovalGate | None = None,
        trace_sink: TraceSink | None = None,
        loop_guard: LoopGuard | None = None,
    ) -> None:
        self._approval_gate = approval_gate
        self._trace_sink = trace_sink
        self._loop_guard = loop_guard
        self._tools: dict[str, tuple[ToolSpec, ToolHandler]] = {}

    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        """Register a tool spec and implementation."""
        if spec.name in self._tools:
            raise ValueError(f"Tool '{spec.name}' is already registered.")
        self._tools[spec.name] = (spec, handler)

    def to_llm_schema(self) -> list[dict[str, JsonValue]]:
        """Return OpenAI-compatible function tool schemas."""
        schemas: list[dict[str, JsonValue]] = []
        for spec, _handler in self._tools.values():
            parameters = cast(dict[str, JsonValue], spec.args_schema.model_json_schema())
            schema = {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": parameters,
                },
            }
            schemas.append(cast(dict[str, JsonValue], schema))
        return schemas

    def dispatch(
        self,
        name: str,
        raw_args: dict[str, JsonValue],
        context: ToolContext,
    ) -> ToolResult:
        """Dispatch one tool call and return a ToolResult envelope.

        Tool execution uses a thread timeout for Windows compatibility. Python cannot forcibly
        stop a running thread, so a timed-out handler may continue in the background; P1 tools are
        read-only, which keeps that limitation acceptable for this phase.
        """
        started_at = perf_counter()
        payload: BaseModel | None = None

        try:
            spec, handler = self._get_tool(name)
            args = self._validate_args(spec, raw_args)
            self._check_loop_guard(context.run_id, name, args)
            self._check_approval(spec, args, context)
            payload = self._execute(handler, args, context, spec.timeout_s)
            if self._loop_guard is not None:
                self._loop_guard.mark_success(context.run_id)
            result = self._build_success(name, started_at, payload)
        except _DispatchFailure as exc:
            result = self._build_failure(name, started_at, exc.error_type, exc.message, exc.details)
        except ToolFailure as exc:
            result = self._build_failure(name, started_at, exc.type, exc.message, exc.details)
        except PathJailViolation as exc:
            result = self._build_failure(
                name,
                started_at,
                ErrorType.PathJailError,
                str(exc),
                None,
            )
        except FutureTimeoutError:
            result = self._build_failure(
                name,
                started_at,
                ErrorType.ToolTimeoutError,
                f"Tool '{name}' exceeded its timeout.",
                {"timeout_s": self._tools[name][0].timeout_s} if name in self._tools else None,
            )
        except Exception as exc:
            result = self._build_failure(
                name,
                started_at,
                ErrorType.InternalToolError,
                f"{exc.__class__.__name__}: {exc}",
                None,
            )

        self._append_trace(context, name, raw_args, result)
        return result

    def _get_tool(self, name: str) -> tuple[ToolSpec, ToolHandler]:
        try:
            return self._tools[name]
        except KeyError as exc:
            available_tools = sorted(self._tools)
            available = ", ".join(available_tools) if available_tools else "none"
            raise _DispatchFailure(
                ErrorType.InvalidArgsError,
                f"Unknown tool '{name}'. Available tools: {available}.",
                {"available_tools": cast(JsonValue, available_tools)},
            ) from exc

    @staticmethod
    def _validate_args(spec: ToolSpec, raw_args: dict[str, JsonValue]) -> BaseModel:
        try:
            return spec.args_schema.model_validate(raw_args)
        except ValidationError as exc:
            details, validation_errors = _validation_details(exc)
            hints = "; ".join(
                f"{error['field']}: {error['message']}" for error in validation_errors
            )
            raise _DispatchFailure(
                ErrorType.InvalidArgsError,
                f"Invalid arguments for tool '{spec.name}': {hints}.",
                details,
            ) from exc

    def _check_loop_guard(self, run_id: str, name: str, args: BaseModel) -> None:
        if self._loop_guard is None or not self._loop_guard.check(run_id, name, args):
            return

        raise _DispatchFailure(
            ErrorType.LoopBlockedError,
            (
                f"Identical repeated call to tool '{name}', or an effectively-identical repeat "
                "of a successful call, was blocked; change the effective arguments before "
                "retrying."
            ),
            {"reason": "duplicate_call"},
        )

    def _check_approval(self, spec: ToolSpec, args: BaseModel, context: ToolContext) -> None:
        if spec.risk_level != "high":
            return

        if self._approval_gate is None:
            message = f"no approval gate configured for high-risk tool '{spec.name}'."
            self._emit_approval_decision(
                spec,
                context,
                decision="denied",
                actor="system",
                reason=message,
            )
            raise _DispatchFailure(
                ErrorType.ApprovalDeniedError,
                message,
                {"risk_level": spec.risk_level},
            )

        outcome = self._approval_gate.check(spec, args, context)
        self._emit_approval_decision(
            spec,
            context,
            decision="approved" if outcome.approved else "denied",
            actor=outcome.actor,
            reason=outcome.reason,
        )
        if not outcome.approved:
            message = outcome.reason or f"Approval denied for high-risk tool '{spec.name}'."
            raise _DispatchFailure(
                ErrorType.ApprovalDeniedError,
                message,
                {"risk_level": spec.risk_level, "reason": outcome.reason},
            )

    def _emit_approval_decision(
        self,
        spec: ToolSpec,
        context: ToolContext,
        *,
        decision: Literal["approved", "denied"],
        actor: str,
        reason: str | None,
    ) -> None:
        if self._trace_sink is None:
            return

        append_approval = getattr(self._trace_sink, "append_approval", None)
        if not callable(append_approval):
            return

        record = ApprovalTraceRecord(
            run_id=context.run_id,
            tool_name=spec.name,
            risk_level=spec.risk_level,
            decision=decision,
            actor=actor,
            reason=reason,
            ts=datetime.now(UTC),
        )
        cast(Callable[[ApprovalTraceRecord], None], append_approval)(record)

    @staticmethod
    def _execute(
        handler: ToolHandler,
        args: BaseModel,
        context: ToolContext,
        timeout_s: int,
    ) -> BaseModel:
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(handler, args, context)
        try:
            return future.result(timeout=timeout_s)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _build_success(name: str, started_at: float, payload: BaseModel) -> ToolResult:
        truncated = bool(getattr(payload, "truncated", False))
        meta = ToolMeta(tool_name=name, latency_ms=_latency_ms(started_at), truncated=truncated)
        return ToolResult.success(data=payload, meta=meta)

    @staticmethod
    def _build_failure(
        name: str,
        started_at: float,
        error_type: ErrorType,
        message: str,
        details: dict[str, JsonValue] | None,
    ) -> ToolResult:
        meta = ToolMeta(tool_name=name, latency_ms=_latency_ms(started_at), truncated=False)
        error = ToolError(type=error_type, message=message, details=details)
        return ToolResult.failure(error=error, meta=meta)

    def _append_trace(
        self,
        context: ToolContext,
        name: str,
        raw_args: dict[str, JsonValue],
        result: ToolResult,
    ) -> None:
        if self._trace_sink is None:
            return

        record = ToolTraceRecord(
            run_id=context.run_id,
            tool_name=name,
            args=raw_args,
            ok=result.ok,
            error_type=result.error.type if result.error is not None else None,
            latency_ms=result.meta.latency_ms,
            truncated=result.meta.truncated,
            ts=datetime.now(UTC),
            outcome=_evidence_outcome(result.data),
        )
        self._trace_sink.append(record)


def _latency_ms(started_at: float) -> int:
    return max(0, int((perf_counter() - started_at) * 1000))


def _evidence_outcome(payload: BaseModel | None) -> dict[str, JsonValue] | None:
    if payload is None:
        return None

    digest = getattr(payload, "evidence_digest", None)
    if not callable(digest):
        return None
    return cast(Callable[[], dict[str, JsonValue]], digest)()


def _validation_details(exc: ValidationError) -> tuple[dict[str, JsonValue], list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    for error in exc.errors(include_url=False, include_context=False, include_input=False):
        location = ".".join(str(part) for part in error.get("loc", ())) or "__root__"
        errors.append(
            {
                "field": location,
                "message": str(error.get("msg", "Invalid value.")),
                "type": str(error.get("type", "value_error")),
            }
        )
    return {"errors": cast(JsonValue, errors)}, errors
