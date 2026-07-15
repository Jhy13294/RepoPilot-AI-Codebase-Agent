from datetime import UTC
from pathlib import Path
from time import sleep

import pytest
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from app.safety.loop_guard import LoopGuard
from app.safety.path_jail import PathJail, PathJailViolation
from app.schemas.tool_io import ErrorType
from app.tools.base import ToolContext, ToolFailure
from app.tools.registry import (
    ApprovalOutcome,
    ApprovalTraceRecord,
    ToolRegistry,
    ToolSpec,
    ToolTraceRecord,
)


class _EchoArgs(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: str


class _EchoPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: str
    truncated: bool = False


class _EvidencePayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: str

    def evidence_digest(self) -> dict[str, JsonValue]:
        return {"source": "generic-payload", "values": [self.value]}


class _CountArgs(BaseModel):
    model_config = ConfigDict(frozen=True)

    count: int = Field(ge=1)


class _CountPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    count: int


class _FakeGate:
    def __init__(
        self,
        approved: bool,
        reason: str | None = None,
        actor: str = "human",
    ) -> None:
        self.approved = approved
        self.reason = reason
        self.actor = actor
        self.calls: list[tuple[ToolSpec, BaseModel, ToolContext]] = []

    def check(
        self,
        spec: ToolSpec,
        args: BaseModel,
        context: ToolContext,
    ) -> ApprovalOutcome:
        self.calls.append((spec, args, context))
        return ApprovalOutcome(approved=self.approved, reason=self.reason, actor=self.actor)


class _FakeSink:
    def __init__(self) -> None:
        self.records: list[ToolTraceRecord] = []

    def append(self, record: ToolTraceRecord) -> None:
        self.records.append(record)


class _ApprovalSink:
    def __init__(self) -> None:
        self.records: list[ToolTraceRecord] = []
        self.approvals: list[ApprovalTraceRecord] = []
        self.emissions: list[ApprovalTraceRecord | ToolTraceRecord] = []

    def append(self, record: ToolTraceRecord) -> None:
        self.records.append(record)
        self.emissions.append(record)

    def append_approval(self, record: ApprovalTraceRecord) -> None:
        self.approvals.append(record)
        self.emissions.append(record)


def _context(tmp_path: Path) -> ToolContext:
    return ToolContext(run_id="run-1", jail=PathJail(tmp_path))


def _spec(
    name: str = "echo",
    risk_level: str = "low",
    timeout_s: int = 60,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="Echo a value.",
        args_schema=_EchoArgs,
        returns_schema=_EchoPayload,
        risk_level=risk_level,
        timeout_s=timeout_s,
    )


def _registry_with_echo() -> ToolRegistry:
    registry = ToolRegistry()

    def handler(args: BaseModel, _context: ToolContext) -> BaseModel:
        parsed = _EchoArgs.model_validate(args)
        return _EchoPayload(value=parsed.value)

    registry.register(_spec(), handler)
    return registry


def test_registry__dispatches_registered_tool_successfully(tmp_path: Path) -> None:
    registry = _registry_with_echo()

    result = registry.dispatch("echo", {"value": "ok"}, _context(tmp_path))

    assert result.ok is True
    assert result.data == _EchoPayload(value="ok")
    assert result.error is None
    assert result.meta.tool_name == "echo"
    assert result.meta.latency_ms >= 0


def test_registry__rejects_duplicate_tool_name() -> None:
    registry = ToolRegistry()

    registry.register(_spec(), lambda _args, _context: _EchoPayload(value="ok"))

    with pytest.raises(ValueError):
        registry.register(_spec(), lambda _args, _context: _EchoPayload(value="again"))


def test_registry__reports_unknown_tool_with_available_names(tmp_path: Path) -> None:
    registry = _registry_with_echo()

    result = registry.dispatch("missing", {}, _context(tmp_path))

    assert result.ok is False
    assert result.error is not None
    assert result.error.type is ErrorType.InvalidArgsError
    assert "echo" in result.error.message


def test_registry__rejects_invalid_args_before_handler(tmp_path: Path) -> None:
    called = False
    registry = ToolRegistry()

    def handler(_args: BaseModel, _context: ToolContext) -> BaseModel:
        nonlocal called
        called = True
        return _EchoPayload(value="never")

    registry.register(_spec(), handler)

    result = registry.dispatch("echo", {}, _context(tmp_path))

    assert called is False
    assert result.ok is False
    assert result.error is not None
    assert result.error.type is ErrorType.InvalidArgsError
    assert "value" in result.error.message


def test_registry__maps_tool_failure_to_declared_error(tmp_path: Path) -> None:
    registry = ToolRegistry()

    def handler(_args: BaseModel, _context: ToolContext) -> BaseModel:
        raise ToolFailure(
            ErrorType.BinaryFileError,
            "Use read_file only on text files.",
            {"path": "image.png"},
        )

    registry.register(_spec(), handler)

    result = registry.dispatch("echo", {"value": "x"}, _context(tmp_path))

    assert result.ok is False
    assert result.error is not None
    assert result.error.type is ErrorType.BinaryFileError
    assert result.error.details == {"path": "image.png"}


def test_registry__maps_path_jail_violation_to_path_jail_error(tmp_path: Path) -> None:
    registry = ToolRegistry()

    def handler(_args: BaseModel, _context: ToolContext) -> BaseModel:
        raise PathJailViolation("Path escaped.")

    registry.register(_spec(), handler)

    result = registry.dispatch("echo", {"value": "x"}, _context(tmp_path))

    assert result.ok is False
    assert result.error is not None
    assert result.error.type is ErrorType.PathJailError
    assert result.error.message == "Path escaped."


def test_registry__maps_runtime_error_to_internal_tool_error(tmp_path: Path) -> None:
    registry = ToolRegistry()

    def handler(_args: BaseModel, _context: ToolContext) -> BaseModel:
        raise RuntimeError("boom")

    registry.register(_spec(), handler)

    result = registry.dispatch("echo", {"value": "x"}, _context(tmp_path))

    assert result.ok is False
    assert result.error is not None
    assert result.error.type is ErrorType.InternalToolError
    assert "RuntimeError" in result.error.message
    assert "Traceback" not in result.error.message


def test_registry__times_out_slow_handler(tmp_path: Path) -> None:
    registry = ToolRegistry()

    def handler(_args: BaseModel, _context: ToolContext) -> BaseModel:
        sleep(2)
        return _EchoPayload(value="late")

    registry.register(_spec(timeout_s=1), handler)

    result = registry.dispatch("echo", {"value": "x"}, _context(tmp_path))

    assert result.ok is False
    assert result.error is not None
    assert result.error.type is ErrorType.ToolTimeoutError


def test_registry__mirrors_payload_truncated_flag(tmp_path: Path) -> None:
    registry = ToolRegistry()

    registry.register(_spec(), lambda _args, _context: _EchoPayload(value="x", truncated=True))

    result = registry.dispatch("echo", {"value": "x"}, _context(tmp_path))

    assert result.ok is True
    assert result.meta.truncated is True


def test_registry__fails_closed_for_high_risk_without_gate(tmp_path: Path) -> None:
    called = False
    sink = _ApprovalSink()
    registry = ToolRegistry(trace_sink=sink)

    def handler(_args: BaseModel, _context: ToolContext) -> BaseModel:
        nonlocal called
        called = True
        return _EchoPayload(value="unsafe")

    registry.register(_spec(risk_level="high"), handler)

    result = registry.dispatch("echo", {"value": "x"}, _context(tmp_path))

    assert called is False
    assert result.ok is False
    assert result.error is not None
    assert result.error.type is ErrorType.ApprovalDeniedError
    assert "no approval gate configured" in result.error.message
    assert len(sink.approvals) == 1
    approval = sink.approvals[0]
    assert approval.run_id == "run-1"
    assert approval.tool_name == "echo"
    assert approval.risk_level == "high"
    assert approval.decision == "denied"
    assert approval.actor == "system"
    assert approval.reason == "no approval gate configured for high-risk tool 'echo'."
    assert approval.ts.tzinfo is UTC
    assert sink.emissions == [approval, sink.records[0]]


def test_registry__executes_high_risk_when_gate_approves(tmp_path: Path) -> None:
    gate = _FakeGate(approved=True, actor="policy:sandbox")
    sink = _ApprovalSink()
    registry = ToolRegistry(approval_gate=gate, trace_sink=sink)
    registry.register(
        _spec(risk_level="high"), lambda args, _context: _EchoArgs.model_validate(args)
    )

    result = registry.dispatch("echo", {"value": "x"}, _context(tmp_path))

    assert result.ok is True
    assert result.data == _EchoArgs(value="x")
    assert len(gate.calls) == 1
    assert len(sink.approvals) == 1
    approval = sink.approvals[0]
    assert approval.run_id == "run-1"
    assert approval.tool_name == "echo"
    assert approval.risk_level == "high"
    assert approval.decision == "approved"
    assert approval.actor == "policy:sandbox"
    assert approval.reason is None
    assert approval.ts.tzinfo is UTC
    assert sink.emissions == [approval, sink.records[0]]


def test_registry__blocks_high_risk_when_gate_denies(tmp_path: Path) -> None:
    called = False
    gate = _FakeGate(approved=False, reason="needs review")
    sink = _ApprovalSink()
    registry = ToolRegistry(approval_gate=gate, trace_sink=sink)

    def handler(_args: BaseModel, _context: ToolContext) -> BaseModel:
        nonlocal called
        called = True
        return _EchoPayload(value="unsafe")

    registry.register(_spec(risk_level="high"), handler)

    result = registry.dispatch("echo", {"value": "x"}, _context(tmp_path))

    assert called is False
    assert result.ok is False
    assert result.error is not None
    assert result.error.type is ErrorType.ApprovalDeniedError
    assert result.error.message == "needs review"
    assert len(gate.calls) == 1
    assert len(sink.approvals) == 1
    approval = sink.approvals[0]
    assert approval.run_id == "run-1"
    assert approval.tool_name == "echo"
    assert approval.risk_level == "high"
    assert approval.decision == "denied"
    assert approval.actor == "human"
    assert approval.reason == "needs review"
    assert approval.ts.tzinfo is UTC
    assert sink.emissions == [approval, sink.records[0]]


@pytest.mark.parametrize("risk_level", ("low", "medium"))
def test_registry__does_not_request_or_trace_approval_below_high_risk(
    tmp_path: Path,
    risk_level: str,
) -> None:
    gate = _FakeGate(approved=True)
    sink = _ApprovalSink()
    registry = ToolRegistry(approval_gate=gate, trace_sink=sink)
    registry.register(
        _spec(risk_level=risk_level),
        lambda _args, _context: _EchoPayload(value="safe"),
    )

    result = registry.dispatch("echo", {"value": "x"}, _context(tmp_path))

    assert result.ok is True
    assert gate.calls == []
    assert sink.approvals == []
    assert sink.emissions == sink.records


def test_registry__high_risk_dispatch_supports_sink_without_append_approval(
    tmp_path: Path,
) -> None:
    sink = _FakeSink()
    registry = ToolRegistry(approval_gate=_FakeGate(approved=True), trace_sink=sink)
    registry.register(
        _spec(risk_level="high"),
        lambda _args, _context: _EchoPayload(value="approved"),
    )

    result = registry.dispatch("echo", {"value": "x"}, _context(tmp_path))

    assert result.ok is True
    assert len(sink.records) == 1
    assert sink.records[0].ok is True


def test_registry__appends_trace_records_for_success_and_failure(tmp_path: Path) -> None:
    sink = _FakeSink()
    registry = ToolRegistry(trace_sink=sink)
    registry.register(_spec(), lambda _args, _context: _EchoPayload(value="ok", truncated=True))

    success = registry.dispatch("echo", {"value": "x"}, _context(tmp_path))
    failure = registry.dispatch("missing", {}, _context(tmp_path))

    assert success.ok is True
    assert failure.ok is False
    assert len(sink.records) == 2
    assert sink.records[0].run_id == "run-1"
    assert sink.records[0].tool_name == "echo"
    assert sink.records[0].args == {"value": "x"}
    assert sink.records[0].ok is True
    assert sink.records[0].error_type is None
    assert sink.records[0].latency_ms >= 0
    assert sink.records[0].truncated is True
    assert sink.records[0].outcome is None
    assert sink.records[0].ts.tzinfo is UTC
    assert sink.records[1].tool_name == "missing"
    assert sink.records[1].ok is False
    assert sink.records[1].error_type is ErrorType.InvalidArgsError
    assert sink.records[1].outcome is None


def test_registry__duck_types_payload_evidence_digest_into_trace_outcome(
    tmp_path: Path,
) -> None:
    sink = _FakeSink()
    registry = ToolRegistry(trace_sink=sink)
    registry.register(
        ToolSpec(
            name="generic_evidence",
            description="Return generic structured evidence.",
            args_schema=_EchoArgs,
            returns_schema=_EvidencePayload,
            risk_level="low",
        ),
        lambda args, _context: _EvidencePayload(value=_EchoArgs.model_validate(args).value),
    )

    result = registry.dispatch(
        "generic_evidence",
        {"value": "objective"},
        _context(tmp_path),
    )

    assert result.ok is True
    assert len(sink.records) == 1
    assert sink.records[0].outcome == {
        "source": "generic-payload",
        "values": ["objective"],
    }


def test_registry__loop_guard_blocks_identical_consecutive_call_and_traces_it(
    tmp_path: Path,
) -> None:
    executions: list[str] = []
    sink = _FakeSink()
    registry = ToolRegistry(trace_sink=sink, loop_guard=LoopGuard())

    def handler(args: BaseModel, _context: ToolContext) -> BaseModel:
        parsed = _EchoArgs.model_validate(args)
        executions.append(parsed.value)
        return _EchoPayload(value=parsed.value)

    registry.register(_spec(), handler)

    first = registry.dispatch("echo", {"value": "same"}, _context(tmp_path))
    second = registry.dispatch("echo", {"value": "same"}, _context(tmp_path))

    assert first.ok is True
    assert second.ok is False
    assert second.error is not None
    assert second.error.type is ErrorType.LoopBlockedError
    assert second.error.details == {"reason": "duplicate_call"}
    assert executions == ["same"]
    assert len(sink.records) == 2
    assert sink.records[1].error_type is ErrorType.LoopBlockedError


def test_registry__loop_guard_allows_non_consecutive_and_different_calls(
    tmp_path: Path,
) -> None:
    executions: list[tuple[str, str]] = []
    registry = ToolRegistry(loop_guard=LoopGuard())

    def handler(args: BaseModel, _context: ToolContext) -> BaseModel:
        parsed = _EchoArgs.model_validate(args)
        executions.append(("echo", parsed.value))
        return _EchoPayload(value=parsed.value)

    def other_handler(args: BaseModel, _context: ToolContext) -> BaseModel:
        parsed = _EchoArgs.model_validate(args)
        executions.append(("other", parsed.value))
        return _EchoPayload(value=parsed.value)

    registry.register(_spec(), handler)
    registry.register(_spec(name="other"), other_handler)

    results = [
        registry.dispatch("echo", {"value": "a"}, _context(tmp_path)),
        registry.dispatch("other", {"value": "b"}, _context(tmp_path)),
        registry.dispatch("echo", {"value": "a"}, _context(tmp_path)),
        registry.dispatch("echo", {"value": "changed"}, _context(tmp_path)),
    ]

    assert all(result.ok for result in results)
    assert executions == [
        ("echo", "a"),
        ("other", "b"),
        ("echo", "a"),
        ("echo", "changed"),
    ]


def test_registry__loop_guard_isolates_calls_by_run(tmp_path: Path) -> None:
    executions: list[str] = []
    registry = ToolRegistry(loop_guard=LoopGuard())

    def handler(args: BaseModel, context: ToolContext) -> BaseModel:
        parsed = _EchoArgs.model_validate(args)
        executions.append(context.run_id)
        return _EchoPayload(value=parsed.value)

    registry.register(_spec(), handler)
    run_one = ToolContext(run_id="run-1", jail=PathJail(tmp_path))
    run_two = ToolContext(run_id="run-2", jail=PathJail(tmp_path))

    first = registry.dispatch("echo", {"value": "same"}, run_one)
    second = registry.dispatch("echo", {"value": "same"}, run_two)

    assert first.ok is True
    assert second.ok is True
    assert executions == ["run-1", "run-2"]


def test_registry__loop_guard_blocks_high_risk_before_second_approval(
    tmp_path: Path,
) -> None:
    executions = 0
    gate = _FakeGate(approved=True)
    sink = _ApprovalSink()
    registry = ToolRegistry(
        approval_gate=gate,
        trace_sink=sink,
        loop_guard=LoopGuard(),
    )

    def handler(args: BaseModel, _context: ToolContext) -> BaseModel:
        nonlocal executions
        executions += 1
        parsed = _EchoArgs.model_validate(args)
        return _EchoPayload(value=parsed.value)

    registry.register(_spec(risk_level="high"), handler)

    first = registry.dispatch("echo", {"value": "same"}, _context(tmp_path))
    second = registry.dispatch("echo", {"value": "same"}, _context(tmp_path))

    assert first.ok is True
    assert second.ok is False
    assert second.error is not None
    assert second.error.type is ErrorType.LoopBlockedError
    assert len(gate.calls) == 1
    assert executions == 1
    assert len(sink.approvals) == 1
    assert sink.approvals[0].decision == "approved"
    assert len(sink.records) == 2
    assert sink.records[1].error_type is ErrorType.LoopBlockedError


def test_registry__loop_guard_is_disabled_by_default(tmp_path: Path) -> None:
    executions = 0
    registry = ToolRegistry()

    def handler(args: BaseModel, _context: ToolContext) -> BaseModel:
        nonlocal executions
        executions += 1
        parsed = _EchoArgs.model_validate(args)
        return _EchoPayload(value=parsed.value)

    registry.register(_spec(), handler)

    first = registry.dispatch("echo", {"value": "same"}, _context(tmp_path))
    second = registry.dispatch("echo", {"value": "same"}, _context(tmp_path))

    assert first.ok is True
    assert second.ok is True
    assert executions == 2


def test_registry__to_llm_schema_uses_function_format_and_args_parameters() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="count",
            description="Count items.",
            args_schema=_CountArgs,
            returns_schema=_CountPayload,
            risk_level="low",
        ),
        lambda args, _context: _CountPayload(count=_CountArgs.model_validate(args).count),
    )

    schema = registry.to_llm_schema()

    assert schema[0]["type"] == "function"
    function = schema[0]["function"]
    assert isinstance(function, dict)
    assert function["name"] == "count"
    assert function["description"] == "Count items."
    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    assert parameters["type"] == "object"
    assert parameters["properties"]["count"]["minimum"] == 1
    assert parameters["required"] == ["count"]
