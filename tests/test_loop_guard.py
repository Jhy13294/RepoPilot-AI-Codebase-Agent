from pathlib import Path

from pydantic import BaseModel, ConfigDict

from app.safety.loop_guard import LoopGuard
from app.safety.path_jail import PathJail
from app.schemas.tool_io import ErrorType
from app.tools.base import ToolContext
from app.tools.registry import ApprovalOutcome, ToolRegistry, ToolSpec


class _NestedArgs(BaseModel):
    model_config = ConfigDict(frozen=True)

    options: dict[str, int]


class _RationaleArgs(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: str
    rationale: str


class _RationalePayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: str


class _FakeGate:
    def __init__(self) -> None:
        self.calls: list[tuple[ToolSpec, BaseModel, ToolContext]] = []

    def check(
        self,
        spec: ToolSpec,
        args: BaseModel,
        context: ToolContext,
    ) -> ApprovalOutcome:
        self.calls.append((spec, args, context))
        return ApprovalOutcome(approved=True)


def _rationale_spec() -> ToolSpec:
    return ToolSpec(
        name="act",
        description="Perform one approval-gated action.",
        args_schema=_RationaleArgs,
        returns_schema=_RationalePayload,
        risk_level="high",
    )


def _context(root: Path) -> ToolContext:
    return ToolContext(run_id="run-1", jail=PathJail(root))


def test_loop_guard__blocks_identical_consecutive_calls_without_updating_last() -> None:
    guard = LoopGuard()
    args = _NestedArgs(options={"first": 1, "second": 2})

    assert guard.check("run-1", "inspect", args) is False
    assert guard.check("run-1", "inspect", args) is True
    assert guard.check("run-1", "inspect", args) is True


def test_loop_guard__canonicalizes_argument_key_order() -> None:
    guard = LoopGuard()
    first = _NestedArgs(options={"first": 1, "second": 2})
    reordered = _NestedArgs(options={"second": 2, "first": 1})

    assert guard.check("run-1", "inspect", first) is False
    assert guard.check("run-1", "inspect", reordered) is True


def test_loop_guard__allows_different_calls_and_records_each_allowed_call() -> None:
    guard = LoopGuard()
    first = _NestedArgs(options={"value": 1})
    second = _NestedArgs(options={"value": 2})

    assert guard.check("run-1", "inspect", first) is False
    assert guard.check("run-1", "inspect", second) is False
    assert guard.check("run-1", "inspect", first) is False
    assert guard.check("run-1", "other_tool", first) is False


def test_loop_guard__isolates_call_history_by_run() -> None:
    guard = LoopGuard()
    args = _NestedArgs(options={"value": 1})

    assert guard.check("run-1", "inspect", args) is False
    assert guard.check("run-2", "inspect", args) is False
    assert guard.check("run-1", "inspect", args) is True
    assert guard.check("run-2", "inspect", args) is True


def test_loop_guard__blocks_successful_effective_repeat_before_approval(
    tmp_path: Path,
) -> None:
    gate = _FakeGate()
    executions: list[str] = []
    registry = ToolRegistry(approval_gate=gate, loop_guard=LoopGuard())

    def handler(args: BaseModel, _context: ToolContext) -> BaseModel:
        parsed = _RationaleArgs.model_validate(args)
        executions.append(parsed.rationale)
        return _RationalePayload(value=parsed.value)

    registry.register(_rationale_spec(), handler)
    context = _context(tmp_path)

    first = registry.dispatch(
        "act",
        {"value": "same", "rationale": "First wording."},
        context,
    )
    second = registry.dispatch(
        "act",
        {"value": "same", "rationale": "Different wording."},
        context,
    )
    third = registry.dispatch(
        "act",
        {"value": "same", "rationale": "Third wording."},
        context,
    )

    assert first.ok is True
    assert second.ok is False
    assert second.error is not None
    assert second.error.type is ErrorType.LoopBlockedError
    assert "effectively-identical" in second.error.message
    assert third.ok is False
    assert third.error is not None
    assert third.error.type is ErrorType.LoopBlockedError
    assert executions == ["First wording."]
    assert len(gate.calls) == 1


def test_loop_guard__allows_changed_rationale_after_failure_but_blocks_verbatim_retry(
    tmp_path: Path,
) -> None:
    gate = _FakeGate()
    executions: list[str] = []
    registry = ToolRegistry(approval_gate=gate, loop_guard=LoopGuard())

    def handler(args: BaseModel, _context: ToolContext) -> BaseModel:
        parsed = _RationaleArgs.model_validate(args)
        executions.append(parsed.rationale)
        if len(executions) == 1:
            raise RuntimeError("injected execution failure")
        return _RationalePayload(value=parsed.value)

    registry.register(_rationale_spec(), handler)
    context = _context(tmp_path)
    first_args = {"value": "same", "rationale": "Initial attempt."}

    first = registry.dispatch("act", first_args, context)
    verbatim_retry = registry.dispatch("act", first_args, context)
    repaired_retry = registry.dispatch(
        "act",
        {"value": "same", "rationale": "Retry after the injected failure."},
        context,
    )

    assert first.ok is False
    assert first.error is not None
    assert first.error.type is ErrorType.InternalToolError
    assert verbatim_retry.ok is False
    assert verbatim_retry.error is not None
    assert verbatim_retry.error.type is ErrorType.LoopBlockedError
    assert repaired_retry.ok is True
    assert executions == ["Initial attempt.", "Retry after the injected failure."]
    assert len(gate.calls) == 2
