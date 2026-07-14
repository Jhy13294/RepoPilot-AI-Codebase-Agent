from pydantic import BaseModel, ConfigDict

from app.safety.loop_guard import LoopGuard


class _NestedArgs(BaseModel):
    model_config = ConfigDict(frozen=True)

    options: dict[str, int]


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
