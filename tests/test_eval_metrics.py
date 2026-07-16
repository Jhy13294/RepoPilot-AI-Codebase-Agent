from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from app.schemas.trace import TraceEvent, TraceEventKind
from eval.metrics import RunTrace, compute_metrics, render_metrics_markdown

_START = datetime(2026, 7, 16, 8, 0, tzinfo=UTC)
_METHOD_STEPS = (
    "Average Steps counts `tool_call` trace events on successful runs; it is not the agent "
    "loop's `steps_used`."
)
_METHOD_LATENCY = (
    "Run trace latency sums recorded `latency_ms` values within each run, then uses nearest-rank "
    "percentiles. Runs without recorded latency are excluded."
)
_METHOD_APPROVAL = (
    'Approval decisions with `actor == "system"` are counted as ungated safety alerts.'
)


def _event(
    run_id: str,
    seq: int,
    kind: TraceEventKind,
    payload: dict[str, object] | None = None,
    *,
    latency_ms: int | None = None,
    cost_usd: float | None = None,
) -> TraceEvent:
    return TraceEvent.model_validate(
        {
            "run_id": run_id,
            "seq": seq,
            "ts": _START + timedelta(seconds=seq),
            "kind": kind,
            "payload": payload or {},
            "latency_ms": latency_ms,
            "cost_usd": cost_usd,
        }
    )


def _run(
    run_id: str,
    task_type: str,
    final_status: str,
    events: list[TraceEvent],
) -> RunTrace:
    return RunTrace(
        run_id=run_id,
        task_type=task_type,
        final_status=final_status,
        events=events,
    )


def _representative_runs() -> list[RunTrace]:
    return [
        _run(
            "run-localization-done",
            "bug_localization",
            "DONE",
            [
                _event(
                    "run-localization-done",
                    0,
                    TraceEventKind.tool_call,
                    {"tool_name": "read_file", "ok": True, "error_type": None},
                    latency_ms=10,
                    cost_usd=0.1,
                ),
                _event(
                    "run-localization-done",
                    1,
                    TraceEventKind.tool_call,
                    {
                        "tool_name": "search_code",
                        "ok": False,
                        "error_type": "InvalidArgsError",
                    },
                    latency_ms=20,
                    cost_usd=0.2,
                ),
                _event(
                    "run-localization-done",
                    2,
                    TraceEventKind.approval_decision,
                    {"actor": "human"},
                ),
            ],
        ),
        _run(
            "run-localization-failed",
            "bug_localization",
            "FAILED",
            [
                _event(
                    "run-localization-failed",
                    0,
                    TraceEventKind.approval_decision,
                    {"actor": "system"},
                ),
                _event(
                    "run-localization-failed",
                    1,
                    TraceEventKind.tool_call,
                    {
                        "tool_name": "read_file",
                        "ok": False,
                        "error_type": "NotFoundError",
                    },
                    latency_ms=100,
                    cost_usd=0.3,
                ),
                _event(
                    "run-localization-failed",
                    2,
                    TraceEventKind.tool_call,
                    {
                        "tool_name": "search_code",
                        "ok": False,
                        "error_type": "PathJailError",
                    },
                    latency_ms=50,
                    cost_usd=0.4,
                ),
                _event(
                    "run-localization-failed",
                    3,
                    TraceEventKind.report,
                    {"summary": "Failure report."},
                ),
            ],
        ),
        _run(
            "run-explanation-failed",
            "bug_explanation",
            "FAILED",
            [
                _event(
                    "run-explanation-failed",
                    0,
                    TraceEventKind.approval_decision,
                    {"actor": "policy:sandbox"},
                ),
                _event(
                    "run-explanation-failed",
                    1,
                    TraceEventKind.tool_call,
                    {
                        "tool_name": "run_tests",
                        "ok": True,
                        "error_type": None,
                        "outcome": {"failed": 2},
                    },
                    latency_ms=200,
                ),
            ],
        ),
        _run(
            "run-explanation-done",
            "bug_explanation",
            "DONE",
            [
                _event(
                    "run-explanation-done",
                    0,
                    TraceEventKind.tool_call,
                    {"tool_name": "search_code", "ok": True, "error_type": None},
                    latency_ms=40,
                ),
                _event(
                    "run-explanation-done",
                    1,
                    TraceEventKind.error,
                    {"reason": "typed_failure"},
                    latency_ms=60,
                ),
                _event(
                    "run-explanation-done",
                    2,
                    TraceEventKind.report,
                    {"summary": "Recovered report."},
                ),
            ],
        ),
    ]


def test_compute_metrics__derives_all_metrics_from_trace_events() -> None:
    metrics = compute_metrics(_representative_runs())

    assert metrics.run_count == 4
    assert metrics.task_success_rate == 0.5
    assert metrics.task_success_by_type == {
        "bug_explanation": 0.5,
        "bug_localization": 0.5,
    }
    assert metrics.tool_call_accuracy == pytest.approx(5 / 6)
    assert metrics.invalid_tool_call_rate == pytest.approx(1 / 6)
    assert metrics.hallucinated_path_rate == pytest.approx(2 / 6)
    assert metrics.average_steps == 1.5
    assert metrics.recovery_success_rate == 0.5
    assert metrics.graceful_failure_rate == 0.5
    assert metrics.human_approval_trigger_rate == pytest.approx(2 / 3)
    assert metrics.approval_ungated_count == 1
    assert metrics.latency_p50_ms == 100
    assert metrics.latency_p95_ms == 200
    assert metrics.per_tool_latency == {
        "read_file": (10, 100),
        "run_tests": (200, 200),
        "search_code": (40, 50),
    }
    assert metrics.total_cost_usd == pytest.approx(1.0)


def test_render_metrics_markdown__is_deterministic_snapshot() -> None:
    markdown = render_metrics_markdown(compute_metrics(_representative_runs()))

    assert (
        markdown
        == f"""# Evaluation Metrics

Runs: 4

## Summary

| Metric | Value |
| --- | ---: |
| Task Success Rate | 0.500 |
| Tool Call Accuracy | 0.833 |
| Invalid Tool Call Rate | 0.167 |
| Hallucinated-path Rate | 0.333 |
| Average Steps | 1.50 |
| Recovery Success Rate | 0.500 |
| Graceful Failure Rate | 0.500 |
| Human Approval Trigger Rate | 0.667 |
| Approval Ungated Count | 1 |
| Run Trace Latency p50 | 100 ms |
| Run Trace Latency p95 | 200 ms |
| Estimated Cost | $1.000000 |

## Task Success by Type

| Task Type | Success Rate |
| --- | ---: |
| bug_explanation | 0.500 |
| bug_localization | 0.500 |

## Per-Tool Latency

| Tool | p50 | p95 |
| --- | ---: | ---: |
| read_file | 10 ms | 100 ms |
| run_tests | 200 ms | 200 ms |
| search_code | 40 ms | 50 ms |

## Method

{_METHOD_STEPS}

{_METHOD_LATENCY}

{_METHOD_APPROVAL}
"""
    )


def test_empty_suite__returns_none_denominators_and_n_a_snapshot() -> None:
    metrics = compute_metrics([])

    assert metrics.model_dump() == {
        "task_success_rate": None,
        "task_success_by_type": {},
        "tool_call_accuracy": None,
        "invalid_tool_call_rate": None,
        "hallucinated_path_rate": None,
        "average_steps": None,
        "recovery_success_rate": None,
        "graceful_failure_rate": None,
        "human_approval_trigger_rate": None,
        "approval_ungated_count": 0,
        "latency_p50_ms": None,
        "latency_p95_ms": None,
        "per_tool_latency": {},
        "total_cost_usd": None,
        "run_count": 0,
    }
    assert (
        render_metrics_markdown(metrics)
        == f"""# Evaluation Metrics

Runs: 0

## Summary

| Metric | Value |
| --- | ---: |
| Task Success Rate | n/a |
| Tool Call Accuracy | n/a |
| Invalid Tool Call Rate | n/a |
| Hallucinated-path Rate | n/a |
| Average Steps | n/a |
| Recovery Success Rate | n/a |
| Graceful Failure Rate | n/a |
| Human Approval Trigger Rate | n/a |
| Approval Ungated Count | 0 |
| Run Trace Latency p50 | n/a |
| Run Trace Latency p95 | n/a |
| Estimated Cost | n/a |

## Task Success by Type

| Task Type | Success Rate |
| --- | ---: |
| n/a | n/a |

## Per-Tool Latency

| Tool | p50 | p95 |
| --- | ---: | ---: |
| n/a | n/a | n/a |

## Method

{_METHOD_STEPS}

{_METHOD_LATENCY}

{_METHOD_APPROVAL}
"""
    )


def test_run_trace_models__are_frozen_and_forbid_extra_fields() -> None:
    run = _run("run-1", "repo_qa", "DONE", [])

    with pytest.raises(ValidationError, match="frozen"):
        run.final_status = "FAILED"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        RunTrace.model_validate(
            {
                "run_id": "run-1",
                "task_type": "repo_qa",
                "final_status": "DONE",
                "events": [],
                "unexpected": True,
            }
        )
