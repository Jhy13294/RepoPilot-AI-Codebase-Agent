"""Trace-derived evaluation metrics and deterministic Markdown rendering."""

from collections.abc import Sequence
from math import ceil

from pydantic import BaseModel, ConfigDict

from app.schemas.trace import TraceEvent

_INVALID_TOOL_CALL_ERROR = "InvalidArgsError"
_HALLUCINATED_PATH_ERRORS = frozenset({"NotFoundError", "PathJailError"})


class RunTrace(BaseModel):
    """Caller-supplied run facts paired with the run's persisted trace events."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    task_type: str
    final_status: str
    events: list[TraceEvent]


class SuiteMetrics(BaseModel):
    """Trace-derived metrics for an evaluation suite."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_success_rate: float | None
    task_success_by_type: dict[str, float]
    tool_call_accuracy: float | None
    invalid_tool_call_rate: float | None
    hallucinated_path_rate: float | None
    average_steps: float | None
    recovery_success_rate: float | None
    graceful_failure_rate: float | None
    human_approval_trigger_rate: float | None
    approval_ungated_count: int
    latency_p50_ms: int | None
    latency_p95_ms: int | None
    per_tool_latency: dict[str, tuple[int, int]]
    total_cost_usd: float | None
    run_count: int


def compute_metrics(runs: Sequence[RunTrace]) -> SuiteMetrics:
    """Compute suite metrics from caller-owned run facts and trace events."""
    run_list = list(runs)
    tool_calls = [event for run in run_list for event in run.events if _is_kind(event, "tool_call")]
    invalid_calls = [
        event for event in tool_calls if event.payload.get("error_type") == _INVALID_TOOL_CALL_ERROR
    ]
    hallucinated_path_calls = [event for event in tool_calls if _is_hallucinated_path(event)]
    successful_runs = [run for run in run_list if run.final_status == "DONE"]
    recovery_runs = [run for run in run_list if _has_typed_failure(run)]
    failed_runs = [run for run in run_list if run.final_status == "FAILED"]
    approval_events = [
        event for run in run_list for event in run.events if _is_kind(event, "approval_decision")
    ]
    ungated_approvals = [
        event for event in approval_events if event.payload.get("actor") == "system"
    ]

    run_latencies: list[int] = []
    per_tool_samples: dict[str, list[int]] = {}
    recorded_costs: list[float] = []
    for run in run_list:
        latencies = [event.latency_ms for event in run.events if event.latency_ms is not None]
        if latencies:
            run_latencies.append(sum(latencies))

        for event in run.events:
            if event.cost_usd is not None:
                recorded_costs.append(event.cost_usd)
            if not _is_kind(event, "tool_call") or event.latency_ms is None:
                continue
            tool_name = event.payload.get("tool_name")
            if isinstance(tool_name, str):
                per_tool_samples.setdefault(tool_name, []).append(event.latency_ms)

    latency_percentiles = _p50_p95(run_latencies)
    per_tool_latency = {
        tool_name: _required_percentiles(samples)
        for tool_name, samples in sorted(per_tool_samples.items())
    }

    return SuiteMetrics(
        task_success_rate=_ratio(len(successful_runs), len(run_list)),
        task_success_by_type=_success_by_type(run_list),
        tool_call_accuracy=_inverse_rate(len(invalid_calls), len(tool_calls)),
        invalid_tool_call_rate=_ratio(len(invalid_calls), len(tool_calls)),
        hallucinated_path_rate=_ratio(len(hallucinated_path_calls), len(tool_calls)),
        average_steps=_mean(
            [sum(_is_kind(event, "tool_call") for event in run.events) for run in successful_runs]
        ),
        recovery_success_rate=_ratio(
            sum(run.final_status == "DONE" for run in recovery_runs),
            len(recovery_runs),
        ),
        graceful_failure_rate=_ratio(
            sum(any(_is_kind(event, "report") for event in run.events) for run in failed_runs),
            len(failed_runs),
        ),
        human_approval_trigger_rate=_inverse_rate(
            len(ungated_approvals),
            len(approval_events),
        ),
        approval_ungated_count=len(ungated_approvals),
        latency_p50_ms=latency_percentiles[0] if latency_percentiles is not None else None,
        latency_p95_ms=latency_percentiles[1] if latency_percentiles is not None else None,
        per_tool_latency=per_tool_latency,
        total_cost_usd=sum(recorded_costs) if recorded_costs else None,
        run_count=len(run_list),
    )


def render_metrics_markdown(metrics: SuiteMetrics) -> str:
    """Render metrics as deterministic Markdown without timestamps or environment data."""
    lines = [
        "# Evaluation Metrics",
        "",
        f"Runs: {metrics.run_count}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Task Success Rate | {_format_rate(metrics.task_success_rate)} |",
        f"| Tool Call Accuracy | {_format_rate(metrics.tool_call_accuracy)} |",
        f"| Invalid Tool Call Rate | {_format_rate(metrics.invalid_tool_call_rate)} |",
        f"| Hallucinated-path Rate | {_format_rate(metrics.hallucinated_path_rate)} |",
        f"| Average Steps | {_format_steps(metrics.average_steps)} |",
        f"| Recovery Success Rate | {_format_rate(metrics.recovery_success_rate)} |",
        f"| Graceful Failure Rate | {_format_rate(metrics.graceful_failure_rate)} |",
        (f"| Human Approval Trigger Rate | {_format_rate(metrics.human_approval_trigger_rate)} |"),
        f"| Approval Ungated Count | {metrics.approval_ungated_count} |",
        f"| Run Trace Latency p50 | {_format_latency(metrics.latency_p50_ms)} |",
        f"| Run Trace Latency p95 | {_format_latency(metrics.latency_p95_ms)} |",
        f"| Estimated Cost | {_format_cost(metrics.total_cost_usd)} |",
        "",
        "## Task Success by Type",
        "",
        "| Task Type | Success Rate |",
        "| --- | ---: |",
    ]
    if metrics.task_success_by_type:
        lines.extend(
            f"| {_escape_cell(task_type)} | {_format_rate(rate)} |"
            for task_type, rate in sorted(metrics.task_success_by_type.items())
        )
    else:
        lines.append("| n/a | n/a |")

    lines.extend(
        [
            "",
            "## Per-Tool Latency",
            "",
            "| Tool | p50 | p95 |",
            "| --- | ---: | ---: |",
        ]
    )
    if metrics.per_tool_latency:
        lines.extend(
            (
                f"| {_escape_cell(tool_name)} | {_format_latency(percentiles[0])} | "
                f"{_format_latency(percentiles[1])} |"
            )
            for tool_name, percentiles in sorted(metrics.per_tool_latency.items())
        )
    else:
        lines.append("| n/a | n/a | n/a |")

    lines.extend(
        [
            "",
            "## Method",
            "",
            (
                "Average Steps counts `tool_call` trace events on successful runs; it is not "
                "the agent loop's `steps_used`."
            ),
            "",
            (
                "Run trace latency sums recorded `latency_ms` values within each run, then uses "
                "nearest-rank percentiles. Runs without recorded latency are excluded."
            ),
            "",
            ('Approval decisions with `actor == "system"` are counted as ungated safety alerts.'),
            "",
        ]
    )
    return "\n".join(lines)


def _success_by_type(runs: Sequence[RunTrace]) -> dict[str, float]:
    counts: dict[str, tuple[int, int]] = {}
    for run in runs:
        successful, total = counts.get(run.task_type, (0, 0))
        counts[run.task_type] = (successful + (run.final_status == "DONE"), total + 1)
    return {
        task_type: successful / total for task_type, (successful, total) in sorted(counts.items())
    }


def _has_typed_failure(run: RunTrace) -> bool:
    for event in run.events:
        if _is_kind(event, "error"):
            return True
        if not _is_kind(event, "tool_call"):
            continue
        if event.payload.get("ok") is False or _outcome_has_failures(event):
            return True
    return False


def _outcome_has_failures(event: TraceEvent) -> bool:
    outcome = event.payload.get("outcome")
    if not isinstance(outcome, dict):
        return False
    failed = outcome.get("failed")
    return isinstance(failed, (int, float)) and not isinstance(failed, bool) and failed > 0


def _is_hallucinated_path(event: TraceEvent) -> bool:
    error_type = event.payload.get("error_type")
    return isinstance(error_type, str) and error_type in _HALLUCINATED_PATH_ERRORS


def _is_kind(event: TraceEvent, kind: str) -> bool:
    return event.kind.value == kind


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _inverse_rate(failures: int, total: int) -> float | None:
    rate = _ratio(failures, total)
    return None if rate is None else 1 - rate


def _mean(values: Sequence[int]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _p50_p95(values: Sequence[int]) -> tuple[int, int] | None:
    if not values:
        return None
    ordered = sorted(values)
    return _nearest_rank(ordered, 50), _nearest_rank(ordered, 95)


def _required_percentiles(values: Sequence[int]) -> tuple[int, int]:
    percentiles = _p50_p95(values)
    if percentiles is None:
        raise ValueError("percentile samples must not be empty")
    return percentiles


def _nearest_rank(ordered_values: Sequence[int], percentile: int) -> int:
    index = max(0, ceil(percentile / 100 * len(ordered_values)) - 1)
    return ordered_values[index]


def _format_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _format_steps(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _format_latency(value: int | None) -> str:
    return "n/a" if value is None else f"{value} ms"


def _format_cost(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.6f}"


def _escape_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\r\n", "<br>").replace("\n", "<br>")
