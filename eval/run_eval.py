"""Minimal issue-only evaluation runner for RepoPilot."""

import argparse
import json
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.agent.critic import Critic
from app.agent.executor import Executor
from app.agent.loop import run_agent_loop
from app.agent.planner import Planner
from app.agent.reporter import Reporter
from app.agent.state import Budgets, TaskSpec
from app.config import load_settings
from app.safety.path_jail import PathJail
from app.schemas.agent_io import CitationGrounding, RunResult
from app.services.llm_client import LLMClient, build_llm_client
from app.storage.db import Database
from app.storage.trace_store import RegistryTraceSink, TraceStore
from app.tools.get_file_tree import register as register_get_file_tree
from app.tools.read_file import register as register_read_file
from app.tools.registry import ToolRegistry, TraceSink
from app.tools.search_code import register as register_search_code
from eval.metrics import RunTrace, compute_metrics, render_metrics_markdown
from eval.scorers import (
    ExplanationScore,
    LocalizationScore,
    score_bug_explanation,
    score_bug_localization,
)

IssueTaskType = Literal["bug_localization", "bug_explanation"]
EvalType = Literal["issue"]
_ISSUE_TASK_TYPES = frozenset({"bug_localization", "bug_explanation"})


class EvalExpected(BaseModel):
    """Gold data for an issue-analysis eval task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gold_file: str
    gold_line_range: tuple[int, int] | None = None
    rubric_keywords: list[str] = Field(default_factory=list)
    requires_valid_citation: bool = False

    @model_validator(mode="after")
    def validate_line_range(self) -> Self:
        if self.gold_line_range is None:
            return self
        start, end = self.gold_line_range
        if start < 1 or end < start:
            raise ValueError("gold_line_range must be a 1-based inclusive range.")
        return self


class EvalBudgets(BaseModel):
    """Per-task run budget used by the issue-only runner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_steps: int = Field(default=20, ge=1)
    max_replans: int = Field(default=3, ge=1)
    max_fix_cycles: int = Field(default=2, ge=1)


class EvalTask(BaseModel):
    """One issue-analysis task selected from eval/tasks.json."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    type: IssueTaskType
    fixture: str
    issue: str
    expected: EvalExpected
    budgets: EvalBudgets = Field(default_factory=EvalBudgets)


class TaskRunReport(BaseModel):
    """Scored result for one task repetition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    task_type: IssueTaskType
    repeat_index: int = Field(ge=0)
    run_id: str
    status: str
    steps: int = Field(ge=0)
    cost_usd: float | None
    suspects: list[str]
    citations: list[str]
    grounding_checks: list[CitationGrounding]
    localization: LocalizationScore | None = None
    explanation: ExplanationScore | None = None


class SuiteReport(BaseModel):
    """Aggregate report for an issue-only eval suite run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    generated_at: str
    task_filter: EvalType
    repeat: int = Field(ge=1)
    task_count: int = Field(ge=0)
    run_count: int = Field(ge=0)
    top3_hit_rate: float | None
    citation_valid_rate: float | None
    mean_steps: float | None
    total_cost_usd: float | None
    results: list[TaskRunReport]


def load_eval_tasks(tasks_path: Path, *, task_filter: EvalType = "issue") -> list[EvalTask]:
    """Load and validate the issue-analysis subset of eval/tasks.json."""
    if task_filter != "issue":
        raise ValueError("Only --type issue is supported by this runner.")

    raw_suite = json.loads(tasks_path.read_text(encoding="utf-8"))
    raw_tasks = raw_suite.get("tasks") if isinstance(raw_suite, dict) else None
    if not isinstance(raw_tasks, list):
        raise ValueError("Task suite must contain a tasks list.")

    tasks: list[EvalTask] = []
    for raw_task in raw_tasks:
        if not isinstance(raw_task, dict) or raw_task.get("type") not in _ISSUE_TASK_TYPES:
            continue
        tasks.append(EvalTask.model_validate(raw_task))
    return tasks


def run_suite(
    tasks: Sequence[EvalTask],
    fixtures_root: Path,
    *,
    client: LLMClient,
    repeat: int = 1,
    task_filter: EvalType = "issue",
    work_dir: Path | None = None,
) -> SuiteReport:
    """Run issue-analysis tasks through the agent loop and score their reports."""
    if task_filter != "issue":
        raise ValueError("Only issue tasks are supported.")
    if repeat < 1:
        raise ValueError("repeat must be at least one.")

    run_root = (
        work_dir if work_dir is not None else Path(tempfile.mkdtemp(prefix="repopilot-eval-"))
    )
    run_root.mkdir(parents=True, exist_ok=True)
    store = TraceStore(run_root / "traces")
    database = Database(run_root / "runs.sqlite3")
    results: list[TaskRunReport] = []

    for task in tasks:
        fixture_root = fixtures_root / task.fixture
        jail = PathJail(fixture_root)
        _validate_gold_reference(task, jail)
        for repeat_index in range(repeat):
            results.append(
                _run_one_task(
                    task,
                    fixture_root,
                    jail,
                    client=client,
                    store=store,
                    database=database,
                    repeat_index=repeat_index,
                )
            )

    return _aggregate_report(
        results,
        task_filter=task_filter,
        repeat=repeat,
        task_count=len(tasks),
    )


def write_suite_report(report: SuiteReport, report_dir: Path) -> Path:
    """Write a SuiteReport to report_dir/results.json."""
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "results.json"
    path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def write_metrics_report(report: SuiteReport, store: TraceStore, report_dir: Path) -> Path:
    """Derive trace metrics and write report_dir/report.md."""
    runs = [
        RunTrace(
            run_id=result.run_id,
            task_type=result.task_type,
            final_status=result.status,
            events=store.read(result.run_id),
        )
        for result in report.results
    ]
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "report.md"
    path.write_text(render_metrics_markdown(compute_metrics(runs)), encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for python -m eval.run_eval."""
    parser = argparse.ArgumentParser(description="Run RepoPilot issue-analysis eval tasks.")
    parser.add_argument("--tasks", type=Path, default=Path("eval/tasks.json"))
    parser.add_argument("--type", choices=["issue"], default="issue", dest="task_filter")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("eval/reports"))
    args = parser.parse_args(argv)

    settings = load_settings()
    client = build_llm_client(settings)
    tasks_path = cast(Path, args.tasks)
    report_dir = cast(Path, args.out) / _timestamp()
    state_dir = report_dir / "state"
    tasks = load_eval_tasks(tasks_path, task_filter=cast(EvalType, args.task_filter))
    report = run_suite(
        tasks,
        tasks_path.parent / "fixtures",
        client=client,
        repeat=cast(int, args.repeat),
        task_filter=cast(EvalType, args.task_filter),
        work_dir=state_dir,
    )
    results_path = write_suite_report(report, report_dir)
    write_metrics_report(report, TraceStore(state_dir / "traces"), report_dir)
    _print_summary(report, results_path)


def _run_one_task(
    task: EvalTask,
    fixture_root: Path,
    jail: PathJail,
    *,
    client: LLMClient,
    store: TraceStore,
    database: Database,
    repeat_index: int,
) -> TaskRunReport:
    registry = _build_read_only_registry(trace_sink=RegistryTraceSink(store))
    task_spec = TaskSpec(task_type="issue", prompt=task.issue, repo=str(fixture_root))
    result = run_agent_loop(
        task_spec,
        planner=Planner(client, store, tools_doc=_tools_doc(registry)),
        executor=Executor(client, registry, jail, store),
        critic=Critic(client, store),
        store=store,
        database=database,
        budgets=Budgets(
            max_steps=task.budgets.max_steps,
            max_replans=task.budgets.max_replans,
            max_fix_cycles=task.budgets.max_fix_cycles,
        ),
        reporter=Reporter(client),
        jail=jail,
    )
    return _score_result(task, result, repeat_index=repeat_index)


def _score_result(
    task: EvalTask,
    result: RunResult,
    *,
    repeat_index: int,
) -> TaskRunReport:
    suspects = [suspect.path for suspect in result.report.suspects] if result.report else []
    citations = list(result.report.citations) if result.report else []
    grounding_checks = list(result.grounding.checks) if result.grounding else []
    localization: LocalizationScore | None = None
    explanation: ExplanationScore | None = None

    if task.type == "bug_localization":
        localization = score_bug_localization(suspects, task.expected.gold_file, top_k=3)
    if task.type == "bug_explanation":
        explanation = score_bug_explanation(
            citations,
            grounding_checks,
            _analysis_text(result),
            task.expected.rubric_keywords,
            require_valid_citation=task.expected.requires_valid_citation,
        )

    return TaskRunReport(
        task_id=task.id,
        task_type=task.type,
        repeat_index=repeat_index,
        run_id=result.run_id,
        status=result.status.value,
        steps=result.steps_used,
        cost_usd=result.usage.cost_usd,
        suspects=suspects,
        citations=citations,
        grounding_checks=grounding_checks,
        localization=localization,
        explanation=explanation,
    )


def _aggregate_report(
    results: Sequence[TaskRunReport],
    *,
    task_filter: EvalType,
    repeat: int,
    task_count: int,
) -> SuiteReport:
    localization_scores = [result.localization for result in results if result.localization]
    explanation_scores = [result.explanation for result in results if result.explanation]
    costs = [result.cost_usd for result in results]
    total_cost = None
    if all(cost is not None for cost in costs):
        total_cost = sum(cost for cost in costs if cost is not None)
    return SuiteReport(
        generated_at=_timestamp(),
        task_filter=task_filter,
        repeat=repeat,
        task_count=task_count,
        run_count=len(results),
        top3_hit_rate=_rate([score.hit for score in localization_scores]),
        citation_valid_rate=_rate([score.citation_valid for score in explanation_scores]),
        mean_steps=_mean([result.steps for result in results]),
        total_cost_usd=total_cost,
        results=list(results),
    )


def _validate_gold_reference(task: EvalTask, jail: PathJail) -> None:
    gold_path = jail.resolve(task.expected.gold_file)
    if not gold_path.is_file():
        raise ValueError(f"Gold file for {task.id} is not a file: {task.expected.gold_file}")

    if task.expected.gold_line_range is None:
        return

    start, end = task.expected.gold_line_range
    total_lines = _count_lines(gold_path)
    if start > total_lines or end > total_lines:
        raise ValueError(
            f"Gold line range for {task.id} exceeds {task.expected.gold_file} "
            f"line count {total_lines}."
        )


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        return sum(1 for _line in stream)


def _analysis_text(result: RunResult) -> str:
    if result.report is None:
        return result.summary
    return f"{result.report.headline}\n\n{result.report.analysis}"


def _build_read_only_registry(trace_sink: TraceSink | None = None) -> ToolRegistry:
    registry = ToolRegistry(trace_sink=trace_sink)
    register_get_file_tree(registry)
    register_read_file(registry)
    register_search_code(registry)
    return registry


def _tools_doc(registry: ToolRegistry) -> str:
    lines: list[str] = []
    for schema in registry.to_llm_schema():
        function = schema.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        description = function.get("description")
        if not isinstance(name, str):
            continue
        description_text = _one_line(description) if isinstance(description, str) else ""
        lines.append(f"- {name}: {description_text}")
    return "\n".join(lines)


def _rate(values: Sequence[bool]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _mean(values: Sequence[int]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _print_summary(report: SuiteReport, results_path: Path) -> None:
    print("Issue eval summary")
    print(f"tasks={report.task_count} repeats={report.repeat} runs={report.run_count}")
    print(f"top3_hit_rate={_format_rate(report.top3_hit_rate)}")
    print(f"citation_valid_rate={_format_rate(report.citation_valid_rate)}")
    print(f"mean_steps={_format_float(report.mean_steps)}")
    print(f"total_cost_usd={_format_cost(report.total_cost_usd)}")
    print(f"results={results_path}")


def _format_rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _format_cost(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:.6f}"


def _one_line(message: str) -> str:
    return " ".join(message.split())


if __name__ == "__main__":
    main()
