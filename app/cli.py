"""Command line interface for RepoPilot."""

from pathlib import Path
from typing import Annotated, Literal, NoReturn

import typer
from rich.console import Console
from rich.text import Text

from app.agent.critic import Critic
from app.agent.executor import Executor
from app.agent.loop import run_agent_loop
from app.agent.planner import Planner
from app.agent.reporter import Reporter
from app.agent.state import Budgets, RunStatus, TaskSpec
from app.agent.tool_loop import run_tool_loop
from app.config import ConfigError, load_settings
from app.safety.approval import CliApprovalGate
from app.safety.loop_guard import LoopGuard
from app.safety.path_jail import PathJail
from app.schemas.agent_io import AskResult, AskStatus, RunResult
from app.services.llm_client import LLMError, build_llm_client
from app.storage.db import Database
from app.storage.trace_store import RegistryTraceSink, TraceStore, render_timeline
from app.tools.apply_patch import register as register_apply_patch
from app.tools.get_file_tree import register as register_get_file_tree
from app.tools.git_commit import register as register_git_commit
from app.tools.git_create_branch import register as register_git_create_branch
from app.tools.propose_patch import register as register_propose_patch
from app.tools.read_file import register as register_read_file
from app.tools.registry import ApprovalGate, ToolRegistry, TraceSink
from app.tools.run_tests import register as register_run_tests
from app.tools.search_code import register as register_search_code

app = typer.Typer()
_STDOUT = Console(highlight=False)
_STDERR = Console(stderr=True, highlight=False)


@app.callback()
def _root() -> None:
    """RepoPilot command line interface."""


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question to ask about the repository.")],
    repo: Annotated[
        Path,
        typer.Option("--repo", "-r", help="Local repository path to inspect."),
    ] = Path("."),
    max_steps: Annotated[
        int | None,
        typer.Option("--max-steps", min=1, help="Maximum LLM/tool loop steps."),
    ] = None,
) -> None:
    """Ask a repository-grounded question."""
    try:
        settings = load_settings()
        registry = _build_read_only_registry()
        jail = PathJail(repo)
        client = build_llm_client(settings)
    except ConfigError as exc:
        _exit_with_error("Configuration error", exc)
    except ValueError as exc:
        _exit_with_error("Repository error", exc)
    except NotImplementedError as exc:
        _exit_with_error("LLM client error", exc)
    except LLMError as exc:
        _exit_with_error("LLM client error", exc)

    result = run_tool_loop(
        question,
        client=client,
        registry=registry,
        jail=jail,
        max_steps=max_steps or settings.max_steps,
    )
    _render_result(result, _STDOUT)

    if result.status is AskStatus.answered:
        raise typer.Exit(code=0)
    raise typer.Exit(code=1)


@app.command()
def run(
    task: Annotated[str, typer.Argument(help="Task prompt to run through the full agent.")],
    repo: Annotated[
        Path,
        typer.Option("--repo", "-r", help="Local repository path to inspect."),
    ] = Path("."),
    task_type: Annotated[
        Literal["question", "issue", "fix"],
        typer.Option("--task-type", help="Task type for the agent run."),
    ] = "question",
    max_steps: Annotated[
        int | None,
        typer.Option("--max-steps", min=1, help="Maximum agent execution steps."),
    ] = None,
) -> None:
    """Run the full Planner-Executor-Critic agent loop."""
    try:
        settings = load_settings()
        store = TraceStore(settings.trace_dir)
        database = Database(settings.db_path)
        trace_sink = RegistryTraceSink(store)
        if task_type == "fix":
            registry = _build_fix_registry(
                trace_sink=trace_sink,
                approval_gate=CliApprovalGate(),
                test_command=settings.test_command,
                test_timeout_s=settings.test_timeout_s,
            )
        else:
            registry = _build_read_only_registry(trace_sink=trace_sink)
        jail = PathJail(repo)
        client = build_llm_client(settings)
        planner = Planner(client, store, tools_doc=_tools_doc(registry))
        executor = Executor(client, registry, jail, store)
        critic = Critic(client, store)
        reporter = Reporter(client)
        task_spec = TaskSpec(task_type=task_type, prompt=task, repo=str(repo))
        budgets = Budgets(
            max_steps=max_steps or settings.max_steps,
            max_replans=settings.max_replans,
            max_fix_cycles=settings.max_fix_cycles,
        )
    except ConfigError as exc:
        _exit_with_error("Configuration error", exc)
    except ValueError as exc:
        _exit_with_error("Repository error", exc)
    except NotImplementedError as exc:
        _exit_with_error("LLM client error", exc)
    except LLMError as exc:
        _exit_with_error("LLM client error", exc)

    result = run_agent_loop(
        task_spec,
        planner=planner,
        executor=executor,
        critic=critic,
        store=store,
        database=database,
        budgets=budgets,
        reporter=reporter,
        jail=jail,
    )
    _render_run_result(result, _STDOUT)

    if result.status is RunStatus.DONE:
        raise typer.Exit(code=0)
    raise typer.Exit(code=1)


@app.command()
def replay(
    run_id: Annotated[str, typer.Argument(help="Run id whose JSONL trace should be replayed.")],
) -> None:
    """Replay a run trace as a compact timeline."""
    try:
        settings = load_settings()
        events = TraceStore(settings.trace_dir).read(run_id)
    except ConfigError as exc:
        _exit_with_error("Configuration error", exc)

    if not events:
        _STDERR.print(f"no trace found for {run_id}")
        raise typer.Exit(code=1)

    _STDOUT.print(render_timeline(events))
    raise typer.Exit(code=0)


def main() -> None:
    """Run the RepoPilot CLI."""
    app()


def _build_read_only_registry(trace_sink: TraceSink | None = None) -> ToolRegistry:
    registry = ToolRegistry(trace_sink=trace_sink)
    register_get_file_tree(registry)
    register_read_file(registry)
    register_search_code(registry)
    return registry


def _build_fix_registry(
    *,
    trace_sink: TraceSink | None = None,
    approval_gate: ApprovalGate,
    test_command: str = "pytest -q",
    test_timeout_s: int = 120,
) -> ToolRegistry:
    registry = ToolRegistry(
        approval_gate=approval_gate,
        trace_sink=trace_sink,
        loop_guard=LoopGuard(),
    )
    register_get_file_tree(registry)
    register_read_file(registry)
    register_search_code(registry)
    register_git_create_branch(registry)
    register_propose_patch(registry)
    register_apply_patch(registry)
    register_run_tests(
        registry,
        test_command=test_command,
        test_timeout_s=test_timeout_s,
    )
    register_git_commit(registry)
    return registry


def _render_result(result: AskResult, console: Console) -> None:
    console.print(Text("Answer", style="bold"))
    console.print(result.answer)
    console.print()
    console.print(Text(_summary_line(result), style="dim"), soft_wrap=True)


def _render_run_result(result: RunResult, console: Console) -> None:
    console.print(Text("Summary", style="bold"))
    console.print(result.summary, soft_wrap=True)
    if result.report is not None and result.report.suspects:
        console.print()
        console.print(Text("Suspects", style="bold"))
        for index, suspect in enumerate(result.report.suspects, start=1):
            console.print(f"{index}. {suspect.path} — {suspect.reason}", soft_wrap=True)
    if result.report is not None and result.report.citations:
        console.print()
        console.print(Text("Citations", style="bold"))
        for citation in result.report.citations:
            console.print(f"- {citation}", soft_wrap=True)
    if result.grounding is not None and result.grounding.ungrounded:
        console.print()
        console.print(Text("Unverified citations", style="bold yellow"))
        for check in result.grounding.ungrounded:
            console.print(f"- {check.citation} ({check.status})", soft_wrap=True)
    console.print()
    console.print(f"run_id={result.run_id}")
    console.print(f"Replay: repopilot replay {result.run_id}")
    console.print(Text(_run_summary_line(result), style="dim"), soft_wrap=True)


def _summary_line(result: AskResult) -> str:
    return (
        f"status={result.status.value} | "
        f"steps={result.steps} | "
        f"tool_calls={len(result.tool_invocations)} | "
        f"tokens_in={result.usage.tokens_in} | "
        f"tokens_out={result.usage.tokens_out} | "
        f"cost={_format_cost(result.usage.cost_usd)}"
    )


def _run_summary_line(result: RunResult) -> str:
    return (
        f"status={result.status.value} | "
        f"steps={result.steps_used} | "
        f"replans={result.replans_used} | "
        f"fix_cycles={result.fix_cycles_used} | "
        f"tokens_in={result.usage.tokens_in} | "
        f"tokens_out={result.usage.tokens_out} | "
        f"cost={_format_cost(result.usage.cost_usd)}"
    )


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


def _format_cost(cost_usd: float | None) -> str:
    if cost_usd is None:
        return "n/a"
    return f"${cost_usd:.6f}"


def _exit_with_error(label: str, exc: Exception) -> NoReturn:
    _STDERR.print(f"{label}: {_one_line(str(exc))}")
    raise typer.Exit(code=1)


def _one_line(message: str) -> str:
    return " ".join(message.split())
