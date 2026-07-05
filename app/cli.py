"""Command line interface for RepoPilot."""

from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.text import Text

from app.agent.tool_loop import run_tool_loop
from app.config import ConfigError, load_settings
from app.safety.path_jail import PathJail
from app.schemas.agent_io import AskResult, RunStatus
from app.services.llm_client import LLMError, build_llm_client
from app.tools.get_file_tree import register as register_get_file_tree
from app.tools.read_file import register as register_read_file
from app.tools.registry import ToolRegistry
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

    if result.status is RunStatus.answered:
        raise typer.Exit(code=0)
    raise typer.Exit(code=1)


def main() -> None:
    """Run the RepoPilot CLI."""
    app()


def _build_read_only_registry() -> ToolRegistry:
    registry = ToolRegistry()
    register_get_file_tree(registry)
    register_read_file(registry)
    register_search_code(registry)
    return registry


def _render_result(result: AskResult, console: Console) -> None:
    console.print(Text("Answer", style="bold"))
    console.print(result.answer)
    console.print()
    console.print(Text(_summary_line(result), style="dim"))


def _summary_line(result: AskResult) -> str:
    return (
        f"status={result.status.value} | "
        f"steps={result.steps} | "
        f"tool_calls={len(result.tool_invocations)} | "
        f"tokens_in={result.usage.tokens_in} | "
        f"tokens_out={result.usage.tokens_out} | "
        f"cost={_format_cost(result.usage.cost_usd)}"
    )


def _format_cost(cost_usd: float | None) -> str:
    if cost_usd is None:
        return "n/a"
    return f"${cost_usd:.6f}"


def _exit_with_error(label: str, exc: Exception) -> NoReturn:
    _STDERR.print(f"{label}: {_one_line(str(exc))}")
    raise typer.Exit(code=1)


def _one_line(message: str) -> str:
    return " ".join(message.split())
