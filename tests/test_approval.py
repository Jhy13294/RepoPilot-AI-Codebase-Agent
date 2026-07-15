from io import StringIO
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict
from rich.console import Console

from app.safety.approval import CliApprovalGate, _render_request
from app.safety.path_jail import PathJail
from app.tools.base import ToolContext
from app.tools.registry import ToolSpec


class _PatchArgs(BaseModel):
    model_config = ConfigDict(frozen=True)

    diff: str
    rationale: str


class _GenericArgs(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    query: str


class _Payload(BaseModel):
    model_config = ConfigDict(frozen=True)

    accepted: bool


def _spec(args_schema: type[BaseModel] = _PatchArgs) -> ToolSpec:
    return ToolSpec(
        name="apply_patch",
        description="Apply a patch.",
        args_schema=args_schema,
        returns_schema=_Payload,
        risk_level="high",
    )


def _context(tmp_path: Path) -> ToolContext:
    return ToolContext(run_id="run-approval-1", jail=PathJail(tmp_path))


def _console() -> tuple[Console, StringIO]:
    stream = StringIO()
    return (
        Console(
            file=stream,
            color_system=None,
            force_terminal=False,
            highlight=False,
            width=160,
        ),
        stream,
    )


@pytest.mark.parametrize("response", ("y", "Y", "yes", " YES "))
def test_cli_approval_gate__only_explicit_yes_responses_approve(
    tmp_path: Path,
    response: str,
) -> None:
    console, _stream = _console()
    gate = CliApprovalGate(console=console, prompt=lambda _message: response)

    outcome = gate.check(
        _spec(),
        _PatchArgs(diff="--- a/file.txt\n+++ b/file.txt\n", rationale="Fix the bug."),
        _context(tmp_path),
    )

    assert outcome.approved is True
    assert outcome.reason is None
    assert outcome.actor == "human"


def test_cli_approval_gate__explicit_denial_returns_reason(tmp_path: Path) -> None:
    console, _stream = _console()
    gate = CliApprovalGate(console=console, prompt=lambda _message: "n")

    outcome = gate.check(
        _spec(),
        _PatchArgs(diff="--- a/file.txt\n+++ b/file.txt\n", rationale="Fix the bug."),
        _context(tmp_path),
    )

    assert outcome.approved is False
    assert outcome.reason


def test_cli_approval_gate__denial_note_is_preserved(tmp_path: Path) -> None:
    console, _stream = _console()
    gate = CliApprovalGate(console=console, prompt=lambda _message: "no: add a regression test")

    outcome = gate.check(
        _spec(),
        _PatchArgs(diff="--- a/file.txt\n+++ b/file.txt\n", rationale="Fix the bug."),
        _context(tmp_path),
    )

    assert outcome.approved is False
    assert outcome.reason is not None
    assert "add a regression test" in outcome.reason


@pytest.mark.parametrize("response", ("", "maybe", "yes please"))
def test_cli_approval_gate__ambiguous_or_empty_input_fails_closed(
    tmp_path: Path,
    response: str,
) -> None:
    console, _stream = _console()
    gate = CliApprovalGate(console=console, prompt=lambda _message: response)

    outcome = gate.check(
        _spec(),
        _PatchArgs(diff="--- a/file.txt\n+++ b/file.txt\n", rationale="Fix the bug."),
        _context(tmp_path),
    )

    assert outcome.approved is False
    assert outcome.reason


def test_cli_approval_gate__eof_fails_closed_without_reading_real_stdin(tmp_path: Path) -> None:
    console, _stream = _console()

    def raise_eof(_message: str) -> str:
        raise EOFError

    gate = CliApprovalGate(console=console, prompt=raise_eof)

    outcome = gate.check(
        _spec(),
        _PatchArgs(diff="--- a/file.txt\n+++ b/file.txt\n", rationale="Fix the bug."),
        _context(tmp_path),
    )

    assert outcome.approved is False
    assert outcome.reason


def test_render_request__shows_patch_context_rationale_and_highlighted_diff(
    tmp_path: Path,
) -> None:
    console, stream = _console()
    args = _PatchArgs(
        diff=("--- a/sample.txt\n+++ b/sample.txt\n@@ -1 +1 @@\n-old value\n+new value\n"),
        rationale="Correct the stale value.",
    )

    console.print(_render_request(_spec(), args, _context(tmp_path)))
    rendered = stream.getvalue()

    assert "HIGH RISK" in rendered
    assert "apply_patch" in rendered
    assert "run-approval-1" in rendered
    assert "Correct the stale value." in rendered
    assert "Diff" in rendered
    assert "-old value" in rendered
    assert "+new value" in rendered


def test_render_request__uses_generic_dump_when_diff_is_absent(tmp_path: Path) -> None:
    console, stream = _console()
    args = _GenericArgs(path="tests/test_sample.py", query="failing test")

    console.print(_render_request(_spec(_GenericArgs), args, _context(tmp_path)))
    rendered = stream.getvalue()

    assert "Arguments" in rendered
    assert "path" in rendered
    assert "tests/test_sample.py" in rendered
    assert "query" in rendered
    assert "failing test" in rendered
