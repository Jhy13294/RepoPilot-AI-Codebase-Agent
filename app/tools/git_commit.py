"""Approval-gated commit creation on the run work branch."""

import ast
import subprocess
from pathlib import Path
from typing import Literal, NoReturn

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.tool_io import ErrorType
from app.tools.base import ToolContext, ToolFailure
from app.tools.registry import ToolRegistry, ToolSpec

__all__ = ["register"]

_GIT_TIMEOUT_S = 10
_GitOperation = Literal["current_branch", "worktree_status", "commit", "head"]
_GitFailureReason = Literal["commit_failed", "git_error"]


class _GitCommitArgs(BaseModel):
    """Arguments for git_commit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    message: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


class _GitCommitPayload(BaseModel):
    """Payload returned by git_commit."""

    model_config = ConfigDict(frozen=True)

    branch: str
    commit: str
    message: str
    files_committed: list[str]
    committed: bool


def register(registry: ToolRegistry) -> None:
    """Register git_commit in a ToolRegistry."""
    registry.register(
        ToolSpec(
            name="git_commit",
            description=(
                "Commit tracked changes only on the current run's repopilot/fix-<run_id> work "
                "branch after explicit human approval. The tool uses a fixed RepoPilot author "
                "identity and never pushes, amends, commits untracked files, or deletes branches. "
                "Creating and committing new untracked files is outside this tool's contract."
            ),
            args_schema=_GitCommitArgs,
            returns_schema=_GitCommitPayload,
            risk_level="high",
        ),
        _handle,
    )


def _handle(args: BaseModel, context: ToolContext) -> BaseModel:
    parsed = _GitCommitArgs.model_validate(args)
    branch = f"repopilot/fix-{context.run_id}"
    root = context.jail.root

    _require_work_branch(root, branch)

    status = _run_git(root, "worktree_status")
    if status.returncode != 0:
        _raise_completed_failure(status, "git_error", "worktree_status")
    files_committed = _parse_status_files(status.stdout)
    if not files_committed:
        raise ToolFailure(
            ErrorType.GitError,
            "There are no tracked changes to commit on the run work branch.",
            {
                "reason": "nothing_to_commit",
                "operation": "worktree_status",
                "returncode": status.returncode,
                "stderr": _decode_output(status.stderr),
            },
        )

    committed = _run_git(root, "commit", parsed.message)
    if committed.returncode != 0:
        _raise_completed_failure(committed, "commit_failed", "commit")

    head = _run_git(root, "head")
    if head.returncode != 0 or not _decode_output(head.stdout):
        _raise_completed_failure(head, "git_error", "head")

    return _GitCommitPayload(
        branch=branch,
        commit=_decode_output(head.stdout),
        message=parsed.message,
        files_committed=files_committed,
        committed=True,
    )


def _require_work_branch(root: Path, expected_branch: str) -> None:
    completed = _run_git(root, "current_branch")
    if completed.returncode != 0:
        _raise_completed_failure(completed, "git_error", "current_branch")

    current_branch = _decode_output(completed.stdout)
    if current_branch != expected_branch:
        raise ToolFailure(
            ErrorType.GitError,
            (
                f"git_commit requires branch '{expected_branch}', but the current branch is "
                f"'{current_branch}'. Create or switch to the run work branch first."
            ),
            {
                "reason": "wrong_branch",
                "current_branch": current_branch,
                "expected_branch": expected_branch,
            },
        )


def _parse_status_files(output: bytes | str | None) -> list[str]:
    if output is None:
        return []
    status = output.decode("utf-8", errors="replace") if isinstance(output, bytes) else output
    if not status:
        return []

    files: list[str] = []
    for line in status.splitlines():
        if len(line) < 4 or line[2] != " ":
            raise _malformed_status_failure()
        state = line[:2]
        path = line[3:]
        if "R" in state or "C" in state:
            marker = " -> "
            if marker not in path:
                raise _malformed_status_failure()
            path = path.rsplit(marker, maxsplit=1)[1]
        files.append(_decode_status_path(path))
    return files


def _decode_status_path(path: str) -> str:
    if not path.startswith('"'):
        return path
    try:
        decoded = ast.literal_eval(f"b{path}")
        if not isinstance(decoded, bytes):
            raise ValueError("Git path did not decode to bytes")
        return decoded.decode("utf-8", errors="strict")
    except (SyntaxError, UnicodeDecodeError, ValueError) as exc:
        raise _malformed_status_failure() from exc


def _run_git(
    root: Path,
    operation: _GitOperation,
    message: str | None = None,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            _git_argv(operation, message),
            cwd=root,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            shell=False,
            timeout=_GIT_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolFailure(
            ErrorType.GitError,
            "Git timed out while committing the approved changes; report this blocker.",
            {
                "reason": "git_timeout",
                "operation": operation,
                "returncode": None,
                "stderr": _decode_output(exc.stderr),
                "timeout_s": _GIT_TIMEOUT_S,
            },
        ) from exc
    except OSError as exc:
        raise ToolFailure(
            ErrorType.GitError,
            "Git could not be started while committing the approved changes.",
            {
                "reason": "git_error",
                "operation": operation,
                "returncode": None,
                "stderr": str(exc),
            },
        ) from exc


def _git_argv(operation: _GitOperation, message: str | None) -> tuple[str, ...]:
    if operation == "current_branch":
        return ("git", "rev-parse", "--abbrev-ref", "HEAD")
    if operation == "worktree_status":
        return ("git", "status", "--porcelain", "--untracked-files=no")
    if operation == "head":
        return ("git", "rev-parse", "HEAD")
    if message is None:
        raise ValueError("message is required for the commit operation")
    return (
        "git",
        "-c",
        "user.name=RepoPilot",
        "-c",
        "user.email=noreply@repopilot.invalid",
        "commit",
        "-a",
        "-m",
        message,
    )


def _raise_completed_failure(
    completed: subprocess.CompletedProcess[bytes],
    reason: _GitFailureReason,
    operation: _GitOperation,
) -> NoReturn:
    messages = {
        "commit_failed": "Git failed to commit the approved tracked changes; report this blocker.",
        "git_error": "Git could not inspect the repository while committing; report this blocker.",
    }
    raise ToolFailure(
        ErrorType.GitError,
        messages[reason],
        {
            "reason": reason,
            "operation": operation,
            "returncode": completed.returncode,
            "stderr": _decode_output(completed.stderr),
        },
    )


def _malformed_status_failure() -> ToolFailure:
    return ToolFailure(
        ErrorType.GitError,
        "Git returned an unreadable tracked-change status; report this blocker.",
        {
            "reason": "git_error",
            "operation": "worktree_status",
            "returncode": 0,
            "stderr": "malformed git status output",
        },
    )


def _decode_output(output: bytes | str | None) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace").strip()
    return output.strip()
