"""Approval-gated creation and switching of the run work branch."""

import subprocess
from pathlib import Path
from typing import Literal, NoReturn

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.tool_io import ErrorType
from app.tools.base import ToolContext, ToolFailure
from app.tools.registry import ToolRegistry, ToolSpec

__all__ = ["register"]

_GIT_TIMEOUT_S = 10
_GitOperation = Literal[
    "current_branch",
    "worktree_status",
    "branch_exists",
    "create_branch",
    "switch_branch",
]
_GitFailureReason = Literal["create_failed", "switch_failed", "git_error"]


class _GitCreateBranchArgs(BaseModel):
    """Arguments for git_create_branch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rationale: str = Field(min_length=1)


class _GitCreateBranchPayload(BaseModel):
    """Payload returned by git_create_branch."""

    model_config = ConfigDict(frozen=True)

    branch: str
    created: bool
    switched: bool
    detail: str


def register(registry: ToolRegistry) -> None:
    """Register git_create_branch in a ToolRegistry."""
    registry.register(
        ToolSpec(
            name="git_create_branch",
            description=(
                "Create or switch to the current run's repopilot/fix-<run_id> work branch after "
                "explicit human approval. The branch name is derived from the run ID and cannot "
                "be supplied by the model. Creating or switching requires a clean tracked "
                "worktree; the tool never stashes, commits, deletes branches, or pushes. If the "
                "branch is already active, no further branch action is needed; do not call "
                "git_create_branch again."
            ),
            args_schema=_GitCreateBranchArgs,
            returns_schema=_GitCreateBranchPayload,
            risk_level="high",
        ),
        _handle,
    )


def _handle(args: BaseModel, context: ToolContext) -> BaseModel:
    _GitCreateBranchArgs.model_validate(args)
    branch = f"repopilot/fix-{context.run_id}"
    root = context.jail.root

    current = _run_git(root, "current_branch")
    if current.returncode != 0:
        _raise_completed_failure(current, "git_error", "current_branch")

    current_branch = _decode_output(current.stdout)
    if current_branch == branch:
        return _GitCreateBranchPayload(
            branch=branch,
            created=False,
            switched=False,
            detail=(
                f"Already on {branch}; the work branch is active and no further action is "
                "needed. Do not call git_create_branch again."
            ),
        )

    status = _run_git(root, "worktree_status")
    if status.returncode != 0:
        _raise_completed_failure(status, "git_error", "worktree_status")
    if _decode_output(status.stdout):
        raise ToolFailure(
            ErrorType.GitError,
            (
                "Tracked working-tree changes block branch switching. Report this blocker; "
                "RepoPilot will not stash or discard user changes."
            ),
            {
                "reason": "dirty_worktree",
                "operation": "worktree_status",
                "returncode": status.returncode,
                "stderr": _decode_output(status.stderr),
            },
        )

    exists = _run_git(root, "branch_exists", branch)
    if exists.returncode == 0:
        switched = _run_git(root, "switch_branch", branch)
        if switched.returncode != 0:
            _raise_completed_failure(switched, "switch_failed", "switch_branch")
        return _GitCreateBranchPayload(
            branch=branch,
            created=False,
            switched=True,
            detail=f"Switched to existing {branch}.",
        )

    if exists.returncode != 1:
        _raise_completed_failure(exists, "git_error", "branch_exists")

    created = _run_git(root, "create_branch", branch)
    if created.returncode != 0:
        _raise_completed_failure(created, "create_failed", "create_branch")
    return _GitCreateBranchPayload(
        branch=branch,
        created=True,
        switched=True,
        detail=f"Created and switched to {branch}.",
    )


def _run_git(
    root: Path,
    operation: _GitOperation,
    branch: str | None = None,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            _git_argv(operation, branch),
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
            (
                "Git timed out while preparing the run work branch. Report the repository "
                "blocker instead of retrying automatically."
            ),
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
            "Git could not be started while preparing the run work branch; report this blocker.",
            {
                "reason": "git_error",
                "operation": operation,
                "returncode": None,
                "stderr": str(exc),
            },
        ) from exc


def _git_argv(operation: _GitOperation, branch: str | None) -> tuple[str, ...]:
    if operation == "current_branch":
        return ("git", "rev-parse", "--abbrev-ref", "HEAD")
    if operation == "worktree_status":
        return ("git", "status", "--porcelain", "--untracked-files=no")
    if branch is None:
        raise ValueError(f"branch is required for git operation '{operation}'")
    if operation == "branch_exists":
        return ("git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}")
    if operation == "create_branch":
        return ("git", "checkout", "--quiet", "-b", branch)
    return ("git", "checkout", "--quiet", branch)


def _raise_completed_failure(
    completed: subprocess.CompletedProcess[bytes],
    reason: _GitFailureReason,
    operation: _GitOperation,
) -> NoReturn:
    messages = {
        "create_failed": "Git failed to create the run work branch; report this blocker.",
        "switch_failed": "Git failed to switch branches; report this blocker.",
        "git_error": "Git could not inspect the repository; report this blocker.",
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


def _decode_output(output: bytes | str | None) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace").strip()
    return output.strip()
