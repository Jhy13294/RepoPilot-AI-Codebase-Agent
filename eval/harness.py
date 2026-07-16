"""Trusted setup and approval primitives for offline patch evaluation."""

import shutil
import subprocess
from pathlib import Path

from pydantic import BaseModel

from app.tools.base import ToolContext
from app.tools.registry import ApprovalOutcome, ToolSpec

_GIT_TIMEOUT_S = 30


class AutoApprovalGate:
    """Approve eval-only high-risk calls while preserving registry gate semantics."""

    def check(
        self,
        spec: ToolSpec,
        args: BaseModel,
        context: ToolContext,
    ) -> ApprovalOutcome:
        """Return the auditable non-interactive decision used by the eval harness."""
        del spec, args, context
        return ApprovalOutcome(
            approved=True,
            actor="eval:auto",
            reason="auto-approved by eval harness",
        )


def prepare_git_workspace(fixture_root: Path, dest: Path) -> Path:
    """Copy a fixture into a clean, locally configured Git workspace."""
    shutil.copytree(fixture_root, dest, ignore=shutil.ignore_patterns("__pycache__"))
    _run_git(dest, "init", "--initial-branch=main", "--quiet")
    _run_git(dest, "config", "core.autocrlf", "false")
    _run_git(dest, "config", "user.email", "eval@repopilot.local")
    _run_git(dest, "config", "user.name", "RepoPilot Eval Harness")
    _run_git(dest, "add", "-A")
    _run_git(
        dest,
        "commit",
        "--quiet",
        "--no-gpg-sign",
        "--allow-empty",
        "-m",
        "Baseline fixture",
    )
    return dest


def _run_git(workspace: Path, *args: str) -> None:
    subprocess.run(
        ("git", *args),
        cwd=workspace,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=True,
        shell=False,
        timeout=_GIT_TIMEOUT_S,
    )
