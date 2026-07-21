"""Trusted setup, approval, and fault injection for offline evaluation."""

import shutil
import subprocess
from pathlib import Path
from threading import Lock
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.safety.loop_guard import LoopGuard
from app.schemas.tool_io import ErrorType
from app.tools.apply_patch import register as register_apply_patch
from app.tools.base import ToolContext, ToolFailure
from app.tools.get_file_tree import register as register_get_file_tree
from app.tools.git_commit import register as register_git_commit
from app.tools.git_create_branch import register as register_git_create_branch
from app.tools.propose_patch import register as register_propose_patch
from app.tools.read_file import register as register_read_file
from app.tools.registry import (
    ApprovalGate,
    ApprovalOutcome,
    ToolHandler,
    ToolRegistry,
    ToolSpec,
    TraceSink,
)
from app.tools.run_tests import register as register_run_tests
from app.tools.search_code import register as register_search_code

_GIT_TIMEOUT_S = 30
_FAULT_MARKER = "  # repopilot recovery fault injection"


class FaultInjection(BaseModel):
    """One bounded handler-level fault injected by the recovery harness."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["patch_conflict", "tool_timeout"]
    target: str = Field(min_length=1)
    times: int = Field(default=1, ge=1)


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


class FaultInjectingRegistry(ToolRegistry):
    """Tool registry that injects bounded faults inside normal handler execution."""

    def __init__(
        self,
        *,
        injection: FaultInjection | None = None,
        approval_gate: ApprovalGate | None = None,
        trace_sink: TraceSink | None = None,
        loop_guard: LoopGuard | None = None,
    ) -> None:
        super().__init__(
            approval_gate=approval_gate,
            trace_sink=trace_sink,
            loop_guard=loop_guard,
        )
        self._injection = injection
        self._remaining_injections = injection.times if injection is not None else 0
        self._injection_lock = Lock()

    def _get_tool(self, name: str) -> tuple[ToolSpec, ToolHandler]:
        spec, real_handler = super()._get_tool(name)
        injection = self._injection
        if injection is None or name != _injected_tool_name(injection):
            return spec, real_handler

        def wrapped_handler(args: BaseModel, context: ToolContext) -> BaseModel:
            if not self._consume_injection():
                return real_handler(args, context)
            if injection.kind == "tool_timeout":
                raise ToolFailure(
                    ErrorType.ToolTimeoutError,
                    f"Injected registry timeout for tool '{name}'.",
                    {"reason": "fault_injection", "target": injection.target},
                )

            _inject_patch_conflict(args, context, injection.target)
            return real_handler(args, context)

        return spec, wrapped_handler

    def _consume_injection(self) -> bool:
        with self._injection_lock:
            if self._remaining_injections < 1:
                return False
            self._remaining_injections -= 1
            return True


def build_recovery_registry(
    injection: FaultInjection | None = None,
    *,
    trace_sink: TraceSink | None = None,
    approval_gate: ApprovalGate,
    test_command: str = "pytest -q",
    test_timeout_s: int = 120,
) -> FaultInjectingRegistry:
    """Build the complete fix registry with optional handler-level fault injection."""
    registry = FaultInjectingRegistry(
        injection=injection,
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


def _injected_tool_name(injection: FaultInjection) -> str:
    if injection.kind == "patch_conflict":
        return "apply_patch"
    return injection.target


def _inject_patch_conflict(args: BaseModel, context: ToolContext, target: str) -> None:
    target_path = context.jail.resolve(target)
    content = target_path.read_bytes().decode("utf-8", errors="strict")
    raw_diff = args.model_dump().get("diff")
    diff = raw_diff if isinstance(raw_diff, str) else ""
    candidates = _removed_target_lines(diff, target)
    lines = content.splitlines(keepends=True)

    for candidate in candidates:
        for index, line in enumerate(lines):
            body = line.rstrip("\r\n")
            if body != candidate:
                continue
            ending = line[len(body) :]
            lines[index] = f"{body}{_FAULT_MARKER}{ending}"
            target_path.write_bytes("".join(lines).encode("utf-8", errors="strict"))
            return

    fallback = f"# repopilot recovery fault injection\n{content}"
    target_path.write_bytes(fallback.encode("utf-8", errors="strict"))


def _removed_target_lines(diff: str, target: str) -> list[str]:
    normalized_target = target.replace("\\", "/").lstrip("./")
    in_target = False
    in_hunk = False
    candidates: list[str] = []

    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[4:].split("\t", maxsplit=1)[0].strip()
            normalized_path = path.replace("\\", "/")
            if normalized_path.startswith("b/"):
                normalized_path = normalized_path[2:]
            in_target = normalized_path == normalized_target
            in_hunk = False
            continue
        if in_target and line.startswith("@@"):
            in_hunk = True
            continue
        if in_target and in_hunk and line.startswith("-") and line[1:].strip():
            candidates.append(line[1:])

    return candidates


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
