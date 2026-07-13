"""Approval-gated unified diff application through the shared registry."""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.tool_io import ErrorType
from app.tools.base import ToolContext, ToolFailure, workspace_relative_path
from app.tools.registry import ToolRegistry, ToolSpec

__all__ = ["register"]

_GIT_TIMEOUT_S = 10
_GitOperation = Literal["branch", "numstat_raw", "numstat_apply", "check", "apply"]
_GIT_ARGV: dict[_GitOperation, tuple[str, ...]] = {
    "branch": ("git", "rev-parse", "--abbrev-ref", "HEAD"),
    "numstat_raw": ("git", "apply", "--numstat", "-z", "-p0", "-"),
    "numstat_apply": ("git", "apply", "--numstat", "-z", "-p1", "-"),
    "check": ("git", "apply", "--check", "-p1", "-"),
    "apply": ("git", "apply", "-p1", "-"),
}


class _ApplyPatchArgs(BaseModel):
    """Arguments for apply_patch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    diff: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


class _ApplyPatchPayload(BaseModel):
    """Payload returned by apply_patch."""

    model_config = ConfigDict(frozen=True)

    files_changed: list[str]
    insertions: int = Field(ge=0)
    deletions: int = Field(ge=0)
    applied: bool


@dataclass(frozen=True, slots=True)
class _NumstatEntry:
    insertions: int
    deletions: int
    paths: tuple[str, ...]


def register(registry: ToolRegistry) -> None:
    """Register apply_patch in a ToolRegistry."""
    registry.register(
        ToolSpec(
            name="apply_patch",
            description=(
                "Apply a git-compatible unified diff to the current working tree after explicit "
                "human approval. Use only on the run's repopilot/fix-<run_id> work branch. The "
                "tool validates every path and checks the full patch before writing; it does not "
                "commit, switch branches, or push."
            ),
            args_schema=_ApplyPatchArgs,
            returns_schema=_ApplyPatchPayload,
            risk_level="high",
        ),
        _handle,
    )


def _handle(args: BaseModel, context: ToolContext) -> BaseModel:
    parsed = _ApplyPatchArgs.model_validate(args)
    _require_work_branch(context)
    patch = _encode_diff(parsed.diff)

    raw_entries = _read_numstat(context.jail.root, patch, "numstat_raw")
    _resolve_changed_paths(raw_entries, context)

    applied_entries = _read_numstat(context.jail.root, patch, "numstat_apply")
    files_changed = _resolve_changed_paths(applied_entries, context)
    insertions = sum(entry.insertions for entry in applied_entries)
    deletions = sum(entry.deletions for entry in applied_entries)

    checked = _run_git(context.jail.root, "check", patch)
    if checked.returncode != 0:
        stderr = _decode_output(checked.stderr)
        reject = stderr or _decode_output(checked.stdout)
        raise ToolFailure(
            ErrorType.PatchApplyError,
            "Patch does not apply cleanly; re-read the affected files and generate a fresh patch.",
            {
                "reason": "check_failed",
                "returncode": checked.returncode,
                "reject": reject,
                "stderr": stderr,
            },
        )

    applied = _run_git(context.jail.root, "apply", patch)
    if applied.returncode != 0:
        raise ToolFailure(
            ErrorType.PatchApplyError,
            "Git failed while applying the checked patch; inspect the repository and retry.",
            {
                "reason": "apply_failed",
                "returncode": applied.returncode,
                "stderr": _decode_output(applied.stderr),
            },
        )

    return _ApplyPatchPayload(
        files_changed=files_changed,
        insertions=insertions,
        deletions=deletions,
        applied=True,
    )


def _require_work_branch(context: ToolContext) -> None:
    completed = _run_git(context.jail.root, "branch")
    if completed.returncode != 0:
        raise ToolFailure(
            ErrorType.PatchApplyError,
            "Git could not determine the current branch; apply_patch requires a work branch.",
            {
                "reason": "git_error",
                "operation": "rev_parse_branch",
                "returncode": completed.returncode,
                "stderr": _decode_output(completed.stderr),
            },
        )

    current_branch = _decode_output(completed.stdout)
    expected_branch = f"repopilot/fix-{context.run_id}"
    if current_branch != expected_branch:
        raise ToolFailure(
            ErrorType.PatchApplyError,
            (
                f"apply_patch requires branch '{expected_branch}', but the current branch is "
                f"'{current_branch}'. Create or switch to the run work branch first."
            ),
            {
                "reason": "wrong_branch",
                "current_branch": current_branch,
                "expected_branch": expected_branch,
            },
        )


def _encode_diff(diff: str) -> bytes:
    try:
        return diff.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ToolFailure(
            ErrorType.InvalidArgsError,
            "diff must be valid UTF-8 text.",
            {"reason": "invalid_diff"},
        ) from exc


def _read_numstat(
    root: Path,
    patch: bytes,
    operation: Literal["numstat_raw", "numstat_apply"],
) -> list[_NumstatEntry]:
    completed = _run_git(root, operation, patch)
    if completed.returncode != 0:
        raise ToolFailure(
            ErrorType.InvalidArgsError,
            "diff is not a parseable git unified patch.",
            {
                "reason": "invalid_diff",
                "returncode": completed.returncode,
                "stderr": _decode_output(completed.stderr),
            },
        )
    return _parse_numstat(completed.stdout)


def _parse_numstat(output: bytes) -> list[_NumstatEntry]:
    if not output or not output.endswith(b"\x00"):
        raise _invalid_numstat_failure()

    records = output.split(b"\x00")
    entries: list[_NumstatEntry] = []
    index = 0
    final_index = len(records) - 1
    while index < final_index:
        record = records[index]
        index += 1
        fields = record.split(b"\t", maxsplit=2)
        if len(fields) != 3:
            raise _invalid_numstat_failure()

        insertions = _parse_change_count(fields[0])
        deletions = _parse_change_count(fields[1])
        path_field = fields[2]
        paths: tuple[str, ...]
        if path_field:
            paths = (_decode_path(path_field),)
        else:
            if index + 1 >= final_index:
                raise _invalid_numstat_failure()
            paths = (_decode_path(records[index]), _decode_path(records[index + 1]))
            index += 2

        entries.append(
            _NumstatEntry(
                insertions=insertions,
                deletions=deletions,
                paths=paths,
            )
        )

    if not entries:
        raise _invalid_numstat_failure()
    return entries


def _parse_change_count(value: bytes) -> int:
    try:
        count = int(value.decode("ascii", errors="strict"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise _invalid_numstat_failure() from exc
    if count < 0:
        raise _invalid_numstat_failure()
    return count


def _decode_path(value: bytes) -> str:
    if not value:
        raise _invalid_numstat_failure()
    try:
        return value.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise _invalid_numstat_failure() from exc


def _resolve_changed_paths(entries: list[_NumstatEntry], context: ToolContext) -> list[str]:
    files_changed: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        for candidate in entry.paths:
            resolved = context.jail.resolve(candidate)
            relative = workspace_relative_path(context.jail.root, resolved)
            if relative not in seen:
                seen.add(relative)
                files_changed.append(relative)
    return files_changed


def _run_git(
    root: Path,
    operation: _GitOperation,
    patch: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            _GIT_ARGV[operation],
            cwd=root,
            input=patch,
            capture_output=True,
            check=False,
            shell=False,
            timeout=_GIT_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolFailure(
            ErrorType.PatchApplyError,
            "Git timed out while processing the patch; inspect the repository before retrying.",
            {
                "reason": "git_timeout",
                "operation": operation,
                "timeout_s": _GIT_TIMEOUT_S,
            },
        ) from exc
    except OSError as exc:
        raise ToolFailure(
            ErrorType.PatchApplyError,
            "Git could not be started for apply_patch.",
            {
                "reason": "git_error",
                "operation": operation,
                "stderr": str(exc),
            },
        ) from exc


def _decode_output(output: bytes) -> str:
    return output.decode("utf-8", errors="replace").strip()


def _invalid_numstat_failure() -> ToolFailure:
    return ToolFailure(
        ErrorType.InvalidArgsError,
        "diff is not a parseable text patch with numeric change statistics.",
        {"reason": "invalid_diff"},
    )
