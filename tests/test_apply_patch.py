import subprocess
from pathlib import Path

import pytest
from pydantic import BaseModel

from app.safety.path_jail import PathJail
from app.schemas.tool_io import ErrorType, ToolResult
from app.tools.apply_patch import register as register_apply_patch
from app.tools.base import ToolContext
from app.tools.propose_patch import register as register_propose_patch
from app.tools.registry import ApprovalOutcome, ToolRegistry, ToolSpec

_RUN_ID = "test-run"
_ORIGINAL = b"alpha\nold\nomega\n"


class _FakeGate:
    def __init__(self, approved: bool, reason: str | None = None) -> None:
        self.approved = approved
        self.reason = reason
        self.calls: list[tuple[ToolSpec, BaseModel, ToolContext]] = []

    def check(
        self,
        spec: ToolSpec,
        args: BaseModel,
        context: ToolContext,
    ) -> ApprovalOutcome:
        self.calls.append((spec, args, context))
        return ApprovalOutcome(approved=self.approved, reason=self.reason)


def _run_git(
    repo: Path,
    *args: str,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        input=input_bytes,
        capture_output=True,
        check=False,
        shell=False,
    )


def _require_git_success(
    repo: Path,
    *args: str,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    completed = _run_git(repo, *args, input_bytes=input_bytes)
    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
    return completed


def _init_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _require_git_success(repo, "init", "--initial-branch=main", "--quiet")
    _require_git_success(repo, "config", "core.autocrlf", "false")
    _require_git_success(repo, "config", "user.name", "RepoPilot Tests")
    _require_git_success(repo, "config", "user.email", "tests@repopilot.local")
    target = repo / "sample.txt"
    target.write_bytes(_ORIGINAL)
    _require_git_success(repo, "add", "sample.txt")
    _require_git_success(repo, "commit", "--quiet", "-m", "test fixture")
    return repo, target


def _switch_to_work_branch(repo: Path, run_id: str = _RUN_ID) -> None:
    _require_git_success(repo, "checkout", "--quiet", "-b", f"repopilot/fix-{run_id}")


def _context(repo: Path, run_id: str = _RUN_ID) -> ToolContext:
    return ToolContext(run_id=run_id, jail=PathJail(repo))


def _registry(gate: _FakeGate | None) -> ToolRegistry:
    registry = ToolRegistry(approval_gate=gate)
    register_apply_patch(registry)
    return registry


def _dispatch(
    repo: Path,
    raw_args: dict[str, object],
    gate: _FakeGate | None,
) -> ToolResult:
    return _registry(gate).dispatch("apply_patch", raw_args, _context(repo))


def _payload(result: ToolResult) -> dict[str, object]:
    assert result.ok is True
    assert result.data is not None
    payload = result.data.model_dump()
    assert isinstance(payload, dict)
    return payload


def _assert_error(result: ToolResult, error_type: ErrorType) -> None:
    assert result.ok is False
    assert result.data is None
    assert result.error is not None
    assert result.error.type is error_type


def _patch(
    *,
    old: str = "old",
    new: str = "new",
    old_path: str = "a/sample.txt",
    new_path: str = "b/sample.txt",
) -> str:
    return f"--- {old_path}\n+++ {new_path}\n@@ -1,3 +1,3 @@\n alpha\n-{old}\n+{new}\n omega\n"


def test_apply_patch__approved_patch_changes_only_the_worktree(tmp_path: Path) -> None:
    repo, target = _init_repo(tmp_path)
    _switch_to_work_branch(repo)
    gate = _FakeGate(approved=True)
    head_before = _require_git_success(repo, "rev-parse", "HEAD").stdout
    branch_before = _require_git_success(repo, "branch", "--show-current").stdout

    result = _dispatch(
        repo,
        {"diff": _patch(), "rationale": "Replace the stale value."},
        gate,
    )

    assert _payload(result) == {
        "files_changed": ["sample.txt"],
        "insertions": 1,
        "deletions": 1,
        "applied": True,
    }
    assert target.read_bytes() == b"alpha\nnew\nomega\n"
    assert len(gate.calls) == 1
    assert gate.calls[0][0].name == "apply_patch"
    assert gate.calls[0][0].risk_level == "high"
    assert _require_git_success(repo, "rev-parse", "HEAD").stdout == head_before
    assert _require_git_success(repo, "branch", "--show-current").stdout == branch_before
    assert _run_git(repo, "diff", "--cached", "--quiet").returncode == 0


def test_apply_patch__denial_preserves_file_bytes(tmp_path: Path) -> None:
    repo, target = _init_repo(tmp_path)
    _switch_to_work_branch(repo)
    before = target.read_bytes()
    gate = _FakeGate(approved=False, reason="Add a regression test first.")

    result = _dispatch(
        repo,
        {"diff": _patch(), "rationale": "Replace the stale value."},
        gate,
    )

    _assert_error(result, ErrorType.ApprovalDeniedError)
    assert result.error is not None
    assert result.error.message == "Add a regression test first."
    assert target.read_bytes() == before
    assert len(gate.calls) == 1


def test_apply_patch__missing_gate_fails_closed_and_preserves_file_bytes(
    tmp_path: Path,
) -> None:
    repo, target = _init_repo(tmp_path)
    _switch_to_work_branch(repo)
    before = target.read_bytes()

    result = _dispatch(
        repo,
        {"diff": _patch(), "rationale": "Replace the stale value."},
        None,
    )

    _assert_error(result, ErrorType.ApprovalDeniedError)
    assert result.error is not None
    assert "no approval gate configured" in result.error.message
    assert target.read_bytes() == before


def test_apply_patch__wrong_branch_is_rejected_without_writing(tmp_path: Path) -> None:
    repo, target = _init_repo(tmp_path)
    before = target.read_bytes()
    gate = _FakeGate(approved=True)

    result = _dispatch(
        repo,
        {"diff": _patch(), "rationale": "Replace the stale value."},
        gate,
    )

    _assert_error(result, ErrorType.PatchApplyError)
    assert result.error is not None
    assert result.error.details is not None
    assert result.error.details["reason"] == "wrong_branch"
    assert target.read_bytes() == before
    assert len(gate.calls) == 1


def test_apply_patch__mismatched_patch_fails_check_without_writing(tmp_path: Path) -> None:
    repo, target = _init_repo(tmp_path)
    _switch_to_work_branch(repo)
    before = target.read_bytes()
    gate = _FakeGate(approved=True)

    result = _dispatch(
        repo,
        {
            "diff": _patch(old="content that is not present"),
            "rationale": "Replace the stale value.",
        },
        gate,
    )

    _assert_error(result, ErrorType.PatchApplyError)
    assert result.error is not None
    assert result.error.details is not None
    assert result.error.details["reason"] == "check_failed"
    assert "reject" in result.error.details
    assert "stderr" in result.error.details
    assert target.read_bytes() == before
    assert len(gate.calls) == 1


@pytest.mark.parametrize("escaped_kind", ("parent", "absolute"))
def test_apply_patch__escaped_target_is_rejected_by_jail_without_writing(
    tmp_path: Path,
    escaped_kind: str,
) -> None:
    repo, target = _init_repo(tmp_path)
    _switch_to_work_branch(repo)
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside sentinel\n")
    outside_before = outside.read_bytes()
    target_before = target.read_bytes()
    escaped_path = "b/../outside.txt" if escaped_kind == "parent" else outside.as_posix()

    result = _dispatch(
        repo,
        {
            "diff": _patch(new_path=escaped_path),
            "rationale": "Attempt to move the target.",
        },
        _FakeGate(approved=True),
    )

    _assert_error(result, ErrorType.PathJailError)
    assert target.read_bytes() == target_before
    assert outside.read_bytes() == outside_before


@pytest.mark.parametrize(
    "raw_args",
    (
        {"diff": "", "rationale": "Replace the stale value."},
        {"diff": _patch(), "rationale": ""},
    ),
)
def test_apply_patch__empty_required_text_is_invalid_before_approval(
    tmp_path: Path,
    raw_args: dict[str, object],
) -> None:
    repo, target = _init_repo(tmp_path)
    _switch_to_work_branch(repo)
    before = target.read_bytes()
    gate = _FakeGate(approved=True)

    result = _dispatch(repo, raw_args, gate)

    _assert_error(result, ErrorType.InvalidArgsError)
    assert target.read_bytes() == before
    assert gate.calls == []


def test_apply_patch__unparseable_nonempty_diff_is_invalid_without_writing(
    tmp_path: Path,
) -> None:
    repo, target = _init_repo(tmp_path)
    _switch_to_work_branch(repo)
    before = target.read_bytes()
    gate = _FakeGate(approved=True)

    result = _dispatch(
        repo,
        {"diff": "not a patch\n", "rationale": "Replace the stale value."},
        gate,
    )

    _assert_error(result, ErrorType.InvalidArgsError)
    assert target.read_bytes() == before
    assert len(gate.calls) == 1


def test_apply_patch__propose_then_apply_round_trip_reproduces_new_content(
    tmp_path: Path,
) -> None:
    repo, target = _init_repo(tmp_path)
    new_content = "alpha\nnew value\nomega"
    proposal_registry = ToolRegistry()
    register_propose_patch(proposal_registry)
    proposal = proposal_registry.dispatch(
        "propose_patch",
        {"path": "sample.txt", "new_content": new_content},
        _context(repo),
    )
    proposal_payload = _payload(proposal)
    diff = proposal_payload["diff"]
    assert isinstance(diff, str)
    assert target.read_bytes() == _ORIGINAL
    _switch_to_work_branch(repo)
    gate = _FakeGate(approved=True)

    applied = _dispatch(
        repo,
        {"diff": diff, "rationale": "Apply the deterministic proposal."},
        gate,
    )

    assert _payload(applied)["applied"] is True
    assert target.read_bytes() == new_content.encode("utf-8")
    assert len(gate.calls) == 1


def test_apply_patch__later_escaped_file_keeps_entire_multi_file_patch_atomic(
    tmp_path: Path,
) -> None:
    repo, target = _init_repo(tmp_path)
    second = repo / "second.txt"
    second.write_bytes(b"before\n")
    _require_git_success(repo, "add", "second.txt")
    _require_git_success(repo, "commit", "--quiet", "-m", "add second fixture")
    _switch_to_work_branch(repo)
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside sentinel\n")
    target_before = target.read_bytes()
    second_before = second.read_bytes()
    outside_before = outside.read_bytes()
    diff = (
        _patch()
        + "--- a/second.txt\n"
        + "+++ b/../outside.txt\n"
        + "@@ -1 +1 @@\n"
        + "-before\n"
        + "+after\n"
    )

    result = _dispatch(
        repo,
        {"diff": diff, "rationale": "Apply two related changes."},
        _FakeGate(approved=True),
    )

    _assert_error(result, ErrorType.PathJailError)
    assert target.read_bytes() == target_before
    assert second.read_bytes() == second_before
    assert outside.read_bytes() == outside_before
