import subprocess
from pathlib import Path

import pytest
from pydantic import BaseModel

from app.safety.path_jail import PathJail
from app.schemas.tool_io import ErrorType, ToolResult
from app.tools import git_create_branch
from app.tools.base import ToolContext
from app.tools.git_create_branch import register
from app.tools.registry import ApprovalOutcome, ToolRegistry, ToolSpec

_RUN_ID = "test-run"
_BRANCH = f"repopilot/fix-{_RUN_ID}"
_ORIGINAL = b"tracked content\n"


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


def _run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        check=False,
        shell=False,
        timeout=10,
    )


def _require_git_success(repo: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    completed = _run_git(repo, *args)
    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
    return completed


def _init_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _require_git_success(repo, "init", "--initial-branch=main", "--quiet")
    _require_git_success(repo, "config", "core.autocrlf", "false")
    _require_git_success(repo, "config", "user.name", "RepoPilot Tests")
    _require_git_success(repo, "config", "user.email", "tests@repopilot.local")
    target = repo / "tracked.txt"
    target.write_bytes(_ORIGINAL)
    _require_git_success(repo, "add", "tracked.txt")
    _require_git_success(repo, "commit", "--quiet", "-m", "test fixture")
    return repo, target


def _context(root: Path) -> ToolContext:
    return ToolContext(run_id=_RUN_ID, jail=PathJail(root))


def _registry(gate: _FakeGate | None) -> ToolRegistry:
    registry = ToolRegistry(approval_gate=gate)
    register(registry)
    return registry


def _dispatch(
    root: Path,
    raw_args: dict[str, object],
    gate: _FakeGate | None,
) -> ToolResult:
    return _registry(gate).dispatch("git_create_branch", raw_args, _context(root))


def _payload(result: ToolResult) -> dict[str, object]:
    assert result.ok is True
    assert result.data is not None
    assert result.error is None
    payload = result.data.model_dump()
    assert isinstance(payload, dict)
    return payload


def _assert_error(result: ToolResult, error_type: ErrorType) -> None:
    assert result.ok is False
    assert result.data is None
    assert result.error is not None
    assert result.error.type is error_type


def _current_branch(repo: Path) -> str:
    completed = _require_git_success(repo, "branch", "--show-current")
    return completed.stdout.decode("utf-8", errors="strict").strip()


def _branch_exists(repo: Path, branch: str = _BRANCH) -> bool:
    completed = _run_git(repo, "show-ref", "--verify", "--quiet", f"refs/heads/{branch}")
    assert completed.returncode in (0, 1), completed.stderr.decode("utf-8", errors="replace")
    return completed.returncode == 0


def _switch_to_work_branch(repo: Path) -> None:
    _require_git_success(repo, "checkout", "--quiet", "-b", _BRANCH)


def test_git_create_branch__approved_create_switches_without_committing(
    tmp_path: Path,
) -> None:
    repo, _target = _init_repo(tmp_path)
    gate = _FakeGate(approved=True)
    head_before = _require_git_success(repo, "rev-parse", "HEAD").stdout

    result = _dispatch(repo, {"rationale": "Create the isolated work branch."}, gate)

    assert _payload(result) == {
        "branch": _BRANCH,
        "created": True,
        "switched": True,
    }
    assert result.meta.tool_name == "git_create_branch"
    assert result.meta.truncated is False
    assert _current_branch(repo) == _BRANCH
    assert _require_git_success(repo, "rev-parse", "HEAD").stdout == head_before
    assert _require_git_success(repo, "status", "--porcelain", "--untracked-files=no").stdout == b""
    assert len(gate.calls) == 1
    spec, args, context = gate.calls[0]
    assert spec.name == "git_create_branch"
    assert spec.risk_level == "high"
    assert args.model_dump() == {"rationale": "Create the isolated work branch."}
    assert context.run_id == _RUN_ID


def test_git_create_branch__existing_branch_is_switched_to_without_recreation(
    tmp_path: Path,
) -> None:
    repo, _target = _init_repo(tmp_path)
    _switch_to_work_branch(repo)
    _require_git_success(repo, "checkout", "--quiet", "main")
    gate = _FakeGate(approved=True)

    result = _dispatch(repo, {"rationale": "Resume the existing work branch."}, gate)

    assert _payload(result) == {
        "branch": _BRANCH,
        "created": False,
        "switched": True,
    }
    assert _current_branch(repo) == _BRANCH
    assert len(gate.calls) == 1


def test_git_create_branch__current_branch_is_a_clean_no_op(tmp_path: Path) -> None:
    repo, _target = _init_repo(tmp_path)
    _switch_to_work_branch(repo)
    gate = _FakeGate(approved=True)

    result = _dispatch(repo, {"rationale": "Confirm the current work branch."}, gate)

    assert _payload(result) == {
        "branch": _BRANCH,
        "created": False,
        "switched": False,
    }
    assert _current_branch(repo) == _BRANCH
    assert len(gate.calls) == 1


def test_git_create_branch__current_branch_is_a_no_op_with_dirty_tracked_file(
    tmp_path: Path,
) -> None:
    repo, target = _init_repo(tmp_path)
    _switch_to_work_branch(repo)
    dirty_content = b"work already applied\n"
    target.write_bytes(dirty_content)
    gate = _FakeGate(approved=True)

    result = _dispatch(repo, {"rationale": "Resume after applying the patch."}, gate)

    assert _payload(result) == {
        "branch": _BRANCH,
        "created": False,
        "switched": False,
    }
    assert _current_branch(repo) == _BRANCH
    assert target.read_bytes() == dirty_content
    assert _require_git_success(repo, "status", "--porcelain", "--untracked-files=no").stdout
    assert len(gate.calls) == 1


def test_git_create_branch__dirty_tracked_tree_is_rejected_without_mutation(
    tmp_path: Path,
) -> None:
    repo, target = _init_repo(tmp_path)
    dirty_content = b"uncommitted user work\n"
    target.write_bytes(dirty_content)
    gate = _FakeGate(approved=True)
    head_before = _require_git_success(repo, "rev-parse", "HEAD").stdout

    result = _dispatch(repo, {"rationale": "Create the isolated work branch."}, gate)

    _assert_error(result, ErrorType.GitError)
    assert result.error is not None
    assert result.error.details is not None
    assert result.error.details["reason"] == "dirty_worktree"
    assert _current_branch(repo) == "main"
    assert _branch_exists(repo) is False
    assert _require_git_success(repo, "rev-parse", "HEAD").stdout == head_before
    assert target.read_bytes() == dirty_content
    assert len(gate.calls) == 1


def test_git_create_branch__untracked_files_do_not_block_creation(tmp_path: Path) -> None:
    repo, _target = _init_repo(tmp_path)
    untracked = repo / "notes.txt"
    untracked.write_bytes(b"user notes\n")

    result = _dispatch(
        repo,
        {"rationale": "Create the isolated work branch."},
        _FakeGate(approved=True),
    )

    assert _payload(result)["created"] is True
    assert _current_branch(repo) == _BRANCH
    assert untracked.read_bytes() == b"user notes\n"


def test_git_create_branch__denial_preserves_branch_state(tmp_path: Path) -> None:
    repo, _target = _init_repo(tmp_path)
    gate = _FakeGate(approved=False, reason="Review the branch operation first.")

    result = _dispatch(repo, {"rationale": "Create the isolated work branch."}, gate)

    _assert_error(result, ErrorType.ApprovalDeniedError)
    assert result.error is not None
    assert result.error.message == "Review the branch operation first."
    assert _current_branch(repo) == "main"
    assert _branch_exists(repo) is False
    assert len(gate.calls) == 1


def test_git_create_branch__missing_gate_fails_closed_without_mutation(
    tmp_path: Path,
) -> None:
    repo, _target = _init_repo(tmp_path)

    result = _dispatch(repo, {"rationale": "Create the isolated work branch."}, None)

    _assert_error(result, ErrorType.ApprovalDeniedError)
    assert result.error is not None
    assert "no approval gate configured" in result.error.message
    assert _current_branch(repo) == "main"
    assert _branch_exists(repo) is False


def test_git_create_branch__empty_rationale_is_invalid_before_approval(
    tmp_path: Path,
) -> None:
    repo, _target = _init_repo(tmp_path)
    gate = _FakeGate(approved=True)

    result = _dispatch(repo, {"rationale": ""}, gate)

    _assert_error(result, ErrorType.InvalidArgsError)
    assert gate.calls == []
    assert _current_branch(repo) == "main"
    assert _branch_exists(repo) is False


def test_git_create_branch__model_cannot_supply_branch_name(tmp_path: Path) -> None:
    repo, _target = _init_repo(tmp_path)
    gate = _FakeGate(approved=True)

    result = _dispatch(
        repo,
        {"rationale": "Create the isolated work branch.", "branch": "user/choice"},
        gate,
    )

    _assert_error(result, ErrorType.InvalidArgsError)
    assert gate.calls == []
    assert _current_branch(repo) == "main"
    assert _branch_exists(repo) is False


def test_git_create_branch__llm_schema_exposes_only_rationale() -> None:
    schema = _registry(_FakeGate(approved=True)).to_llm_schema()[0]
    function = schema["function"]
    assert isinstance(function, dict)
    parameters = function["parameters"]
    assert isinstance(parameters, dict)

    assert parameters["required"] == ["rationale"]
    assert parameters["additionalProperties"] is False
    properties = parameters["properties"]
    assert isinstance(properties, dict)
    assert tuple(properties) == ("rationale",)


def test_git_create_branch__registry_result_round_trips_as_an_envelope(
    tmp_path: Path,
) -> None:
    repo, _target = _init_repo(tmp_path)

    result = _dispatch(
        repo,
        {"rationale": "Create the branch through the registry."},
        _FakeGate(approved=True),
    )

    decoded = ToolResult.model_validate_json(result.model_dump_json())
    assert decoded.ok is True
    assert decoded.error is None
    assert decoded.data is not None
    assert decoded.data.model_dump() == {
        "branch": _BRANCH,
        "created": True,
        "switched": True,
    }
    assert decoded.meta.tool_name == "git_create_branch"


def test_git_create_branch__non_git_directory_returns_typed_git_error(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    gate = _FakeGate(approved=True)

    result = _dispatch(
        workspace,
        {"rationale": "Create the isolated work branch."},
        gate,
    )

    _assert_error(result, ErrorType.GitError)
    assert result.error is not None
    assert result.error.details is not None
    assert result.error.details["reason"] == "git_error"
    assert result.error.details["returncode"] != 0
    assert isinstance(result.error.details["stderr"], str)
    assert result.meta.tool_name == "git_create_branch"
    assert len(gate.calls) == 1


def test_git_create_branch__git_timeout_returns_typed_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _target = _init_repo(tmp_path)
    gate = _FakeGate(approved=True)

    def raise_timeout(
        *_args: object,
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.TimeoutExpired(
            cmd=("git", "rev-parse"),
            timeout=10,
            stderr=b"simulated timeout stderr",
        )

    with monkeypatch.context() as scoped_patch:
        scoped_patch.setattr(git_create_branch.subprocess, "run", raise_timeout)
        result = _dispatch(
            repo,
            {"rationale": "Create the isolated work branch."},
            gate,
        )

    _assert_error(result, ErrorType.GitError)
    assert result.error is not None
    assert result.error.details is not None
    assert result.error.details["reason"] == "git_timeout"
    assert result.error.details["returncode"] is None
    assert result.error.details["stderr"] == "simulated timeout stderr"
    assert len(gate.calls) == 1
    assert _current_branch(repo) == "main"
    assert _branch_exists(repo) is False


def test_git_create_branch__git_os_error_returns_typed_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _target = _init_repo(tmp_path)
    gate = _FakeGate(approved=True)

    def raise_os_error(
        *_args: object,
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        raise OSError("simulated missing git executable")

    with monkeypatch.context() as scoped_patch:
        scoped_patch.setattr(git_create_branch.subprocess, "run", raise_os_error)
        result = _dispatch(
            repo,
            {"rationale": "Create the isolated work branch."},
            gate,
        )

    _assert_error(result, ErrorType.GitError)
    assert result.error is not None
    assert result.error.details is not None
    assert result.error.details["reason"] == "git_error"
    assert result.error.details["returncode"] is None
    assert result.error.details["stderr"] == "simulated missing git executable"
    assert len(gate.calls) == 1
    assert _current_branch(repo) == "main"
    assert _branch_exists(repo) is False


@pytest.mark.parametrize(
    ("branch_exists", "expected_reason", "expected_checkout"),
    (
        (False, "create_failed", ("git", "checkout", "--quiet", "-b", _BRANCH)),
        (True, "switch_failed", ("git", "checkout", "--quiet", _BRANCH)),
    ),
)
def test_git_create_branch__checkout_failure_is_typed_and_subprocess_is_locked_down(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    branch_exists: bool,
    expected_reason: str,
    expected_checkout: tuple[str, ...],
) -> None:
    repo, _target = _init_repo(tmp_path)
    gate = _FakeGate(approved=True)
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def fake_run(
        argv: tuple[str, ...],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append((argv, kwargs))
        if argv[1] == "rev-parse":
            return subprocess.CompletedProcess(argv, 0, stdout=b"main\n", stderr=b"")
        if argv[1] == "status":
            return subprocess.CompletedProcess(argv, 0, stdout=b"", stderr=b"")
        if argv[1] == "show-ref":
            return subprocess.CompletedProcess(
                argv,
                0 if branch_exists else 1,
                stdout=b"",
                stderr=b"",
            )
        return subprocess.CompletedProcess(
            argv,
            128,
            stdout=b"",
            stderr=b"simulated checkout failure",
        )

    with monkeypatch.context() as scoped_patch:
        scoped_patch.setattr(git_create_branch.subprocess, "run", fake_run)
        result = _dispatch(
            repo,
            {"rationale": "Create the isolated work branch."},
            gate,
        )

    _assert_error(result, ErrorType.GitError)
    assert result.error is not None
    assert result.error.details is not None
    assert result.error.details["reason"] == expected_reason
    assert result.error.details["returncode"] == 128
    assert result.error.details["stderr"] == "simulated checkout failure"
    assert calls[-1][0] == expected_checkout
    for _argv, kwargs in calls:
        assert kwargs == {
            "cwd": repo,
            "stdin": subprocess.DEVNULL,
            "capture_output": True,
            "check": False,
            "shell": False,
            "timeout": 10,
        }
    assert len(gate.calls) == 1
    assert _current_branch(repo) == "main"
    assert _branch_exists(repo) is False
