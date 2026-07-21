import subprocess
from pathlib import Path

import pytest
from pydantic import BaseModel

from app.safety.path_jail import PathJail
from app.schemas.tool_io import ErrorType, ToolResult
from app.tools import git_commit
from app.tools.base import ToolContext
from app.tools.git_commit import register
from app.tools.registry import ApprovalOutcome, ToolRegistry, ToolSpec

_RUN_ID = "test-run"
_BRANCH = f"repopilot/fix-{_RUN_ID}"
_MESSAGE = "fix: update tracked content"
_RATIONALE = "Commit the verified tracked change."
_ORIGINAL = b"tracked content\n"
_UPDATED = b"updated content\n"


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
        stdin=subprocess.DEVNULL,
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


def _switch_to_work_branch(repo: Path) -> None:
    _require_git_success(repo, "checkout", "--quiet", "-b", _BRANCH)


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
    return _registry(gate).dispatch("git_commit", raw_args, _context(root))


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


def _head(repo: Path) -> str:
    return _require_git_success(repo, "rev-parse", "HEAD").stdout.decode().strip()


def _tracked_status(repo: Path) -> bytes:
    return _require_git_success(
        repo,
        "status",
        "--porcelain",
        "--untracked-files=no",
    ).stdout


def test_git_commit__approved_commit_advances_head_and_returns_payload(
    tmp_path: Path,
) -> None:
    repo, target = _init_repo(tmp_path)
    _switch_to_work_branch(repo)
    target.write_bytes(_UPDATED)
    gate = _FakeGate(approved=True)
    head_before = _head(repo)

    result = _dispatch(
        repo,
        {"message": _MESSAGE, "rationale": _RATIONALE},
        gate,
    )

    head_after = _head(repo)
    assert head_after != head_before
    assert _payload(result) == {
        "branch": _BRANCH,
        "commit": head_after,
        "message": _MESSAGE,
        "files_committed": ["tracked.txt"],
        "committed": True,
    }
    assert result.meta.tool_name == "git_commit"
    assert result.meta.truncated is False
    assert _tracked_status(repo) == b""
    assert _require_git_success(repo, "show", "HEAD:tracked.txt").stdout == _UPDATED
    author = _require_git_success(repo, "show", "-s", "--format=%an%x00%ae", "HEAD").stdout
    assert author.rstrip(b"\n") == b"RepoPilot\x00noreply@repopilot.invalid"
    assert len(gate.calls) == 1
    spec, args, context = gate.calls[0]
    assert spec.name == "git_commit"
    assert spec.risk_level == "high"
    assert args.model_dump() == {"message": _MESSAGE, "rationale": _RATIONALE}
    assert context.run_id == _RUN_ID


def test_git_commit__untracked_secret_is_not_committed(tmp_path: Path) -> None:
    repo, target = _init_repo(tmp_path)
    _switch_to_work_branch(repo)
    target.write_bytes(_UPDATED)
    secret = repo / "secret.env"
    secret.write_bytes(b"API_KEY=do-not-commit\n")

    result = _dispatch(
        repo,
        {"message": _MESSAGE, "rationale": _RATIONALE},
        _FakeGate(approved=True),
    )

    assert _payload(result)["files_committed"] == ["tracked.txt"]
    assert secret.read_bytes() == b"API_KEY=do-not-commit\n"
    status = _require_git_success(
        repo,
        "status",
        "--porcelain",
        "--untracked-files=all",
    ).stdout
    assert status == b"?? secret.env\n"
    tree = _require_git_success(repo, "ls-tree", "-r", "--name-only", "HEAD").stdout
    assert b"secret.env" not in tree.splitlines()


def test_git_commit__wrong_branch_is_rejected_without_commit(tmp_path: Path) -> None:
    repo, target = _init_repo(tmp_path)
    target.write_bytes(_UPDATED)
    gate = _FakeGate(approved=True)
    head_before = _head(repo)
    status_before = _tracked_status(repo)

    result = _dispatch(
        repo,
        {"message": _MESSAGE, "rationale": _RATIONALE},
        gate,
    )

    _assert_error(result, ErrorType.GitError)
    assert result.error is not None
    assert result.error.details is not None
    assert result.error.details["reason"] == "wrong_branch"
    assert result.error.details["current_branch"] == "main"
    assert result.error.details["expected_branch"] == _BRANCH
    assert _head(repo) == head_before
    assert _tracked_status(repo) == status_before
    assert len(gate.calls) == 1


def test_git_commit__nothing_to_commit_is_rejected(tmp_path: Path) -> None:
    repo, _target = _init_repo(tmp_path)
    _switch_to_work_branch(repo)
    gate = _FakeGate(approved=True)
    head_before = _head(repo)

    result = _dispatch(
        repo,
        {"message": _MESSAGE, "rationale": _RATIONALE},
        gate,
    )

    _assert_error(result, ErrorType.GitError)
    assert result.error is not None
    assert result.error.details is not None
    assert result.error.details["reason"] == "nothing_to_commit"
    assert _head(repo) == head_before
    assert _tracked_status(repo) == b""
    assert len(gate.calls) == 1


def test_git_commit__denial_preserves_head_and_worktree(tmp_path: Path) -> None:
    repo, target = _init_repo(tmp_path)
    _switch_to_work_branch(repo)
    target.write_bytes(_UPDATED)
    gate = _FakeGate(approved=False, reason="Review the commit first.")
    head_before = _head(repo)
    status_before = _tracked_status(repo)

    result = _dispatch(
        repo,
        {"message": _MESSAGE, "rationale": _RATIONALE},
        gate,
    )

    _assert_error(result, ErrorType.ApprovalDeniedError)
    assert result.error is not None
    assert result.error.message == "Review the commit first."
    assert _head(repo) == head_before
    assert _tracked_status(repo) == status_before
    assert target.read_bytes() == _UPDATED
    assert len(gate.calls) == 1


def test_git_commit__missing_gate_fails_closed_without_commit(tmp_path: Path) -> None:
    repo, target = _init_repo(tmp_path)
    _switch_to_work_branch(repo)
    target.write_bytes(_UPDATED)
    head_before = _head(repo)
    status_before = _tracked_status(repo)

    result = _dispatch(
        repo,
        {"message": _MESSAGE, "rationale": _RATIONALE},
        None,
    )

    _assert_error(result, ErrorType.ApprovalDeniedError)
    assert result.error is not None
    assert "no approval gate configured" in result.error.message
    assert _head(repo) == head_before
    assert _tracked_status(repo) == status_before


@pytest.mark.parametrize(
    "raw_args",
    (
        {"message": "", "rationale": _RATIONALE},
        {"message": _MESSAGE, "rationale": ""},
    ),
)
def test_git_commit__empty_required_argument_is_invalid_before_approval(
    tmp_path: Path,
    raw_args: dict[str, object],
) -> None:
    repo, target = _init_repo(tmp_path)
    _switch_to_work_branch(repo)
    target.write_bytes(_UPDATED)
    gate = _FakeGate(approved=True)
    head_before = _head(repo)

    result = _dispatch(repo, raw_args, gate)

    _assert_error(result, ErrorType.InvalidArgsError)
    assert gate.calls == []
    assert _head(repo) == head_before
    assert target.read_bytes() == _UPDATED


@pytest.mark.parametrize(
    "extra",
    (
        {"branch": "user/choice"},
        {"author": "User Supplied"},
        {"unexpected": "value"},
    ),
)
def test_git_commit__model_cannot_supply_extra_fields(
    tmp_path: Path,
    extra: dict[str, object],
) -> None:
    repo, target = _init_repo(tmp_path)
    _switch_to_work_branch(repo)
    target.write_bytes(_UPDATED)
    gate = _FakeGate(approved=True)
    raw_args: dict[str, object] = {
        "message": _MESSAGE,
        "rationale": _RATIONALE,
        **extra,
    }

    result = _dispatch(repo, raw_args, gate)

    _assert_error(result, ErrorType.InvalidArgsError)
    assert gate.calls == []
    assert target.read_bytes() == _UPDATED


def test_git_commit__llm_schema_exposes_only_message_and_rationale() -> None:
    schema = _registry(_FakeGate(approved=True)).to_llm_schema()[0]
    function = schema["function"]
    assert isinstance(function, dict)
    parameters = function["parameters"]
    assert isinstance(parameters, dict)

    assert parameters["required"] == ["message", "rationale"]
    assert parameters["additionalProperties"] is False
    properties = parameters["properties"]
    assert isinstance(properties, dict)
    assert tuple(properties) == ("message", "rationale")


def test_git_commit__registry_result_round_trips_as_an_envelope(
    tmp_path: Path,
) -> None:
    repo, target = _init_repo(tmp_path)
    _switch_to_work_branch(repo)
    target.write_bytes(_UPDATED)

    result = _dispatch(
        repo,
        {"message": _MESSAGE, "rationale": _RATIONALE},
        _FakeGate(approved=True),
    )

    decoded = ToolResult.model_validate_json(result.model_dump_json())
    assert decoded.ok is True
    assert decoded.error is None
    assert decoded.data is not None
    assert decoded.data.model_dump() == _payload(result)
    assert decoded.meta.tool_name == "git_commit"


def test_git_commit__subprocess_argv_is_locked_down(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []
    expected_sha = "a" * 40

    def fake_run(
        argv: tuple[str, ...],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append((argv, kwargs))
        if argv == ("git", "rev-parse", "--abbrev-ref", "HEAD"):
            return subprocess.CompletedProcess(argv, 0, stdout=f"{_BRANCH}\n".encode(), stderr=b"")
        if argv == ("git", "status", "--porcelain", "--untracked-files=no"):
            return subprocess.CompletedProcess(argv, 0, stdout=b" M tracked.txt\n", stderr=b"")
        if argv[-2:] == ("rev-parse", "HEAD"):
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout=f"{expected_sha}\n".encode(),
                stderr=b"",
            )
        if "commit" in argv:
            return subprocess.CompletedProcess(argv, 0, stdout=b"committed\n", stderr=b"")
        raise AssertionError(f"Unexpected subprocess argv: {argv!r}")

    with monkeypatch.context() as scoped_patch:
        scoped_patch.setattr(git_commit.subprocess, "run", fake_run)
        result = _dispatch(
            tmp_path,
            {"message": _MESSAGE, "rationale": _RATIONALE},
            _FakeGate(approved=True),
        )

    assert _payload(result)["commit"] == expected_sha
    assert [argv for argv, _kwargs in calls] == [
        ("git", "rev-parse", "--abbrev-ref", "HEAD"),
        ("git", "status", "--porcelain", "--untracked-files=no"),
        (
            "git",
            "-c",
            "user.name=RepoPilot",
            "-c",
            "user.email=noreply@repopilot.invalid",
            "commit",
            "-a",
            "-m",
            _MESSAGE,
        ),
        ("git", "rev-parse", "HEAD"),
    ]
    for argv, kwargs in calls:
        assert kwargs == {
            "cwd": tmp_path,
            "stdin": subprocess.DEVNULL,
            "capture_output": True,
            "check": False,
            "shell": False,
            "timeout": 10,
        }
        assert "push" not in argv
        assert "--amend" not in argv
        assert "remote" not in argv
        assert not (len(argv) > 1 and argv[1] == "add")


def test_git_commit__commit_failure_returns_typed_git_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_run(
        argv: tuple[str, ...],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append(argv)
        if argv == ("git", "rev-parse", "--abbrev-ref", "HEAD"):
            return subprocess.CompletedProcess(argv, 0, stdout=f"{_BRANCH}\n".encode(), stderr=b"")
        if argv == ("git", "status", "--porcelain", "--untracked-files=no"):
            return subprocess.CompletedProcess(argv, 0, stdout=b" M tracked.txt\n", stderr=b"")
        return subprocess.CompletedProcess(
            argv,
            128,
            stdout=b"",
            stderr=b"simulated commit failure",
        )

    with monkeypatch.context() as scoped_patch:
        scoped_patch.setattr(git_commit.subprocess, "run", fake_run)
        result = _dispatch(
            tmp_path,
            {"message": _MESSAGE, "rationale": _RATIONALE},
            _FakeGate(approved=True),
        )

    _assert_error(result, ErrorType.GitError)
    assert result.error is not None
    assert result.error.details is not None
    assert result.error.details["reason"] == "commit_failed"
    assert result.error.details["returncode"] == 128
    assert result.error.details["stderr"] == "simulated commit failure"
    assert calls[-1][-3:] == ("-a", "-m", _MESSAGE)
    assert ("git", "rev-parse", "HEAD") not in calls


def test_git_commit__git_timeout_returns_typed_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        scoped_patch.setattr(git_commit.subprocess, "run", raise_timeout)
        result = _dispatch(
            tmp_path,
            {"message": _MESSAGE, "rationale": _RATIONALE},
            gate,
        )

    _assert_error(result, ErrorType.GitError)
    assert result.error is not None
    assert result.error.details is not None
    assert result.error.details["reason"] == "git_timeout"
    assert result.error.details["operation"] == "current_branch"
    assert result.error.details["returncode"] is None
    assert result.error.details["stderr"] == "simulated timeout stderr"
    assert len(gate.calls) == 1


def test_git_commit__git_os_error_returns_typed_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _FakeGate(approved=True)

    def raise_os_error(
        *_args: object,
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        raise OSError("simulated missing git executable")

    with monkeypatch.context() as scoped_patch:
        scoped_patch.setattr(git_commit.subprocess, "run", raise_os_error)
        result = _dispatch(
            tmp_path,
            {"message": _MESSAGE, "rationale": _RATIONALE},
            gate,
        )

    _assert_error(result, ErrorType.GitError)
    assert result.error is not None
    assert result.error.details is not None
    assert result.error.details["reason"] == "git_error"
    assert result.error.details["operation"] == "current_branch"
    assert result.error.details["returncode"] is None
    assert result.error.details["stderr"] == "simulated missing git executable"
    assert len(gate.calls) == 1
