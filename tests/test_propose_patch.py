import subprocess
from pathlib import Path

import pytest

from app.safety.path_jail import PathJail
from app.schemas.tool_io import ErrorType, ToolResult
from app.tools.base import ToolContext
from app.tools.propose_patch import register
from app.tools.registry import ToolRegistry

MAX_CONTENT_BYTES = 204_800


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    register(registry)
    return registry


def _context(root: Path) -> ToolContext:
    return ToolContext(run_id="test-run", jail=PathJail(root))


def _dispatch(root: Path, raw_args: dict[str, object]) -> ToolResult:
    return _registry().dispatch("propose_patch", raw_args, _context(root))


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


def _run_git(repo: Path, *args: str, input_bytes: bytes | None = None) -> None:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        input=input_bytes,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")


def test_propose_patch__round_trip_passes_git_apply_and_restores_exact_content(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    target = repo / "src" / "example.py"
    target.parent.mkdir(parents=True)
    original = b"def answer() -> int:\n    return 41\n"
    new_content = "def answer() -> int:\n    value = 42\n    return value"
    target.write_bytes(original)
    _run_git(repo, "init", "--quiet")
    _run_git(repo, "config", "core.autocrlf", "false")

    result = _dispatch(repo, {"path": "src/example.py", "new_content": new_content})

    payload = _payload(result)
    diff = payload["diff"]
    assert isinstance(diff, str)
    assert diff.startswith("--- a/src/example.py\n+++ b/src/example.py\n")
    assert "\\ No newline at end of file\n" in diff
    assert target.read_bytes() == original

    patch_bytes = diff.encode("utf-8")
    _run_git(repo, "apply", "--check", "-", input_bytes=patch_bytes)
    assert target.read_bytes() == original
    _run_git(repo, "apply", "-", input_bytes=patch_bytes)
    assert target.read_bytes() == new_content.encode("utf-8")


def test_propose_patch__counts_changes_and_uses_three_context_lines(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    original_lines = [f"line {number}\n" for number in range(1, 10)]
    new_lines = [
        *original_lines[:4],
        "line five updated\n",
        "line five and a half\n",
        *original_lines[5:],
    ]
    target = workspace / "sample.txt"
    target.write_bytes("".join(original_lines).encode("utf-8"))

    result = _dispatch(
        workspace,
        {"path": "sample.txt", "new_content": "".join(new_lines)},
    )

    payload = _payload(result)
    diff = payload["diff"]
    assert isinstance(diff, str)
    assert payload["insertions"] == 2
    assert payload["deletions"] == 1
    assert payload["is_noop"] is False
    assert "@@ -2,7 +2,8 @@\n" in diff
    assert " line 1\n" not in diff
    assert " line 2\n" in diff
    assert " line 8\n" in diff
    assert " line 9\n" not in diff
    assert target.read_bytes() == "".join(original_lines).encode("utf-8")


def test_propose_patch__identical_content_is_successful_noop(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "sample.txt"
    content = "alpha\r\nbeta\r\n"
    target.write_bytes(content.encode("utf-8"))

    result = _dispatch(workspace, {"path": "sample.txt", "new_content": content})

    payload = _payload(result)
    assert payload == {
        "path": "sample.txt",
        "diff": "",
        "insertions": 0,
        "deletions": 0,
        "is_noop": True,
    }
    assert result.error is None
    assert target.read_bytes() == content.encode("utf-8")


def test_propose_patch__missing_path_maps_to_not_found(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    result = _dispatch(workspace, {"path": "missing.txt", "new_content": "replacement\n"})

    _assert_error(result, ErrorType.NotFoundError)


def test_propose_patch__directory_maps_to_not_found_with_get_file_tree_hint(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "src").mkdir()

    result = _dispatch(workspace, {"path": "src", "new_content": "replacement\n"})

    _assert_error(result, ErrorType.NotFoundError)
    assert result.error is not None
    assert "get_file_tree" in result.error.message


def test_propose_patch__null_byte_binary_maps_to_binary_error(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "binary.dat").write_bytes(b"text\x00payload")

    result = _dispatch(workspace, {"path": "binary.dat", "new_content": "replacement\n"})

    _assert_error(result, ErrorType.BinaryFileError)


def test_propose_patch__invalid_utf8_without_null_maps_to_binary_error(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "latin1.txt").write_bytes(b"caf\xe9\n")

    result = _dispatch(workspace, {"path": "latin1.txt", "new_content": "cafe\n"})

    _assert_error(result, ErrorType.BinaryFileError)


@pytest.mark.parametrize("escaped_path", ("../outside.txt", "/outside.txt"))
def test_propose_patch__jail_escape_maps_to_path_jail_error(
    tmp_path: Path,
    escaped_path: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    result = _dispatch(workspace, {"path": escaped_path, "new_content": "replacement\n"})

    _assert_error(result, ErrorType.PathJailError)


def test_propose_patch__oversized_new_content_maps_to_invalid_args(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "sample.txt").write_text("original\n", encoding="utf-8")
    oversized_content = "界" * ((MAX_CONTENT_BYTES // 3) + 1)

    result = _dispatch(
        workspace,
        {"path": "sample.txt", "new_content": oversized_content},
    )

    _assert_error(result, ErrorType.InvalidArgsError)


def test_propose_patch__oversized_current_file_maps_to_invalid_args(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "large.txt").write_bytes(b"x" * (MAX_CONTENT_BYTES + 1))

    result = _dispatch(workspace, {"path": "large.txt", "new_content": "replacement\n"})

    _assert_error(result, ErrorType.InvalidArgsError)


def test_propose_patch__extra_argument_is_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "sample.txt").write_text("original\n", encoding="utf-8")

    result = _dispatch(
        workspace,
        {"path": "sample.txt", "new_content": "replacement\n", "unknown": True},
    )

    _assert_error(result, ErrorType.InvalidArgsError)


def test_propose_patch__bare_registry_allows_medium_risk_without_approval_gate(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "sample.txt").write_text("old\n", encoding="utf-8")
    registry = ToolRegistry(approval_gate=None)
    register(registry)

    result = registry.dispatch(
        "propose_patch",
        {"path": "sample.txt", "new_content": "new\n"},
        _context(workspace),
    )

    payload = _payload(result)
    assert payload["path"] == "sample.txt"
    assert result.error is None
    assert result.meta.tool_name == "propose_patch"
    assert result.meta.latency_ms >= 0
    assert result.meta.truncated is False
