from pathlib import Path
from typing import Protocol

import pytest

from app.safety.path_jail import PathJail
from app.schemas.tool_io import ErrorType, ToolResult
from app.tools.base import ToolContext
from app.tools.read_file import register
from app.tools.registry import ToolRegistry

MAX_CONTENT_BYTES = 204_800


class _MiniRepoManifest(Protocol):
    long_file_lines: int
    binary_file: str
    parse_date_file: str
    unicode_file: str
    unicode_probe: str


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    register(registry)
    return registry


def _context(root: Path) -> ToolContext:
    return ToolContext(run_id="test-run", jail=PathJail(root))


def _dispatch(root: Path, raw_args: dict[str, object]) -> ToolResult:
    return _registry().dispatch("read_file", raw_args, _context(root))


def _payload(result: ToolResult) -> dict[str, object]:
    assert result.ok is True
    assert result.data is not None
    payload = result.data.model_dump()
    assert isinstance(payload, dict)
    return payload


def _normalized_disk_content(path: Path) -> str:
    return "\n".join(path.read_text(encoding="utf-8").splitlines())


def _lines(payload: dict[str, object]) -> list[str]:
    content = payload["content"]
    assert isinstance(content, str)
    if content == "":
        return []
    return content.split("\n")


def _assert_error(result: ToolResult, error_type: ErrorType) -> None:
    assert result.ok is False
    assert result.error is not None
    assert result.error.type is error_type


def test_read_file__reads_full_dates_file(
    mini_repo: Path,
    mini_repo_manifest: _MiniRepoManifest,
) -> None:
    result = _dispatch(mini_repo, {"path": mini_repo_manifest.parse_date_file})

    payload = _payload(result)
    expected_content = _normalized_disk_content(mini_repo / mini_repo_manifest.parse_date_file)
    expected_lines = expected_content.splitlines()
    assert payload["path"] == mini_repo_manifest.parse_date_file
    assert payload["content"] == expected_content
    assert payload["start_line"] == 1
    assert payload["end_line"] == len(expected_lines)
    assert payload["total_lines"] == len(expected_lines)
    assert payload["truncated"] is False
    assert result.meta.truncated is False


def test_read_file__reads_long_file_window(
    mini_repo: Path,
) -> None:
    result = _dispatch(mini_repo, {"path": "data/long_file.txt", "start_line": 3, "end_line": 6})

    payload = _payload(result)
    assert payload["content"] == "line 003\nline 004\nline 005\nline 006"
    assert payload["start_line"] == 3
    assert payload["end_line"] == 6
    assert payload["total_lines"] == 450
    assert payload["truncated"] is False
    assert result.meta.truncated is False


def test_read_file__caps_default_window_at_400_lines(
    mini_repo: Path,
    mini_repo_manifest: _MiniRepoManifest,
) -> None:
    result = _dispatch(mini_repo, {"path": "data/long_file.txt", "start_line": 1})

    payload = _payload(result)
    lines = _lines(payload)
    assert len(lines) == 400
    assert lines[0] == "line 001"
    assert lines[-1] == "line 400"
    assert payload["end_line"] == 400
    assert payload["total_lines"] == mini_repo_manifest.long_file_lines
    assert payload["truncated"] is True
    assert result.meta.truncated is True


def test_read_file__clamps_end_line_past_eof_without_truncation(
    mini_repo: Path,
) -> None:
    result = _dispatch(
        mini_repo,
        {"path": "data/long_file.txt", "start_line": 440, "end_line": 999},
    )

    payload = _payload(result)
    lines = _lines(payload)
    assert len(lines) == 11
    assert lines[0] == "line 440"
    assert lines[-1] == "line 450"
    assert payload["end_line"] == 450
    assert payload["total_lines"] == 450
    assert payload["truncated"] is False
    assert result.meta.truncated is False


def test_read_file__caps_content_bytes_without_splitting_lines(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    line_values = [f"row {index:03d} " + ("x" * 1010) for index in range(1, 301)]
    (workspace / "wide.txt").write_text("\n".join(line_values), encoding="utf-8")

    result = _dispatch(workspace, {"path": "wide.txt"})

    payload = _payload(result)
    content = payload["content"]
    assert isinstance(content, str)
    returned_lines = _lines(payload)
    assert len(content.encode("utf-8")) <= MAX_CONTENT_BYTES
    assert 0 < len(returned_lines) < len(line_values)
    assert returned_lines == line_values[: len(returned_lines)]
    assert payload["end_line"] == len(returned_lines)
    assert payload["total_lines"] == len(line_values)
    assert payload["truncated"] is True
    assert result.meta.truncated is True


def test_read_file__empty_file_uses_one_zero_convention(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "empty.txt").write_text("", encoding="utf-8")

    result = _dispatch(workspace, {"path": "empty.txt"})

    payload = _payload(result)
    assert payload == {
        "path": "empty.txt",
        "content": "",
        "start_line": 1,
        "end_line": 0,
        "total_lines": 0,
        "truncated": False,
    }
    assert result.meta.truncated is False


def test_read_file__normalizes_crlf_to_lf(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "crlf.txt").write_bytes(b"alpha\r\nbeta\r\ngamma\r\n")

    result = _dispatch(workspace, {"path": "crlf.txt"})

    payload = _payload(result)
    assert payload["content"] == "alpha\nbeta\ngamma"
    assert payload["total_lines"] == 3
    assert payload["end_line"] == 3


def test_read_file__preserves_unicode_probe(
    mini_repo: Path,
    mini_repo_manifest: _MiniRepoManifest,
) -> None:
    usage_path = mini_repo / mini_repo_manifest.unicode_file
    usage_lines = usage_path.read_text(encoding="utf-8").splitlines()
    probe_line = usage_lines.index(mini_repo_manifest.unicode_probe) + 1

    result = _dispatch(
        mini_repo,
        {
            "path": mini_repo_manifest.unicode_file,
            "start_line": probe_line,
            "end_line": probe_line,
        },
    )

    payload = _payload(result)
    assert payload["content"] == mini_repo_manifest.unicode_probe
    assert payload["start_line"] == probe_line
    assert payload["end_line"] == probe_line


def test_read_file__null_byte_binary_maps_to_binary_error(
    mini_repo: Path,
    mini_repo_manifest: _MiniRepoManifest,
) -> None:
    result = _dispatch(mini_repo, {"path": mini_repo_manifest.binary_file})

    _assert_error(result, ErrorType.BinaryFileError)
    assert result.error is not None
    assert "get_file_tree" in result.error.message


def test_read_file__invalid_utf8_without_null_maps_to_binary_error(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "latin1.txt").write_bytes(b"caf\xe9\n")

    result = _dispatch(workspace, {"path": "latin1.txt"})

    _assert_error(result, ErrorType.BinaryFileError)
    assert result.error is not None
    assert "get_file_tree" in result.error.message


def test_read_file__missing_path_maps_to_not_found(mini_repo: Path) -> None:
    result = _dispatch(mini_repo, {"path": "nope"})

    _assert_error(result, ErrorType.NotFoundError)


def test_read_file__directory_maps_to_not_found_with_get_file_tree_hint(mini_repo: Path) -> None:
    result = _dispatch(mini_repo, {"path": "src"})

    _assert_error(result, ErrorType.NotFoundError)
    assert result.error is not None
    assert "get_file_tree" in result.error.message


def test_read_file__jail_escape_maps_to_path_jail_error(mini_repo: Path) -> None:
    result = _dispatch(mini_repo, {"path": "../x"})

    _assert_error(result, ErrorType.PathJailError)


@pytest.mark.parametrize(
    "raw_args",
    (
        {"path": "data/long_file.txt", "start_line": 0},
        {"path": "data/long_file.txt", "start_line": 6, "end_line": 5},
        {"path": "data/long_file.txt", "unknown": True},
    ),
)
def test_read_file__invalid_schema_args_are_rejected(
    mini_repo: Path,
    raw_args: dict[str, object],
) -> None:
    result = _dispatch(mini_repo, raw_args)

    _assert_error(result, ErrorType.InvalidArgsError)


def test_read_file__start_line_past_total_is_rejected(
    mini_repo: Path,
    mini_repo_manifest: _MiniRepoManifest,
) -> None:
    start_line = mini_repo_manifest.long_file_lines + 1

    result = _dispatch(mini_repo, {"path": "data/long_file.txt", "start_line": start_line})

    _assert_error(result, ErrorType.InvalidArgsError)
    assert result.error is not None
    assert str(start_line) in result.error.message
    assert str(mini_repo_manifest.long_file_lines) in result.error.message
