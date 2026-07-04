from pathlib import Path
from typing import Protocol

import pytest

from app.safety.path_jail import PathJail
from app.schemas.tool_io import ErrorType, ToolResult
from app.tools.base import ToolContext
from app.tools.registry import ToolRegistry
from app.tools.search_code import register


class _MiniRepoManifest(Protocol):
    binary_file: str
    parse_date_file: str
    parse_date_line: int
    parse_date_mention_files: frozenset[str]


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    register(registry)
    return registry


def _context(root: Path) -> ToolContext:
    return ToolContext(run_id="test-run", jail=PathJail(root))


def _dispatch(root: Path, raw_args: dict[str, object]) -> ToolResult:
    return _registry().dispatch("search_code", raw_args, _context(root))


def _payload(result: ToolResult) -> dict[str, object]:
    assert result.ok is True
    assert result.data is not None
    payload = result.data.model_dump()
    assert isinstance(payload, dict)
    return payload


def _matches(result: ToolResult) -> list[dict[str, object]]:
    payload = _payload(result)
    matches = payload["matches"]
    assert isinstance(matches, list)
    return matches


def _paths(result: ToolResult) -> list[str]:
    return [str(match["path"]) for match in _matches(result)]


def _path_line_pairs(result: ToolResult) -> list[tuple[str, int]]:
    return [(str(match["path"]), int(match["line"])) for match in _matches(result)]


def _line_number(root: Path, relative_path: str, needle: str) -> int:
    lines = (root / relative_path).read_text(encoding="utf-8").splitlines()
    return next(index for index, line in enumerate(lines, start=1) if needle in line)


def _match_by_path(result: ToolResult, path: str) -> dict[str, object]:
    return next(match for match in _matches(result) if match["path"] == path)


def _assert_error(result: ToolResult, error_type: ErrorType) -> None:
    assert result.ok is False
    assert result.error is not None
    assert result.error.type is error_type


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except OSError as exc:
        pytest.skip(f"Symlink creation is not available in this environment: {exc}")


def test_search_code__literal_parse_date_returns_three_ordered_matches_with_context(
    mini_repo: Path,
    mini_repo_manifest: _MiniRepoManifest,
) -> None:
    result = _dispatch(mini_repo, {"query": "parse_date"})

    assert result.ok is True
    payload = _payload(result)
    assert payload["total_found"] == 4
    assert payload["truncated"] is False
    assert result.meta.truncated is False
    assert list(dict.fromkeys(_paths(result))) == sorted(
        mini_repo_manifest.parse_date_mention_files
    )
    assert _path_line_pairs(result) == [
        ("docs/usage.md", _line_number(mini_repo, "docs/usage.md", "parse_date")),
        ("src/sample_pkg/core.py", _line_number(mini_repo, "src/sample_pkg/core.py", "parse_date")),
        ("src/sample_pkg/core.py", _line_number(mini_repo, "src/sample_pkg/core.py", "return")),
        (mini_repo_manifest.parse_date_file, mini_repo_manifest.parse_date_line),
    ]

    definition = _match_by_path(result, mini_repo_manifest.parse_date_file)
    assert definition["text"] == "def parse_date(value: str) -> date:"
    assert definition["context_before"] == ["", ""]
    assert definition["context_after"] == [
        "    stripped = value.strip()",
        "    for fmt in _INPUT_FORMATS:",
    ]


def test_search_code__regex_finds_unique_parse_date_definition(
    mini_repo: Path,
    mini_repo_manifest: _MiniRepoManifest,
) -> None:
    result = _dispatch(mini_repo, {"query": r"def parse_date\(", "regex": True})

    assert _path_line_pairs(result) == [
        (mini_repo_manifest.parse_date_file, mini_repo_manifest.parse_date_line)
    ]
    assert _payload(result)["total_found"] == 1


def test_search_code__invalid_regex_reports_compiler_message(mini_repo: Path) -> None:
    result = _dispatch(mini_repo, {"query": "[", "regex": True})

    _assert_error(result, ErrorType.InvalidArgsError)
    assert result.error is not None
    assert "unterminated character set" in result.error.message


def test_search_code__recursive_python_glob_only_returns_python_matches(mini_repo: Path) -> None:
    result = _dispatch(mini_repo, {"query": "parse_date", "glob": "**/*.py"})

    assert _paths(result) == [
        "src/sample_pkg/core.py",
        "src/sample_pkg/core.py",
        "src/sample_pkg/dates.py",
    ]
    assert _payload(result)["total_found"] == 3


def test_search_code__top_level_markdown_glob_does_not_cross_directories(mini_repo: Path) -> None:
    result = _dispatch(mini_repo, {"query": "RepoPilot", "glob": "*.md"})

    assert _paths(result) == ["README.md"]
    assert _payload(result)["total_found"] == 1


def test_search_code__src_recursive_glob_limits_matches_to_src(mini_repo: Path) -> None:
    result = _dispatch(mini_repo, {"query": "parse_date", "glob": "src/**/*.py"})

    assert _paths(result) == [
        "src/sample_pkg/core.py",
        "src/sample_pkg/core.py",
        "src/sample_pkg/dates.py",
    ]
    assert _payload(result)["total_found"] == 3


@pytest.mark.parametrize("glob", ("../x", "/", "/etc/*", "C:/x"))
def test_search_code__invalid_glob_is_rejected(mini_repo: Path, glob: str) -> None:
    result = _dispatch(mini_repo, {"query": "parse_date", "glob": glob})

    _assert_error(result, ErrorType.InvalidArgsError)


def test_search_code__zero_matches_is_success_with_empty_matches(mini_repo: Path) -> None:
    result = _dispatch(mini_repo, {"query": "definitely-not-present"})

    payload = _payload(result)
    assert payload["matches"] == []
    assert payload["total_found"] == 0
    assert payload["truncated"] is False
    assert result.meta.truncated is False


def test_search_code__max_results_truncates_matches_but_counts_total(mini_repo: Path) -> None:
    result = _dispatch(mini_repo, {"query": "line ", "max_results": 2})

    matches = _matches(result)
    payload = _payload(result)
    assert len(matches) == 2
    assert payload["total_found"] > 2
    assert payload["truncated"] is True
    assert result.meta.truncated is True


def test_search_code__context_lines_zero_returns_no_neighbors(mini_repo: Path) -> None:
    result = _dispatch(mini_repo, {"query": "def parse_date(", "context_lines": 0})

    match = _matches(result)[0]
    assert match["context_before"] == []
    assert match["context_after"] == []


def test_search_code__first_line_match_has_empty_context_before(mini_repo: Path) -> None:
    result = _dispatch(
        mini_repo,
        {"query": "from sample_pkg.dates", "glob": "src/sample_pkg/core.py"},
    )

    match = _matches(result)[0]
    assert match["line"] == 1
    assert match["context_before"] == []
    assert match["context_after"] == ["", 'DEFAULT_INPUT = "2026-07-04"']


def test_search_code__binary_file_is_silently_skipped(
    mini_repo: Path,
    mini_repo_manifest: _MiniRepoManifest,
) -> None:
    result = _dispatch(mini_repo, {"query": "LOGO", "glob": mini_repo_manifest.binary_file})

    payload = _payload(result)
    assert payload["matches"] == []
    assert payload["total_found"] == 0


def test_search_code__symlink_file_is_not_searched(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (workspace / "visible.txt").write_text("ordinary text\n", encoding="utf-8")
    target = outside / "secret.txt"
    target.write_text("outside-only-token\n", encoding="utf-8")
    _symlink_or_skip(workspace / "linked-secret.txt", target)

    result = _dispatch(workspace, {"query": "outside-only-token"})

    payload = _payload(result)
    assert payload["matches"] == []
    assert payload["total_found"] == 0


@pytest.mark.parametrize("raw_args", ({"query": "parse_date", "unknown": True}, {"query": ""}))
def test_search_code__schema_args_are_rejected(
    mini_repo: Path,
    raw_args: dict[str, object],
) -> None:
    result = _dispatch(mini_repo, raw_args)

    _assert_error(result, ErrorType.InvalidArgsError)
