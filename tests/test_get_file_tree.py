from pathlib import Path
from typing import Protocol

import pytest

from app.safety.path_jail import PathJail
from app.schemas.tool_io import ErrorType, ToolResult
from app.tools.base import ToolContext
from app.tools.get_file_tree import register
from app.tools.registry import ToolRegistry


class _MiniRepoManifest(Protocol):
    files: frozenset[str]
    dirs: frozenset[str]
    parse_date_file: str
    hidden_entries: frozenset[str]


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    register(registry)
    return registry


def _context(root: Path) -> ToolContext:
    return ToolContext(run_id="test-run", jail=PathJail(root))


def _dispatch(root: Path, raw_args: dict[str, object]) -> ToolResult:
    return _registry().dispatch("get_file_tree", raw_args, _context(root))


def _entry_maps(result: ToolResult) -> list[dict[str, object]]:
    assert result.data is not None
    payload = result.data.model_dump()
    entries = payload["entries"]
    assert isinstance(entries, list)
    return entries


def _paths(result: ToolResult) -> set[str]:
    return {str(entry["path"]) for entry in _entry_maps(result)}


def _entry_by_path(result: ToolResult, path: str) -> dict[str, object]:
    return next(entry for entry in _entry_maps(result) if entry["path"] == path)


def _visible_paths(manifest: _MiniRepoManifest) -> set[str]:
    return set(manifest.files | manifest.dirs) - set(manifest.hidden_entries)


def _depth(path: str) -> int:
    return path.count("/") + 1


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except OSError as exc:
        pytest.skip(f"Symlink creation is not available in this environment: {exc}")


def test_get_file_tree__default_lists_visible_full_tree(
    mini_repo: Path,
    mini_repo_manifest: _MiniRepoManifest,
) -> None:
    result = _dispatch(mini_repo, {})

    assert result.ok is True
    assert result.meta.truncated is False
    assert result.data is not None
    payload = result.data.model_dump()
    assert payload["truncated"] is False
    assert _paths(result) == _visible_paths(mini_repo_manifest)
    assert _paths(result).isdisjoint(mini_repo_manifest.hidden_entries)

    known_file = mini_repo_manifest.parse_date_file
    assert (
        _entry_by_path(result, known_file)["size_bytes"] == (mini_repo / known_file).stat().st_size
    )

    known_dir = "src"
    assert _entry_by_path(result, known_dir)["kind"] == "dir"
    assert _entry_by_path(result, known_dir)["size_bytes"] == 0


def test_get_file_tree__include_hidden_lists_all_manifest_entries(
    mini_repo: Path,
    mini_repo_manifest: _MiniRepoManifest,
) -> None:
    result = _dispatch(mini_repo, {"include_hidden": True})

    assert result.ok is True
    assert _paths(result) == set(mini_repo_manifest.files | mini_repo_manifest.dirs)
    assert len(_entry_maps(result)) == 19
    assert ".hidden.cfg" in _paths(result)


def test_get_file_tree__src_subtree_uses_src_root(
    mini_repo: Path,
    mini_repo_manifest: _MiniRepoManifest,
) -> None:
    result = _dispatch(mini_repo, {"path": "src"})

    assert result.ok is True
    assert result.data is not None
    payload = result.data.model_dump()
    assert payload["root"] == "src"
    assert _paths(result) == {
        path for path in _visible_paths(mini_repo_manifest) if path.startswith("src/")
    }
    assert all(path.startswith("src/") for path in _paths(result))


def test_get_file_tree__max_depth_one_lists_only_direct_children(
    mini_repo: Path,
    mini_repo_manifest: _MiniRepoManifest,
) -> None:
    result = _dispatch(mini_repo, {"max_depth": 1})

    assert result.ok is True
    assert _paths(result) == {
        path for path in _visible_paths(mini_repo_manifest) if "/" not in path
    }


def test_get_file_tree__max_entries_sets_payload_and_meta_truncated(
    mini_repo: Path,
) -> None:
    result = _dispatch(mini_repo, {"max_entries": 5})

    assert result.ok is True
    assert len(_entry_maps(result)) == 5
    assert result.data is not None
    assert result.data.model_dump()["truncated"] is True
    assert result.meta.truncated is True


def test_get_file_tree__output_is_deterministic_bfs(
    mini_repo: Path,
) -> None:
    first = _dispatch(mini_repo, {})
    second = _dispatch(mini_repo, {})

    assert first.ok is True
    assert second.ok is True
    assert first.data is not None
    assert second.data is not None
    assert first.data.model_dump_json().encode() == second.data.model_dump_json().encode()

    depths = [_depth(str(entry["path"])) for entry in _entry_maps(first)]
    assert depths == sorted(depths)


def test_get_file_tree__jail_escape_maps_to_path_jail_error(mini_repo: Path) -> None:
    result = _dispatch(mini_repo, {"path": "../x"})

    assert result.ok is False
    assert result.error is not None
    assert result.error.type is ErrorType.PathJailError


def test_get_file_tree__missing_path_maps_to_not_found(mini_repo: Path) -> None:
    result = _dispatch(mini_repo, {"path": "nope"})

    assert result.ok is False
    assert result.error is not None
    assert result.error.type is ErrorType.NotFoundError


def test_get_file_tree__file_path_maps_to_not_found_with_read_file_hint(mini_repo: Path) -> None:
    result = _dispatch(mini_repo, {"path": "README.md"})

    assert result.ok is False
    assert result.error is not None
    assert result.error.type is ErrorType.NotFoundError
    assert "read_file" in result.error.message


@pytest.mark.parametrize("raw_args", ({"max_depth": 0}, {"unknown": True}))
def test_get_file_tree__invalid_args_are_rejected(
    mini_repo: Path,
    raw_args: dict[str, object],
) -> None:
    result = _dispatch(mini_repo, raw_args)

    assert result.ok is False
    assert result.error is not None
    assert result.error.type is ErrorType.InvalidArgsError


def test_get_file_tree__skips_symlink_directories(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (workspace / "visible.txt").write_text("visible\n", encoding="utf-8")
    (outside / "secret.txt").write_text("secret\n", encoding="utf-8")
    _symlink_or_skip(workspace / "external_link", outside)

    result = _dispatch(workspace, {"include_hidden": True})

    assert result.ok is True
    assert "visible.txt" in _paths(result)
    assert not any(path.startswith("external_link") for path in _paths(result))
