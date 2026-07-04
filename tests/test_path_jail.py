from pathlib import Path

import pytest

from app.safety.path_jail import PathJail, PathJailViolation


def _make_root(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    return root


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except OSError as exc:
        pytest.skip(f"Symlink creation is not available in this environment: {exc}")


def test_path_jail__resolves_nested_relative_path(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    nested_dir = root / "src" / "package"
    nested_dir.mkdir(parents=True)
    file_path = nested_dir / "module.py"
    file_path.write_text("VALUE = 1\n", encoding="utf-8")
    jail = PathJail(root)

    resolved = jail.resolve("src/package/module.py")

    assert resolved.is_absolute()
    assert resolved.is_relative_to(jail.root)
    assert resolved == file_path.resolve()


def test_path_jail__rejects_parent_traversal(tmp_path: Path) -> None:
    jail = PathJail(_make_root(tmp_path))

    with pytest.raises(PathJailViolation) as exc_info:
        jail.resolve("../x")

    message = str(exc_info.value)
    assert "../x" in message
    assert "workspace-relative" in message


def test_path_jail__rejects_nested_parent_traversal(tmp_path: Path) -> None:
    jail = PathJail(_make_root(tmp_path))

    with pytest.raises(PathJailViolation):
        jail.resolve("a/../../x")


def test_path_jail__rejects_backslash_parent_traversal(tmp_path: Path) -> None:
    jail = PathJail(_make_root(tmp_path))

    with pytest.raises(PathJailViolation):
        jail.resolve(r"..\x")


def test_path_jail__rejects_posix_absolute_path(tmp_path: Path) -> None:
    jail = PathJail(_make_root(tmp_path))

    with pytest.raises(PathJailViolation):
        jail.resolve("/etc/passwd")


@pytest.mark.parametrize("candidate", (r"C:\Windows\x", "C:/x"))
def test_path_jail__rejects_windows_absolute_path(tmp_path: Path, candidate: str) -> None:
    jail = PathJail(_make_root(tmp_path))

    with pytest.raises(PathJailViolation):
        jail.resolve(candidate)


def test_path_jail__rejects_windows_drive_relative_path(tmp_path: Path) -> None:
    jail = PathJail(_make_root(tmp_path))

    with pytest.raises(PathJailViolation):
        jail.resolve("C:notes.txt")


def test_path_jail__rejects_windows_unc_path(tmp_path: Path) -> None:
    jail = PathJail(_make_root(tmp_path))

    with pytest.raises(PathJailViolation):
        jail.resolve(r"\\server\share\x")


def test_path_jail__rejects_empty_path(tmp_path: Path) -> None:
    jail = PathJail(_make_root(tmp_path))

    with pytest.raises(PathJailViolation):
        jail.resolve("")


def test_path_jail__rejects_symlink_escape(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_file = outside / "secret.txt"
    outside_file.write_text("secret\n", encoding="utf-8")
    _symlink_or_skip(root / "escape", outside)
    jail = PathJail(root)

    with pytest.raises(PathJailViolation) as exc_info:
        jail.resolve("escape/secret.txt")

    assert "escape/secret.txt" in str(exc_info.value)


def test_path_jail__resolves_internal_symlink(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    target = root / "target"
    target.mkdir()
    target_file = target / "note.txt"
    target_file.write_text("safe\n", encoding="utf-8")
    _symlink_or_skip(root / "link", target)
    jail = PathJail(root)

    resolved = jail.resolve("link/note.txt")

    assert resolved == target_file.resolve()
    assert resolved.is_relative_to(jail.root)


def test_path_jail__rejects_missing_or_non_directory_root(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing"
    file_root = tmp_path / "file.txt"
    file_root.write_text("not a directory\n", encoding="utf-8")

    with pytest.raises(ValueError):
        PathJail(missing_root)

    with pytest.raises(ValueError):
        PathJail(file_root)
