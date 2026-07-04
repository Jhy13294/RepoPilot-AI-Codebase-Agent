"""Directory tree tool registered through the shared registry."""

from collections import deque
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.tool_io import ErrorType
from app.tools.base import ToolContext, ToolFailure
from app.tools.registry import ToolRegistry, ToolSpec

__all__ = ["register"]


class _GetFileTreeArgs(BaseModel):
    """Arguments for get_file_tree."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = "."
    max_depth: int = Field(default=4, ge=1, le=8)
    max_entries: int = Field(default=500, ge=1, le=500)
    include_hidden: bool = False


class _TreeEntry(BaseModel):
    """One workspace-relative tree entry."""

    model_config = ConfigDict(frozen=True)

    path: str
    kind: Literal["file", "dir"]
    size_bytes: int = Field(ge=0)


class _TreePayload(BaseModel):
    """Payload returned by get_file_tree."""

    model_config = ConfigDict(frozen=True)

    root: str
    entries: list[_TreeEntry]
    truncated: bool


def register(registry: ToolRegistry) -> None:
    """Register get_file_tree in a ToolRegistry."""
    registry.register(
        ToolSpec(
            name="get_file_tree",
            description=(
                "List a deterministic, breadth-first directory tree inside the workspace. "
                "Use it to understand repository topology before choosing files. "
                "Do not use it to read file contents; use read_file for that."
            ),
            args_schema=_GetFileTreeArgs,
            returns_schema=_TreePayload,
            risk_level="low",
        ),
        _handle,
    )


def _handle(args: BaseModel, context: ToolContext) -> BaseModel:
    """Return a breadth-first directory listing.

    Symlinks are skipped with `is_symlink()` and are never listed or descended, so traversal
    cannot become a second path around the workspace jail.
    """
    parsed = _GetFileTreeArgs.model_validate(args)
    root = context.jail.resolve(parsed.path)

    if not root.exists():
        raise ToolFailure(
            ErrorType.NotFoundError,
            f"Path '{parsed.path}' does not exist; choose an existing workspace directory.",
            {"path": parsed.path},
        )

    if not root.is_dir():
        if root.is_file():
            message = (
                f"Path '{parsed.path}' is a file, not a directory; use read_file to read "
                "file contents."
            )
        else:
            message = f"Path '{parsed.path}' is not a directory; choose a directory path."
        raise ToolFailure(ErrorType.NotFoundError, message, {"path": parsed.path})

    entries: list[_TreeEntry] = []
    queue: deque[tuple[Path, int]] = deque([(root, 0)])

    while queue:
        directory, current_depth = queue.popleft()
        if current_depth >= parsed.max_depth:
            continue

        for child in sorted(directory.iterdir(), key=lambda path: path.name):
            if child.is_symlink():
                continue
            if not parsed.include_hidden and child.name.startswith("."):
                continue

            child_depth = current_depth + 1
            if child_depth > parsed.max_depth:
                continue

            if child.is_dir():
                entries.append(
                    _TreeEntry(
                        path=_workspace_path(context.jail.root, child), kind="dir", size_bytes=0
                    )
                )
                if child_depth < parsed.max_depth:
                    queue.append((child, child_depth))
            else:
                entries.append(
                    _TreeEntry(
                        path=_workspace_path(context.jail.root, child),
                        kind="file",
                        size_bytes=child.stat().st_size,
                    )
                )

            if len(entries) == parsed.max_entries:
                return _TreePayload(
                    root=_workspace_path(context.jail.root, root),
                    entries=entries,
                    truncated=True,
                )

    return _TreePayload(
        root=_workspace_path(context.jail.root, root),
        entries=entries,
        truncated=False,
    )


def _workspace_path(workspace_root: Path, path: Path) -> str:
    relative = path.relative_to(workspace_root)
    if relative.parts == ():
        return "."
    return relative.as_posix()
