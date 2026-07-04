"""Workspace-relative path resolution with escape protection."""

from pathlib import Path, PurePosixPath, PureWindowsPath


class PathJailViolation(Exception):
    """Raised when a candidate path is not contained by the workspace root."""


class PathJail:
    """Resolve workspace-relative paths while preventing lexical and symlink escapes."""

    def __init__(self, root: Path) -> None:
        try:
            resolved_root = root.resolve()
        except (OSError, ValueError) as exc:
            raise ValueError(f"Workspace root '{root}' could not be resolved.") from exc

        if not resolved_root.is_dir():
            raise ValueError(f"Workspace root '{root}' must exist and be a directory.")

        self._root = resolved_root

    @property
    def root(self) -> Path:
        return self._root

    def resolve(self, candidate: str) -> Path:
        self._reject_lexical_escape(candidate)

        try:
            resolved_candidate = (self._root / candidate).resolve()
        except (OSError, ValueError) as exc:
            raise PathJailViolation(
                f"Path '{candidate}' could not be resolved inside the workspace root; "
                "use a valid workspace-relative path."
            ) from exc

        if not resolved_candidate.is_relative_to(self._root):
            raise PathJailViolation(
                f"Path '{candidate}' escapes the workspace root; "
                "use a workspace-relative path that stays inside the workspace root."
            )

        return resolved_candidate

    @staticmethod
    def _reject_lexical_escape(candidate: str) -> None:
        if candidate.strip() == "":
            raise PathJailViolation(
                f"Path '{candidate}' is empty; use a workspace-relative path inside the "
                "workspace root."
            )

        for parsed_candidate in (PureWindowsPath(candidate), PurePosixPath(candidate)):
            if (
                parsed_candidate.is_absolute()
                or parsed_candidate.drive
                or parsed_candidate.root
                or ".." in parsed_candidate.parts
            ):
                raise PathJailViolation(
                    f"Path '{candidate}' escapes the workspace root; use a workspace-relative "
                    "path without drive, root, or '..'."
                )
