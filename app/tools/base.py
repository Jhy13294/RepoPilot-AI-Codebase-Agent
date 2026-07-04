"""Base types shared by tool implementations and the registry."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, JsonValue

from app.safety.path_jail import PathJail
from app.schemas.tool_io import ErrorType


class ToolFailure(Exception):
    """Structured failure raised by tool implementations."""

    def __init__(
        self,
        type: ErrorType,
        message: str,
        details: dict[str, JsonValue] | None = None,
    ) -> None:
        super().__init__(message)
        self.type = type
        self.message = message
        self.details = details


class ToolContext(BaseModel):
    """Execution context passed to every tool handler."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    run_id: str
    jail: PathJail


def workspace_relative_path(root: Path, path: Path) -> str:
    """Return a workspace-relative POSIX path, using '.' for the workspace root."""
    relative = path.relative_to(root)
    if relative.parts == ():
        return "."
    return relative.as_posix()
