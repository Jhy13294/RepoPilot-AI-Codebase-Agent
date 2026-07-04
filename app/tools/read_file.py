"""Text file reader tool registered through the shared registry."""

from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.tool_io import ErrorType
from app.tools.base import ToolContext, ToolFailure
from app.tools.registry import ToolRegistry, ToolSpec

__all__ = ["register"]

MAX_WINDOW_LINES = 400
MAX_CONTENT_BYTES = 204_800
_BINARY_SNIFF_BYTES = 1024


class _ReadFileArgs(BaseModel):
    """Arguments for read_file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    start_line: int = Field(default=1, ge=1)
    end_line: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        if self.end_line is not None and self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        return self


class _ReadFilePayload(BaseModel):
    """Payload returned by read_file.

    Empty files are successful reads with `start_line=1` and `end_line=0`.
    """

    model_config = ConfigDict(frozen=True)

    path: str
    content: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=0)
    total_lines: int = Field(ge=0)
    truncated: bool


def register(registry: ToolRegistry) -> None:
    """Register read_file in a ToolRegistry."""
    registry.register(
        ToolSpec(
            name="read_file",
            description=(
                "Read UTF-8 text file content inside the workspace with an optional inclusive "
                "1-based line window. Use it to ground analysis in real source text. Do not use "
                "it for directory listings or binary files; use get_file_tree to inspect paths."
            ),
            args_schema=_ReadFileArgs,
            returns_schema=_ReadFilePayload,
            risk_level="low",
        ),
        _handle,
    )


def _handle(args: BaseModel, context: ToolContext) -> BaseModel:
    parsed = _ReadFileArgs.model_validate(args)
    path = context.jail.resolve(parsed.path)

    if not path.exists():
        raise ToolFailure(
            ErrorType.NotFoundError,
            f"Path '{parsed.path}' does not exist; choose an existing workspace file.",
            {"path": parsed.path},
        )

    if path.is_dir():
        raise ToolFailure(
            ErrorType.NotFoundError,
            f"Path '{parsed.path}' is a directory; use get_file_tree to inspect directory paths.",
            {"path": parsed.path},
        )

    _reject_binary_sample(path, parsed.path)
    return _read_text_window(path, parsed, context)


def _reject_binary_sample(path: Path, requested_path: str) -> None:
    with path.open("rb") as stream:
        sample = stream.read(_BINARY_SNIFF_BYTES)

    if b"\x00" in sample:
        raise _binary_failure(requested_path)


def _read_text_window(
    path: Path,
    args: _ReadFileArgs,
    context: ToolContext,
) -> _ReadFilePayload:
    collected_lines: list[str] = []
    content_bytes = 0
    returned_end_line = 0
    total_lines = 0
    truncated = False

    try:
        with path.open("r", encoding="utf-8", errors="strict", newline=None) as stream:
            for raw_line in stream:
                total_lines += 1

                if not _is_requested_line(total_lines, args):
                    continue

                line = raw_line.removesuffix("\n")
                if truncated:
                    continue

                if len(collected_lines) == MAX_WINDOW_LINES:
                    truncated = True
                    continue

                added_bytes = len(line.encode("utf-8"))
                if collected_lines:
                    added_bytes += 1
                if content_bytes + added_bytes > MAX_CONTENT_BYTES:
                    truncated = True
                    continue

                collected_lines.append(line)
                content_bytes += added_bytes
                returned_end_line = total_lines
    except UnicodeDecodeError as exc:
        raise _binary_failure(args.path) from exc

    if total_lines == 0:
        if args.start_line != 1:
            raise _invalid_start_line(args.start_line, total_lines)
        return _ReadFilePayload(
            path=_workspace_path(context.jail.root, path),
            content="",
            start_line=1,
            end_line=0,
            total_lines=0,
            truncated=False,
        )

    if args.start_line > total_lines:
        raise _invalid_start_line(args.start_line, total_lines)

    return _ReadFilePayload(
        path=_workspace_path(context.jail.root, path),
        content="\n".join(collected_lines),
        start_line=args.start_line,
        end_line=returned_end_line,
        total_lines=total_lines,
        truncated=truncated,
    )


def _is_requested_line(line_number: int, args: _ReadFileArgs) -> bool:
    if line_number < args.start_line:
        return False
    return args.end_line is None or line_number <= args.end_line


def _binary_failure(requested_path: str) -> ToolFailure:
    return ToolFailure(
        ErrorType.BinaryFileError,
        f"Path '{requested_path}' is not valid UTF-8 text; use get_file_tree to inspect it.",
        {"path": requested_path},
    )


def _invalid_start_line(start_line: int, total_lines: int) -> ToolFailure:
    return ToolFailure(
        ErrorType.InvalidArgsError,
        f"start_line {start_line} is greater than total_lines {total_lines}.",
        {"start_line": start_line, "total_lines": total_lines},
    )


def _workspace_path(workspace_root: Path, path: Path) -> str:
    relative = path.relative_to(workspace_root)
    if relative.parts == ():
        return "."
    return relative.as_posix()
