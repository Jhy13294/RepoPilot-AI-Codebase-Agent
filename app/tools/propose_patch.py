"""Unified diff proposal tool registered through the shared registry."""

from difflib import unified_diff
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.tool_io import ErrorType
from app.tools.base import ToolContext, ToolFailure, workspace_relative_path
from app.tools.read_file import MAX_CONTENT_BYTES
from app.tools.registry import ToolRegistry, ToolSpec

__all__ = ["register"]

_BINARY_SNIFF_BYTES = 1024
_DIFF_CONTEXT_LINES = 3
_NO_NEWLINE_MARKER = "\\ No newline at end of file\n"


class _ProposePatchArgs(BaseModel):
    """Arguments for propose_patch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    new_content: str


class _ProposePatchPayload(BaseModel):
    """Payload returned by propose_patch."""

    model_config = ConfigDict(frozen=True)

    path: str
    diff: str
    insertions: int = Field(ge=0)
    deletions: int = Field(ge=0)
    is_noop: bool


def register(registry: ToolRegistry) -> None:
    """Register propose_patch in a ToolRegistry."""
    registry.register(
        ToolSpec(
            name="propose_patch",
            description=(
                "Build a deterministic git-apply-compatible unified diff that replaces the full "
                "content of one existing UTF-8 workspace file. This tool only proposes a patch; "
                "it does not write files or invoke git."
            ),
            args_schema=_ProposePatchArgs,
            returns_schema=_ProposePatchPayload,
            risk_level="medium",
        ),
        _handle,
    )


def _handle(args: BaseModel, context: ToolContext) -> BaseModel:
    parsed = _ProposePatchArgs.model_validate(args)
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

    _validate_new_content(parsed.new_content, parsed.path)
    current_content = _read_current_content(path, parsed.path)
    relative_path = workspace_relative_path(context.jail.root, path)

    if parsed.new_content == current_content:
        return _ProposePatchPayload(
            path=relative_path,
            diff="",
            insertions=0,
            deletions=0,
            is_noop=True,
        )

    diff_records = list(
        unified_diff(
            current_content.splitlines(keepends=True),
            parsed.new_content.splitlines(keepends=True),
            fromfile=f"a/{relative_path}",
            tofile=f"b/{relative_path}",
            n=_DIFF_CONTEXT_LINES,
            lineterm="\n",
        )
    )
    insertions, deletions = _count_changes(diff_records)
    return _ProposePatchPayload(
        path=relative_path,
        diff=_render_diff(diff_records),
        insertions=insertions,
        deletions=deletions,
        is_noop=False,
    )


def _validate_new_content(content: str, requested_path: str) -> None:
    try:
        content_bytes = len(content.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ToolFailure(
            ErrorType.InvalidArgsError,
            f"new_content for path '{requested_path}' is not valid UTF-8 text.",
            {"path": requested_path},
        ) from exc

    if content_bytes > MAX_CONTENT_BYTES:
        raise _content_too_large_failure("new_content", requested_path, content_bytes)


def _read_current_content(path: Path, requested_path: str) -> str:
    with path.open("rb") as stream:
        content = stream.read(MAX_CONTENT_BYTES + 1)

    if b"\x00" in content[:_BINARY_SNIFF_BYTES]:
        raise _binary_failure(requested_path)

    if len(content) > MAX_CONTENT_BYTES:
        raise _content_too_large_failure("Current file", requested_path, len(content))

    try:
        return content.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise _binary_failure(requested_path) from exc


def _count_changes(diff_records: list[str]) -> tuple[int, int]:
    body_records = diff_records[2:]
    insertions = sum(record.startswith("+") for record in body_records)
    deletions = sum(record.startswith("-") for record in body_records)
    return insertions, deletions


def _render_diff(diff_records: list[str]) -> str:
    rendered: list[str] = []
    for record in diff_records:
        rendered.append(record)
        if not record.endswith("\n"):
            rendered.extend(("\n", _NO_NEWLINE_MARKER))
    return "".join(rendered)


def _binary_failure(requested_path: str) -> ToolFailure:
    return ToolFailure(
        ErrorType.BinaryFileError,
        f"Path '{requested_path}' is not valid UTF-8 text; use get_file_tree to inspect it.",
        {"path": requested_path},
    )


def _content_too_large_failure(
    content_name: str,
    requested_path: str,
    content_bytes: int,
) -> ToolFailure:
    return ToolFailure(
        ErrorType.InvalidArgsError,
        f"{content_name} for path '{requested_path}' exceeds the {MAX_CONTENT_BYTES}-byte limit.",
        {
            "path": requested_path,
            "content_bytes": content_bytes,
            "max_content_bytes": MAX_CONTENT_BYTES,
        },
    )
