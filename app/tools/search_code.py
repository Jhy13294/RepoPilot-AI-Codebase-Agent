"""Code search tool registered through the shared registry."""

import os
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.tool_io import ErrorType
from app.tools.base import ToolContext, ToolFailure, workspace_relative_path
from app.tools.registry import ToolRegistry, ToolSpec

__all__ = ["register"]

_BINARY_SNIFF_BYTES = 1024


class _SearchCodeArgs(BaseModel):
    """Arguments for search_code."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1)
    regex: bool = False
    glob: str | None = None
    max_results: int = Field(default=50, ge=1, le=500)
    context_lines: int = Field(default=2, ge=0, le=10)


class _Match(BaseModel):
    """One matching line with nearby context."""

    model_config = ConfigDict(frozen=True)

    path: str
    line: int = Field(ge=1)
    text: str
    context_before: list[str]
    context_after: list[str]


class _SearchPayload(BaseModel):
    """Payload returned by search_code."""

    model_config = ConfigDict(frozen=True)

    matches: list[_Match]
    total_found: int = Field(ge=0)
    truncated: bool


def register(registry: ToolRegistry) -> None:
    """Register search_code in a ToolRegistry."""
    registry.register(
        ToolSpec(
            name="search_code",
            description=(
                "Search UTF-8 text files in the workspace by literal text or regular expression, "
                "optionally constrained by a workspace-relative glob such as '**/*.py'. Use it to "
                "locate symbols, call sites, and relevant snippets before reading files. Do not "
                "use it for directory topology, long contiguous file reads, binary files, hidden "
                "paths, symlinked paths, or file mutation."
            ),
            args_schema=_SearchCodeArgs,
            returns_schema=_SearchPayload,
            risk_level="low",
        ),
        _handle,
    )


def _handle(args: BaseModel, context: ToolContext) -> BaseModel:
    parsed = _SearchCodeArgs.model_validate(args)
    pattern = _compile_pattern(parsed)
    allowed_paths = _allowed_glob_paths(context.jail.root, parsed.glob)

    matches: list[_Match] = []
    total_found = 0

    for path in _safe_file_paths(context.jail.root, allowed_paths):
        lines = _read_text_lines(path)
        if lines is None:
            continue

        relative_path = workspace_relative_path(context.jail.root, path)
        for line_index, line in enumerate(lines):
            if not _line_matches(parsed, pattern, line):
                continue

            total_found += 1
            if len(matches) < parsed.max_results:
                matches.append(
                    _build_match(relative_path, line_index, line, lines, parsed.context_lines)
                )

    return _SearchPayload(
        matches=matches, total_found=total_found, truncated=total_found > len(matches)
    )


def _compile_pattern(args: _SearchCodeArgs) -> re.Pattern[str] | None:
    if not args.regex:
        return None

    try:
        return re.compile(args.query)
    except re.error as exc:
        raise ToolFailure(
            ErrorType.InvalidArgsError,
            f"Invalid regular expression for query '{args.query}': {exc}.",
            {"query": args.query, "regex_error": str(exc)},
        ) from exc


def _line_matches(args: _SearchCodeArgs, pattern: re.Pattern[str] | None, line: str) -> bool:
    if pattern is None:
        return args.query in line
    return pattern.search(line) is not None


def _allowed_glob_paths(root: Path, pattern: str | None) -> set[Path] | None:
    if pattern is None:
        return None

    _validate_glob(pattern)
    try:
        return {path.resolve() for path in root.glob(pattern) if path.is_file()}
    except ValueError as exc:
        raise ToolFailure(
            ErrorType.InvalidArgsError,
            f"Invalid glob pattern '{pattern}': {exc}.",
            {"glob": pattern},
        ) from exc


def _validate_glob(pattern: str) -> None:
    has_drive_prefix = len(pattern) >= 2 and pattern[1] == ":"
    if (
        ".." in pattern
        or pattern.startswith(("/", "\\"))
        or Path(pattern).drive
        or has_drive_prefix
    ):
        raise ToolFailure(
            ErrorType.InvalidArgsError,
            (
                f"Invalid glob pattern '{pattern}': glob must be a relative workspace pattern "
                "without '..', absolute roots, or drive prefixes."
            ),
            {"glob": pattern},
        )


def _safe_file_paths(root: Path, allowed_paths: set[Path] | None) -> list[Path]:
    paths: list[Path] = []

    for current_root, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        directory = Path(current_root)
        dirnames[:] = sorted(
            dirname
            for dirname in dirnames
            if not dirname.startswith(".") and not (directory / dirname).is_symlink()
        )

        for filename in sorted(filenames):
            if filename.startswith("."):
                continue

            path = directory / filename
            if path.is_symlink():
                continue

            resolved_path = path.resolve()
            if allowed_paths is not None and resolved_path not in allowed_paths:
                continue

            paths.append(path)

    return sorted(paths, key=lambda path: workspace_relative_path(root, path))


def _read_text_lines(path: Path) -> list[str] | None:
    with path.open("rb") as stream:
        if b"\x00" in stream.read(_BINARY_SNIFF_BYTES):
            return None

    try:
        with path.open("r", encoding="utf-8", errors="strict", newline=None) as stream:
            return [line.removesuffix("\n") for line in stream]
    except UnicodeDecodeError:
        return None


def _build_match(
    relative_path: str,
    line_index: int,
    line: str,
    lines: list[str],
    context_lines: int,
) -> _Match:
    return _Match(
        path=relative_path,
        line=line_index + 1,
        text=line,
        context_before=lines[max(0, line_index - context_lines) : line_index],
        context_after=lines[line_index + 1 : line_index + 1 + context_lines],
    )
