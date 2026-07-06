"""Deterministic validation for workspace file-line citations."""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from app.safety.path_jail import PathJail, PathJailViolation

_LINE_SUFFIX_RE = re.compile(r"\A(?P<path>.+):(?P<start>[0-9]+)(?:-(?P<end>[0-9]+))?\Z")
_WINDOWS_DRIVE_PATH_RE = re.compile(r"\A[A-Za-z]:[^:]*\Z")
_BINARY_SNIFF_BYTES = 1024


class CitationStatus(StrEnum):
    """Validation status for one citation."""

    valid = "valid"
    unparseable = "unparseable"
    path_not_found = "path_not_found"
    path_escapes = "path_escapes"
    line_out_of_range = "line_out_of_range"
    binary_file = "binary_file"


@dataclass(frozen=True, slots=True)
class Citation:
    """Parsed citation syntax."""

    raw: str
    path: str
    start_line: int | None
    end_line: int | None


@dataclass(frozen=True, slots=True)
class CitationCheck:
    """Validation result for one raw citation."""

    raw: str
    status: CitationStatus
    citation: Citation | None
    detail: str


@dataclass(frozen=True, slots=True)
class CitationReport:
    """Aggregate validation result for a batch of citations."""

    checks: tuple[CitationCheck, ...]

    @property
    def valid_count(self) -> int:
        return sum(check.status is CitationStatus.valid for check in self.checks)

    @property
    def all_valid(self) -> bool:
        return all(check.status is CitationStatus.valid for check in self.checks)

    @property
    def invalid(self) -> tuple[CitationCheck, ...]:
        return tuple(check for check in self.checks if check.status is not CitationStatus.valid)


def parse_citation(raw: str) -> Citation | None:
    """Parse PATH, PATH:LINE, or PATH:START-END citation syntax."""
    text = raw.strip()
    if text == "":
        return None

    line_match = _LINE_SUFFIX_RE.fullmatch(text)
    if line_match is not None:
        path = line_match.group("path")
        start_line = int(line_match.group("start"))
        raw_end_line = line_match.group("end")
        end_line = int(raw_end_line) if raw_end_line is not None else None
        return Citation(raw=raw, path=path, start_line=start_line, end_line=end_line)

    if ":" in text and _WINDOWS_DRIVE_PATH_RE.fullmatch(text) is None:
        return None

    return Citation(raw=raw, path=text, start_line=None, end_line=None)


def validate_citation(raw: str, jail: PathJail) -> CitationCheck:
    """Validate one citation against a jailed workspace."""
    citation = parse_citation(raw)
    if citation is None:
        return CitationCheck(
            raw=raw,
            status=CitationStatus.unparseable,
            citation=None,
            detail="Citation must use PATH, PATH:LINE, or PATH:START-END syntax.",
        )

    try:
        path = jail.resolve(citation.path)
    except PathJailViolation:
        return CitationCheck(
            raw=raw,
            status=CitationStatus.path_escapes,
            citation=citation,
            detail=f"Path '{citation.path}' escapes the workspace root.",
        )

    if not path.is_file():
        return CitationCheck(
            raw=raw,
            status=CitationStatus.path_not_found,
            citation=citation,
            detail=f"Path '{citation.path}' is not a file inside the workspace.",
        )

    if citation.start_line is None:
        return CitationCheck(
            raw=raw,
            status=CitationStatus.valid,
            citation=citation,
            detail=f"Path '{citation.path}' exists.",
        )

    total_lines = _count_utf8_lines(path)
    if total_lines is None:
        return CitationCheck(
            raw=raw,
            status=CitationStatus.binary_file,
            citation=citation,
            detail=f"Path '{citation.path}' is not valid UTF-8 text.",
        )

    end_line = citation.end_line if citation.end_line is not None else citation.start_line
    if _line_range_is_out_of_range(citation.start_line, end_line, total_lines):
        return CitationCheck(
            raw=raw,
            status=CitationStatus.line_out_of_range,
            citation=citation,
            detail=(
                f"Line range {citation.start_line}-{end_line} is outside "
                f"the file's {total_lines} line(s)."
            ),
        )

    return CitationCheck(
        raw=raw,
        status=CitationStatus.valid,
        citation=citation,
        detail=f"Line range {citation.start_line}-{end_line} exists in '{citation.path}'.",
    )


def validate_citations(raws: Iterable[str], jail: PathJail) -> CitationReport:
    """Validate multiple citations against a jailed workspace."""
    return CitationReport(tuple(validate_citation(raw, jail) for raw in raws))


def _count_utf8_lines(path: Path) -> int | None:
    with path.open("rb") as stream:
        if b"\x00" in stream.read(_BINARY_SNIFF_BYTES):
            return None

    total_lines = 0
    try:
        with path.open("r", encoding="utf-8", errors="strict", newline=None) as stream:
            for _line in stream:
                total_lines += 1
    except UnicodeDecodeError:
        return None

    return total_lines


def _line_range_is_out_of_range(start_line: int, end_line: int, total_lines: int) -> bool:
    return start_line < 1 or end_line < 1 or start_line > end_line or end_line > total_lines
