from pathlib import Path
from typing import Protocol

import pytest

from app.agent.citations import (
    Citation,
    CitationCheck,
    CitationReport,
    CitationStatus,
    parse_citation,
    validate_citation,
    validate_citations,
)
from app.safety.path_jail import PathJail


class _MiniRepoManifest(Protocol):
    binary_file: str
    long_file_lines: int
    parse_date_file: str


def _jail(root: Path) -> PathJail:
    return PathJail(root)


def _assert_status(check: CitationCheck, status: CitationStatus) -> None:
    assert check.status is status


def test_parse_citation__accepts_path_only() -> None:
    assert parse_citation("src/sample_pkg/core.py") == Citation(
        raw="src/sample_pkg/core.py",
        path="src/sample_pkg/core.py",
        start_line=None,
        end_line=None,
    )


def test_parse_citation__accepts_single_line() -> None:
    assert parse_citation("src/sample_pkg/core.py:7") == Citation(
        raw="src/sample_pkg/core.py:7",
        path="src/sample_pkg/core.py",
        start_line=7,
        end_line=None,
    )


def test_parse_citation__accepts_inclusive_range() -> None:
    assert parse_citation("src/sample_pkg/core.py:7-9") == Citation(
        raw="src/sample_pkg/core.py:7-9",
        path="src/sample_pkg/core.py",
        start_line=7,
        end_line=9,
    )


@pytest.mark.parametrize(
    "raw",
    (
        "",
        "   ",
        ":1",
        "src/sample_pkg/core.py:",
        "src/sample_pkg/core.py:line",
        "src/sample_pkg/core.py:1-",
        "src/sample_pkg/core.py:-1",
        "src/sample_pkg/core.py:1-two",
        "src/sample_pkg/core.py:1-2-3",
    ),
)
def test_parse_citation__rejects_malformed_syntax(raw: str) -> None:
    assert parse_citation(raw) is None


@pytest.mark.parametrize(
    ("raw", "status"),
    (
        ("src/sample_pkg/dates.py:6", CitationStatus.valid),
        ("src/sample_pkg/dates.py:", CitationStatus.unparseable),
        ("src/sample_pkg/missing.py:1", CitationStatus.path_not_found),
        ("../secret.txt:1", CitationStatus.path_escapes),
        ("src/sample_pkg/dates.py:999", CitationStatus.line_out_of_range),
        ("data/logo.bin:1", CitationStatus.binary_file),
    ),
)
def test_validate_citation__covers_statuses(
    mini_repo: Path,
    raw: str,
    status: CitationStatus,
) -> None:
    check = validate_citation(raw, _jail(mini_repo))

    _assert_status(check, status)
    assert check.raw == raw


def test_validate_citation__path_only_binary_file_can_be_cited(
    mini_repo: Path,
    mini_repo_manifest: _MiniRepoManifest,
) -> None:
    check = validate_citation(mini_repo_manifest.binary_file, _jail(mini_repo))

    _assert_status(check, CitationStatus.valid)


@pytest.mark.parametrize(
    "raw",
    (
        "../README.md:1",
        "/etc/passwd:1",
        r"C:\Windows\system32\drivers\etc\hosts:1",
        "C:/Windows/win.ini:1",
    ),
)
def test_validate_citation__path_escape_is_reported_not_raised(
    mini_repo: Path,
    raw: str,
) -> None:
    check = validate_citation(raw, _jail(mini_repo))

    _assert_status(check, CitationStatus.path_escapes)
    assert check.citation is not None
    assert check.citation.path


@pytest.mark.parametrize(
    "raw",
    (
        "data/long_file.txt:1",
        "data/long_file.txt:450",
        "data/long_file.txt:1-450",
        "data/long_file.txt:3-6",
    ),
)
def test_validate_citation__line_ranges_are_one_based_inclusive(
    mini_repo: Path,
    raw: str,
) -> None:
    check = validate_citation(raw, _jail(mini_repo))

    _assert_status(check, CitationStatus.valid)


@pytest.mark.parametrize(
    "raw",
    (
        "data/long_file.txt:0",
        "data/long_file.txt:451",
        "data/long_file.txt:440-451",
        "data/long_file.txt:6-5",
    ),
)
def test_validate_citation__line_ranges_outside_file_are_rejected(
    mini_repo: Path,
    raw: str,
) -> None:
    check = validate_citation(raw, _jail(mini_repo))

    _assert_status(check, CitationStatus.line_out_of_range)


def test_validate_citation__empty_file_has_zero_lines(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "empty.txt").write_text("", encoding="utf-8")

    valid_path_only = validate_citation("empty.txt", _jail(workspace))
    line_one = validate_citation("empty.txt:1", _jail(workspace))

    _assert_status(valid_path_only, CitationStatus.valid)
    _assert_status(line_one, CitationStatus.line_out_of_range)


def test_validate_citation__invalid_utf8_with_line_is_binary_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "latin1.txt").write_bytes(b"caf\xe9\n")

    check = validate_citation("latin1.txt:1", _jail(workspace))

    _assert_status(check, CitationStatus.binary_file)


def test_validate_citations__aggregates_batch(mini_repo: Path) -> None:
    report = validate_citations(
        (
            "src/sample_pkg/dates.py:6",
            "README.md",
            "src/sample_pkg/dates.py:999",
            "missing.py:1",
        ),
        _jail(mini_repo),
    )

    assert isinstance(report, CitationReport)
    assert len(report.checks) == 4
    assert report.valid_count == 2
    assert report.all_valid is False
    assert [check.raw for check in report.invalid] == [
        "src/sample_pkg/dates.py:999",
        "missing.py:1",
    ]


def test_validate_citations__all_valid_for_empty_batch(mini_repo: Path) -> None:
    report = validate_citations((), _jail(mini_repo))

    assert report.valid_count == 0
    assert report.all_valid is True
    assert report.invalid == ()
