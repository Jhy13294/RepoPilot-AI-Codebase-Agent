import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

collect_ignore = ["fixtures"]


@dataclass(frozen=True, slots=True)
class MiniRepoManifest:
    files: frozenset[str]
    dirs: frozenset[str]
    long_file_lines: int
    binary_file: str
    parse_date_file: str
    parse_date_line: int
    parse_date_mention_files: frozenset[str]
    unicode_file: str
    unicode_probe: str
    hidden_entries: frozenset[str]


MINI_REPO_MANIFEST = MiniRepoManifest(
    files=frozenset(
        {
            ".cache/pip.log",
            ".hidden.cfg",
            "README.md",
            "data/logo.bin",
            "data/long_file.txt",
            "docs/usage.md",
            "pyproject.toml",
            "src/sample_pkg/__init__.py",
            "src/sample_pkg/core.py",
            "src/sample_pkg/dates.py",
            "src/sample_pkg/i18n/messages.txt",
            "tests/test_dates.py",
        }
    ),
    dirs=frozenset(
        {
            ".cache",
            "data",
            "docs",
            "src",
            "src/sample_pkg",
            "src/sample_pkg/i18n",
            "tests",
        }
    ),
    long_file_lines=450,
    binary_file="data/logo.bin",
    parse_date_file="src/sample_pkg/dates.py",
    parse_date_line=6,
    parse_date_mention_files=frozenset(
        {
            "docs/usage.md",
            "src/sample_pkg/core.py",
            "src/sample_pkg/dates.py",
        }
    ),
    unicode_file="docs/usage.md",
    unicode_probe="中文探针: RepoPilot 可以读取 UTF-8 文档。",
    hidden_entries=frozenset({".cache", ".cache/pip.log", ".hidden.cfg"}),
)


@pytest.fixture
def mini_repo(tmp_path: Path) -> Path:
    source = Path(__file__).parent / "fixtures" / "mini_repo"
    return shutil.copytree(source, tmp_path / "mini_repo")


@pytest.fixture
def mini_repo_manifest() -> MiniRepoManifest:
    return MINI_REPO_MANIFEST
