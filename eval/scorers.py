"""Scorers for issue-analysis, patch, and recovery evaluation tasks."""

import re
import shlex
import subprocess
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.agent_io import CitationGrounding

_CITATION_SUFFIX_RE = re.compile(r":\d+(?:-\d+)?\Z")


class LocalizationScore(BaseModel):
    """Top-k bug localization score for one issue task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gold_file: str
    hit: bool
    rank: int | None = Field(default=None, ge=1)
    top_k: int = Field(ge=1)


class ExplanationScore(BaseModel):
    """Citation and rubric score for one bug explanation task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    citation_valid: bool
    rubric_hits: list[str]
    all_rubric: bool
    passed: bool


class RepoQaScore(BaseModel):
    """Path and rubric score for one repository question task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path_hit: bool
    rubric_hits: list[str]
    all_rubric: bool
    passed: bool


class PatchScore(BaseModel):
    """Independent final-workspace test outcome for one patch task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tests_green: bool
    returncode: int


class RecoveryScore(BaseModel):
    """Independent injected-failure, loop, and final-test recovery outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    injected_error_seen: bool
    loop_done: bool
    tests_green: bool
    returncode: int
    recovered: bool


def score_bug_localization(
    suspect_paths: Sequence[str],
    gold_file: str,
    *,
    top_k: int = 3,
) -> LocalizationScore:
    """Score whether the gold file appears in the first top_k suspect paths."""
    if top_k < 1:
        raise ValueError("top_k must be at least one.")

    normalized_gold = _normalize_path(gold_file)
    rank: int | None = None
    for index, suspect_path in enumerate(suspect_paths[:top_k], start=1):
        if _same_repo_path(_normalize_path(suspect_path), normalized_gold):
            rank = index
            break

    return LocalizationScore(
        gold_file=normalized_gold,
        hit=rank is not None,
        rank=rank,
        top_k=top_k,
    )


def score_bug_explanation(
    citations: Sequence[str],
    grounding_checks: Sequence[CitationGrounding],
    analysis_text: str,
    rubric_keywords: Sequence[str],
    *,
    require_valid_citation: bool,
) -> ExplanationScore:
    """Score grounded citations plus case-insensitive rubric keyword coverage."""
    citation_valid = _citations_are_grounded(citations, grounding_checks)
    rubric_hits = _rubric_hits(analysis_text, rubric_keywords)
    normalized_keywords = [keyword for keyword in rubric_keywords if keyword.strip()]
    all_rubric = len(rubric_hits) == len(normalized_keywords)
    passed = all_rubric and (citation_valid or not require_valid_citation)
    return ExplanationScore(
        citation_valid=citation_valid,
        rubric_hits=rubric_hits,
        all_rubric=all_rubric,
        passed=passed,
    )


def score_repo_qa(
    candidate_paths: Sequence[str],
    analysis_text: str,
    paths_any: Sequence[str],
    rubric_keywords: Sequence[str],
) -> RepoQaScore:
    """Score repository path evidence and case-insensitive rubric coverage."""
    normalized_candidates = [_normalize_path(path) for path in candidate_paths]
    normalized_expected = [_normalize_path(path) for path in paths_any]
    path_hit = any(
        _same_repo_path(candidate, expected)
        for candidate in normalized_candidates
        for expected in normalized_expected
    )
    rubric_hits = _rubric_hits(analysis_text, rubric_keywords)
    nonempty_keywords = [keyword for keyword in rubric_keywords if keyword.strip()]
    all_rubric = len(rubric_hits) == len(nonempty_keywords)
    return RepoQaScore(
        path_hit=path_hit,
        rubric_hits=rubric_hits,
        all_rubric=all_rubric,
        passed=path_hit and all_rubric,
    )


def score_patch(workspace: Path, test_command: str) -> PatchScore:
    """Rerun the configured tests and treat their exit code as patch ground truth."""
    argv = shlex.split(test_command)
    if not argv:
        raise ValueError("test_command must not be empty.")

    completed = subprocess.run(
        argv,
        cwd=workspace,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        shell=False,
    )
    return PatchScore(
        tests_green=completed.returncode == 0,
        returncode=completed.returncode,
    )


def score_recovery(
    observed_error_types: Sequence[str],
    loop_status: str,
    workspace: Path,
    test_command: str,
    recovered_from: str,
) -> RecoveryScore:
    """Score recovery from trace primitives and an independent final test run."""
    patch_score = score_patch(workspace, test_command)
    injected_error_seen = recovered_from in observed_error_types
    loop_done = loop_status == "DONE"
    return RecoveryScore(
        injected_error_seen=injected_error_seen,
        loop_done=loop_done,
        tests_green=patch_score.tests_green,
        returncode=patch_score.returncode,
        recovered=injected_error_seen and loop_done and patch_score.tests_green,
    )


def _normalize_path(raw_path: str) -> str:
    candidate = _CITATION_SUFFIX_RE.sub("", raw_path.strip().strip("\"'"))
    parts = [part for part in candidate.replace("\\", "/").split("/") if part not in {"", "."}]
    return "/".join(parts)


def _same_repo_path(candidate: str, gold_file: str) -> bool:
    return candidate == gold_file or candidate.endswith(f"/{gold_file}")


def _citations_are_grounded(
    citations: Sequence[str],
    grounding_checks: Sequence[CitationGrounding],
) -> bool:
    if not citations:
        return False

    for citation in citations:
        matching_checks = [check for check in grounding_checks if check.citation == citation]
        if not matching_checks or not all(check.grounded for check in matching_checks):
            return False

    return True


def _rubric_hits(analysis_text: str, rubric_keywords: Sequence[str]) -> list[str]:
    haystack = analysis_text.casefold()
    return [
        keyword for keyword in rubric_keywords if keyword.strip() and keyword.casefold() in haystack
    ]
