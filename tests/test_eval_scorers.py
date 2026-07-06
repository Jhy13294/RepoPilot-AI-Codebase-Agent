import pytest
from pydantic import ValidationError

from app.schemas.agent_io import CitationGrounding
from eval.scorers import (
    LocalizationScore,
    score_bug_explanation,
    score_bug_localization,
)


def _grounding(citation: str, *, grounded: bool = True) -> CitationGrounding:
    return CitationGrounding(
        citation=citation,
        status="valid" if grounded else "line_out_of_range",
        grounded=grounded,
        detail="test grounding",
    )


def test_score_bug_localization__normalizes_paths_and_reports_first_top_k_hit() -> None:
    score = score_bug_localization(
        [
            "./calculator/format.py",
            "C:\\tmp\\buggy-calculator\\calculator\\ops.py:22",
            "calculator/stats.py",
        ],
        "calculator/ops.py",
        top_k=3,
    )

    assert score == LocalizationScore(
        gold_file="calculator/ops.py",
        hit=True,
        rank=2,
        top_k=3,
    )


def test_score_bug_localization__ignores_matches_outside_top_k() -> None:
    score = score_bug_localization(
        [
            "calculator/format.py",
            "calculator/stats.py",
            "README.md",
            "calculator/ops.py",
        ],
        "./calculator/ops.py",
        top_k=3,
    )

    assert score.hit is False
    assert score.rank is None


def test_score_bug_localization__rejects_invalid_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        score_bug_localization([], "calculator/ops.py", top_k=0)


def test_score_models_are_frozen_and_forbid_extra_fields() -> None:
    score = LocalizationScore(gold_file="calculator/ops.py", hit=True, rank=1, top_k=3)

    with pytest.raises(ValidationError, match="Extra inputs"):
        LocalizationScore.model_validate(
            {
                "gold_file": "calculator/ops.py",
                "hit": True,
                "rank": 1,
                "top_k": 3,
                "extra": "nope",
            }
        )

    with pytest.raises(ValidationError, match="frozen"):
        score.hit = False


def test_score_bug_explanation__requires_only_cited_grounding_and_all_rubric_hits() -> None:
    score = score_bug_explanation(
        ["calculator/format.py:8"],
        [
            _grounding("calculator/format.py:8"),
            _grounding("calculator/stats.py", grounded=False),
        ],
        "format_percent should multiply by 100 before rendering the percentage.",
        ["multiply by 100", "FORMAT_PERCENT"],
        require_valid_citation=True,
    )

    assert score.citation_valid is True
    assert score.rubric_hits == ["multiply by 100", "FORMAT_PERCENT"]
    assert score.all_rubric is True
    assert score.passed is True


def test_score_bug_explanation__fails_required_invalid_citation() -> None:
    score = score_bug_explanation(
        ["calculator/format.py:999"],
        [_grounding("calculator/format.py:999", grounded=False)],
        "format_percent should multiply by 100.",
        ["multiply by 100"],
        require_valid_citation=True,
    )

    assert score.citation_valid is False
    assert score.all_rubric is True
    assert score.passed is False


def test_score_bug_explanation__can_pass_without_required_citation() -> None:
    score = score_bug_explanation(
        [],
        [],
        "format_percent should multiply by 100.",
        ["multiply by 100"],
        require_valid_citation=False,
    )

    assert score.citation_valid is False
    assert score.all_rubric is True
    assert score.passed is True
