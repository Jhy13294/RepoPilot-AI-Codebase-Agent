import pytest
from pydantic import ValidationError

from app.schemas.agent_io import CitationGrounding
from eval.scorers import (
    LocalizationScore,
    PatchScore,
    RepoQaScore,
    score_bug_explanation,
    score_bug_localization,
    score_repo_qa,
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
    patch_score = PatchScore(tests_green=True, returncode=0)

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

    with pytest.raises(ValidationError, match="Extra inputs"):
        PatchScore.model_validate(
            {"tests_green": True, "returncode": 0, "agent_claimed_green": True}
        )

    with pytest.raises(ValidationError, match="frozen"):
        patch_score.tests_green = False


def test_repo_qa_score_is_frozen_and_forbids_extra_fields() -> None:
    score = RepoQaScore(
        path_hit=True,
        rubric_hits=["validate_payload"],
        all_rubric=True,
        passed=True,
    )

    with pytest.raises(ValidationError, match="Extra inputs"):
        RepoQaScore.model_validate(
            {
                "path_hit": True,
                "rubric_hits": ["validate_payload"],
                "all_rubric": True,
                "passed": True,
                "extra": "nope",
            }
        )

    with pytest.raises(ValidationError, match="frozen"):
        score.passed = False


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


def test_score_repo_qa__hits_any_normalized_candidate_path() -> None:
    score = score_repo_qa(
        [
            "C:\\tmp\\mini-flask-api\\app\\validation.py:12",
            "./app/routes.py",
        ],
        "No rubric is required.",
        ["app/store.py", "app/validation.py"],
        [],
    )

    assert score.path_hit is True
    assert score.all_rubric is True
    assert score.passed is True


def test_score_repo_qa__misses_unrelated_candidate_paths() -> None:
    score = score_repo_qa(
        ["app/routes.py", "README.md"],
        "The store calls insert and commit.",
        ["app/store.py", "app/validation.py"],
        ["insert", "commit"],
    )

    assert score.path_hit is False
    assert score.all_rubric is True
    assert score.passed is False


def test_score_repo_qa__matches_all_rubric_keywords_case_insensitively() -> None:
    score = score_repo_qa(
        ["app/validation.py"],
        "VALIDATE_PAYLOAD delegates date parsing to Parse_Date.",
        ["app/validation.py"],
        ["validate_payload", "parse_date", "  "],
    )

    assert score.rubric_hits == ["validate_payload", "parse_date"]
    assert score.all_rubric is True
    assert score.passed is True


def test_score_repo_qa__reports_partial_rubric_coverage() -> None:
    score = score_repo_qa(
        ["app/validation.py"],
        "validate_payload rejects malformed input.",
        ["app/validation.py"],
        ["validate_payload", "ValueError"],
    )

    assert score.rubric_hits == ["validate_payload"]
    assert score.all_rubric is False
    assert score.passed is False


@pytest.mark.parametrize(
    ("candidate_paths", "analysis_text", "expected_path_hit", "expected_all_rubric", "passed"),
    [
        (["app/routes.py"], "insert and commit", True, True, True),
        (["README.md"], "insert and commit", False, True, False),
        (["app/routes.py"], "insert only", True, False, False),
        (["README.md"], "insert only", False, False, False),
    ],
)
def test_score_repo_qa__passed_requires_path_and_all_rubric(
    candidate_paths: list[str],
    analysis_text: str,
    expected_path_hit: bool,
    expected_all_rubric: bool,
    passed: bool,
) -> None:
    score = score_repo_qa(
        candidate_paths,
        analysis_text,
        ["app/routes.py"],
        ["insert", "commit"],
    )

    assert score.path_hit is expected_path_hit
    assert score.all_rubric is expected_all_rubric
    assert score.passed is passed
