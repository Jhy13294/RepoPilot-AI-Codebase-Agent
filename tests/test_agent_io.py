import pytest
from pydantic import ValidationError

from app.agent.state import RunStatus
from app.schemas.agent_io import (
    AnalysisReport,
    CitationGrounding,
    GroundingReport,
    ReportConfidence,
    RunResult,
    SuspectFile,
)
from app.schemas.llm_io import Usage


def test_analysis_report__suspects_and_citations_round_trip() -> None:
    report = AnalysisReport(
        headline="parse_date failure localized",
        analysis="The failing parser branch is in the date module.",
        confidence=ReportConfidence.high,
        open_questions=["Should timezone formats be accepted?"],
        citations=["src/sample_pkg/dates.py:6", "src/sample_pkg/dates.py:10-12"],
        suspects=[
            SuspectFile(
                path="src/sample_pkg/dates.py",
                reason="The trace shows parse_date rejects the reported input there.",
            )
        ],
        usage=Usage(tokens_in=5, tokens_out=7, cost_usd=0.03),
    )

    round_tripped = AnalysisReport.model_validate_json(report.model_dump_json())

    assert round_tripped == report
    assert round_tripped.suspects[0].path == "src/sample_pkg/dates.py"
    assert round_tripped.citations == ["src/sample_pkg/dates.py:6", "src/sample_pkg/dates.py:10-12"]


def test_agent_io__defaults_keep_existing_callers_additive() -> None:
    report = AnalysisReport(
        headline="Question answered",
        analysis="The trace answered the repository question.",
        confidence=ReportConfidence.medium,
        usage=Usage(tokens_in=1, tokens_out=2, cost_usd=None),
    )
    result = RunResult(
        run_id="run-1",
        status=RunStatus.DONE,
        summary="Question answered",
        steps_used=0,
        replans_used=0,
        fix_cycles_used=0,
        usage=Usage(tokens_in=1, tokens_out=2, cost_usd=None),
    )

    assert report.suspects == []
    assert report.citations == []
    assert result.report is None
    assert result.grounding is None


def test_grounding_report__properties_and_round_trip() -> None:
    grounded = CitationGrounding(
        citation="src/sample_pkg/dates.py:6",
        status="valid",
        grounded=True,
        detail="Line range 6-6 exists in 'src/sample_pkg/dates.py'.",
    )
    ungrounded = CitationGrounding(
        citation="src/sample_pkg/missing.py:1",
        status="path_not_found",
        grounded=False,
        detail="Path is not a file inside the workspace.",
    )
    report = GroundingReport(checks=[grounded, ungrounded])

    round_tripped = GroundingReport.model_validate_json(report.model_dump_json())

    assert round_tripped == report
    assert report.grounded_count == 1
    assert report.all_grounded is False
    assert report.ungrounded == (ungrounded,)
    assert GroundingReport().grounded_count == 0
    assert GroundingReport().all_grounded is True
    assert GroundingReport().ungrounded == ()


def test_suspect_file__rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        SuspectFile.model_validate(
            {
                "path": "src/sample_pkg/dates.py",
                "reason": "The trace points at this parser.",
                "confidence": 0.9,
            }
        )


def test_citation_grounding__rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        CitationGrounding.model_validate(
            {
                "citation": "src/sample_pkg/dates.py:6",
                "status": "valid",
                "grounded": True,
                "detail": "Line exists.",
                "source": "model",
            }
        )
