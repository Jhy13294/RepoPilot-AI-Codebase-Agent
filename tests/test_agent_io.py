import pytest
from pydantic import ValidationError

from app.agent.state import RunStatus
from app.schemas.agent_io import AnalysisReport, ReportConfidence, RunResult, SuspectFile
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


def test_suspect_file__rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        SuspectFile.model_validate(
            {
                "path": "src/sample_pkg/dates.py",
                "reason": "The trace points at this parser.",
                "confidence": 0.9,
            }
        )
