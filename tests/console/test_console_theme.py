from datetime import UTC, datetime

import pytest

from app.api.schemas import ApprovalRequestView, TraceEventView
from app.console.theme import (
    CONSOLE_CSS,
    ActLabel,
    approval_header,
    archive_heading,
    decision_record,
    event_act,
    numbered_entry,
    report_header,
    section_label,
)

_NOW = datetime(2026, 7, 20, 8, 0, tzinfo=UTC)


def _event(
    kind: str,
    *,
    tool_name: str | None = None,
    decision: str | None = None,
    summary: str | None = None,
) -> TraceEventView:
    return TraceEventView(
        seq=7,
        ts=_NOW,
        kind=kind,
        summary=summary,
        tool_name=tool_name,
        decision=decision,
    )


def _approval() -> ApprovalRequestView:
    return ApprovalRequestView(
        request_id='request<script>&"',
        run_id="run-main",
        tool_name='<img onerror="alert(1)">',
        risk_level="high",
        args={"diff": '<script>alert("diff")</script>&'},
        status="approved",
        actor='<img onerror="actor">',
        note='<script>alert("note")</script>&',
        created_at=_NOW,
        decided_at=_NOW,
    )


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("plan", ActLabel.PLAN),
        ("replan", ActLabel.REVISION),
        ("tool_result", ActLabel.STEP_RESULT),
        ("approval_request", ActLabel.DECISION_REQUESTED),
        ("critic_verdict", ActLabel.VERIFICATION),
        ("report", ActLabel.REPORT),
        ("error", ActLabel.FAULT),
    ],
)
def test_event_act__maps_non_branching_trace_kinds(kind: str, expected: ActLabel) -> None:
    assert event_act(_event(kind)) is expected


@pytest.mark.parametrize("tool_name", ["read_file", "search_code", "get_file_tree"])
def test_event_act__read_only_tools_are_observations(tool_name: str) -> None:
    assert event_act(_event("tool_call", tool_name=tool_name)) is ActLabel.OBSERVATION


@pytest.mark.parametrize("tool_name", ["apply_patch", "git_create_branch", "run_tests"])
def test_event_act__mutating_tools_are_interventions(tool_name: str) -> None:
    assert event_act(_event("tool_call", tool_name=tool_name)) is ActLabel.INTERVENTION


def test_event_act__propose_patch_is_a_proposal_not_intervention() -> None:
    # propose_patch drafts a diff without writing; it must not read as INTERVENTION.
    assert event_act(_event("tool_call", tool_name="propose_patch")) is ActLabel.PROPOSAL


@pytest.mark.parametrize("tool_name", ["mystery_tool", None])
def test_event_act__unclassified_tool_stays_intervention(tool_name: str | None) -> None:
    assert event_act(_event("tool_call", tool_name=tool_name)) is ActLabel.INTERVENTION


@pytest.mark.parametrize(
    ("decision", "expected"),
    [
        ("approved", ActLabel.HUMAN_DECISION_APPROVED),
        ("denied", ActLabel.HUMAN_DECISION_DENIED),
    ],
)
def test_event_act__human_decision_includes_outcome(
    decision: str,
    expected: ActLabel,
) -> None:
    assert event_act(_event("approval_decision", decision=decision)) is expected


def test_numbered_entry__keeps_raw_metadata_beneath_editorial_act() -> None:
    event = TraceEventView(
        seq=7,
        ts=_NOW,
        kind="tool_call",
        tool_name="read_file",
        ok=False,
        error_type="read_failed",
        latency_ms=17,
    )

    rendered = numbered_entry(event)

    assert "OBSERVATION" in rendered
    assert "KIND / tool_call" in rendered
    assert "TOOL / read_file" in rendered
    assert "OK / false" in rendered
    assert "ERROR / read_failed" in rendered
    assert "LATENCY / 17 ms" in rendered
    assert _NOW.isoformat() in rendered


def test_archive_helpers__derive_numbering_and_reference_from_real_values() -> None:
    heading = archive_heading("1234567890abcdef", "DONE")
    label = section_label("Execution record")

    assert "RUN / 1234567890ab" in heading
    assert "1234567890abcdef" not in heading
    assert "01" in label
    assert '<span class="rp-section-separator">—</span>' in label
    assert "EXECUTION RECORD" in label


def test_html_helpers__escape_adversarial_prose_and_metadata() -> None:
    request = _approval()
    event = _event(
        '<script>alert("kind")</script>',
        tool_name='<img onerror="tool">',
        summary='<img onerror="summary"> & "quoted"',
    )
    rendered = "".join(
        (
            archive_heading('<script>&"run-reference', '<img onerror="status">'),
            section_label('<script>&"section'),
            numbered_entry(event),
            approval_header(request, '<script>&"rationale'),
            decision_record(request),
            report_header('<img onerror="run-reference">', '<script>&"status'),
        )
    )

    assert "<script>" not in rendered
    assert "<img onerror=" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "&lt;img onerror=&quot;" in rendered
    assert "&amp;" in rendered
    assert "&quot;" in rendered


def test_console_css__stays_offline_static_and_cache_class_free() -> None:
    assert "st-emotion-cache" not in CONSOLE_CSS
    assert "@import" not in CONSOLE_CSS
    assert "@keyframes" not in CONSOLE_CSS
