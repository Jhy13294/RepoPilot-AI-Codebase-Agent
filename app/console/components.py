"""Network-free Streamlit renderers for the run archive console."""

import json
from collections.abc import Callable
from dataclasses import dataclass

import streamlit as st

from app.api.schemas import ApprovalRequestView, RunSummaryView
from app.console.client import ApprovalDecision
from app.console.state import ConsoleError, ConsolePhase, ConsoleState
from app.console.theme import (
    approval_header,
    archive_heading,
    decision_record,
    empty_note,
    error_banner,
    metadata_row,
    numbered_entry,
    section_label,
    status_badge,
)


@dataclass(frozen=True, slots=True)
class ArchiveAction:
    """A local archive-index selection with no network side effect."""

    run_id: str | None


type DecisionCallback = Callable[[str, ApprovalDecision, str | None], None]


def render_header(state: ConsoleState) -> None:
    """Render the archive masthead."""
    st.markdown(
        archive_heading(state.active_run_id, state.status),
        unsafe_allow_html=True,
    )


def render_error(error: ConsoleError | None) -> None:
    """Render one page-level error block when present."""
    if error is None:
        return
    st.markdown(
        error_banner(error.code.value, error.message),
        unsafe_allow_html=True,
    )


def render_approval_panel(
    state: ConsoleState,
    queue_decision: DecisionCallback,
) -> None:
    """Render pending approvals as the first full-width focal section."""
    if not state.pending:
        return
    st.markdown(section_label("Decision desk"), unsafe_allow_html=True)
    for request in state.pending:
        rationale = _approval_rationale(request)
        st.markdown(
            approval_header(request, rationale),
            unsafe_allow_html=True,
        )
        diff_value = request.args.get("diff")
        if diff_value is not None:
            diff_text = (
                diff_value
                if isinstance(diff_value, str)
                else json.dumps(diff_value, indent=2, ensure_ascii=False)
            )
            st.code(diff_text, language="diff", wrap_lines=False)
        note = str(
            st.text_input(
                "Decision note (optional)",
                key=f"approval_note_{request.request_id}",
            )
        )
        approve_column, reject_column = st.columns(2)
        with approve_column:
            st.button(
                "Approve",
                key=f"approve_{request.request_id}",
                type="primary",
                use_container_width=True,
                on_click=queue_decision,
                args=(request.request_id, "approve", note),
            )
        with reject_column:
            st.button(
                "Reject",
                key=f"reject_{request.request_id}",
                use_container_width=True,
                on_click=queue_decision,
                args=(request.request_id, "deny", note),
            )


def render_active_archive(state: ConsoleState) -> ArchiveAction | None:
    """Render the three-column run archive and return a local index action."""
    index_column, timeline_column, metadata_column = st.columns(
        (0.85, 1.8, 1.0),
        gap="large",
    )
    with index_column:
        action = render_archive_index(state)
    with timeline_column:
        render_timeline(state)
        if state.phase is ConsolePhase.TERMINAL:
            render_report_view(state)
    with metadata_column:
        render_metadata(state)
    return action


def render_archive_index(state: ConsoleState) -> ArchiveAction | None:
    """Render persisted runs without fetching or mutating state."""
    st.markdown(section_label("Archive index"), unsafe_allow_html=True)
    if state.active_run_id is not None and bool(
        st.button("New run", key="new_run", use_container_width=True)
    ):
        return ArchiveAction(run_id=None)
    if not state.history:
        st.markdown(
            empty_note("No persisted runs are indexed yet."),
            unsafe_allow_html=True,
        )
        return None
    for summary in state.history:
        label = f"{summary.status.value} · {summary.run_id}"
        selected = bool(
            st.button(
                label,
                key=f"select_{summary.run_id}",
                disabled=summary.run_id == state.active_run_id,
                use_container_width=True,
            )
        )
        if selected:
            return ArchiveAction(run_id=summary.run_id)
    return None


def render_timeline(state: ConsoleState) -> None:
    """Render the complete deduplicated session timeline."""
    st.markdown(section_label("Execution record"), unsafe_allow_html=True)
    if not state.events:
        st.markdown(
            empty_note("No events have been recorded for this run yet."),
            unsafe_allow_html=True,
        )
        return
    for event in state.events:
        st.markdown(numbered_entry(event), unsafe_allow_html=True)


def render_metadata(state: ConsoleState) -> None:
    """Render run identity, lifecycle, counters, and decision records."""
    st.markdown(section_label("Metadata"), unsafe_allow_html=True)
    st.markdown(status_badge(state.status), unsafe_allow_html=True)
    summary = _active_summary(state)
    detail = state.run_detail
    rows: tuple[tuple[str, object], ...] = (
        ("Run ID", state.active_run_id),
        ("Task type", detail.task_type if detail is not None else state.task_type),
        ("Repository", detail.repo if detail is not None else state.repo),
        (
            "Step count",
            detail.step_count
            if detail is not None
            else summary.step_count
            if summary is not None
            else None,
        ),
        (
            "Steps used",
            detail.steps_used
            if detail is not None
            else summary.steps_used
            if summary is not None
            else None,
        ),
        (
            "Replans used",
            detail.replans_used
            if detail is not None
            else summary.replans_used
            if summary is not None
            else None,
        ),
        (
            "Fix cycles",
            detail.fix_cycles_used
            if detail is not None
            else summary.fix_cycles_used
            if summary is not None
            else None,
        ),
        (
            "Updated",
            detail.updated_at
            if detail is not None
            else summary.updated_at
            if summary is not None
            else None,
        ),
    )
    for key, value in rows:
        st.markdown(metadata_row(key, value), unsafe_allow_html=True)
    if state.prompt:
        st.caption(state.prompt)
    if state.decision_records:
        st.markdown(section_label("Decision record"), unsafe_allow_html=True)
        for request in state.decision_records:
            st.markdown(decision_record(request), unsafe_allow_html=True)


def render_report_view(state: ConsoleState) -> None:
    """Render the terminal report or the explicit no-report state."""
    st.markdown(section_label("Terminal report"), unsafe_allow_html=True)
    if state.run_detail is None or state.run_detail.summary is None:
        status = state.status or "terminal"
        st.markdown(
            empty_note(f"Run is {status}; no report is available."),
            unsafe_allow_html=True,
        )
        return
    st.markdown(state.run_detail.summary)


def _approval_rationale(request: ApprovalRequestView) -> str:
    for key in ("rationale", "reason"):
        value = request.args.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return "No rationale was supplied."


def _active_summary(state: ConsoleState) -> RunSummaryView | None:
    return next(
        (item for item in state.history if item.run_id == state.active_run_id),
        None,
    )
