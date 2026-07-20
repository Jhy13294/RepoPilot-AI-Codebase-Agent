"""Editorial archive theme and HTML helpers for the Streamlit console."""

from enum import StrEnum
from html import escape

import streamlit as st

from app.api.schemas import ApprovalRequestView, TraceEventView

CONSOLE_CSS = """
.rp-archive-header {
  background: #f8f3ea;
  border-bottom: 1px solid #a89d8d;
  border-top: 5px solid #7d3126;
  box-shadow: inset 0 -4px 0 #eee5d7;
  margin: 0 0 1.45rem 0;
  padding: 1.15rem 1.2rem 1.05rem 1.2rem;
}
.rp-archive-grid {
  align-items: end;
  display: grid;
  gap: 1rem 1.4rem;
  grid-template-columns: minmax(16rem, 1.8fr) minmax(9rem, 0.7fr) auto;
}
.rp-archive-kicker,
.rp-section-label,
.rp-entry-number,
.rp-entry-kind,
.rp-entry-act,
.rp-meta-key,
.rp-status,
.rp-error-code,
.rp-approval-risk,
.rp-approval-docket,
.rp-archive-reference-label,
.rp-archive-reference-value,
.rp-report-kicker,
.rp-report-meta {
  font-family: "IBM Plex Mono", "Cascadia Mono", Consolas, monospace;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.rp-archive-kicker {
  color: #7d3126;
  font-size: 0.68rem;
  font-weight: 700;
  margin-bottom: 0.42rem;
}
.rp-archive-title {
  color: #211e1a;
  font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Palatino, Georgia, serif;
  font-size: clamp(2.25rem, 4.6vw, 4.15rem);
  font-weight: 500;
  letter-spacing: -0.045em;
  line-height: 0.92;
}
.rp-archive-reference {
  border-left: 1px solid #c9beae;
  min-width: 0;
  padding-left: 1rem;
}
.rp-archive-reference-label {
  color: #756d62;
  font-size: 0.56rem;
  margin-bottom: 0.28rem;
}
.rp-archive-reference-value {
  color: #2d2924;
  font-size: 0.72rem;
  overflow-wrap: anywhere;
}
.rp-archive-status {
  justify-self: end;
}
.rp-section-label {
  align-items: center;
  color: #4f4941;
  display: grid;
  font-size: 0.62rem;
  font-weight: 700;
  gap: 0.55rem;
  grid-template-columns: auto auto minmax(1.5rem, 1fr);
  margin: 0.35rem 0 0.9rem 0;
}
.rp-section-number { color: #7d3126; }
.rp-section-rule { border-top: 1px solid #b8ad9b; }
.rp-section-title { white-space: nowrap; }
.rp-section-separator { color: #9f9484; }
.rp-status {
  border: 1px solid currentColor;
  display: inline-block;
  font-size: 0.61rem;
  font-weight: 750;
  line-height: 1;
  padding: 0.38rem 0.48rem 0.34rem 0.48rem;
}
.rp-status--progress { color: #315c6d; background: #e3ecec; }
.rp-status--decision { color: #8a5a16; background: #f2e8cf; }
.rp-status--complete { color: #356043; background: #e1ebe2; }
.rp-status--failed { color: #8a3b2e; background: #f1dfda; }
.rp-status--cancelled { color: #625b51; background: #e7e1d7; }
.rp-timeline-entry {
  border-bottom: 1px solid #cec3b3;
  display: grid;
  gap: 0.24rem 0.9rem;
  grid-template-columns: 3.35rem minmax(8.8rem, 0.72fr) minmax(0, 1.5fr);
  padding: 0.88rem 0;
}
.rp-entry-number {
  color: #7d3126;
  font-size: 0.78rem;
  grid-row: 1 / span 2;
  padding-top: 0.15rem;
}
.rp-entry-number::before {
  color: #9b9183;
  content: "SEQ ";
  display: block;
  font-size: 0.48rem;
  letter-spacing: 0.12em;
  margin-bottom: 0.08rem;
}
.rp-entry-act {
  color: #211e1a;
  font-size: 0.82rem;
  font-weight: 780;
  grid-column: 2;
  line-height: 1.15;
}
.rp-entry-act::before {
  color: #7d3126;
  content: "ACT / ";
  font-size: 0.53rem;
}
.rp-entry-kind {
  color: #756d62;
  font-size: 0.58rem;
  grid-column: 2 / 4;
  grid-row: 2;
  line-height: 1.45;
  overflow-wrap: anywhere;
}
.rp-entry-summary {
  align-self: start;
  color: #2b2722;
  font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Palatino, Georgia, serif;
  font-size: 0.98rem;
  grid-column: 3;
  grid-row: 1;
  line-height: 1.45;
  overflow-wrap: anywhere;
}
.rp-empty-note {
  border: 1px dashed #b8ad9b;
  color: #5f584e;
  font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Palatino, Georgia, serif;
  font-size: 0.9rem;
  padding: 0.9rem 1rem;
}
.rp-meta-row {
  border-bottom: 1px solid #d2c8b8;
  padding: 0.5rem 0;
}
.rp-meta-key {
  color: #756d62;
  font-size: 0.62rem;
}
.rp-meta-value {
  color: #24211d;
  font-family: "IBM Plex Mono", "Cascadia Mono", Consolas, monospace;
  font-size: 0.75rem;
  overflow-wrap: anywhere;
}
.rp-error-banner {
  background: #f1dfda;
  border-left: 4px solid #8a3b2e;
  color: #43251f;
  margin: 0.6rem 0 1rem 0;
  padding: 0.75rem 0.9rem;
}
.rp-error-code {
  font-size: 0.68rem;
  font-weight: 700;
  margin-bottom: 0.2rem;
}
.rp-error-message { font-size: 0.88rem; }
.rp-approval-focus {
  background: #f7f1e7;
  border: 1px solid #9f9483;
  box-shadow: inset 0 0 0 4px #eee5d7;
  margin: 0.5rem 0 0.9rem 0;
  padding: 1rem 1.1rem 1.05rem 1.1rem;
}
.rp-approval-topline {
  align-items: start;
  border-bottom: 1px solid #c8bdad;
  display: flex;
  gap: 1rem;
  justify-content: space-between;
  margin-bottom: 0.8rem;
  padding-bottom: 0.65rem;
}
.rp-approval-docket {
  color: #655e54;
  font-size: 0.58rem;
  overflow-wrap: anywhere;
}
.rp-approval-title {
  color: #211e1a;
  font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Palatino, Georgia, serif;
  font-size: 1.45rem;
  font-weight: 600;
  letter-spacing: -0.02em;
  line-height: 1.05;
}
.rp-approval-risk {
  color: #8a3b2e;
  border: 2px solid currentColor;
  font-size: 0.58rem;
  font-weight: 800;
  line-height: 1;
  padding: 0.36rem 0.42rem 0.31rem 0.42rem;
  white-space: nowrap;
}
.rp-approval-tool {
  color: #625b51;
  font-family: "IBM Plex Mono", "Cascadia Mono", Consolas, monospace;
  font-size: 0.68rem;
  margin-top: 0.38rem;
}
.rp-approval-rationale {
  color: #3d3832;
  font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Palatino, Georgia, serif;
  font-size: 0.96rem;
  line-height: 1.45;
  margin-top: 0.7rem;
}
.rp-decision-record {
  border-left: 2px solid #8a3b2e;
  color: #504a42;
  font-family: "IBM Plex Mono", "Cascadia Mono", Consolas, monospace;
  font-size: 0.7rem;
  margin: 0.45rem 0;
  padding-left: 0.55rem;
}
.rp-report-document {
  background: #fbf8f1;
  border-bottom: 3px double #9f9483;
  padding: 0.45rem 0 0.9rem 0;
}
.rp-report-kicker {
  color: #7d3126;
  font-size: 0.58rem;
  font-weight: 750;
  margin-bottom: 0.55rem;
}
.rp-report-title {
  color: #211e1a;
  font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Palatino, Georgia, serif;
  font-size: clamp(1.7rem, 3vw, 2.55rem);
  font-weight: 500;
  letter-spacing: -0.035em;
  line-height: 1;
  margin-bottom: 0.65rem;
}
.rp-report-meta {
  color: #6b6359;
  font-size: 0.55rem;
  overflow-wrap: anywhere;
}
@media (max-width: 900px) {
  .rp-archive-grid { grid-template-columns: 1fr; }
  .rp-archive-reference { border-left: 0; border-top: 1px solid #c9beae; padding: 0.7rem 0 0; }
  .rp-archive-status { justify-self: start; }
  .rp-timeline-entry { grid-template-columns: 3rem minmax(0, 1fr); }
  .rp-entry-number { grid-row: 1 / span 3; }
  .rp-entry-kind { grid-column: 2; grid-row: 2; }
  .rp-entry-summary { grid-column: 2; grid-row: 3; }
}
"""


class ActLabel(StrEnum):
    """Editorial action labels derived only from public trace metadata."""

    PLAN = "PLAN"
    REVISION = "REVISION"
    OBSERVATION = "OBSERVATION"
    PROPOSAL = "PROPOSAL"
    INTERVENTION = "INTERVENTION"
    STEP_RESULT = "STEP RESULT"
    DECISION_REQUESTED = "DECISION REQUESTED"
    HUMAN_DECISION = "HUMAN DECISION"
    HUMAN_DECISION_APPROVED = "HUMAN DECISION · APPROVED"
    HUMAN_DECISION_DENIED = "HUMAN DECISION · DENIED"
    VERIFICATION = "VERIFICATION"
    REPORT = "REPORT"
    FAULT = "FAULT"
    ARCHIVE_EVENT = "ARCHIVE EVENT"


_OBSERVATION_TOOLS = frozenset({"read_file", "search_code", "get_file_tree"})
_PROPOSAL_TOOLS = frozenset({"propose_patch"})
_INTERVENTION_TOOLS = frozenset({"apply_patch", "git_create_branch", "run_tests"})
_TOOL_ACTS: dict[str, ActLabel] = {
    **{name: ActLabel.OBSERVATION for name in _OBSERVATION_TOOLS},
    **{name: ActLabel.PROPOSAL for name in _PROPOSAL_TOOLS},
    **{name: ActLabel.INTERVENTION for name in _INTERVENTION_TOOLS},
}
_EVENT_ACTS: dict[str, ActLabel] = {
    "plan": ActLabel.PLAN,
    "replan": ActLabel.REVISION,
    "tool_result": ActLabel.STEP_RESULT,
    "approval_request": ActLabel.DECISION_REQUESTED,
    "critic_verdict": ActLabel.VERIFICATION,
    "report": ActLabel.REPORT,
    "error": ActLabel.FAULT,
}
_SECTION_NUMBERS = {
    "execution record": "01",
    "archive index": "02",
    "metadata": "03",
    "decision desk": "04",
    "decision record": "05",
    "terminal report": "06",
}

_STATUS_PRESENTATION: dict[str, tuple[str, str, str]] = {
    "PLANNING": ("progress", "▶", "In progress"),
    "EXECUTING": ("progress", "▶", "In progress"),
    "VERIFYING": ("progress", "▶", "In progress"),
    "REPLANNING": ("progress", "▶", "In progress"),
    "REPORTING": ("progress", "▶", "In progress"),
    "AWAITING_APPROVAL": ("decision", "◆", "Decision required"),
    "DONE": ("complete", "✓", "Complete"),
    "FAILED": ("failed", "!", "Failed"),
    "CANCELLED": ("cancelled", "X", "Cancelled"),
}


def inject_theme() -> None:
    """Inject the console CSS exactly once per full page execution."""
    st.markdown(f"<style>{CONSOLE_CSS}</style>", unsafe_allow_html=True)


def archive_heading(run_id: str | None, status: str | None) -> str:
    """Return the archive masthead HTML."""
    reference = escape(_short_run_reference(run_id))
    badge = status_badge(status) if status is not None else ""
    return (
        '<div class="rp-archive-header">'
        '<div class="rp-archive-grid">'
        "<div>"
        '<div class="rp-archive-kicker">RepoPilot / Editorial run archive</div>'
        '<div class="rp-archive-title">Execution Record</div>'
        "</div>"
        '<div class="rp-archive-reference">'
        '<div class="rp-archive-reference-label">File reference</div>'
        f'<div class="rp-archive-reference-value">RUN / {reference}</div>'
        "</div>"
        f'<div class="rp-archive-status">{badge}</div>'
        "</div>"
        "</div>"
    )


def status_badge(status: str | None) -> str:
    """Encode lifecycle through color, text, and a glyph."""
    normalized = status or "UNKNOWN"
    bucket, glyph, label = _STATUS_PRESENTATION.get(
        normalized,
        ("cancelled", "·", "Unknown"),
    )
    return (
        f'<span class="rp-status rp-status--{bucket}" title="{escape(normalized)}">'
        f"{glyph} {escape(label)} / {escape(normalized)}</span>"
    )


def section_label(label: str) -> str:
    """Return a numbered editorial section label with a rule."""
    number = _SECTION_NUMBERS.get(label.casefold(), "—")
    return (
        '<div class="rp-section-label">'
        f'<span class="rp-section-number">{escape(number)}</span>'
        f'<span class="rp-section-title"><span class="rp-section-separator">—</span> '
        f"{escape(label.upper())}</span>"
        '<span class="rp-section-rule" aria-hidden="true"></span>'
        "</div>"
    )


def event_act(event: TraceEventView) -> ActLabel:
    """Translate one public trace event into its editorial archive action."""
    if event.kind == "tool_call":
        # propose_patch only drafts a diff (no write); unknown tools stay
        # INTERVENTION so an unclassified action is never understated.
        if event.tool_name is None:
            return ActLabel.INTERVENTION
        return _TOOL_ACTS.get(event.tool_name, ActLabel.INTERVENTION)
    if event.kind == "approval_decision":
        decision = (event.decision or "").casefold()
        if decision in {"approve", "approved"}:
            return ActLabel.HUMAN_DECISION_APPROVED
        if decision in {"deny", "denied"}:
            return ActLabel.HUMAN_DECISION_DENIED
        return ActLabel.HUMAN_DECISION
    return _EVENT_ACTS.get(event.kind, ActLabel.ARCHIVE_EVENT)


def numbered_entry(event: TraceEventView) -> str:
    """Return one numbered, safely escaped timeline entry."""
    act = event_act(event)
    detail = _event_narrative(event)
    metadata = _event_metadata(event)
    return (
        '<div class="rp-timeline-entry">'
        f'<div class="rp-entry-number">{event.seq:02d}</div>'
        f'<div class="rp-entry-act">{escape(act.value)}</div>'
        f'<div class="rp-entry-kind">{escape(metadata)}</div>'
        f'<div class="rp-entry-summary">{escape(detail)}</div>'
        "</div>"
    )


def empty_note(message: str) -> str:
    """Return a restrained empty-state block."""
    return f'<div class="rp-empty-note">{escape(message)}</div>'


def metadata_row(key: str, value: object) -> str:
    """Return one escaped metadata row."""
    rendered = "—" if value is None else str(value)
    return (
        '<div class="rp-meta-row">'
        f'<div class="rp-meta-key">{escape(key)}</div>'
        f'<div class="rp-meta-value">{escape(rendered)}</div>'
        "</div>"
    )


def error_banner(code: str, message: str) -> str:
    """Return a safe archive-style page error banner."""
    return (
        '<div class="rp-error-banner">'
        f'<div class="rp-error-code">{escape(code)}</div>'
        f'<div class="rp-error-message">{escape(message)}</div>'
        "</div>"
    )


def approval_header(request: ApprovalRequestView, rationale: str) -> str:
    """Return the focal header for one pending approval."""
    return (
        '<div class="rp-approval-focus">'
        '<div class="rp-approval-topline">'
        f'<div class="rp-approval-docket">Pending document / {escape(request.request_id)}</div>'
        f'<div class="rp-approval-risk">{escape(request.risk_level)} risk</div>'
        "</div>"
        '<div class="rp-approval-title">Human authorization required</div>'
        f'<div class="rp-approval-tool">Protected action / {escape(request.tool_name)}</div>'
        f'<div class="rp-approval-rationale">{escape(rationale)}</div>'
        "</div>"
    )


def report_header(run_id: str | None, status: str | None) -> str:
    """Return a safely escaped document masthead for the terminal report."""
    reference = escape(_short_run_reference(run_id))
    rendered_status = escape(status or "UNKNOWN")
    return (
        '<div class="rp-report-document">'
        '<div class="rp-report-kicker">Final analysis / Terminal record</div>'
        '<div class="rp-report-title">Run Report</div>'
        f'<div class="rp-report-meta">RUN / {reference} · STATUS / {rendered_status}</div>'
        "</div>"
    )


def decision_record(request: ApprovalRequestView) -> str:
    """Return one immutable decision ledger entry."""
    note = f" · {request.note}" if request.note else ""
    return (
        '<div class="rp-decision-record">'
        f"{escape(request.request_id)} · {escape(request.status)} · "
        f"{escape(request.actor or 'unknown')}{escape(note)}"
        "</div>"
    )


def _short_run_reference(run_id: str | None) -> str:
    if run_id is None:
        return "NO ACTIVE RUN"
    return run_id[:12]


def _event_narrative(event: TraceEventView) -> str:
    if event.summary:
        return event.summary
    if event.kind == "tool_call":
        if event.ok is True:
            return "The tool operation completed successfully."
        if event.ok is False:
            return "The tool operation returned a failure."
        return "A tool operation was recorded."
    if event.kind == "approval_request":
        return "A protected action is awaiting human authorization."
    if event.kind == "approval_decision":
        decision = event.decision or "a decision"
        actor = event.actor or "an unknown actor"
        return f"{actor} recorded {decision} for the protected action."
    if event.kind == "critic_verdict" and event.decision:
        return f"The verifier recorded a {event.decision} verdict."
    return "Recorded event."


def _event_metadata(event: TraceEventView) -> str:
    parts = [f"KIND / {event.kind}"]
    optional_parts = (
        ("TOOL", event.tool_name),
        ("REQUEST", event.request_id),
        ("RISK", event.risk_level),
        ("DECISION", event.decision),
        ("ACTOR", event.actor),
        ("ERROR", event.error_type),
    )
    parts.extend(f"{label} / {value}" for label, value in optional_parts if value is not None)
    if event.ok is not None:
        parts.append(f"OK / {str(event.ok).lower()}")
    if event.latency_ms is not None:
        parts.append(f"LATENCY / {event.latency_ms} ms")
    parts.append(f"UTC / {event.ts.isoformat()}")
    return " · ".join(parts)
