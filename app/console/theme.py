"""Restrained archive theme and HTML helpers for the Streamlit console."""

from html import escape

import streamlit as st

from app.api.schemas import ApprovalRequestView, TraceEventView

CONSOLE_CSS = """
.rp-archive-header {
  border-top: 4px solid #8a3b2e;
  border-bottom: 1px solid #b8ad9b;
  margin: 0 0 1.15rem 0;
  padding: 0.85rem 0 0.75rem 0;
}
.rp-archive-kicker,
.rp-section-label,
.rp-entry-number,
.rp-entry-kind,
.rp-meta-key,
.rp-status,
.rp-error-code,
.rp-approval-risk {
  font-family: "Source Code Pro", "Cascadia Mono", Consolas, monospace;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.rp-archive-kicker {
  color: #8a3b2e;
  font-size: 0.72rem;
  margin-bottom: 0.2rem;
}
.rp-archive-title {
  color: #24211d;
  font-size: clamp(1.55rem, 3vw, 2.5rem);
  font-weight: 650;
  letter-spacing: -0.025em;
  line-height: 1.05;
}
.rp-archive-subtitle {
  color: #625b51;
  font-size: 0.9rem;
  margin-top: 0.35rem;
}
.rp-section-label {
  border-bottom: 1px solid #b8ad9b;
  color: #625b51;
  font-size: 0.68rem;
  margin: 0.3rem 0 0.7rem 0;
  padding-bottom: 0.3rem;
}
.rp-status {
  border: 1px solid currentColor;
  display: inline-block;
  font-size: 0.66rem;
  font-weight: 700;
  line-height: 1;
  padding: 0.32rem 0.45rem;
}
.rp-status--progress { color: #315c6d; background: #e3ecec; }
.rp-status--decision { color: #8a5a16; background: #f2e8cf; }
.rp-status--complete { color: #356043; background: #e1ebe2; }
.rp-status--failed { color: #8a3b2e; background: #f1dfda; }
.rp-status--cancelled { color: #625b51; background: #e7e1d7; }
.rp-timeline-entry {
  border-bottom: 1px solid #d2c8b8;
  display: grid;
  gap: 0.15rem 0.7rem;
  grid-template-columns: 3.4rem minmax(0, 1fr);
  padding: 0.65rem 0;
}
.rp-entry-number {
  color: #8a3b2e;
  font-size: 0.74rem;
  grid-row: 1 / span 2;
}
.rp-entry-kind {
  color: #625b51;
  font-size: 0.64rem;
}
.rp-entry-summary {
  color: #24211d;
  font-size: 0.91rem;
  overflow-wrap: anywhere;
}
.rp-empty-note {
  border: 1px dashed #b8ad9b;
  color: #625b51;
  font-size: 0.85rem;
  padding: 0.8rem;
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
  font-family: "Source Code Pro", "Cascadia Mono", Consolas, monospace;
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
  background: #eee6d8;
  border: 1px solid #a99d89;
  border-top: 4px solid #8a3b2e;
  margin: 0.45rem 0 0.75rem 0;
  padding: 0.8rem 0.9rem;
}
.rp-approval-title {
  color: #24211d;
  font-size: 1.05rem;
  font-weight: 650;
}
.rp-approval-risk {
  color: #8a3b2e;
  font-size: 0.65rem;
  margin-top: 0.25rem;
}
.rp-approval-rationale {
  color: #504a42;
  font-size: 0.88rem;
  margin-top: 0.55rem;
}
.rp-decision-record {
  border-left: 2px solid #8a3b2e;
  color: #504a42;
  font-family: "Source Code Pro", "Cascadia Mono", Consolas, monospace;
  font-size: 0.7rem;
  margin: 0.45rem 0;
  padding-left: 0.55rem;
}
"""

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
    reference = escape(run_id) if run_id is not None else "No active run"
    badge = status_badge(status) if status is not None else ""
    return (
        '<div class="rp-archive-header">'
        '<div class="rp-archive-kicker">RepoPilot / Autonomous run archive</div>'
        '<div class="rp-archive-title">Execution Record</div>'
        f'<div class="rp-archive-subtitle">{reference}&nbsp;&nbsp;{badge}</div>'
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
    """Return a compact archival section label."""
    return f'<div class="rp-section-label">{escape(label)}</div>'


def numbered_entry(event: TraceEventView) -> str:
    """Return one numbered, safely escaped timeline entry."""
    detail = event.summary or event.tool_name or event.decision or "Recorded event"
    return (
        '<div class="rp-timeline-entry">'
        f'<div class="rp-entry-number">{event.seq:02d}</div>'
        f'<div class="rp-entry-kind">{escape(event.kind)}</div>'
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
        f'<div class="rp-approval-title">Decision required · {escape(request.tool_name)}</div>'
        f'<div class="rp-approval-risk">{escape(request.risk_level)} risk / '
        f"{escape(request.request_id)}</div>"
        f'<div class="rp-approval-rationale">{escape(rationale)}</div>'
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
