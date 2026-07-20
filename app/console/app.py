"""Streamlit entry point for the RepoPilot HTTP run archive."""

# ruff: noqa: E402

import sys
import time
from pathlib import Path
from typing import cast

# Streamlit prepends this script directory, so the repository package path must win.
_REPOSITORY_ROOT = str(Path(__file__).resolve().parents[2])
if _REPOSITORY_ROOT in sys.path:
    sys.path.remove(_REPOSITORY_ROOT)
sys.path.insert(0, _REPOSITORY_ROOT)

import streamlit as st

from app.console.client import (
    ApprovalDecision,
    ConsoleClientLike,
    TaskType,
    build_console_client,
)
from app.console.components import (
    ArchiveAction,
    render_active_archive,
    render_approval_panel,
    render_archive_index,
    render_error,
    render_header,
)
from app.console.state import (
    ConsolePhase,
    ConsoleState,
    advance,
    decide_approval,
    reset_to_empty,
    select_run,
    start_run,
)
from app.console.theme import inject_theme

_STATE_KEY = "_console_state"
_CLIENT_KEY = "_client"
_DECISION_KEY = "_approval_action"
_DEFAULT_REFRESH_SECONDS = 1.0

QueuedDecision = tuple[str, ApprovalDecision, str | None]


def main() -> None:
    """Execute one complete Streamlit script pass."""
    st.set_page_config(
        page_title="RepoPilot Run Archive",
        page_icon="📂",
        layout="wide",
    )
    inject_theme()

    client = _console_client()
    state = advance(_console_state(), client)
    queued_decision = _take_queued_decision()
    if queued_decision is not None:
        request_id, decision, note = queued_decision
        state = decide_approval(
            state,
            client,
            request_id=request_id,
            decision=decision,
            note=note,
        )
    _save_state(state)

    render_header(state)
    render_error(state.page_error)

    if state.phase is ConsolePhase.EMPTY:
        _render_empty(state, client)
    else:
        render_approval_panel(state, _queue_decision)
        action = render_active_archive(state)
        _apply_archive_action(state, action)

    _schedule_refresh(_console_state())


def _render_empty(state: ConsoleState, client: ConsoleClientLike) -> None:
    form_column, index_column = st.columns((1.6, 1.0), gap="large")
    with form_column:
        render_error(state.form_error)
        with st.form("create_run"):
            st.subheader("Open a new run record")
            selected_type = str(
                st.selectbox(
                    "Task type",
                    options=("question", "issue", "fix"),
                )
            )
            prompt = str(st.text_area("Prompt", height=150))
            repo = str(st.text_input("Repository path"))
            submitted = bool(st.form_submit_button("Create run", type="primary"))
        if submitted:
            task_type = cast(TaskType, selected_type)
            updated = start_run(
                state,
                client,
                task_type=task_type,
                prompt=prompt,
                repo=repo,
            )
            _save_state(updated)
            if updated.form_error is not None:
                if state.form_error is None:
                    render_error(updated.form_error)
            else:
                st.rerun()
    with index_column:
        action = render_archive_index(state)
    _apply_archive_action(state, action)


def _apply_archive_action(
    state: ConsoleState,
    action: ArchiveAction | None,
) -> None:
    if action is None:
        return
    updated = reset_to_empty(state) if action.run_id is None else select_run(state, action.run_id)
    _save_state(updated)
    st.rerun()


def _console_state() -> ConsoleState:
    value = st.session_state.get(_STATE_KEY)
    return value if isinstance(value, ConsoleState) else ConsoleState()


def _save_state(state: ConsoleState) -> None:
    st.session_state[_STATE_KEY] = state


def _console_client() -> ConsoleClientLike:
    injected = st.session_state.get(_CLIENT_KEY)
    client = cast(ConsoleClientLike, injected or build_console_client())
    if injected is None:
        st.session_state[_CLIENT_KEY] = client
    return client


def _queue_decision(
    request_id: str,
    decision: ApprovalDecision,
    note: str | None,
) -> None:
    st.session_state[_DECISION_KEY] = (request_id, decision, note)


def _take_queued_decision() -> QueuedDecision | None:
    value = st.session_state.pop(_DECISION_KEY, None)
    if not isinstance(value, tuple) or len(value) != 3:
        return None
    request_id, decision, note = value
    if not isinstance(request_id, str) or decision not in {"approve", "deny"}:
        return None
    if note is not None and not isinstance(note, str):
        return None
    return request_id, decision, note


def _schedule_refresh(state: ConsoleState) -> None:
    auto_refresh = bool(st.session_state.get("_auto_refresh", True))
    if (
        not auto_refresh
        or state.active_run_id is None
        or state.terminal
        or state.page_error is not None
    ):
        return
    interval_value = st.session_state.get(
        "_refresh_interval_seconds",
        _DEFAULT_REFRESH_SECONDS,
    )
    interval = (
        float(interval_value)
        if isinstance(interval_value, int | float)
        else _DEFAULT_REFRESH_SECONDS
    )
    time.sleep(max(0.1, interval))
    st.rerun()


if __name__ == "__main__":
    main()
