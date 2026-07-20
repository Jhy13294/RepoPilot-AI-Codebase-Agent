import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import cast

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from app.agent.state import AgentState, Budgets, RunStatus, TaskSpec
from app.api.app import _stream_run_events, create_app
from app.api.schemas import RunEventsPage, TraceEventView
from app.api.service import RunService
from app.config import Settings
from app.schemas.trace import TraceEventKind
from app.storage.db import Database
from app.storage.trace_store import TraceStore

_SENTINEL = "PRIVATE_DIFF_SENTINEL_7f3b"


class _ConnectedRequest:
    async def is_disconnected(self) -> bool:
        return False


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        OPENAI_API_KEY="sk-test-api",
        db_path=tmp_path / "runs.sqlite",
        trace_dir=tmp_path / "traces",
    )


def _service(tmp_path: Path) -> tuple[RunService, Database, TraceStore]:
    settings = _settings(tmp_path)
    database = Database(settings.db_path)
    store = TraceStore(settings.trace_dir)

    def discard_spawn(_job: Callable[[], None]) -> None:
        return None

    return (
        RunService(settings, database=database, store=store, spawn=discard_spawn),
        database,
        store,
    )


def _save_run(
    database: Database,
    tmp_path: Path,
    run_id: str,
    status: RunStatus,
) -> AgentState:
    state = AgentState(
        run_id=run_id,
        task=TaskSpec(task_type="question", prompt="Inspect the repository.", repo=str(tmp_path)),
        plan=[],
        cursor=0,
        tool_history=[],
        budgets=Budgets(),
        status=status,
    )
    database.save_state(state)
    return state


def _trace_views_from_sse(body: str) -> list[TraceEventView]:
    views: list[TraceEventView] = []
    for frame in body.split("\n\n"):
        lines = frame.splitlines()
        if not lines or lines[0].startswith(":"):
            continue
        event_line = next((line for line in lines if line.startswith("event: ")), None)
        if event_line in {"event: terminal", "event: timeout"}:
            continue
        data_line = next(line for line in lines if line.startswith("data: "))
        views.append(TraceEventView.model_validate_json(data_line.removeprefix("data: ")))
    return views


def test_events_endpoint__cursor_pages_compose_without_duplicates(
    tmp_path: Path,
) -> None:
    service, database, store = _service(tmp_path)
    run_id = "run-cursor"
    _save_run(database, tmp_path, run_id, RunStatus.PLANNING)
    for index in range(3):
        store.append(run_id, TraceEventKind.plan, {"summary": f"Event {index}"})

    with TestClient(create_app(service)) as client:
        first_response = client.get(f"/runs/{run_id}/events", params={"after_seq": -1})
        first_page = RunEventsPage.model_validate(first_response.json())
        assert [event.seq for event in first_page.events] == [0, 1, 2]
        assert first_page.next_cursor == 2

        for index in range(3, 6):
            store.append(run_id, TraceEventKind.plan, {"summary": f"Event {index}"})

        resumed_response = client.get(
            f"/runs/{run_id}/events",
            params={"after_seq": first_page.next_cursor},
        )
        resumed_page = RunEventsPage.model_validate(resumed_response.json())
        assert [event.seq for event in resumed_page.events] == [3, 4, 5]
        assert resumed_page.next_cursor == 5

        empty_page = RunEventsPage.model_validate(
            client.get(f"/runs/{run_id}/events", params={"after_seq": 5}).json()
        )
        future_page = RunEventsPage.model_validate(
            client.get(f"/runs/{run_id}/events", params={"after_seq": 99}).json()
        )
        full_page = RunEventsPage.model_validate(
            client.get(f"/runs/{run_id}/events", params={"after_seq": -1}).json()
        )

    assert first_response.status_code == 200
    assert resumed_response.status_code == 200
    assert first_page.events + resumed_page.events == full_page.events
    assert empty_page.events == []
    assert empty_page.next_cursor == 5
    assert future_page.events == []
    assert future_page.next_cursor == 99
    assert not full_page.terminal


def test_events_and_stream__share_safe_projection_without_diff_leakage(
    tmp_path: Path,
) -> None:
    service, database, store = _service(tmp_path)
    run_id = "run-safe-stream"
    _save_run(database, tmp_path, run_id, RunStatus.DONE)
    store.append(
        run_id,
        TraceEventKind.tool_call,
        {
            "tool_name": "apply_patch",
            "args": {"diff": _SENTINEL},
            "ok": False,
            "error_type": "PatchApplyError",
            "outcome": {"diff": _SENTINEL},
        },
        latency_ms=23,
    )
    store.append(
        run_id,
        TraceEventKind.approval_request,
        {
            "request_id": "request-safe",
            "tool_name": "apply_patch",
            "risk_level": "high",
            "args": {"diff": _SENTINEL},
        },
    )

    with TestClient(create_app(service)) as client:
        events_response = client.get(f"/runs/{run_id}/events")
        stream_response = client.get(f"/runs/{run_id}/stream")
        resumed_stream_response = client.get(
            f"/runs/{run_id}/stream",
            headers={"Last-Event-ID": "0"},
        )

    assert events_response.status_code == 200
    assert stream_response.status_code == 200
    assert stream_response.headers["content-type"].startswith("text/event-stream")
    assert _SENTINEL not in events_response.text
    assert _SENTINEL not in stream_response.text

    page = RunEventsPage.model_validate(events_response.json())
    stream_views = _trace_views_from_sse(stream_response.text)
    assert stream_views == page.events
    assert "event: terminal\ndata: DONE\n\n" in stream_response.text

    approval_view = page.events[1]
    assert approval_view.model_dump(exclude_none=True) == {
        "seq": 1,
        "ts": approval_view.ts,
        "kind": "approval_request",
        "tool_name": "apply_patch",
        "risk_level": "high",
        "request_id": "request-safe",
    }
    assert _trace_views_from_sse(resumed_stream_response.text) == [approval_view]


def test_events_and_stream__terminal_comes_only_from_sqlite_and_closes_immediately(
    tmp_path: Path,
) -> None:
    service, database, store = _service(tmp_path)
    run_id = "run-terminal"
    state = _save_run(database, tmp_path, run_id, RunStatus.PLANNING)
    report = store.append(run_id, TraceEventKind.report, {"summary": "Trace looks complete."})

    with TestClient(create_app(service)) as client:
        running_page = RunEventsPage.model_validate(client.get(f"/runs/{run_id}/events").json())
        database.save_state(state.model_copy(update={"status": RunStatus.DONE}))
        terminal_page = RunEventsPage.model_validate(
            client.get(
                f"/runs/{run_id}/events",
                params={"after_seq": report.seq},
            ).json()
        )
        stream_response = client.get(
            f"/runs/{run_id}/stream",
            headers={"Last-Event-ID": str(report.seq)},
        )

    assert not running_page.terminal
    assert running_page.status is RunStatus.PLANNING
    assert terminal_page.terminal
    assert terminal_page.status is RunStatus.DONE
    assert terminal_page.events == []
    assert terminal_page.next_cursor == report.seq
    assert stream_response.text == "event: terminal\ndata: DONE\n\n"


def test_run_event_endpoints__return_404_for_unknown_run(tmp_path: Path) -> None:
    service, _database, _store = _service(tmp_path)

    with TestClient(create_app(service)) as client:
        events_response = client.get("/runs/missing/events")
        stream_response = client.get("/runs/missing/stream")

    assert events_response.status_code == 404
    assert stream_response.status_code == 404


def test_stream_generator__heartbeats_then_observes_terminal_without_real_sleep(
    tmp_path: Path,
) -> None:
    service, database, _store = _service(tmp_path)
    run_id = "run-heartbeat"
    state = _save_run(database, tmp_path, run_id, RunStatus.PLANNING)
    initial_page = service.list_events_since(run_id)
    assert initial_page is not None

    async def exercise_stream() -> None:
        stream = _stream_run_events(
            cast(Request, _ConnectedRequest()),
            service,
            initial_page,
            poll_interval_s=0,
            timeout_s=1,
        )
        assert await anext(stream) == ": keep-alive\n\n"
        database.save_state(state.model_copy(update={"status": RunStatus.DONE}))
        assert await anext(stream) == "event: terminal\ndata: DONE\n\n"
        with pytest.raises(StopAsyncIteration):
            await anext(stream)

    asyncio.run(exercise_stream())


def test_stream_generator__emits_timeout_frame_and_closes(tmp_path: Path) -> None:
    service, database, _store = _service(tmp_path)
    run_id = "run-timeout"
    _save_run(database, tmp_path, run_id, RunStatus.PLANNING)
    initial_page = service.list_events_since(run_id)
    assert initial_page is not None

    async def exercise_stream() -> None:
        stream = _stream_run_events(
            cast(Request, _ConnectedRequest()),
            service,
            initial_page,
            poll_interval_s=0,
            timeout_s=0,
        )
        assert await anext(stream) == "event: timeout\ndata: timeout\n\n"
        with pytest.raises(StopAsyncIteration):
            await anext(stream)

    asyncio.run(exercise_stream())


def test_list_events_since__concurrent_append_never_reads_a_torn_json_line(
    tmp_path: Path,
) -> None:
    service, database, store = _service(tmp_path)
    run_id = "run-concurrent"
    _save_run(database, tmp_path, run_id, RunStatus.PLANNING)
    writer_started = Event()
    writer_continue = Event()

    def append_events() -> None:
        store.append(run_id, TraceEventKind.plan, {"summary": "Event 0"})
        writer_started.set()
        assert writer_continue.wait(timeout=5)
        for index in range(1, 100):
            store.append(run_id, TraceEventKind.plan, {"summary": f"Event {index}"})

    cursor = -1
    observed: list[TraceEventView] = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        writer = executor.submit(append_events)
        assert writer_started.wait(timeout=5)
        first_page = service.list_events_since(run_id, cursor)
        assert first_page is not None
        observed.extend(first_page.events)
        cursor = first_page.next_cursor
        writer_continue.set()

        while not writer.done():
            page = service.list_events_since(run_id, cursor)
            assert page is not None
            observed.extend(page.events)
            cursor = page.next_cursor
        writer.result()

    final_page = service.list_events_since(run_id, cursor)
    assert final_page is not None
    observed.extend(final_page.events)
    assert [event.seq for event in observed] == list(range(100))
