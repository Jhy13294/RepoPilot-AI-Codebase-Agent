"""FastAPI application factory for the RepoPilot run service."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import monotonic
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from app.api.events import format_sse_frame
from app.api.schemas import (
    ApprovalRequestView,
    CreateRunRequest,
    CreateRunResponse,
    DecideApprovalRequest,
    RunEventsPage,
    RunListResponse,
    RunView,
)
from app.api.service import InvalidRepositoryError, RunService
from app.config import load_settings
from app.storage.db import ApprovalRequestAlreadyDecidedError, ApprovalRequestNotFoundError

_STREAM_POLL_INTERVAL_S = 1.0
_STREAM_TIMEOUT_S = 300.0
_KEEP_ALIVE_FRAME = ": keep-alive\n\n"
_TIMEOUT_FRAME = "event: timeout\ndata: timeout\n\n"
_LOGGER = logging.getLogger(__name__)


def create_app(service: RunService | None = None) -> FastAPI:
    """Build the HTTP application around an injected or configured run service."""
    owns_service = service is None
    run_service = service or RunService(load_settings())

    @asynccontextmanager
    async def lifespan(_application: FastAPI) -> AsyncIterator[None]:
        yield
        if owns_service:
            run_service.close()

    application = FastAPI(
        title="RepoPilot API",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.state.run_service = run_service

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.post(
        "/runs",
        response_model=CreateRunResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def create_run(request: CreateRunRequest, response: Response) -> CreateRunResponse:
        try:
            created = run_service.create_run(request)
        except InvalidRepositoryError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        response.headers["Location"] = f"/runs/{created.run_id}"
        return created

    @application.get("/runs/{run_id}", response_model=RunView)
    def get_run(run_id: str) -> RunView:
        run = run_service.get_run(run_id)
        if run is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Run '{run_id}' was not found.",
            )
        return run

    @application.get("/runs/{run_id}/events", response_model=RunEventsPage)
    def list_run_events(run_id: str, after_seq: int = -1) -> RunEventsPage:
        page = run_service.list_events_since(run_id, after_seq)
        if page is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Run '{run_id}' was not found.",
            )
        return page

    @application.get("/runs/{run_id}/stream")
    async def stream_run_events(
        run_id: str,
        request: Request,
        last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
    ) -> StreamingResponse:
        cursor = _parse_last_event_id(last_event_id)
        page = await run_in_threadpool(run_service.list_events_since, run_id, cursor)
        if page is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Run '{run_id}' was not found.",
            )
        return StreamingResponse(
            _stream_run_events(
                request,
                run_service,
                page,
                poll_interval_s=_STREAM_POLL_INTERVAL_S,
                timeout_s=_STREAM_TIMEOUT_S,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    @application.get("/runs", response_model=RunListResponse)
    def list_runs() -> RunListResponse:
        return run_service.list_runs()

    @application.get("/approvals", response_model=list[ApprovalRequestView])
    def list_approvals(run_id: str | None = None) -> list[ApprovalRequestView]:
        return run_service.list_pending_approvals(run_id)

    @application.post("/approvals/{request_id}", response_model=ApprovalRequestView)
    def decide_approval(
        request_id: str,
        request: DecideApprovalRequest,
    ) -> ApprovalRequestView:
        existing = run_service.get_approval_request(request_id)
        if existing is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Approval request '{request_id}' was not found.",
            )
        if existing.status != "pending":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Approval request '{request_id}' has already been decided.",
            )

        try:
            return run_service.decide_approval(
                request_id,
                approved=request.decision == "approve",
                actor="human",
                note=request.note,
            )
        except ApprovalRequestNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Approval request '{request_id}' was not found.",
            ) from exc
        except ApprovalRequestAlreadyDecidedError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Approval request '{request_id}' has already been decided.",
            ) from exc

    return application


def _parse_last_event_id(last_event_id: str | None) -> int:
    if last_event_id is None or not last_event_id.strip():
        return -1
    try:
        return int(last_event_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Last-Event-ID must be an integer sequence cursor.",
        ) from exc


async def _stream_run_events(
    request: Request,
    run_service: RunService,
    initial_page: RunEventsPage,
    *,
    poll_interval_s: float,
    timeout_s: float,
) -> AsyncIterator[str]:
    page = initial_page
    cursor = page.next_cursor
    deadline = monotonic() + timeout_s
    try:
        while True:
            for event in page.events:
                yield format_sse_frame(event)
            cursor = page.next_cursor

            if page.terminal:
                yield f"event: terminal\ndata: {page.status.value}\n\n"
                return
            if await request.is_disconnected():
                return

            remaining_s = deadline - monotonic()
            if remaining_s <= 0:
                yield _TIMEOUT_FRAME
                return
            if not page.events:
                yield _KEEP_ALIVE_FRAME

            await asyncio.sleep(min(poll_interval_s, remaining_s))
            if await request.is_disconnected():
                return
            if monotonic() >= deadline:
                yield _TIMEOUT_FRAME
                return

            next_page = await run_in_threadpool(
                run_service.list_events_since,
                initial_page.run_id,
                cursor,
            )
            if next_page is None:
                return
            page = next_page
    finally:
        _LOGGER.debug("Run event stream closed for %s at cursor %d.", page.run_id, cursor)
