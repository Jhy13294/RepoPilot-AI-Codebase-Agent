"""FastAPI application factory for the RepoPilot run service."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response, status

from app.api.schemas import (
    ApprovalRequestView,
    CreateRunRequest,
    CreateRunResponse,
    DecideApprovalRequest,
    RunListResponse,
    RunView,
)
from app.api.service import InvalidRepositoryError, RunService
from app.config import load_settings
from app.storage.db import ApprovalRequestAlreadyDecidedError, ApprovalRequestNotFoundError


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
