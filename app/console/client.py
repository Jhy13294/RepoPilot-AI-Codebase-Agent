"""Thin HTTP client for the RepoPilot Streamlit console."""

import os
from typing import Literal, Protocol

import httpx

from app.api.schemas import (
    ApprovalRequestView,
    CreateRunRequest,
    CreateRunResponse,
    DecideApprovalRequest,
    RunEventsPage,
    RunListResponse,
    RunView,
)

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
_DEFAULT_TIMEOUT_SECONDS = 10.0

type TaskType = Literal["question", "issue", "fix"]
type ApprovalDecision = Literal["approve", "deny"]


class ConsoleClientLike(Protocol):
    """Structural seam used by the console state machine and AppTest fakes."""

    def create_run(
        self,
        task_type: TaskType,
        prompt: str,
        repo: str,
        max_steps: int | None = None,
    ) -> CreateRunResponse:
        """Create one run through the HTTP API."""

    def get_run(self, run_id: str) -> RunView:
        """Fetch one detailed run view."""

    def list_runs(self) -> RunListResponse:
        """Fetch the persisted run archive index."""

    def get_events(self, run_id: str, after_seq: int) -> RunEventsPage:
        """Fetch one incremental event page."""

    def list_pending(self, run_id: str) -> list[ApprovalRequestView]:
        """Fetch pending approvals for one run."""

    def decide(
        self,
        request_id: str,
        decision: ApprovalDecision,
        note: str | None,
    ) -> ApprovalRequestView:
        """Submit one human approval decision."""


class ConsoleClient:
    """Small DTO-validating wrapper around ``httpx.Client``."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._http = httpx.Client(
            base_url=base_url,
            timeout=timeout_seconds,
            trust_env=False,
        )

    def create_run(
        self,
        task_type: TaskType,
        prompt: str,
        repo: str,
        max_steps: int | None = None,
    ) -> CreateRunResponse:
        """Create one run and validate the acknowledgement DTO."""
        request = CreateRunRequest(
            task_type=task_type,
            prompt=prompt,
            repo=repo,
            max_steps=max_steps,
        )
        response = self._http.post(
            "/runs",
            json=request.model_dump(mode="json", exclude_none=True),
        )
        response.raise_for_status()
        return CreateRunResponse.model_validate(response.json())

    def get_run(self, run_id: str) -> RunView:
        """Fetch and validate one detailed run DTO."""
        response = self._http.get(f"/runs/{run_id}")
        response.raise_for_status()
        return RunView.model_validate(response.json())

    def list_runs(self) -> RunListResponse:
        """Fetch and validate the archive index DTO."""
        response = self._http.get("/runs")
        response.raise_for_status()
        return RunListResponse.model_validate(response.json())

    def get_events(self, run_id: str, after_seq: int) -> RunEventsPage:
        """Fetch and validate one cursor page."""
        response = self._http.get(
            f"/runs/{run_id}/events",
            params={"after_seq": after_seq},
        )
        response.raise_for_status()
        return RunEventsPage.model_validate(response.json())

    def list_pending(self, run_id: str) -> list[ApprovalRequestView]:
        """Fetch and validate all pending approval DTOs for one run."""
        response = self._http.get("/approvals", params={"run_id": run_id})
        response.raise_for_status()
        payload: object = response.json()
        if not isinstance(payload, list):
            raise ValueError("The approvals endpoint did not return a list.")
        return [ApprovalRequestView.model_validate(item) for item in payload]

    def decide(
        self,
        request_id: str,
        decision: ApprovalDecision,
        note: str | None,
    ) -> ApprovalRequestView:
        """Submit and validate one resolved approval DTO."""
        request = DecideApprovalRequest(decision=decision, note=note)
        response = self._http.post(
            f"/approvals/{request_id}",
            json=request.model_dump(mode="json", exclude_none=True),
        )
        response.raise_for_status()
        return ApprovalRequestView.model_validate(response.json())

    def close(self) -> None:
        """Release the underlying connection pool."""
        self._http.close()


def build_console_client() -> ConsoleClient:
    """Build the production client from the console-specific environment variable."""
    base_url = os.environ.get("REPOPILOT_API_BASE_URL", DEFAULT_API_BASE_URL)
    return ConsoleClient(base_url.rstrip("/"))
