from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import JsonValue

from app.api.app import create_app
from app.api.service import RunService
from app.config import Settings
from app.services.llm_client import LLMClient
from app.storage.db import Database
from app.storage.trace_store import TraceStore

_APPROVAL_ARGS: dict[str, JsonValue] = {
    "diff": "--- a/src/value.py\n+++ b/src/value.py\n@@ -1 +1 @@\n-old\n+new\n",
    "rationale": "Correct the verified defect.",
    "path": "src/value.py",
}


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        OPENAI_API_KEY="sk-test-api",
        llm_provider="openai_compatible",
        model="test-model",
        db_path=tmp_path / "runs.sqlite",
        trace_dir=tmp_path / "traces",
        max_steps=6,
        max_replans=2,
        max_fix_cycles=2,
    )


def _unexpected_spawn(_job: Callable[[], None]) -> None:
    raise AssertionError("Approval reads and decisions must not spawn a run.")


def _unexpected_client_factory(_settings: Settings) -> LLMClient:
    raise AssertionError("Approval reads and decisions must not construct an LLM client.")


def _restart_service(
    settings: Settings,
    database: Database,
    store: TraceStore,
) -> RunService:
    return RunService(
        settings,
        database=database,
        store=store,
        spawn=_unexpected_spawn,
        client_factory=_unexpected_client_factory,
    )


def test_approvals_api__lists_pending_requests_with_full_args_and_run_filter(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    database = Database(settings.db_path)
    store = TraceStore(settings.trace_dir)
    database.create_approval_request(
        "approval-main",
        run_id="run-main",
        tool_name="apply_patch",
        risk_level="high",
        args=_APPROVAL_ARGS,
        created_at=datetime(2026, 7, 20, 8, 0, tzinfo=UTC),
    )
    database.create_approval_request(
        "approval-other",
        run_id="run-other",
        tool_name="run_tests",
        risk_level="high",
        args={"rationale": "Verify another run."},
        created_at=datetime(2026, 7, 20, 8, 1, tzinfo=UTC),
    )
    service = _restart_service(settings, database, store)

    with TestClient(create_app(service)) as client:
        all_response = client.get("/approvals")
        filtered_response = client.get("/approvals", params={"run_id": "run-main"})

    assert all_response.status_code == 200
    all_requests = all_response.json()
    assert [item["request_id"] for item in all_requests] == [
        "approval-main",
        "approval-other",
    ]
    assert all_requests[0]["args"] == _APPROVAL_ARGS
    assert all_requests[0]["status"] == "pending"
    assert all_requests[0]["actor"] is None
    assert all_requests[0]["note"] is None
    assert all_requests[0]["decided_at"] is None

    assert filtered_response.status_code == 200
    assert filtered_response.json() == [all_requests[0]]


def test_approvals_api__decides_restart_pending_and_rejects_unknown_or_redecide(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    database = Database(settings.db_path)
    store = TraceStore(settings.trace_dir)
    database.create_approval_request(
        "approval-restart",
        run_id="run-restart",
        tool_name="apply_patch",
        risk_level="high",
        args=_APPROVAL_ARGS,
    )
    service = _restart_service(settings, database, store)

    with TestClient(create_app(service)) as client:
        missing_response = client.post(
            "/approvals/unknown",
            json={"decision": "approve"},
        )
        approved_response = client.post(
            "/approvals/approval-restart",
            json={"decision": "approve", "note": "Reviewed and accepted."},
        )
        conflict_response = client.post(
            "/approvals/approval-restart",
            json={"decision": "deny", "note": "Too late."},
        )

    assert missing_response.status_code == 404
    assert approved_response.status_code == 200
    approved = approved_response.json()
    assert approved["request_id"] == "approval-restart"
    assert approved["status"] == "approved"
    assert approved["actor"] == "human"
    assert approved["note"] == "Reviewed and accepted."
    assert approved["decided_at"] is not None
    assert approved["args"] == _APPROVAL_ARGS

    assert conflict_response.status_code == 409
    persisted = database.get_approval_request("approval-restart")
    assert persisted is not None
    assert persisted.status == "approved"
    assert persisted.actor == "human"
    assert persisted.note == "Reviewed and accepted."
