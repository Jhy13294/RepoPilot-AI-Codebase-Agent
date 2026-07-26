import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import pytest
from fastapi.testclient import TestClient
from pydantic import JsonValue

from app.agent.state import RunStatus
from app.api.app import create_app
from app.api.service import RunService
from app.config import Settings
from app.schemas.llm_io import LLMMessage, LLMResponse, Role, StopReason, ToolCall, Usage
from app.schemas.trace import TraceEventKind
from app.services.llm_client import LLMClient, ToolSchema
from app.storage.db import Database
from app.storage.trace_store import TraceStore


class _DeferredSpawn:
    def __init__(self) -> None:
        self.jobs: list[Callable[[], None]] = []

    def __call__(self, job: Callable[[], None]) -> None:
        self.jobs.append(job)

    def run_next(self) -> None:
        if not self.jobs:
            raise AssertionError("No deferred run is available.")
        self.jobs.pop(0)()


class _ScriptedClient:
    def __init__(self, responses: Sequence[LLMResponse]) -> None:
        self._responses = list(responses)

    def complete(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[ToolSchema] | None = None,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        del messages, tools, system, temperature, max_tokens
        if not self._responses:
            raise AssertionError("No scripted LLM response remains.")
        return self._responses.pop(0)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        OPENAI_API_KEY="sk-test-api",
        llm_provider="openai_compatible",
        model="test-model",
        db_path=tmp_path / "runs.sqlite",
        trace_dir=tmp_path / "traces",
        workspace_dir=tmp_path,
        max_steps=6,
        max_replans=2,
        max_fix_cycles=2,
    )


def _service(
    tmp_path: Path,
    *,
    scripted_client: _ScriptedClient | None = None,
) -> tuple[RunService, Database, TraceStore, _DeferredSpawn, Settings]:
    settings = _settings(tmp_path)
    database = Database(settings.db_path)
    store = TraceStore(settings.trace_dir)
    spawn = _DeferredSpawn()

    def client_factory(_settings: Settings) -> LLMClient:
        if scripted_client is None:
            raise AssertionError("The client factory must not run in this test.")
        return scripted_client

    service = RunService(
        settings,
        database=database,
        store=store,
        spawn=spawn,
        client_factory=client_factory,
    )
    return service, database, store, spawn, settings


def _response(
    content: str = "",
    *,
    stop_reason: StopReason = StopReason.end_turn,
    tool_calls: list[ToolCall] | None = None,
) -> LLMResponse:
    return LLMResponse(
        message=LLMMessage(
            role=Role.assistant,
            content=content,
            tool_calls=tool_calls or [],
        ),
        stop_reason=stop_reason,
        usage=Usage(tokens_in=1, tokens_out=1, cost_usd=0.001),
        model="scripted-model",
        raw_finish_reason=stop_reason.value,
    )


def _tool_call(call_id: str, name: str, arguments: dict[str, object]) -> ToolCall:
    return ToolCall(
        id=call_id,
        name=name,
        arguments=cast(dict[str, JsonValue], arguments),
    )


def _completed_run_client() -> _ScriptedClient:
    return _ScriptedClient(
        [
            _response(
                json.dumps(
                    {
                        "steps": [
                            {
                                "intent": "Locate the parse_date definition.",
                                "suggested_tools": ["search_code"],
                                "success_check": "The definition path and line are known.",
                            }
                        ]
                    }
                )
            ),
            _response(
                stop_reason=StopReason.tool_use,
                tool_calls=[
                    _tool_call(
                        "call-search",
                        "search_code",
                        {"query": "def parse_date", "glob": "**/*.py", "context_lines": 0},
                    )
                ],
            ),
            _response(
                json.dumps(
                    {
                        "findings": "parse_date is defined in src/sample_pkg/dates.py:6.",
                        "evidence": ["search_code: src/sample_pkg/dates.py:6"],
                    }
                )
            ),
            _response(
                json.dumps(
                    {
                        "decision": "proceed",
                        "reason": "The raw tool evidence proves the definition location.",
                        "hint": "",
                    }
                )
            ),
            _response(
                json.dumps(
                    {
                        "headline": "parse_date definition found",
                        "analysis": "The verified search trace identifies the implementation.",
                        "confidence": "high",
                        "open_questions": [],
                        "suspects": [],
                        "citations": ["src/sample_pkg/dates.py:6"],
                    }
                )
            ),
        ]
    )


def _create_body(repo: Path, *, task_type: str = "question") -> dict[str, object]:
    return {
        "task_type": task_type,
        "prompt": "Where is parse_date defined?",
        "repo": str(repo),
        "max_steps": 3,
    }


def test_runs_api__post_prepersists_planning_before_deferred_execution(
    mini_repo: Path,
    tmp_path: Path,
) -> None:
    service, database, _store, spawn, _settings_value = _service(tmp_path)

    with TestClient(create_app(service)) as client:
        response = client.post("/runs", json=_create_body(mini_repo))

        assert response.status_code == 202
        created = response.json()
        run_id = created["run_id"]
        assert created == {"run_id": run_id, "status": "PLANNING"}
        assert response.headers["location"] == f"/runs/{run_id}"

        detail_response = client.get(f"/runs/{run_id}")
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["run_id"] == run_id
        assert detail["task_type"] == "question"
        assert detail["repo"] == str(mini_repo.resolve())
        assert detail["status"] == "PLANNING"
        assert detail["step_count"] == 0
        assert detail["steps_used"] == 0
        assert detail["summary"] is None
        assert detail["tool_calls"] == []

        list_response = client.get("/runs")
        assert list_response.status_code == 200
        [listed] = list_response.json()["runs"]
        assert listed["run_id"] == run_id
        assert listed["status"] == "PLANNING"
        assert "summary" not in listed
        assert "tool_calls" not in listed

    assert len(spawn.jobs) == 1
    state = database.load_state(run_id)
    assert state is not None
    assert state.budgets.max_steps == 3


def test_runs_api__running_view_reads_live_jsonl_without_exposing_payload(
    mini_repo: Path,
    tmp_path: Path,
) -> None:
    service, database, store, _spawn, _settings_value = _service(tmp_path)

    with TestClient(create_app(service)) as client:
        created = client.post("/runs", json=_create_body(mini_repo)).json()
        run_id = created["run_id"]
        store.append(run_id, TraceEventKind.report, {"summary": "not terminal yet"})
        store.append(
            run_id,
            TraceEventKind.tool_call,
            {
                "tool_name": "read_file",
                "args": {"path": "secret.py", "diff": "private diff"},
                "ok": False,
                "error_type": "NotFoundError",
                "truncated": True,
            },
            latency_ms=12,
        )

        response = client.get(f"/runs/{run_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "PLANNING"
    assert body["summary"] is None
    assert database.tool_calls(run_id) == []
    assert body["tool_calls"] == [
        {
            "seq": 1,
            "ts": body["tool_calls"][0]["ts"],
            "tool_name": "read_file",
            "ok": False,
            "error_type": "NotFoundError",
            "latency_ms": 12,
        }
    ]
    assert set(body["tool_calls"][0]) == {
        "seq",
        "ts",
        "tool_name",
        "ok",
        "error_type",
        "latency_ms",
    }


@pytest.mark.parametrize("report_payload", [{}, {"summary": 7}])
def test_runs_api__terminal_report_without_string_summary_is_tolerated(
    mini_repo: Path,
    tmp_path: Path,
    report_payload: dict[str, JsonValue],
) -> None:
    service, database, store, _spawn, _settings_value = _service(tmp_path)

    with TestClient(create_app(service)) as client:
        created = client.post("/runs", json=_create_body(mini_repo)).json()
        run_id = created["run_id"]
        state = database.load_state(run_id)
        assert state is not None
        database.save_state(state.model_copy(update={"status": RunStatus.FAILED}))
        store.append(run_id, TraceEventKind.report, report_payload)

        response = client.get(f"/runs/{run_id}")

    assert response.status_code == 200
    assert response.json()["status"] == "FAILED"
    assert response.json()["summary"] is None


def test_runs_api__completed_run_and_timeline_survive_storage_restart(
    mini_repo: Path,
    tmp_path: Path,
) -> None:
    service, first_database, _first_store, spawn, settings = _service(
        tmp_path,
        scripted_client=_completed_run_client(),
    )

    with TestClient(create_app(service)) as client:
        created = client.post("/runs", json=_create_body(mini_repo, task_type="issue")).json()
        run_id = created["run_id"]
        spawn.run_next()
        first_response = client.get(f"/runs/{run_id}")

    assert first_response.status_code == 200
    first_view = first_response.json()
    assert first_view["status"] == "DONE"
    assert first_view["step_count"] == 1
    assert first_view["steps_used"] == 1
    assert first_view["summary"] == (
        "parse_date definition found\n\nThe verified search trace identifies the implementation."
    )
    assert len(first_view["tool_calls"]) == 1
    assert first_view["tool_calls"][0]["tool_name"] == "search_code"
    assert first_view["tool_calls"][0]["ok"] is True

    first_database.engine.dispose()
    restarted_database = Database(settings.db_path)
    restarted_store = TraceStore(settings.trace_dir)
    restarted_spawn = _DeferredSpawn()

    def unexpected_client_factory(_settings: Settings) -> LLMClient:
        raise AssertionError("Reading a restarted run must not construct an LLM client.")

    restarted_service = RunService(
        settings,
        database=restarted_database,
        store=restarted_store,
        spawn=restarted_spawn,
        client_factory=unexpected_client_factory,
    )
    with TestClient(create_app(restarted_service)) as client:
        restarted_response = client.get(f"/runs/{run_id}")

    assert restarted_response.status_code == 200
    assert restarted_response.json() == first_view
    assert restarted_database.get_run(run_id) is not None
    assert restarted_spawn.jobs == []


def test_runs_api__repository_outside_workspace_returns_400_without_side_effects(
    tmp_path: Path,
) -> None:
    service, database, _store, spawn, _settings_value = _service(tmp_path)
    outside_repo = tmp_path.parent / f"{tmp_path.name}-outside"
    outside_repo.mkdir()

    with TestClient(create_app(service)) as client:
        response = client.post("/runs", json=_create_body(outside_repo))

    assert response.status_code == 400
    assert response.json() == {
        "detail": (
            f"Repository path '{outside_repo}' is outside the configured API workspace root "
            f"'{tmp_path.resolve()}'."
        )
    }
    assert database.list_runs() == []
    assert spawn.jobs == []


def test_runs_api__parent_traversal_outside_workspace_returns_400(tmp_path: Path) -> None:
    service, database, _store, spawn, _settings_value = _service(tmp_path)
    traversal_repo = tmp_path / "inside" / ".." / ".." / "outside"

    with TestClient(create_app(service)) as client:
        response = client.post("/runs", json=_create_body(traversal_repo))

    assert response.status_code == 400
    assert response.json() == {
        "detail": (
            f"Repository path '{traversal_repo}' is outside the configured API workspace root "
            f"'{tmp_path.resolve()}'."
        )
    }
    assert database.list_runs() == []
    assert spawn.jobs == []


def test_runs_api__workspace_root_repository_is_allowed(tmp_path: Path) -> None:
    service, database, _store, spawn, _settings_value = _service(tmp_path)

    with TestClient(create_app(service)) as client:
        response = client.post("/runs", json=_create_body(tmp_path))

    assert response.status_code == 202
    run_id = response.json()["run_id"]
    assert database.get_run(run_id) is not None
    assert len(spawn.jobs) == 1


@pytest.mark.parametrize("repo_kind", ["missing", "file"])
def test_runs_api__invalid_repository_returns_400_without_side_effects(
    tmp_path: Path,
    repo_kind: str,
) -> None:
    service, database, _store, spawn, _settings_value = _service(tmp_path)
    repo = tmp_path / "missing"
    if repo_kind == "file":
        repo = tmp_path / "not-a-directory.txt"
        repo.write_text("not a repository", encoding="utf-8")

    with TestClient(create_app(service)) as client:
        response = client.post("/runs", json=_create_body(repo))

    assert response.status_code == 400
    assert "directory" in response.json()["detail"]
    assert database.list_runs() == []
    assert spawn.jobs == []


def test_runs_api__unknown_task_type_is_rejected_without_creating_a_run(
    mini_repo: Path,
    tmp_path: Path,
) -> None:
    service, database, _store, spawn, _settings_value = _service(tmp_path)

    with TestClient(create_app(service)) as client:
        response = client.post("/runs", json=_create_body(mini_repo, task_type="unknown"))

    assert response.status_code == 422
    assert database.list_runs() == []
    assert spawn.jobs == []


def test_runs_api__unknown_run_returns_404_and_openapi_documents_run_routes(
    tmp_path: Path,
) -> None:
    service, _database, _store, _spawn, _settings_value = _service(tmp_path)

    with TestClient(create_app(service)) as client:
        missing_response = client.get("/runs/unknown")
        openapi_response = client.get("/openapi.json")

    assert missing_response.status_code == 404
    assert openapi_response.status_code == 200
    paths = openapi_response.json()["paths"]
    assert set(paths["/runs"]) == {"get", "post"}
    assert set(paths["/runs/{run_id}"]) == {"get"}
