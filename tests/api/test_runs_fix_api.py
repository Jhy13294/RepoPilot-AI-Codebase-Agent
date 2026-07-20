import json
import multiprocessing
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from threading import Thread
from time import monotonic, sleep
from typing import cast

import pytest
from fastapi.testclient import TestClient
from pydantic import JsonValue

from app.agent.state import AgentState, RunStatus
from app.api.app import create_app
from app.api.schemas import CreateRunRequest
from app.api.service import RunService
from app.config import Settings
from app.schemas.llm_io import LLMMessage, LLMResponse, Role, StopReason, ToolCall, Usage
from app.schemas.trace import TraceEvent, TraceEventKind
from app.services.llm_client import LLMClient, ToolSchema
from app.storage.db import Database
from app.storage.trace_store import TraceStore

_ORIGINAL = b"old\n"
_UPDATED = b"new\n"
_DIFF = "--- a/tracked.txt\n+++ b/tracked.txt\n@@ -1 +1 @@\n-old\n+new\n"
_THREAD_TIMEOUT_S = 10.0
_POLL_INTERVAL_S = 0.01
_SAFE_TOOL_CALL_FIELDS = {
    "seq",
    "ts",
    "tool_name",
    "ok",
    "error_type",
    "latency_ms",
}


class _ScriptedClient:
    def __init__(self, responses: Sequence[LLMResponse]) -> None:
        self._responses = list(responses)

    @property
    def remaining(self) -> int:
        return len(self._responses)

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


class _ThreadSpawn:
    def __init__(self) -> None:
        self.threads: list[Thread] = []
        self.errors: list[BaseException] = []

    def __call__(self, job: Callable[[], None]) -> None:
        def target() -> None:
            try:
                job()
            except BaseException as exc:
                self.errors.append(exc)

        thread = Thread(target=target, name="fix-api-test-worker", daemon=True)
        self.threads.append(thread)
        thread.start()

    def assert_finished(self) -> None:
        for thread in self.threads:
            thread.join(timeout=_THREAD_TIMEOUT_S)
        assert self.errors == []
        assert all(not thread.is_alive() for thread in self.threads)


class _RecordingDatabase(Database):
    def __init__(self, db_path: Path) -> None:
        super().__init__(db_path)
        self.saved_statuses: list[RunStatus] = []

    def save_state(self, state: AgentState) -> None:
        self.saved_statuses.append(state.status)
        super().save_state(state)


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
        test_command="pytest -q",
        test_timeout_s=30,
    )


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
        usage=Usage(tokens_in=1, tokens_out=1, cost_usd=None),
        model="scripted-model",
        raw_finish_reason=stop_reason.value,
    )


def _tool_call(call_id: str, name: str, arguments: dict[str, object]) -> ToolCall:
    return ToolCall(
        id=call_id,
        name=name,
        arguments=cast(dict[str, JsonValue], arguments),
    )


def _plan(*, tools: list[str], intent: str, success_check: str) -> LLMResponse:
    return _response(
        json.dumps(
            {
                "steps": [
                    {
                        "intent": intent,
                        "suggested_tools": tools,
                        "success_check": success_check,
                    }
                ]
            }
        )
    )


def _branch_call() -> LLMResponse:
    return _response(
        stop_reason=StopReason.tool_use,
        tool_calls=[
            _tool_call(
                "create-branch",
                "git_create_branch",
                {"rationale": "Create the isolated work branch."},
            )
        ],
    )


def _apply_call() -> LLMResponse:
    return _response(
        stop_reason=StopReason.tool_use,
        tool_calls=[
            _tool_call(
                "apply-patch",
                "apply_patch",
                {
                    "diff": _DIFF,
                    "rationale": "Apply the reviewed correction.",
                },
            )
        ],
    )


def _read_call() -> LLMResponse:
    return _response(
        stop_reason=StopReason.tool_use,
        tool_calls=[_tool_call("read-unchanged", "read_file", {"path": "tracked.txt"})],
    )


def _step_result(findings: str) -> LLMResponse:
    return _response(json.dumps({"findings": findings, "evidence": [findings]}))


def _verdict(reason: str) -> LLMResponse:
    return _response(
        json.dumps(
            {
                "decision": "proceed",
                "reason": reason,
                "hint": "",
            }
        )
    )


def _report(headline: str, analysis: str) -> LLMResponse:
    return _response(
        json.dumps(
            {
                "headline": headline,
                "analysis": analysis,
                "confidence": "high",
                "open_questions": [],
                "suspects": [],
                "citations": [],
            }
        )
    )


def _approved_client() -> _ScriptedClient:
    return _ScriptedClient(
        [
            _plan(
                tools=["git_create_branch", "apply_patch"],
                intent="Create the work branch and apply the reviewed patch.",
                success_check="The approved patch is present on the run work branch.",
            ),
            _branch_call(),
            _apply_call(),
            _step_result("The approved patch was applied on the run work branch."),
            _verdict("The tool evidence proves the approved patch was applied."),
            _report(
                "Patch run completed",
                "The approved patch was applied on the isolated work branch.",
            ),
        ]
    )


def _denied_client() -> _ScriptedClient:
    return _ScriptedClient(
        [
            _plan(
                tools=["git_create_branch", "apply_patch"],
                intent="Create the work branch and apply the reviewed patch.",
                success_check="The approved patch is present on the run work branch.",
            ),
            _branch_call(),
            _apply_call(),
            _plan(
                tools=["read_file"],
                intent="Read the unchanged file and report the review outcome.",
                success_check="The unchanged file is grounded in tool evidence.",
            ),
            _read_call(),
            _step_result("tracked.txt remains unchanged after the denial."),
            _verdict("The denial was respected and the unchanged file was verified."),
            _report(
                "Patch was not applied",
                "The reviewer denied the change and tracked.txt remained unchanged.",
            ),
        ]
    )


def _branch_only_client() -> _ScriptedClient:
    return _ScriptedClient(
        [
            _plan(
                tools=["git_create_branch"],
                intent="Create the isolated work branch.",
                success_check="The run work branch is current.",
            ),
            _branch_call(),
            _step_result("The run work branch is current."),
            _verdict("The branch tool evidence satisfies the success check."),
            _report(
                "Work branch created",
                "The isolated run work branch was created after approval.",
            ),
        ]
    )


def _question_client() -> _ScriptedClient:
    return _ScriptedClient(
        [
            _plan(
                tools=["read_file"],
                intent="Read tracked.txt.",
                success_check="The current value is grounded in tool evidence.",
            ),
            _read_call(),
            _step_result("tracked.txt contains old."),
            _verdict("The read_file evidence proves the current value."),
            _report(
                "Current value found",
                "The read-only run found the current tracked.txt value.",
            ),
        ]
    )


def _run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        check=False,
        shell=False,
        timeout=10,
    )


def _require_git_success(repo: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    completed = _run_git(repo, *args)
    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
    return completed


def _init_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _require_git_success(repo, "init", "--initial-branch=main", "--quiet")
    _require_git_success(repo, "config", "core.autocrlf", "false")
    _require_git_success(repo, "config", "user.name", "RepoPilot Tests")
    _require_git_success(repo, "config", "user.email", "tests@repopilot.local")
    target = repo / "tracked.txt"
    target.write_bytes(_ORIGINAL)
    _require_git_success(repo, "add", "tracked.txt")
    _require_git_success(
        repo,
        "commit",
        "--quiet",
        "--no-gpg-sign",
        "-m",
        "test fixture",
    )
    return repo, target


def _create_body(repo: Path) -> dict[str, object]:
    return {
        "task_type": "fix",
        "prompt": "Replace old with new in tracked.txt.",
        "repo": str(repo),
        "max_steps": 4,
    }


def _wait_for_approval(
    client: TestClient,
    run_id: str,
    tool_name: str,
) -> tuple[dict[str, object], dict[str, object]]:
    deadline = monotonic() + _THREAD_TIMEOUT_S
    last_run: object = None
    last_approvals: object = None
    while monotonic() < deadline:
        run_response = client.get(f"/runs/{run_id}")
        approvals_response = client.get("/approvals", params={"run_id": run_id})
        assert run_response.status_code == 200
        assert approvals_response.status_code == 200
        last_run = run_response.json()
        last_approvals = approvals_response.json()
        if (
            isinstance(last_run, dict)
            and last_run.get("status") == RunStatus.AWAITING_APPROVAL.value
            and isinstance(last_approvals, list)
            and len(last_approvals) == 1
            and isinstance(last_approvals[0], dict)
            and last_approvals[0].get("tool_name") == tool_name
        ):
            return last_run, last_approvals[0]
        sleep(_POLL_INTERVAL_S)
    raise AssertionError(
        f"Run {run_id!r} did not park for {tool_name!r}; "
        f"last_run={last_run!r}, last_approvals={last_approvals!r}"
    )


def _wait_for_terminal(client: TestClient, run_id: str) -> dict[str, object]:
    deadline = monotonic() + _THREAD_TIMEOUT_S
    last_run: object = None
    while monotonic() < deadline:
        response = client.get(f"/runs/{run_id}")
        assert response.status_code == 200
        last_run = response.json()
        if isinstance(last_run, dict) and last_run.get("status") in {
            status.value for status in (RunStatus.DONE, RunStatus.FAILED, RunStatus.CANCELLED)
        }:
            return last_run
        sleep(_POLL_INTERVAL_S)
    raise AssertionError(f"Run {run_id!r} did not reach a terminal state; last_run={last_run!r}")


def _approve(client: TestClient, request_id: object, *, note: str) -> dict[str, object]:
    assert isinstance(request_id, str)
    response = client.post(
        f"/approvals/{request_id}",
        json={"decision": "approve", "note": note},
    )
    assert response.status_code == 200
    decided = response.json()
    assert isinstance(decided, dict)
    return decided


def _assert_audit_pair(
    events: Sequence[TraceEvent],
    *,
    tool_name: str,
    decision: str,
    reason: str,
) -> None:
    request_indexes = [
        index
        for index, event in enumerate(events)
        if event.kind is TraceEventKind.approval_request
        and event.payload.get("tool_name") == tool_name
    ]
    decision_events = [
        event
        for event in events
        if event.kind is TraceEventKind.approval_decision
        and event.payload.get("tool_name") == tool_name
    ]
    assert len(request_indexes) == 1
    assert len(decision_events) == 1
    [request_index] = request_indexes
    request_event = events[request_index]
    decision_event = events[request_index + 1]
    assert "diff" not in request_event.payload
    assert decision_event.kind is TraceEventKind.approval_decision
    assert decision_event.run_id == request_event.run_id
    assert decision_event.payload == {
        "tool_name": tool_name,
        "risk_level": "high",
        "decision": decision,
        "actor": "human",
        "reason": reason,
    }


def _park_fix_run_process(
    db_path: Path,
    trace_dir: Path,
    repo: Path,
    run_id_path: Path,
) -> None:
    settings = Settings(
        _env_file=None,
        OPENAI_API_KEY="sk-test-api",
        llm_provider="openai_compatible",
        model="test-model",
        db_path=db_path,
        trace_dir=trace_dir,
        max_steps=4,
        max_replans=2,
        max_fix_cycles=2,
    )
    database = Database(db_path)
    scripted = _ScriptedClient(
        [
            _plan(
                tools=["git_create_branch"],
                intent="Create the isolated work branch.",
                success_check="The run work branch is current.",
            ),
            _branch_call(),
        ]
    )
    service = RunService(
        settings,
        database=database,
        store=TraceStore(trace_dir),
        client_factory=lambda _settings: scripted,
    )
    created = service.create_run(
        CreateRunRequest(
            task_type="fix",
            prompt="Create the isolated work branch.",
            repo=str(repo),
            max_steps=2,
        )
    )
    deadline = monotonic() + _THREAD_TIMEOUT_S
    while monotonic() < deadline:
        run = database.get_run(created.run_id)
        pending = service.list_pending_approvals(created.run_id)
        if run is not None and run.status is RunStatus.AWAITING_APPROVAL and len(pending) == 1:
            temporary_path = run_id_path.with_suffix(".tmp")
            temporary_path.write_text(created.run_id, encoding="utf-8")
            temporary_path.replace(run_id_path)
            return
        sleep(_POLL_INTERVAL_S)
    temporary_path = run_id_path.with_suffix(".tmp")
    temporary_path.write_text(f"ERROR:{created.run_id}", encoding="utf-8")
    temporary_path.replace(run_id_path)


def test_runs_fix_api__approve_round_trip_applies_patch_and_preserves_safe_view(
    tmp_path: Path,
) -> None:
    repo, target = _init_repo(tmp_path)
    settings = _settings(tmp_path)
    database = Database(settings.db_path)
    store = TraceStore(settings.trace_dir)
    spawn = _ThreadSpawn()
    scripted = _approved_client()
    service = RunService(
        settings,
        database=database,
        store=store,
        spawn=spawn,
        client_factory=lambda _settings: scripted,
    )

    with TestClient(create_app(service)) as client:
        create_response = client.post("/runs", json=_create_body(repo))
        assert create_response.status_code == 202
        run_id = create_response.json()["run_id"]

        branch_run, branch_request = _wait_for_approval(client, run_id, "git_create_branch")
        assert branch_run["status"] == RunStatus.AWAITING_APPROVAL.value
        _approve(client, branch_request["request_id"], note="Create the isolated branch.")

        parked_run, patch_request = _wait_for_approval(client, run_id, "apply_patch")
        assert patch_request["run_id"] == run_id
        assert patch_request["risk_level"] == "high"
        assert patch_request["status"] == "pending"
        assert patch_request["args"] == {
            "diff": _DIFF,
            "rationale": "Apply the reviewed correction.",
        }
        assert all(set(call) == _SAFE_TOOL_CALL_FIELDS for call in parked_run["tool_calls"])
        assert _DIFF not in json.dumps(parked_run["tool_calls"])

        approved = _approve(
            client,
            patch_request["request_id"],
            note="The patch was reviewed and approved.",
        )
        assert approved["status"] == "approved"
        terminal = _wait_for_terminal(client, run_id)

    spawn.assert_finished()
    assert terminal["status"] == RunStatus.DONE.value
    assert target.read_bytes() == _UPDATED
    branch = _require_git_success(repo, "branch", "--show-current").stdout.decode().strip()
    assert branch == f"repopilot/fix-{run_id}"
    assert scripted.remaining == 0
    assert [call["tool_name"] for call in terminal["tool_calls"]] == [
        "git_create_branch",
        "apply_patch",
    ]
    assert all(set(call) == _SAFE_TOOL_CALL_FIELDS for call in terminal["tool_calls"])
    _assert_audit_pair(
        store.read(run_id),
        tool_name="apply_patch",
        decision="approved",
        reason="The patch was reviewed and approved.",
    )


def test_runs_fix_api__denial_replans_without_writing_and_preserves_note(
    tmp_path: Path,
) -> None:
    repo, target = _init_repo(tmp_path)
    settings = _settings(tmp_path)
    database = Database(settings.db_path)
    store = TraceStore(settings.trace_dir)
    spawn = _ThreadSpawn()
    scripted = _denied_client()
    service = RunService(
        settings,
        database=database,
        store=store,
        spawn=spawn,
        client_factory=lambda _settings: scripted,
    )
    denial_note = "Add a regression test before applying this diff."

    with TestClient(create_app(service)) as client:
        create_response = client.post("/runs", json=_create_body(repo))
        assert create_response.status_code == 202
        run_id = create_response.json()["run_id"]

        _branch_run, branch_request = _wait_for_approval(client, run_id, "git_create_branch")
        _approve(client, branch_request["request_id"], note="Create the isolated branch.")
        _patch_run, patch_request = _wait_for_approval(client, run_id, "apply_patch")
        assert target.read_bytes() == _ORIGINAL
        deny_response = client.post(
            f"/approvals/{patch_request['request_id']}",
            json={"decision": "deny", "note": denial_note},
        )
        assert deny_response.status_code == 200
        assert deny_response.json()["status"] == "denied"
        terminal = _wait_for_terminal(client, run_id)

    spawn.assert_finished()
    assert terminal["status"] == RunStatus.DONE.value
    assert terminal["replans_used"] == 1
    assert target.read_bytes() == _ORIGINAL
    assert scripted.remaining == 0
    events = store.read(run_id)
    assert any(event.kind is TraceEventKind.replan for event in events)
    _assert_audit_pair(
        events,
        tool_name="apply_patch",
        decision="denied",
        reason=denial_note,
    )
    [denied_call] = [call for call in terminal["tool_calls"] if call["tool_name"] == "apply_patch"]
    assert denied_call["ok"] is False
    assert denied_call["error_type"] == "ApprovalDeniedError"


def test_runs_fix_api__background_construction_exception_reaches_failed(
    tmp_path: Path,
) -> None:
    repo, target = _init_repo(tmp_path)
    settings = _settings(tmp_path)
    database = _RecordingDatabase(settings.db_path)
    store = TraceStore(settings.trace_dir)
    spawn = _ThreadSpawn()

    def exploding_client_factory(_settings: Settings) -> LLMClient:
        raise RuntimeError("client construction exploded")

    service = RunService(
        settings,
        database=database,
        store=store,
        spawn=spawn,
        client_factory=exploding_client_factory,
    )

    with TestClient(create_app(service)) as client:
        create_response = client.post("/runs", json=_create_body(repo))
        assert create_response.status_code == 202
        run_id = create_response.json()["run_id"]
        terminal = _wait_for_terminal(client, run_id)

    spawn.assert_finished()
    assert terminal["status"] == RunStatus.FAILED.value
    assert terminal["summary"] is None
    assert target.read_bytes() == _ORIGINAL
    assert database.saved_statuses == [
        RunStatus.PLANNING,
        RunStatus.REPORTING,
        RunStatus.FAILED,
    ]
    error_events = [event for event in store.read(run_id) if event.kind is TraceEventKind.error]
    assert len(error_events) == 1
    assert error_events[0].payload["reason"] == "background_exception"
    assert "client construction exploded" in str(error_events[0].payload["message"])


def test_runs_fix_api__trace_failure_does_not_block_background_terminalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    settings = _settings(tmp_path)
    database = _RecordingDatabase(settings.db_path)
    store = TraceStore(settings.trace_dir)
    spawn = _ThreadSpawn()

    def fail_append(*_args: object, **_kwargs: object) -> TraceEvent:
        raise OSError("trace append failed")

    def exploding_client_factory(_settings: Settings) -> LLMClient:
        raise RuntimeError("client construction exploded")

    monkeypatch.setattr(store, "append", fail_append)
    service = RunService(
        settings,
        database=database,
        store=store,
        spawn=spawn,
        client_factory=exploding_client_factory,
    )

    with TestClient(create_app(service)) as client:
        create_response = client.post("/runs", json=_create_body(repo))
        assert create_response.status_code == 202
        run_id = create_response.json()["run_id"]
        terminal = _wait_for_terminal(client, run_id)

    spawn.assert_finished()
    assert terminal["status"] == RunStatus.FAILED.value
    assert database.saved_statuses == [
        RunStatus.PLANNING,
        RunStatus.REPORTING,
        RunStatus.FAILED,
    ]
    assert store.read(run_id) == []


def test_runs_fix_api__restart_preserves_pending_state_but_cannot_resume_worker(
    tmp_path: Path,
) -> None:
    repo, target = _init_repo(tmp_path)
    db_path = tmp_path / "runs.sqlite"
    trace_dir = tmp_path / "traces"
    run_id_path = tmp_path / "parked-run-id.txt"
    process = multiprocessing.get_context("spawn").Process(
        target=_park_fix_run_process,
        args=(db_path, trace_dir, repo, run_id_path),
        name="fix-api-restart-boundary",
    )
    process.start()
    observer: Database | None = None
    try:
        deadline = monotonic() + _THREAD_TIMEOUT_S
        while monotonic() < deadline and not run_id_path.exists():
            sleep(_POLL_INTERVAL_S)
        assert run_id_path.exists(), (
            f"Child did not persist a parked run; exitcode={process.exitcode}"
        )
        run_id = run_id_path.read_text(encoding="utf-8")
        assert not run_id.startswith("ERROR:"), run_id
        observer = Database(db_path)
        run_before_restart = observer.get_run(run_id)
        pending_before_restart = observer.list_pending_approvals(run_id)
        assert run_before_restart is not None
        assert run_before_restart.status is RunStatus.AWAITING_APPROVAL
        assert len(pending_before_restart) == 1
    finally:
        if observer is not None:
            observer.engine.dispose()
        if process.is_alive():
            process.terminate()
        process.join(timeout=_THREAD_TIMEOUT_S)
        if process.is_alive():
            process.kill()
            process.join(timeout=_THREAD_TIMEOUT_S)
        assert not process.is_alive()
        process.close()

    settings = _settings(tmp_path)
    restarted_database = Database(db_path)
    restarted_store = TraceStore(trace_dir)

    def unexpected_client_factory(_settings: Settings) -> LLMClient:
        raise AssertionError("Restarted reads and decisions must not construct a client.")

    restarted_service = RunService(
        settings,
        database=restarted_database,
        store=restarted_store,
        client_factory=unexpected_client_factory,
    )
    try:
        with TestClient(create_app(restarted_service)) as client:
            run_response = client.get(f"/runs/{run_id}")
            approvals_response = client.get("/approvals", params={"run_id": run_id})
            assert run_response.status_code == 200
            assert run_response.json()["status"] == RunStatus.AWAITING_APPROVAL.value
            assert approvals_response.status_code == 200
            [pending_after_restart] = approvals_response.json()
            assert pending_after_restart["request_id"] == (pending_before_restart[0].request_id)
            assert pending_after_restart["status"] == "pending"

            decision_response = client.post(
                f"/approvals/{pending_after_restart['request_id']}",
                json={"decision": "approve", "note": "Persist the decision only."},
            )
            assert decision_response.status_code == 200
            assert client.get(f"/runs/{run_id}").json()["status"] == (
                RunStatus.AWAITING_APPROVAL.value
            )
    finally:
        restarted_service.close()

    assert target.read_bytes() == _ORIGINAL
    branch = _require_git_success(repo, "branch", "--show-current").stdout.decode().strip()
    assert branch == "main"
    restarted_events = restarted_store.read(run_id)
    assert [event.kind for event in restarted_events] == [
        TraceEventKind.plan,
        TraceEventKind.approval_request,
    ]


def test_run_service__parked_fix_does_not_starve_a_read_only_run(tmp_path: Path) -> None:
    repo, target = _init_repo(tmp_path)
    settings = _settings(tmp_path)
    fix_client = _branch_only_client()
    question_client = _question_client()
    scripted_clients = [fix_client, question_client]

    def client_factory(_settings: Settings) -> LLMClient:
        if not scripted_clients:
            raise AssertionError("No scripted run client remains.")
        return scripted_clients.pop(0)

    service = RunService(settings, client_factory=client_factory)
    assert service._executor is not None
    assert 1 < service._executor._max_workers <= 8
    fix_run_id: str | None = None
    fix_finished = False
    try:
        with TestClient(create_app(service)) as client:
            try:
                fix_response = client.post("/runs", json=_create_body(repo))
                assert fix_response.status_code == 202
                fix_run_id = fix_response.json()["run_id"]
                _fix_run, branch_request = _wait_for_approval(
                    client,
                    fix_run_id,
                    "git_create_branch",
                )

                question_response = client.post(
                    "/runs",
                    json={
                        "task_type": "question",
                        "prompt": "What is the current tracked.txt value?",
                        "repo": str(repo),
                        "max_steps": 2,
                    },
                )
                assert question_response.status_code == 202
                question_run_id = question_response.json()["run_id"]
                question_terminal = _wait_for_terminal(client, question_run_id)

                assert question_terminal["status"] == RunStatus.DONE.value
                assert client.get(f"/runs/{fix_run_id}").json()["status"] == (
                    RunStatus.AWAITING_APPROVAL.value
                )
                assert target.read_bytes() == _ORIGINAL
                assert question_client.remaining == 0

                _approve(client, branch_request["request_id"], note="Create the branch.")
                fix_terminal = _wait_for_terminal(client, fix_run_id)
                assert fix_terminal["status"] == RunStatus.DONE.value
                fix_finished = True
            finally:
                if fix_run_id is not None and not fix_finished:
                    pending_response = client.get(
                        "/approvals",
                        params={"run_id": fix_run_id},
                    )
                    if pending_response.status_code == 200:
                        for pending in pending_response.json():
                            _approve(
                                client,
                                pending["request_id"],
                                note="Release the parked test worker.",
                            )
                    _wait_for_terminal(client, fix_run_id)
    finally:
        service.close()

    assert fix_client.remaining == 0
    assert scripted_clients == []
