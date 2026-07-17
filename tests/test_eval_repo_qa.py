import json
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError

import eval.run_eval as run_eval_module
from app.schemas.llm_io import LLMMessage, LLMResponse, Role, StopReason, Usage
from app.services.llm_client import ToolSchema
from app.storage.trace_store import TraceStore
from eval.harness import prepare_git_workspace
from eval.run_eval import (
    RepoQaEvalTask,
    RepoQaExpected,
    load_repo_qa_tasks,
    run_repo_qa_suite,
    write_repo_qa_metrics_report,
    write_repo_qa_suite_report,
)

_FIXTURES_ROOT = Path("eval/fixtures")
_FIXTURE_ROOT = _FIXTURES_ROOT / "mini-flask-api"


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


def _response(content: str) -> LLMResponse:
    return LLMResponse(
        message=LLMMessage(role=Role.assistant, content=content),
        stop_reason=StopReason.end_turn,
        usage=Usage(tokens_in=1, tokens_out=1, cost_usd=0.01),
        model="fake-model",
        raw_finish_reason=StopReason.end_turn.value,
    )


def _tool_response(call_id: str, name: str, arguments: dict[str, object]) -> LLMResponse:
    return LLMResponse.model_validate(
        {
            "message": {
                "role": Role.assistant,
                "tool_calls": [{"id": call_id, "name": name, "arguments": arguments}],
            },
            "stop_reason": StopReason.tool_use,
            "usage": {"tokens_in": 1, "tokens_out": 1, "cost_usd": 0.01},
            "model": "fake-model",
            "raw_finish_reason": StopReason.tool_use.value,
        }
    )


def _plan_json() -> str:
    return json.dumps(
        {
            "steps": [
                {
                    "intent": "Inspect the request-validation implementation.",
                    "suggested_tools": ["read_file"],
                    "success_check": "The validation entry point and date failure are identified.",
                }
            ]
        }
    )


def _step_result() -> str:
    return json.dumps(
        {
            "findings": (
                "app/validation.py defines validate_payload and parse_date; parse_date raises "
                "ValueError for an invalid date."
            ),
            "evidence": ["app/validation.py:1"],
        }
    )


def _proceed_verdict() -> str:
    return json.dumps(
        {
            "decision": "proceed",
            "reason": "The validation source directly answers the repository question.",
            "hint": "",
        }
    )


def _report_json(*, suspect_path: str, citation: str) -> str:
    return json.dumps(
        {
            "headline": "Request validation is centralized in app/validation.py",
            "analysis": (
                "Incoming payloads pass through validate_payload. Its parse_date helper rejects "
                "invalid dates by raising ValueError."
            ),
            "confidence": "high",
            "open_questions": [],
            "suspects": [
                {
                    "path": suspect_path,
                    "reason": "The scripted source inspection identified this file.",
                }
            ],
            "citations": [citation],
        }
    )


def _repo_qa_responses(*, suspect_path: str, citation: str, call_id: str) -> list[LLMResponse]:
    return [
        _response(_plan_json()),
        _tool_response(call_id, "read_file", {"path": "app/validation.py"}),
        _response(_step_result()),
        _response(_proceed_verdict()),
        _response(_report_json(suspect_path=suspect_path, citation=citation)),
    ]


def _repo_qa_task() -> RepoQaEvalTask:
    return next(
        task for task in load_repo_qa_tasks(Path("eval/tasks.json")) if task.id == "EV-QA-001"
    )


def _task_data() -> dict[str, object]:
    return {
        "id": "EV-QA-TEST",
        "type": "repo_qa",
        "fixture": "mini-flask-api",
        "prompt": "Where is request validation implemented?",
        "expected": {
            "paths_any": ["app/validation.py"],
            "rubric_keywords": ["validate_payload"],
        },
        "budgets": {"max_steps": 5},
    }


def test_load_repo_qa_tasks__filters_tasks_and_models_are_frozen_with_forbidden_extras(
    tmp_path: Path,
) -> None:
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {"id": "ignored-issue", "type": "bug_localization"},
                    _task_data(),
                    {"id": "ignored-patch", "type": "patch"},
                ]
            }
        ),
        encoding="utf-8",
    )

    tasks = load_repo_qa_tasks(tasks_path)

    assert [task.id for task in tasks] == ["EV-QA-TEST"]
    assert tasks[0].prompt == "Where is request validation implemented?"
    assert tasks[0].expected.paths_any == ["app/validation.py"]
    with pytest.raises(ValidationError, match="frozen"):
        tasks[0].prompt = "Mutation must fail."
    with pytest.raises(ValidationError, match="frozen"):
        tasks[0].expected.paths_any = ["app/routes.py"]

    expected_with_extra = {
        "paths_any": ["app/validation.py"],
        "rubric_keywords": [],
        "unexpected": True,
    }
    with pytest.raises(ValidationError, match="extra_forbidden"):
        RepoQaExpected.model_validate(expected_with_extra)
    with pytest.raises(ValidationError, match="too_short"):
        RepoQaExpected(paths_any=[], rubric_keywords=[])
    with pytest.raises(ValidationError, match="Field required"):
        RepoQaExpected.model_validate({"paths_any": ["app/validation.py"]})

    task_with_extra = _task_data() | {"unexpected": True}
    with pytest.raises(ValidationError, match="extra_forbidden"):
        RepoQaEvalTask.model_validate(task_with_extra)


def test_load_repo_qa_tasks__actual_ev_qa_contracts_are_complete() -> None:
    tasks = load_repo_qa_tasks(Path("eval/tasks.json"))

    assert [task.id for task in tasks] == ["EV-QA-001", "EV-QA-002"]
    assert tasks[0].type == "repo_qa"
    assert tasks[0].fixture == "mini-flask-api"
    assert tasks[0].prompt.startswith("Where are incoming request payloads validated")
    assert tasks[0].expected.paths_any == ["app/validation.py"]
    assert tasks[0].expected.rubric_keywords == [
        "validate_payload",
        "parse_date",
        "ValueError",
    ]
    for task in tasks:
        fixture_root = _FIXTURES_ROOT / task.fixture
        assert fixture_root.is_dir()
        assert all((fixture_root / path).is_file() for path in task.expected.paths_any)


def test_main__routes_repo_qa_type_to_repo_qa_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_repo_qa_cli(args: object) -> None:
        captured["args"] = args

    monkeypatch.setattr(run_eval_module, "_run_repo_qa_cli", fake_run_repo_qa_cli)

    run_eval_module.main(["--type", "repo_qa", "--repeat", "2"])

    args = captured["args"]
    assert args.task_filter == "repo_qa"
    assert args.repeat == 2


def test_repo_qa_eval__scripts_real_fixture_scores_union_and_writes_reports(
    tmp_path: Path,
) -> None:
    client = _ScriptedClient(
        [
            *_repo_qa_responses(
                suspect_path="app/routes.py",
                citation="app/validation.py:1",
                call_id="read-via-citation",
            ),
            *_repo_qa_responses(
                suspect_path="app/validation.py",
                citation="app/routes.py:1",
                call_id="read-via-suspect",
            ),
        ]
    )
    state_dir = tmp_path / "state"

    report = run_repo_qa_suite(
        [_repo_qa_task()],
        _FIXTURES_ROOT,
        client=client,
        repeat=2,
        work_dir=state_dir,
    )
    results_path = write_repo_qa_suite_report(report, tmp_path / "reports")
    metrics_path = write_repo_qa_metrics_report(
        report,
        TraceStore(state_dir / "traces"),
        tmp_path / "reports",
    )

    assert report.task_count == 1
    assert report.run_count == 2
    assert report.path_hit_rate == 1.0
    assert report.rubric_all_rate == 1.0
    assert report.success_rate == 1.0
    assert report.mean_steps == 1.0
    assert client.remaining == 0
    assert [result.task_id for result in report.results] == ["EV-QA-001", "EV-QA-001"]
    assert [result.loop_status for result in report.results] == ["DONE", "DONE"]
    assert [result.candidate_paths for result in report.results] == [
        ["app/routes.py", "app/validation.py"],
        ["app/validation.py", "app/routes.py"],
    ]
    assert all(result.path_hit for result in report.results)
    assert all(result.all_rubric for result in report.results)
    assert all(result.passed for result in report.results)
    assert [result.rubric_hits for result in report.results] == [
        ["validate_payload", "parse_date", "ValueError"],
        ["validate_payload", "parse_date", "ValueError"],
    ]

    persisted = json.loads(results_path.read_text(encoding="utf-8"))
    assert persisted["task_filter"] == "repo_qa"
    assert persisted["success_rate"] == 1.0
    assert persisted["results"][0]["candidate_paths"] == [
        "app/routes.py",
        "app/validation.py",
    ]
    assert persisted["results"][0]["path_hit"] is True
    assert persisted["results"][0]["all_rubric"] is True
    assert persisted["results"][0]["passed"] is True
    markdown = metrics_path.read_text(encoding="utf-8")
    assert "| Task Success Rate | 1.000 |" in markdown
    assert "| repo_qa | 1.000 |" in markdown

    for result in report.results:
        events = TraceStore(state_dir / "traces").read(result.run_id)
        read_calls = [
            event
            for event in events
            if event.kind.value == "tool_call" and event.payload.get("tool_name") == "read_file"
        ]
        assert len(read_calls) == 1
        assert read_calls[0].payload.get("ok") is True


def test_mini_flask_api_fixture__clean_workspace_has_exactly_one_seed_failure(
    tmp_path: Path,
) -> None:
    workspace = prepare_git_workspace(_FIXTURE_ROOT, tmp_path / "workspace")

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=workspace,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    output = f"{completed.stdout}\n{completed.stderr}"
    failed_counts = [int(count) for count in re.findall(r"(?<!\d)(\d+) failed\b", output)]
    assert completed.returncode != 0, output
    assert failed_counts and failed_counts[-1] == 1, output
    assert "FAILED tests/test_routes.py::test_create_todo__empty_title_returns_422" in output, (
        output
    )
