import json
import shlex
import sys
from collections.abc import Sequence
from difflib import unified_diff
from pathlib import Path

import pytest
from pydantic import ValidationError

import eval.run_eval as run_eval_module
from app.cli import _build_fix_registry
from app.safety.path_jail import PathJail
from app.schemas.llm_io import LLMMessage, LLMResponse, Role, StopReason, Usage
from app.schemas.tool_io import ErrorType
from app.services.llm_client import ToolSchema
from app.storage.trace_store import RegistryTraceSink, TraceStore
from app.tools.base import ToolContext
from app.tools.registry import ToolRegistry
from eval.harness import (
    AutoApprovalGate,
    FaultInjection,
    build_recovery_registry,
    prepare_git_workspace,
)
from eval.run_eval import (
    RecoveryEvalTask,
    RecoveryExpected,
    RecoveryInjection,
    load_recovery_tasks,
    run_recovery_suite,
    write_recovery_metrics_report,
    write_recovery_suite_report,
)

_FIXTURES_ROOT = Path("eval/fixtures")
_CALCULATOR_ROOT = _FIXTURES_ROOT / "buggy-calculator"
_FLASK_ROOT = _FIXTURES_ROOT / "mini-flask-api"
_FAULT_MARKER = "  # repopilot recovery fault injection"


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
        usage=Usage(tokens_in=1, tokens_out=1, cost_usd=None),
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
            "usage": {"tokens_in": 1, "tokens_out": 1, "cost_usd": None},
            "model": "fake-model",
            "raw_finish_reason": StopReason.tool_use.value,
        }
    )


def _plan_json() -> str:
    return json.dumps(
        {
            "steps": [
                {
                    "intent": "Apply the fix, recover from any tool failure, and verify tests.",
                    "suggested_tools": [
                        "git_create_branch",
                        "read_file",
                        "propose_patch",
                        "apply_patch",
                        "run_tests",
                    ],
                    "success_check": "run_tests reports zero failures and errors.",
                }
            ]
        }
    )


def _step_result() -> str:
    return json.dumps(
        {
            "findings": "The injected failure was handled and the real tests passed.",
            "evidence": ["run_tests reported zero failures and errors."],
        }
    )


def _proceed_verdict() -> str:
    return json.dumps(
        {
            "decision": "proceed",
            "reason": "The scripted recovery path completed with green test evidence.",
            "hint": "",
        }
    )


def _report_json() -> str:
    return json.dumps(
        {
            "headline": "Recovery run completed",
            "analysis": "The tool failure was observed, retried, and followed by green tests.",
            "confidence": "high",
            "open_questions": [],
            "suspects": [],
            "citations": [],
        }
    )


def _fixture_text(root: Path, path: str) -> str:
    return (root / path).read_bytes().decode("utf-8")


def _replace_block(content: str, old: str, new: str) -> str:
    newline = "\r\n" if "\r\n" in content else "\n"
    native_old = old.replace("\n", newline)
    native_new = new.replace("\n", newline)
    assert native_old in content
    return content.replace(native_old, native_new, 1)


def _render_diff(path: str, old_content: str, new_content: str) -> str:
    return "".join(
        unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=3,
            lineterm="\n",
        )
    )


def _pytest_command() -> str:
    return shlex.join((Path(sys.executable).as_posix(), "-m", "pytest", "-q"))


def _tool_names(registry: ToolRegistry) -> list[str]:
    schemas = registry.to_llm_schema()
    return [str(schema["function"]["name"]) for schema in schemas]


def _task(task_id: str) -> RecoveryEvalTask:
    return next(task for task in load_recovery_tasks(Path("eval/tasks.json")) if task.id == task_id)


def test_fault_injection_model__is_frozen_forbids_extra_and_defaults_once() -> None:
    injection = FaultInjection(kind="tool_timeout", target="run_tests")

    assert injection.times == 1
    with pytest.raises(ValidationError, match="frozen"):
        injection.times = 2
    with pytest.raises(ValidationError, match="extra_forbidden"):
        FaultInjection.model_validate(
            {
                "kind": "tool_timeout",
                "target": "run_tests",
                "times": 1,
                "unexpected": True,
            }
        )
    with pytest.raises(ValidationError, match="greater_than_equal"):
        FaultInjection(kind="tool_timeout", target="run_tests", times=0)
    with pytest.raises(ValidationError, match="literal_error"):
        FaultInjection.model_validate({"kind": "network_error", "target": "run_tests"})


def test_build_recovery_registry__without_injection_matches_fix_registry_tools() -> None:
    gate = AutoApprovalGate()
    recovery_registry = build_recovery_registry(approval_gate=gate)
    fix_registry = _build_fix_registry(approval_gate=gate)

    assert _tool_names(recovery_registry) == _tool_names(fix_registry)


def test_fault_injecting_registry__timeout_is_gated_then_real_retry_runs(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "test_sample.py").write_text("def test_passes():\n    assert True\n", encoding="utf-8")
    store = TraceStore(tmp_path / "traces")
    registry = build_recovery_registry(
        FaultInjection(kind="tool_timeout", target="run_tests"),
        trace_sink=RegistryTraceSink(store),
        approval_gate=AutoApprovalGate(),
        test_command=_pytest_command(),
    )
    context = ToolContext(run_id="timeout-red-face", jail=PathJail(repo))

    first = registry.dispatch(
        "run_tests",
        {"rationale": "First verification attempt."},
        context,
    )
    second = registry.dispatch(
        "run_tests",
        {"rationale": "Retry verification after the timeout."},
        context,
    )

    assert first.ok is False
    assert first.error is not None
    assert first.error.type is ErrorType.ToolTimeoutError
    assert second.ok is True
    events = store.read(context.run_id)
    approvals = [
        event
        for event in events
        if event.kind.value == "approval_decision" and event.payload.get("tool_name") == "run_tests"
    ]
    calls = [
        event
        for event in events
        if event.kind.value == "tool_call" and event.payload.get("tool_name") == "run_tests"
    ]
    assert len(approvals) == 2
    assert [event.payload.get("decision") for event in approvals] == ["approved", "approved"]
    assert [event.payload.get("ok") for event in calls] == [False, True]
    assert [event.payload.get("error_type") for event in calls] == [
        "ToolTimeoutError",
        None,
    ]
    assert all(approval.seq < call.seq for approval, call in zip(approvals, calls, strict=True))


def test_fault_injecting_registry__patch_conflict_mutates_then_real_apply_fails(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    original = "def value():\n    return 1\n"
    corrected = "def value():\n    return 2\n"
    (fixture / "sample.py").write_text(original, encoding="utf-8")
    workspace = prepare_git_workspace(fixture, tmp_path / "workspace")
    store = TraceStore(tmp_path / "traces")
    registry = build_recovery_registry(
        FaultInjection(kind="patch_conflict", target="sample.py"),
        trace_sink=RegistryTraceSink(store),
        approval_gate=AutoApprovalGate(),
    )
    context = ToolContext(run_id="patch-red-face", jail=PathJail(workspace))
    branch = registry.dispatch(
        "git_create_branch",
        {"rationale": "Create the isolated recovery branch."},
        context,
    )
    before = (workspace / "sample.py").read_bytes()

    applied = registry.dispatch(
        "apply_patch",
        {
            "diff": _render_diff("sample.py", original, corrected),
            "rationale": "Apply the scripted correction.",
        },
        context,
    )

    assert branch.ok is True
    assert applied.ok is False
    assert applied.error is not None
    assert applied.error.type is ErrorType.PatchApplyError
    assert applied.error.details is not None
    assert applied.error.details.get("reason") == "check_failed"
    after = (workspace / "sample.py").read_bytes()
    assert after != before
    assert _FAULT_MARKER.encode() in after
    events = store.read(context.run_id)
    apply_approval = next(
        event
        for event in events
        if event.kind.value == "approval_decision"
        and event.payload.get("tool_name") == "apply_patch"
    )
    apply_call = next(
        event
        for event in events
        if event.kind.value == "tool_call" and event.payload.get("tool_name") == "apply_patch"
    )
    assert apply_approval.payload.get("decision") == "approved"
    assert apply_approval.seq < apply_call.seq
    assert apply_call.payload.get("ok") is False
    assert apply_call.payload.get("error_type") == "PatchApplyError"


def test_load_recovery_tasks__filters_models_and_applies_defaults(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {"id": "ignored-patch", "type": "patch"},
                    {
                        "id": "EV-REC-TEST",
                        "type": "recovery",
                        "fixture": "buggy-calculator",
                        "issue": "Recover from the injected failure.",
                        "inject": {"kind": "tool_timeout", "target": "run_tests"},
                        "expected": {
                            "tests_green": True,
                            "recovered_from": "ToolTimeoutError",
                        },
                    },
                    {"id": "ignored-question", "type": "repo_qa"},
                ]
            }
        ),
        encoding="utf-8",
    )

    tasks = load_recovery_tasks(tasks_path)

    assert [task.id for task in tasks] == ["EV-REC-TEST"]
    assert tasks[0].inject == RecoveryInjection(kind="tool_timeout", target="run_tests")
    assert tasks[0].expected.test_command == "pytest -q"
    with pytest.raises(ValidationError, match="frozen"):
        tasks[0].issue = "Mutation must fail."
    with pytest.raises(ValidationError, match="extra_forbidden"):
        RecoveryExpected.model_validate(
            {
                "tests_green": True,
                "recovered_from": "ToolTimeoutError",
                "unexpected": True,
            }
        )
    with pytest.raises(ValidationError, match="recovered_from"):
        RecoveryExpected(tests_green=True, recovered_from=" ")


def test_load_recovery_tasks__actual_ev_rec_contracts_are_complete() -> None:
    tasks = {task.id: task for task in load_recovery_tasks(Path("eval/tasks.json"))}

    assert set(tasks) == {"EV-REC-001", "EV-REC-002"}
    patch_conflict = tasks["EV-REC-001"]
    assert patch_conflict.fixture == "buggy-calculator"
    assert patch_conflict.inject == RecoveryInjection(
        kind="patch_conflict",
        target="calculator/ops.py",
    )
    assert patch_conflict.expected.recovered_from == "PatchApplyError"
    timeout = tasks["EV-REC-002"]
    assert timeout.fixture == "mini-flask-api"
    assert timeout.inject == RecoveryInjection(
        kind="tool_timeout",
        target="run_tests",
    )
    assert timeout.expected.recovered_from == "ToolTimeoutError"
    assert all(task.expected.test_command == "pytest -q" for task in tasks.values())
    assert all((_FIXTURES_ROOT / task.fixture).is_dir() for task in tasks.values())


def test_main__routes_recovery_type_to_recovery_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_recovery_cli(args: object) -> None:
        captured["args"] = args

    monkeypatch.setattr(run_eval_module, "_run_recovery_cli", fake_run_recovery_cli)

    run_eval_module.main(["--type", "recovery", "--repeat", "2"])

    args = captured["args"]
    assert args.task_filter == "recovery"
    assert args.repeat == 2


def _calculator_recovery_responses() -> list[LLMResponse]:
    ops_path = "calculator/ops.py"
    format_path = "calculator/format.py"
    stats_path = "calculator/stats.py"
    original_ops = _fixture_text(_CALCULATOR_ROOT, ops_path)
    corrected_ops = _replace_block(
        original_ops,
        ("    if (left < 0) == (right < 0):\n        return -quotient\n\n    return quotient\n"),
        ("    if (left < 0) == (right < 0):\n        return quotient\n\n    return -quotient\n"),
    )
    disturbed_ops = _replace_block(
        original_ops,
        "        return -quotient\n",
        f"        return -quotient{_FAULT_MARKER}\n",
    )
    corrected_disturbed_ops = _replace_block(
        disturbed_ops,
        (
            f"    if (left < 0) == (right < 0):\n"
            f"        return -quotient{_FAULT_MARKER}\n\n"
            "    return quotient\n"
        ),
        (
            f"    if (left < 0) == (right < 0):\n"
            f"        return quotient{_FAULT_MARKER}\n\n"
            "    return -quotient\n"
        ),
    )
    original_format = _fixture_text(_CALCULATOR_ROOT, format_path)
    corrected_format = _replace_block(
        original_format,
        '    return f"{value:g}%"\n',
        '    return f"{value * 100:g}%"\n',
    )
    original_stats = _fixture_text(_CALCULATOR_ROOT, stats_path)
    corrected_stats = _replace_block(
        original_stats,
        "        return float(ordered[middle])\n",
        "        return float((ordered[middle - 1] + ordered[middle]) / 2)\n",
    )
    first_diff = _render_diff(ops_path, original_ops, corrected_ops)
    recovered_diff = "".join(
        [
            _render_diff(ops_path, disturbed_ops, corrected_disturbed_ops),
            _render_diff(format_path, original_format, corrected_format),
            _render_diff(stats_path, original_stats, corrected_stats),
        ]
    )
    return [
        _response(_plan_json()),
        _tool_response(
            "calculator-branch",
            "git_create_branch",
            {"rationale": "Create the calculator recovery branch."},
        ),
        _tool_response(
            "calculator-propose-first",
            "propose_patch",
            {"path": ops_path, "new_content": corrected_ops},
        ),
        _tool_response(
            "calculator-apply-conflict",
            "apply_patch",
            {
                "diff": first_diff,
                "rationale": "Apply the first calculator correction.",
            },
        ),
        _tool_response(
            "calculator-read-after-conflict",
            "read_file",
            {"path": ops_path},
        ),
        _tool_response(
            "calculator-propose-recovered",
            "propose_patch",
            {"path": ops_path, "new_content": corrected_disturbed_ops},
        ),
        _tool_response(
            "calculator-apply-recovered",
            "apply_patch",
            {
                "diff": recovered_diff,
                "rationale": "Apply the refreshed full-suite calculator correction.",
            },
        ),
        _tool_response(
            "calculator-run-tests",
            "run_tests",
            {"rationale": "Verify all calculator regressions after recovery."},
        ),
        _response(_step_result()),
        _response(_proceed_verdict()),
        _response(_report_json()),
    ]


def _flask_recovery_responses() -> list[LLMResponse]:
    path = "app/validation.py"
    original = _fixture_text(_FLASK_ROOT, path)
    corrected = _replace_block(
        original,
        (
            "    if not isinstance(title, str):\n"
            '        raise ValueError("title must be a string")\n\n'
            "    return {\n"
        ),
        (
            "    if not isinstance(title, str):\n"
            '        raise ValueError("title must be a string")\n'
            "    if not title:\n"
            '        raise ValueError("title must not be empty")\n\n'
            "    return {\n"
        ),
    )
    diff = _render_diff(path, original, corrected)
    return [
        _response(_plan_json()),
        _tool_response(
            "flask-branch",
            "git_create_branch",
            {"rationale": "Create the Flask recovery branch."},
        ),
        _tool_response(
            "flask-propose",
            "propose_patch",
            {"path": path, "new_content": corrected},
        ),
        _tool_response(
            "flask-apply",
            "apply_patch",
            {"diff": diff, "rationale": "Apply empty-title validation."},
        ),
        _tool_response(
            "flask-run-tests-timeout",
            "run_tests",
            {"rationale": "Run Flask tests for the first verification attempt."},
        ),
        _tool_response(
            "flask-run-tests-retry",
            "run_tests",
            {"rationale": "Retry Flask tests after the injected timeout."},
        ),
        _response(_step_result()),
        _response(_proceed_verdict()),
        _response(_report_json()),
    ]


def test_recovery_eval__patch_conflict_recovers_to_done_and_green(tmp_path: Path) -> None:
    fixture_before = {
        path: (_CALCULATOR_ROOT / path).read_bytes()
        for path in ["calculator/ops.py", "calculator/format.py", "calculator/stats.py"]
    }
    client = _ScriptedClient(_calculator_recovery_responses())
    state_dir = tmp_path / "state"

    report = run_recovery_suite(
        [_task("EV-REC-001")],
        _FIXTURES_ROOT,
        client=client,
        work_dir=state_dir,
    )
    results_path = write_recovery_suite_report(report, tmp_path / "reports")
    metrics_path = write_recovery_metrics_report(
        report,
        TraceStore(state_dir / "traces"),
        tmp_path / "reports",
    )

    result = report.results[0]
    assert result.task_id == "EV-REC-001"
    assert result.loop_status == "DONE"
    assert result.injected_error_seen is True
    assert result.loop_done is True
    assert result.tests_green is True
    assert result.returncode == 0
    assert result.recovered is True
    assert report.recovery_rate == 1.0
    assert report.loop_done_rate == 1.0
    assert report.tests_green_rate == 1.0
    assert report.injection_fired_rate == 1.0
    assert client.remaining == 0

    persisted = json.loads(results_path.read_text(encoding="utf-8"))
    assert persisted["recovery_rate"] == 1.0
    assert persisted["loop_done_rate"] == 1.0
    assert persisted["tests_green_rate"] == 1.0
    assert persisted["injection_fired_rate"] == 1.0
    assert persisted["results"][0]["recovered"] is True
    markdown = metrics_path.read_text(encoding="utf-8")
    assert "| Recovery Success Rate | 1.000 |" in markdown
    assert "| recovery | 1.000 |" in markdown

    store = TraceStore(state_dir / "traces")
    events = store.read(result.run_id)
    tool_calls = [event for event in events if event.kind.value == "tool_call"]
    assert [event.payload.get("tool_name") for event in tool_calls] == [
        "git_create_branch",
        "propose_patch",
        "apply_patch",
        "read_file",
        "propose_patch",
        "apply_patch",
        "run_tests",
    ]
    apply_calls = [event for event in tool_calls if event.payload.get("tool_name") == "apply_patch"]
    assert [event.payload.get("ok") for event in apply_calls] == [False, True]
    assert [event.payload.get("error_type") for event in apply_calls] == [
        "PatchApplyError",
        None,
    ]
    apply_approvals = [
        event
        for event in events
        if event.kind.value == "approval_decision"
        and event.payload.get("tool_name") == "apply_patch"
    ]
    assert len(apply_approvals) == 2
    assert all(
        approval.seq < call.seq for approval, call in zip(apply_approvals, apply_calls, strict=True)
    )
    terminal_results = [event for event in events if event.kind.value == "tool_result"]
    assert len(terminal_results) == 1
    assert terminal_results[0].payload.get("status") == "completed"
    assert "reason" not in terminal_results[0].payload
    assert {
        path: (_CALCULATOR_ROOT / path).read_bytes()
        for path in ["calculator/ops.py", "calculator/format.py", "calculator/stats.py"]
    } == fixture_before
    assert not (_CALCULATOR_ROOT / ".git").exists()


def test_recovery_eval__tool_timeout_retries_real_tests_and_recovers(tmp_path: Path) -> None:
    fixture_before = (_FLASK_ROOT / "app/validation.py").read_bytes()
    client = _ScriptedClient(_flask_recovery_responses())
    state_dir = tmp_path / "state"

    report = run_recovery_suite(
        [_task("EV-REC-002")],
        _FIXTURES_ROOT,
        client=client,
        work_dir=state_dir,
    )

    result = report.results[0]
    assert result.task_id == "EV-REC-002"
    assert result.loop_status == "DONE"
    assert result.injected_error_seen is True
    assert result.loop_done is True
    assert result.tests_green is True
    assert result.returncode == 0
    assert result.recovered is True
    assert report.recovery_rate == 1.0
    assert report.injection_fired_rate == 1.0
    assert client.remaining == 0

    store = TraceStore(state_dir / "traces")
    run_test_calls = [
        event
        for event in store.read(result.run_id)
        if event.kind.value == "tool_call" and event.payload.get("tool_name") == "run_tests"
    ]
    assert [event.payload.get("ok") for event in run_test_calls] == [False, True]
    assert [event.payload.get("error_type") for event in run_test_calls] == [
        "ToolTimeoutError",
        None,
    ]
    approvals = [
        event
        for event in store.read(result.run_id)
        if event.kind.value == "approval_decision" and event.payload.get("tool_name") == "run_tests"
    ]
    assert len(approvals) == 2
    assert all(
        approval.seq < call.seq for approval, call in zip(approvals, run_test_calls, strict=True)
    )
    assert (_FLASK_ROOT / "app/validation.py").read_bytes() == fixture_before
    assert not (_FLASK_ROOT / ".git").exists()
