import json
import subprocess
from collections.abc import Sequence
from difflib import unified_diff
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

import eval.run_eval as run_eval_module
from app.safety.path_jail import PathJail
from app.schemas.llm_io import LLMMessage, LLMResponse, Role, StopReason, Usage
from app.services.llm_client import ToolSchema
from app.storage.trace_store import TraceStore
from app.tools.base import ToolContext
from app.tools.registry import ApprovalOutcome, ToolSpec
from eval.harness import AutoApprovalGate, prepare_git_workspace
from eval.run_eval import (
    PatchEvalTask,
    load_patch_tasks,
    run_patch_suite,
    write_patch_metrics_report,
    write_patch_suite_report,
)

_FIXTURES_ROOT = Path("eval/fixtures")
_FIXTURE_ROOT = _FIXTURES_ROOT / "buggy-calculator"


class _GateArgs(BaseModel):
    rationale: str


class _GateResult(BaseModel):
    applied: bool


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
                    "intent": "Create a work branch, apply the patch, and verify the test suite.",
                    "suggested_tools": [
                        "git_create_branch",
                        "propose_patch",
                        "apply_patch",
                        "run_tests",
                    ],
                    "success_check": "run_tests reports zero failures and errors.",
                }
            ]
        }
    )


def _step_result(findings: str) -> str:
    return json.dumps({"findings": findings, "evidence": [findings]})


def _proceed_verdict() -> str:
    return json.dumps(
        {
            "decision": "proceed",
            "reason": "The scripted tool sequence completed.",
            "hint": "",
        }
    )


def _report_json() -> str:
    return json.dumps(
        {
            "headline": "Patch run completed",
            "analysis": "The scripted patch workflow reached its terminal report.",
            "confidence": "high",
            "open_questions": [],
            "suspects": [],
            "citations": [],
        }
    )


def _fixture_text(path: str) -> str:
    return (_FIXTURE_ROOT / path).read_bytes().decode("utf-8")


def _replace_block(content: str, old: str, new: str) -> str:
    newline = "\r\n" if "\r\n" in content else "\n"
    old_with_native_newlines = old.replace("\n", newline)
    new_with_native_newlines = new.replace("\n", newline)
    assert old_with_native_newlines in content
    return content.replace(old_with_native_newlines, new_with_native_newlines, 1)


# EV-PATCH-001 freezes the full pytest suite, so the known-green script repairs all three seeds.
def _corrected_contents() -> list[tuple[str, str]]:
    ops = _replace_block(
        _fixture_text("calculator/ops.py"),
        ("    if (left < 0) == (right < 0):\n        return -quotient\n\n    return quotient\n"),
        ("    if (left < 0) == (right < 0):\n        return quotient\n\n    return -quotient\n"),
    )
    percent = _replace_block(
        _fixture_text("calculator/format.py"),
        '    return f"{value:g}%"\n',
        '    return f"{value * 100:g}%"\n',
    )
    stats = _replace_block(
        _fixture_text("calculator/stats.py"),
        "        return float(ordered[middle])\n",
        "        return float((ordered[middle - 1] + ordered[middle]) / 2)\n",
    )
    return [
        ("calculator/ops.py", ops),
        ("calculator/format.py", percent),
        ("calculator/stats.py", stats),
    ]


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


def _positive_responses() -> list[LLMResponse]:
    responses = [
        _response(_plan_json()),
        _tool_response(
            "create-branch",
            "git_create_branch",
            {"rationale": "Create the isolated eval work branch."},
        ),
    ]
    diffs: list[str] = []
    for index, (path, new_content) in enumerate(_corrected_contents(), start=1):
        diffs.append(_render_diff(path, _fixture_text(path), new_content))
        responses.append(
            _tool_response(
                f"propose-{index}",
                "propose_patch",
                {"path": path, "new_content": new_content},
            )
        )
    responses.extend(
        [
            _tool_response(
                "apply-combined",
                "apply_patch",
                {
                    "diff": "".join(diffs),
                    "rationale": "Apply the three verified fixture corrections together.",
                },
            ),
            _tool_response(
                "run-tests",
                "run_tests",
                {"rationale": "Verify all fixture regressions."},
            ),
            _response(_step_result("run_tests reported zero failures and errors.")),
            _response(_proceed_verdict()),
            _response(_report_json()),
        ]
    )
    return responses


def _ineffective_responses() -> list[LLMResponse]:
    path = "calculator/ops.py"
    original = _fixture_text(path)
    unchanged_bug = _replace_block(
        original,
        '    """Return left divided by right."""\n',
        '    """Return left divided by right as a float."""\n',
    )
    diff = _render_diff(path, original, unchanged_bug)
    return [
        _response(_plan_json()),
        _tool_response(
            "create-branch",
            "git_create_branch",
            {"rationale": "Create the isolated eval work branch."},
        ),
        _tool_response(
            "propose-ineffective",
            "propose_patch",
            {"path": path, "new_content": unchanged_bug},
        ),
        _tool_response(
            "apply-ineffective",
            "apply_patch",
            {"diff": diff, "rationale": "Apply the proposed documentation-only patch."},
        ),
        _tool_response(
            "run-failing-tests",
            "run_tests",
            {"rationale": "Run the configured tests after the patch."},
        ),
        _response(_step_result("The patch applied and the workflow completed.")),
        _response(_proceed_verdict()),
        _response(_report_json()),
    ]


def _patch_task() -> PatchEvalTask:
    return next(
        task for task in load_patch_tasks(Path("eval/tasks.json")) if task.id == "EV-PATCH-001"
    )


def _git(workspace: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=workspace,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=True,
        text=True,
        timeout=30,
    )
    return completed.stdout.strip()


def test_auto_approval_gate__returns_auditable_constant_outcome(tmp_path: Path) -> None:
    spec = ToolSpec(
        name="apply_patch",
        description="Test high-risk tool.",
        args_schema=_GateArgs,
        returns_schema=_GateResult,
        risk_level="high",
    )

    outcome = AutoApprovalGate().check(
        spec,
        _GateArgs(rationale="Exercise the eval gate."),
        ToolContext(run_id="eval-gate-test", jail=PathJail(tmp_path)),
    )

    assert outcome == ApprovalOutcome(
        approved=True,
        actor="eval:auto",
        reason="auto-approved by eval harness",
    )


def test_prepare_git_workspace__copies_without_pycache_and_commits_clean_baseline(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "fixture"
    (fixture / "package" / "__pycache__").mkdir(parents=True)
    (fixture / "package" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (fixture / "package" / "__pycache__" / "module.pyc").write_bytes(b"cached")

    workspace = prepare_git_workspace(fixture, tmp_path / "workspace")

    assert workspace == tmp_path / "workspace"
    assert (workspace / "package" / "module.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert not list(workspace.rglob("__pycache__"))
    assert _git(workspace, "status", "--porcelain") == ""
    assert _git(workspace, "rev-list", "--count", "HEAD") == "1"
    assert _git(workspace, "config", "--local", "user.email") == "eval@repopilot.local"
    assert _git(workspace, "config", "--local", "user.name") == "RepoPilot Eval Harness"


def test_load_patch_tasks__filters_patch_tasks_and_returns_frozen_models(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {"id": "ignored-issue", "type": "bug_localization"},
                    {
                        "id": "EV-PATCH-TEST",
                        "type": "patch",
                        "fixture": "buggy-calculator",
                        "issue": "Fix the divide regression.",
                        "expected": {"tests_green": True, "test_command": "pytest -q"},
                    },
                    {"id": "ignored-recovery", "type": "recovery"},
                ]
            }
        ),
        encoding="utf-8",
    )

    tasks = load_patch_tasks(tasks_path)

    assert [task.id for task in tasks] == ["EV-PATCH-TEST"]
    assert tasks[0].expected.tests_green is True
    assert tasks[0].expected.test_command == "pytest -q"
    with pytest.raises(ValidationError, match="frozen"):
        tasks[0].issue = "Mutation must fail."
    with pytest.raises(ValidationError, match="frozen"):
        tasks[0].expected.tests_green = False


def test_load_patch_tasks__actual_ev_patch_001_contract_is_frozen() -> None:
    tasks = {task.id: task for task in load_patch_tasks(Path("eval/tasks.json"))}

    task = tasks["EV-PATCH-001"]
    assert task.type == "patch"
    assert task.fixture == "buggy-calculator"
    assert "tests/test_ops.py::test_divide_two_negative_numbers_is_positive" in task.issue
    assert task.expected.tests_green is True
    assert task.expected.test_command == "pytest -q"


def test_main__routes_patch_type_to_patch_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_patch_cli(args: object) -> None:
        captured["args"] = args

    monkeypatch.setattr(run_eval_module, "_run_patch_cli", fake_run_patch_cli)

    run_eval_module.main(["--type", "patch", "--repeat", "2"])

    args = captured["args"]
    assert args.task_filter == "patch"
    assert args.repeat == 2


def test_patch_eval__gated_real_patch_reaches_done_and_independent_tests_green(
    tmp_path: Path,
) -> None:
    fixture_before = {
        path: (_FIXTURE_ROOT / path).read_bytes() for path, _new_content in _corrected_contents()
    }
    state_dir = tmp_path / "state"
    client = _ScriptedClient(_positive_responses())

    report = run_patch_suite(
        [_patch_task()],
        _FIXTURES_ROOT,
        client=client,
        work_dir=state_dir,
    )
    results_path = write_patch_suite_report(report, tmp_path / "reports")
    metrics_path = write_patch_metrics_report(
        report,
        TraceStore(state_dir / "traces"),
        tmp_path / "reports",
    )

    result = report.results[0]
    assert result.task_id == "EV-PATCH-001"
    assert result.loop_status == "DONE"
    assert result.tests_green is True
    assert result.returncode == 0
    assert client.remaining == 0
    persisted = json.loads(results_path.read_text(encoding="utf-8"))
    assert persisted["results"][0]["loop_status"] == "DONE"
    assert persisted["results"][0]["tests_green"] is True
    assert "| Human Approval Trigger Rate | 1.000 |" in metrics_path.read_text(encoding="utf-8")

    events = TraceStore(state_dir / "traces").read(result.run_id)
    approvals = [event for event in events if event.kind.value == "approval_decision"]
    assert [event.payload.get("tool_name") for event in approvals] == [
        "git_create_branch",
        "apply_patch",
        "run_tests",
    ]
    assert approvals
    assert {event.payload.get("actor") for event in approvals} == {"eval:auto"}
    assert {event.payload.get("decision") for event in approvals} == {"approved"}
    assert {event.payload.get("reason") for event in approvals} == {"auto-approved by eval harness"}
    tool_calls = [event for event in events if event.kind.value == "tool_call"]
    assert [event.payload.get("tool_name") for event in tool_calls] == [
        "git_create_branch",
        "propose_patch",
        "propose_patch",
        "propose_patch",
        "apply_patch",
        "run_tests",
    ]
    assert all(event.payload.get("ok") is True for event in tool_calls)
    gated_tool_calls = [
        event
        for event in tool_calls
        if event.payload.get("tool_name") in {"git_create_branch", "apply_patch", "run_tests"}
    ]
    assert [event.payload.get("tool_name") for event in gated_tool_calls] == [
        event.payload.get("tool_name") for event in approvals
    ]
    assert all(
        approval.seq < tool_call.seq
        for approval, tool_call in zip(approvals, gated_tool_calls, strict=True)
    )
    tool_results = [event for event in events if event.kind.value == "tool_result"]
    assert len(tool_results) == 1
    assert tool_results[0].payload.get("status") == "completed"
    assert "reason" not in tool_results[0].payload
    assert {
        path: (_FIXTURE_ROOT / path).read_bytes() for path, _new_content in _corrected_contents()
    } == fixture_before
    assert not (_FIXTURE_ROOT / ".git").exists()


def test_patch_eval__loop_done_does_not_override_independent_red_tests(tmp_path: Path) -> None:
    client = _ScriptedClient(_ineffective_responses())
    state_dir = tmp_path / "state"

    report = run_patch_suite(
        [_patch_task()],
        _FIXTURES_ROOT,
        client=client,
        work_dir=state_dir,
    )
    results_path = write_patch_suite_report(report, tmp_path / "reports")

    result = report.results[0]
    assert result.loop_status == "DONE"
    assert result.tests_green is False
    assert result.returncode != 0
    assert report.loop_done_rate == 1.0
    assert report.tests_green_rate == 0.0
    assert client.remaining == 0
    persisted = json.loads(results_path.read_text(encoding="utf-8"))
    assert persisted["results"][0]["loop_status"] == "DONE"
    assert persisted["results"][0]["tests_green"] is False
    events = TraceStore(state_dir / "traces").read(result.run_id)
    apply_calls = [
        event
        for event in events
        if event.kind.value == "tool_call" and event.payload.get("tool_name") == "apply_patch"
    ]
    assert len(apply_calls) == 1
    assert apply_calls[0].payload.get("ok") is True
