import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest
from pydantic import BaseModel, JsonValue, ValidationError
from typer.testing import CliRunner

import app.cli as cli
import app.tools.run_tests as run_tests_module
from app.agent.critic import Critic
from app.agent.executor import Executor
from app.agent.loop import run_agent_loop
from app.agent.planner import Planner
from app.agent.state import Budgets, RunStatus, TaskSpec
from app.safety.loop_guard import LoopGuard
from app.safety.path_jail import PathJail
from app.schemas.agent_io import RunResult
from app.schemas.llm_io import LLMMessage, LLMResponse, Role, StopReason, ToolCall, Usage
from app.services.llm_client import ToolSchema
from app.storage.db import Database
from app.storage.trace_store import RegistryTraceSink, TraceStore
from app.tools.base import ToolContext
from app.tools.registry import ApprovalOutcome, ToolRegistry, ToolSpec

_ORIGINAL = b"old\n"
_INTERMEDIATE = b"broken\n"
_UPDATED = b"new\n"
_DIFF = "--- a/tracked.txt\n+++ b/tracked.txt\n@@ -1 +1 @@\n-old\n+new\n"
_FIRST_FIX_DIFF = "--- a/tracked.txt\n+++ b/tracked.txt\n@@ -1 +1 @@\n-old\n+broken\n"
_SECOND_FIX_DIFF = "--- a/tracked.txt\n+++ b/tracked.txt\n@@ -1 +1 @@\n-broken\n+new\n"


class _ScriptedClient:
    def __init__(self, responses: Sequence[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[list[LLMMessage]] = []

    def complete(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[ToolSchema] | None = None,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        del tools, system, temperature, max_tokens
        if not self._responses:
            raise AssertionError("No scripted LLM response remains.")
        return self._responses.pop(0)


class _FakeGate:
    def __init__(self, approved: bool = True, reason: str | None = None) -> None:
        self.approved = approved
        self.reason = reason
        self.calls: list[tuple[ToolSpec, BaseModel, ToolContext]] = []

    def check(
        self,
        spec: ToolSpec,
        args: BaseModel,
        context: ToolContext,
    ) -> ApprovalOutcome:
        self.calls.append((spec, args, context))
        return ApprovalOutcome(approved=self.approved, reason=self.reason)


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
        model="fake-model",
        raw_finish_reason=stop_reason.value,
    )


def _tool_call(call_id: str, name: str, arguments: dict[str, object]) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=cast(dict[str, JsonValue], arguments))


def _plan_json() -> str:
    return json.dumps(
        {
            "steps": [
                {
                    "intent": "Create the isolated work branch.",
                    "suggested_tools": ["git_create_branch"],
                    "success_check": "The run work branch is current.",
                },
                {
                    "intent": "Read the file and propose the corrected content.",
                    "suggested_tools": ["read_file", "propose_patch"],
                    "success_check": "A grounded patch is proposed without writing.",
                },
                {
                    "intent": "Apply the approved patch on the work branch.",
                    "suggested_tools": ["apply_patch"],
                    "success_check": "The approved patch is present in the worktree.",
                },
            ]
        }
    )


def _fix_cycle_plan_json() -> str:
    return json.dumps(
        {
            "steps": [
                {
                    "intent": "Create the isolated work branch.",
                    "suggested_tools": ["git_create_branch"],
                    "success_check": "The run work branch is current.",
                },
                {
                    "intent": "Patch tracked.txt and run the configured test suite.",
                    "suggested_tools": ["propose_patch", "apply_patch", "run_tests"],
                    "success_check": "The test report has zero failures.",
                },
            ]
        }
    )


def _step_result(findings: str) -> str:
    return json.dumps({"findings": findings, "evidence": [findings]})


def _verdict() -> str:
    return json.dumps(
        {
            "decision": "proceed",
            "reason": "The tool evidence satisfies the step success check.",
            "hint": "",
        }
    )


def _retry_verdict() -> str:
    return json.dumps(
        {
            "decision": "retry",
            "reason": "The test report still contains a failing test.",
            "hint": "Revise the patch and run the tests again.",
        }
    )


def _raw_evidence_section(prompt: str) -> str:
    return prompt.split("Raw evidence:\n", maxsplit=1)[1].split(
        "\n\nRepo overview:",
        maxsplit=1,
    )[0]


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
    _require_git_success(repo, "commit", "--quiet", "-m", "test fixture")
    return repo, target


def _tool_names(registry: ToolRegistry) -> list[str]:
    names: list[str] = []
    for schema in registry.to_llm_schema():
        function = schema["function"]
        assert isinstance(function, dict)
        name = function["name"]
        assert isinstance(name, str)
        names.append(name)
    return names


def test_task_spec__fix_is_valid_and_unknown_type_is_rejected() -> None:
    task = TaskSpec(task_type="fix", prompt="Correct tracked.txt.", repo=".")

    assert task.task_type == "fix"
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(
            {"task_type": "unknown", "prompt": "Correct tracked.txt.", "repo": "."}
        )


def test_fix_registry__denied_apply_cannot_write(tmp_path: Path) -> None:
    repo, target = _init_repo(tmp_path)
    gate = _FakeGate()
    registry = cli._build_fix_registry(approval_gate=gate)
    context = ToolContext(run_id="denied-run", jail=PathJail(repo))

    branch_result = registry.dispatch(
        "git_create_branch",
        {"rationale": "Create the isolated work branch."},
        context,
    )
    proposal_result = registry.dispatch(
        "propose_patch",
        {"path": "tracked.txt", "new_content": _UPDATED.decode()},
        context,
    )
    assert branch_result.ok is True
    assert proposal_result.ok is True
    assert proposal_result.data is not None
    proposed_diff = proposal_result.data.model_dump()["diff"]
    assert proposed_diff == _DIFF
    before = target.read_bytes()

    gate.approved = False
    gate.reason = "Patch needs another review."
    apply_result = registry.dispatch(
        "apply_patch",
        {"diff": proposed_diff, "rationale": "Apply the reviewed correction."},
        context,
    )

    assert apply_result.ok is False
    assert apply_result.error is not None
    assert apply_result.error.type.value == "ApprovalDeniedError"
    assert target.read_bytes() == before == _ORIGINAL
    assert [spec.name for spec, _args, _context in gate.calls] == [
        "git_create_branch",
        "apply_patch",
    ]
    assert _tool_names(registry) == [
        "get_file_tree",
        "read_file",
        "search_code",
        "git_create_branch",
        "propose_patch",
        "apply_patch",
        "run_tests",
        "git_commit",
    ]


def test_fix_registry__adds_high_risk_tools_and_loop_guard_only_to_fix_registry() -> None:
    fix_registry = cli._build_fix_registry(approval_gate=_FakeGate())
    read_only_registry = cli._build_read_only_registry()

    assert "run_tests" in _tool_names(fix_registry)
    assert "git_commit" in _tool_names(fix_registry)
    assert isinstance(fix_registry._loop_guard, LoopGuard)
    assert "run_tests" not in _tool_names(read_only_registry)
    assert "git_commit" not in _tool_names(read_only_registry)
    assert read_only_registry._loop_guard is None


def test_fix_loop__approved_create_propose_apply_reaches_done(tmp_path: Path) -> None:
    repo, target = _init_repo(tmp_path)
    store = TraceStore(tmp_path / "traces")
    gate = _FakeGate()
    registry = cli._build_fix_registry(
        trace_sink=RegistryTraceSink(store),
        approval_gate=gate,
    )
    planner = Planner(_ScriptedClient([_response(_plan_json())]), store)
    executor = Executor(
        _ScriptedClient(
            [
                _response(
                    stop_reason=StopReason.tool_use,
                    tool_calls=[
                        _tool_call(
                            "create-branch",
                            "git_create_branch",
                            {"rationale": "Create the isolated work branch."},
                        )
                    ],
                ),
                _response(_step_result("The run work branch is current.")),
                _response(
                    stop_reason=StopReason.tool_use,
                    tool_calls=[
                        _tool_call(
                            "propose-patch",
                            "propose_patch",
                            {"path": "tracked.txt", "new_content": _UPDATED.decode()},
                        )
                    ],
                ),
                _response(_step_result("A deterministic patch was proposed.")),
                _response(
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
                ),
                _response(_step_result("The approved patch was applied.")),
            ]
        ),
        registry,
        PathJail(repo),
        store,
    )
    critic = Critic(
        _ScriptedClient([_response(_verdict()), _response(_verdict()), _response(_verdict())]),
        store,
    )

    result = run_agent_loop(
        TaskSpec(task_type="fix", prompt="Replace old with new.", repo=str(repo)),
        planner=planner,
        executor=executor,
        critic=critic,
        store=store,
        database=Database(tmp_path / "runs.sqlite"),
        budgets=Budgets(max_steps=3),
        jail=PathJail(repo),
    )

    assert result.status is RunStatus.DONE
    assert result.steps_used == 3
    assert target.read_bytes() == _UPDATED
    branch = _require_git_success(repo, "branch", "--show-current").stdout.decode().strip()
    assert branch == f"repopilot/fix-{result.run_id}"
    assert [spec.name for spec, _args, _context in gate.calls] == [
        "git_create_branch",
        "apply_patch",
    ]
    assert {context.run_id for _spec, _args, context in gate.calls} == {result.run_id}


def test_fix_loop__failed_tests_trigger_repatch_then_passing_tests_finish(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo, target = _init_repo(tmp_path)
    store = TraceStore(tmp_path / "traces")
    gate = _FakeGate()
    failing_outcome = run_tests_module._RunTestsPayload(
        passed=0,
        failed=1,
        errors=0,
        skipped=0,
        total=1,
        exit_code=1,
        duration_ms=5,
        failures=[
            run_tests_module._TestFailure(
                test_id="tests.test_tracked::test_value",
                message="expected new but found broken",
            )
        ],
        failures_truncated=False,
    )
    passing_outcome = run_tests_module._RunTestsPayload(
        passed=1,
        failed=0,
        errors=0,
        skipped=0,
        total=1,
        exit_code=0,
        duration_ms=4,
        failures=[],
        failures_truncated=False,
    )
    test_outcomes = iter([failing_outcome, passing_outcome])

    def fake_run_tests(
        args: BaseModel,
        context: ToolContext,
        *,
        test_command: str,
        test_timeout_s: int,
    ) -> BaseModel:
        del args, context, test_command, test_timeout_s
        try:
            return next(test_outcomes)
        except StopIteration as exc:
            raise AssertionError("run_tests was dispatched more than twice") from exc

    monkeypatch.setattr(run_tests_module, "_handle", fake_run_tests)
    registry = cli._build_fix_registry(
        trace_sink=RegistryTraceSink(store),
        approval_gate=gate,
    )
    planner = Planner(_ScriptedClient([_response(_fix_cycle_plan_json())]), store)
    false_success_claim = "All tests passed with zero failures."
    executor = Executor(
        _ScriptedClient(
            [
                _response(
                    stop_reason=StopReason.tool_use,
                    tool_calls=[
                        _tool_call(
                            "create-branch",
                            "git_create_branch",
                            {"rationale": "Create the isolated work branch."},
                        )
                    ],
                ),
                _response(_step_result("The run work branch is current.")),
                _response(
                    stop_reason=StopReason.tool_use,
                    tool_calls=[
                        _tool_call(
                            "propose-first-patch",
                            "propose_patch",
                            {
                                "path": "tracked.txt",
                                "new_content": _INTERMEDIATE.decode(),
                            },
                        )
                    ],
                ),
                _response(
                    stop_reason=StopReason.tool_use,
                    tool_calls=[
                        _tool_call(
                            "apply-first-patch",
                            "apply_patch",
                            {
                                "diff": _FIRST_FIX_DIFF,
                                "rationale": "Apply the first reviewed correction.",
                            },
                        )
                    ],
                ),
                _response(
                    stop_reason=StopReason.tool_use,
                    tool_calls=[
                        _tool_call(
                            "run-failing-tests",
                            "run_tests",
                            {"rationale": "Verify the first correction."},
                        )
                    ],
                ),
                _response(_step_result(false_success_claim)),
                _response(
                    stop_reason=StopReason.tool_use,
                    tool_calls=[
                        _tool_call(
                            "propose-second-patch",
                            "propose_patch",
                            {"path": "tracked.txt", "new_content": _UPDATED.decode()},
                        )
                    ],
                ),
                _response(
                    stop_reason=StopReason.tool_use,
                    tool_calls=[
                        _tool_call(
                            "apply-second-patch",
                            "apply_patch",
                            {
                                "diff": _SECOND_FIX_DIFF,
                                "rationale": "Apply the revised correction.",
                            },
                        )
                    ],
                ),
                _response(
                    stop_reason=StopReason.tool_use,
                    tool_calls=[
                        _tool_call(
                            "run-passing-tests",
                            "run_tests",
                            {"rationale": "Verify the revised correction."},
                        )
                    ],
                ),
                _response(_step_result("run_tests reported passed=1, failed=0, errors=0.")),
            ]
        ),
        registry,
        PathJail(repo),
        store,
    )
    critic_client = _ScriptedClient(
        [_response(_verdict()), _response(_retry_verdict()), _response(_verdict())]
    )
    critic = Critic(critic_client, store)

    result = run_agent_loop(
        TaskSpec(task_type="fix", prompt="Replace old with new and verify it.", repo=str(repo)),
        planner=planner,
        executor=executor,
        critic=critic,
        store=store,
        database=Database(tmp_path / "runs.sqlite"),
        budgets=Budgets(max_steps=4, max_fix_cycles=2),
        jail=PathJail(repo),
    )

    assert result.status is RunStatus.DONE
    assert result.fix_cycles_used >= 1
    assert target.read_bytes() == _UPDATED
    assert [spec.name for spec, _args, _context in gate.calls] == [
        "git_create_branch",
        "apply_patch",
        "run_tests",
        "apply_patch",
        "run_tests",
    ]
    raw_evidence = _raw_evidence_section(critic_client.calls[1][0].content)
    assert "outcome=failed:1,passed:0" in raw_evidence
    assert "tests.test_tracked::test_value" in raw_evidence
    assert false_success_claim not in raw_evidence

    tool_call_events = [
        event for event in store.read(result.run_id) if event.kind.value == "tool_call"
    ]
    run_tests_events = [
        event for event in tool_call_events if event.payload.get("tool_name") == "run_tests"
    ]
    assert [event.payload.get("ok") for event in run_tests_events] == [True, True]
    assert [event.payload.get("outcome") for event in run_tests_events] == [
        failing_outcome.evidence_digest(),
        passing_outcome.evidence_digest(),
    ]
    assert all(
        "outcome" not in event.payload
        for event in tool_call_events
        if event.payload.get("tool_name") != "run_tests"
    )
    with pytest.raises(StopIteration):
        next(test_outcomes)


def test_cli_run__fix_uses_gated_registry_for_planner_and_executor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    gate = _FakeGate()
    client = _ScriptedClient([])
    captured: dict[str, object] = {}
    original_builder = cli._build_fix_registry

    def build_fix_registry(
        *,
        trace_sink: object = None,
        approval_gate: object,
        test_command: str = "pytest -q",
        test_timeout_s: int = 120,
    ) -> ToolRegistry:
        captured["approval_gate"] = approval_gate
        captured["test_command"] = test_command
        captured["test_timeout_s"] = test_timeout_s
        registry = original_builder(
            trace_sink=cast(RegistryTraceSink | None, trace_sink),
            approval_gate=cast(_FakeGate, approval_gate),
            test_command=test_command,
            test_timeout_s=test_timeout_s,
        )
        captured["registry"] = registry
        return registry

    def fake_run_agent_loop(task: TaskSpec, **kwargs: object) -> RunResult:
        captured["task"] = task
        captured.update(kwargs)
        return RunResult(
            run_id="cli-fix-run",
            status=RunStatus.DONE,
            summary="Fix wiring smoke test completed.",
            steps_used=0,
            replans_used=0,
            fix_cycles_used=0,
            usage=Usage(tokens_in=0, tokens_out=0, cost_usd=None),
        )

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-cli")
    monkeypatch.setenv("REPOPILOT_TRACE_DIR", str(tmp_path / "traces"))
    monkeypatch.setenv("REPOPILOT_DB_PATH", str(tmp_path / "runs.sqlite"))
    monkeypatch.setenv("REPOPILOT_TEST_COMMAND", "pytest tests/test_fix_wiring.py -q")
    monkeypatch.setenv("REPOPILOT_TEST_TIMEOUT_S", "45")
    monkeypatch.setattr(cli, "build_llm_client", lambda _settings: client)
    monkeypatch.setattr(cli, "CliApprovalGate", lambda: gate)
    monkeypatch.setattr(cli, "_build_fix_registry", build_fix_registry)
    monkeypatch.setattr(cli, "run_agent_loop", fake_run_agent_loop)

    result = CliRunner().invoke(
        cli.app,
        ["run", "Correct tracked.txt.", "--task-type", "fix", "--repo", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    assert captured["approval_gate"] is gate
    assert captured["test_command"] == "pytest tests/test_fix_wiring.py -q"
    assert captured["test_timeout_s"] == 45
    registry = cast(ToolRegistry, captured["registry"])
    executor = cast(Executor, captured["executor"])
    planner = cast(Planner, captured["planner"])
    task = cast(TaskSpec, captured["task"])
    assert executor._registry is registry
    assert "- apply_patch:" in planner._system_prompt
    assert "- run_tests:" in planner._system_prompt
    assert "create the work branch first" in planner._system_prompt
    assert "test report has zero failures" in planner._system_prompt
    assert "commit the changes on the work branch with a clear message" in planner._system_prompt
    assert task.task_type == "fix"


def test_console_script_run_help_lists_fix_task_type() -> None:
    result = subprocess.run(
        ["uv", "run", "repopilot", "run", "--help"],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )

    output = f"{result.stdout}{result.stderr}"
    assert result.returncode == 0, output
    assert "--task-type" in output
    assert "fix" in output
