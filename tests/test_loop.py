import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from pydantic import BaseModel, JsonValue

from app.agent.critic import Critic
from app.agent.executor import Executor
from app.agent.loop import _raw_evidence, run_agent_loop
from app.agent.planner import Planner
from app.agent.reporter import Reporter
from app.agent.state import Budgets, RunStatus, TaskSpec
from app.safety.path_jail import PathJail
from app.schemas.llm_io import LLMMessage, LLMResponse, Role, StopReason, ToolCall, Usage
from app.schemas.trace import TraceEvent, TraceEventKind
from app.services.llm_client import ToolSchema
from app.storage.db import Database
from app.storage.trace_store import RegistryTraceSink, TraceStore, render_timeline
from app.tools.apply_patch import register as register_apply_patch
from app.tools.base import ToolContext
from app.tools.get_file_tree import register as register_get_file_tree
from app.tools.git_create_branch import register as register_git_create_branch
from app.tools.read_file import register as register_read_file
from app.tools.registry import ApprovalOutcome, ToolRegistry, ToolSpec
from app.tools.search_code import register as register_search_code

_ORIGINAL = b"old\n"
_SECOND_APPROACH = b"second\n"
_FIRST_DIFF = "--- a/tracked.txt\n+++ b/tracked.txt\n@@ -1 +1 @@\n-old\n+first\n"
_SECOND_DIFF = "--- a/tracked.txt\n+++ b/tracked.txt\n@@ -1 +1 @@\n-old\n+second\n"


@dataclass(frozen=True, slots=True)
class _ClientCall:
    messages: list[LLMMessage]
    tools: list[ToolSchema] | None
    system: str | None
    temperature: float | None
    max_tokens: int | None


class _ScriptedClient:
    def __init__(self, responses: Sequence[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[_ClientCall] = []

    def complete(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[ToolSchema] | None = None,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls.append(
            _ClientCall(
                messages=list(messages),
                tools=list(tools) if tools is not None else None,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
        if not self._responses:
            raise AssertionError("No scripted LLM response remains.")
        return self._responses.pop(0)


class _SequencedGate:
    def __init__(self, outcomes: Sequence[ApprovalOutcome], snapshot_path: Path) -> None:
        self._outcomes = list(outcomes)
        self._snapshot_path = snapshot_path
        self.calls: list[tuple[ToolSpec, BaseModel, ToolContext]] = []
        self.snapshots: list[bytes] = []

    def check(
        self,
        spec: ToolSpec,
        args: BaseModel,
        context: ToolContext,
    ) -> ApprovalOutcome:
        self.calls.append((spec, args, context))
        self.snapshots.append(self._snapshot_path.read_bytes())
        if not self._outcomes:
            raise AssertionError("No scripted approval outcome remains.")
        return self._outcomes.pop(0)


def _task() -> TaskSpec:
    return TaskSpec(
        task_type="issue",
        prompt="Find where parse_date is implemented.",
        repo=".",
    )


def _plan_json(*, steps: int = 1, intent_prefix: str = "Inspect evidence") -> str:
    return json.dumps(
        {
            "steps": [
                {
                    "intent": f"{intent_prefix} {index}.",
                    "suggested_tools": ["search_code", "read_file"],
                    "success_check": f"Evidence for part {index} is cited.",
                }
                for index in range(steps)
            ]
        }
    )


def _fix_plan_json(*, include_branch: bool, approach: str) -> str:
    steps: list[dict[str, object]] = []
    if include_branch:
        steps.append(
            {
                "intent": "Create the isolated work branch.",
                "suggested_tools": ["git_create_branch"],
                "success_check": "The run work branch is current.",
            }
        )
    steps.append(
        {
            "intent": f"Apply the {approach} patch approach.",
            "suggested_tools": ["apply_patch"],
            "success_check": "The approved replacement is present in tracked.txt.",
        }
    )
    return json.dumps({"steps": steps})


def _verdict_json(
    decision: str = "proceed",
    *,
    reason: str = "The raw evidence satisfies the success check.",
    hint: str = "",
) -> str:
    return json.dumps({"decision": decision, "reason": reason, "hint": hint})


def _report_json(
    *,
    headline: str = "Model-authored run report",
    analysis: str = "The model synthesized the trace into an analysis.",
    confidence: str = "medium",
    open_questions: list[str] | None = None,
    suspects: list[dict[str, str]] | None = None,
    citations: list[str] | None = None,
) -> str:
    payload: dict[str, object] = {
        "headline": headline,
        "analysis": analysis,
        "confidence": confidence,
        "open_questions": open_questions or [],
    }
    if suspects is not None:
        payload["suspects"] = suspects
    if citations is not None:
        payload["citations"] = citations
    return json.dumps(payload)


def _result_json(
    findings: str = "parse_date is implemented in src/sample_pkg/dates.py:6.",
    *,
    evidence: str = "self-reported fake evidence",
) -> str:
    return json.dumps({"findings": findings, "evidence": [evidence]})


def _response(
    content: str = "",
    *,
    stop_reason: StopReason = StopReason.end_turn,
    tool_calls: list[ToolCall] | None = None,
    tokens_in: int = 1,
    tokens_out: int = 1,
    cost_usd: float | None = 0.01,
) -> LLMResponse:
    return LLMResponse(
        message=LLMMessage(
            role=Role.assistant,
            content=content,
            tool_calls=tool_calls or [],
        ),
        stop_reason=stop_reason,
        usage=Usage(tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=cost_usd),
        model="fake-model",
        raw_finish_reason=stop_reason.value,
    )


def _tool_call(call_id: str, name: str, arguments: dict[str, object]) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=cast(dict[str, JsonValue], arguments))


def _registry(store: TraceStore) -> ToolRegistry:
    registry = ToolRegistry(trace_sink=RegistryTraceSink(store))
    register_get_file_tree(registry)
    register_read_file(registry)
    register_search_code(registry)
    return registry


def _fix_registry(store: TraceStore, gate: _SequencedGate) -> ToolRegistry:
    registry = ToolRegistry(approval_gate=gate, trace_sink=RegistryTraceSink(store))
    register_git_create_branch(registry)
    register_apply_patch(registry)
    return registry


def _executor(client: _ScriptedClient, mini_repo: Path, store: TraceStore) -> Executor:
    return Executor(client, _registry(store), PathJail(mini_repo), store)


def _database(tmp_path: Path) -> Database:
    return Database(tmp_path / "runs.sqlite")


def _require_git_success(repo: Path, *args: str) -> None:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        check=False,
        shell=False,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")


def _init_fix_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "fix-repo"
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


def _trace_kinds(events: Sequence[TraceEvent]) -> list[TraceEventKind]:
    return [event.kind for event in events]


def _report_events(events: Sequence[TraceEvent]) -> list[TraceEvent]:
    return [event for event in events if event.kind is TraceEventKind.report]


def _verdict_decisions(events: Sequence[TraceEvent]) -> list[str]:
    return [
        str(event.payload["decision"])
        for event in events
        if event.kind is TraceEventKind.critic_verdict
    ]


def _raw_evidence_section(prompt: str) -> str:
    return prompt.split("Raw evidence:\n", maxsplit=1)[1].split(
        "\n\nRepo overview:",
        maxsplit=1,
    )[0]


def test_raw_evidence__renders_objective_outcome_and_ignores_executor_result(
    tmp_path: Path,
) -> None:
    store = TraceStore(tmp_path / "traces")
    run_id = "run-objective-evidence"
    store.append(
        run_id,
        TraceEventKind.tool_call,
        {
            "tool_name": "run_tests",
            "args": {"rationale": "Verify the fix."},
            "ok": True,
            "error_type": None,
            "truncated": False,
            "outcome": {
                "passed": 0,
                "failed": 1,
                "errors": 0,
                "skipped": 0,
                "total": 1,
                "failing_test_ids": ["tests.test_tracked::test_value"],
            },
        },
    )
    store.append(
        run_id,
        TraceEventKind.tool_result,
        {"findings": "Executor falsely claims all tests passed."},
    )

    evidence = _raw_evidence(store.read(run_id))

    assert len(evidence) == 1
    assert 'outcome=failed:1,passed:0,failing:["tests.test_tracked::test_value"]' in evidence[0]
    assert "Executor falsely claims all tests passed." not in evidence[0]


def test_raw_evidence__bounds_failing_test_id_rendering(tmp_path: Path) -> None:
    store = TraceStore(tmp_path / "traces")
    failing_test_ids = [f"tests.test_many::test_{index}" for index in range(12)]
    store.append(
        "run-bounded-evidence",
        TraceEventKind.tool_call,
        {
            "tool_name": "run_tests",
            "args": {"rationale": "Verify the fix."},
            "ok": True,
            "error_type": None,
            "truncated": False,
            "outcome": {
                "passed": 0,
                "failed": 10,
                "errors": 2,
                "skipped": 0,
                "total": 12,
                "failing_test_ids": failing_test_ids,
            },
        },
    )

    [evidence] = _raw_evidence(store.read("run-bounded-evidence"))
    visible_ids = json.dumps(failing_test_ids[:10], separators=(",", ":"))

    assert evidence.endswith(f"outcome=failed:10,passed:0,failing:{visible_ids}(+2 more),errors:2")
    assert failing_test_ids[10] not in evidence
    assert failing_test_ids[11] not in evidence


def test_loop__multi_step_happy_path_persists_trace_and_timeline(
    mini_repo: Path,
    tmp_path: Path,
) -> None:
    store = TraceStore(tmp_path / "traces")
    database = _database(tmp_path)
    planner_client = _ScriptedClient(
        [_response(_plan_json(steps=2), tokens_in=1, tokens_out=2, cost_usd=0.01)]
    )
    executor_client = _ScriptedClient(
        [
            _response(
                stop_reason=StopReason.tool_use,
                tool_calls=[
                    _tool_call(
                        "call-search",
                        "search_code",
                        {"query": "def parse_date", "glob": "**/*.py"},
                    )
                ],
                tokens_in=3,
                tokens_out=4,
                cost_usd=0.03,
            ),
            _response(_result_json(), tokens_in=5, tokens_out=6, cost_usd=0.05),
            _response(
                stop_reason=StopReason.tool_use,
                tool_calls=[
                    _tool_call(
                        "call-read",
                        "read_file",
                        {
                            "path": "src/sample_pkg/dates.py",
                            "start_line": 1,
                            "end_line": 12,
                        },
                    )
                ],
                tokens_in=11,
                tokens_out=12,
                cost_usd=0.11,
            ),
            _response(
                _result_json("parse_date implementation lines were read."),
                tokens_in=13,
                tokens_out=14,
                cost_usd=0.13,
            ),
        ]
    )
    critic_client = _ScriptedClient(
        [
            _response(_verdict_json(), tokens_in=7, tokens_out=8, cost_usd=0.07),
            _response(_verdict_json(), tokens_in=17, tokens_out=18, cost_usd=0.17),
        ]
    )

    result = run_agent_loop(
        _task(),
        planner=Planner(planner_client, store),
        executor=_executor(executor_client, mini_repo, store),
        critic=Critic(critic_client, store),
        store=store,
        database=database,
    )

    assert result.status is RunStatus.DONE
    assert result.steps_used == 2
    assert result.replans_used == 0
    assert result.fix_cycles_used == 0
    assert result.usage.tokens_in == 57
    assert result.usage.tokens_out == 64
    assert result.usage.cost_usd == pytest.approx(0.57)

    events = store.read(result.run_id)
    assert {event.run_id for event in events} == {result.run_id}
    assert [event.seq for event in events] == list(range(len(events)))
    assert _trace_kinds(events) == [
        TraceEventKind.plan,
        TraceEventKind.tool_call,
        TraceEventKind.tool_result,
        TraceEventKind.critic_verdict,
        TraceEventKind.tool_call,
        TraceEventKind.tool_result,
        TraceEventKind.critic_verdict,
        TraceEventKind.report,
    ]
    assert len(_report_events(events)) == 1

    terminal_state = database.load_state(result.run_id)
    assert terminal_state is not None
    assert terminal_state.status is RunStatus.DONE
    assert [step.status.value for step in terminal_state.plan] == ["done", "done"]
    assert [tool_call.tool_name for tool_call in database.tool_calls(result.run_id)] == [
        "search_code",
        "read_file",
    ]

    raw_evidence = _raw_evidence_section(critic_client.calls[0].messages[0].content)
    assert "tool=search_code" in raw_evidence
    assert "self-reported fake evidence" not in raw_evidence

    timeline = render_timeline(events)
    assert "plan - Planned 2 step(s)." in timeline
    assert "tool_call - search_code ok" in timeline
    assert "tool_result - Executor step 0 completed." in timeline
    assert "critic_verdict - Critic verdict for step 0: proceed." in timeline
    assert "report - Run " in timeline


def test_loop__caller_supplied_run_id_reaches_result_trace_and_database(
    mini_repo: Path,
    tmp_path: Path,
) -> None:
    store = TraceStore(tmp_path / "traces")
    database = _database(tmp_path)

    result = run_agent_loop(
        _task(),
        planner=Planner(_ScriptedClient([_response(_plan_json())]), store),
        executor=_executor(
            _ScriptedClient([_response(_result_json())]),
            mini_repo,
            store,
        ),
        critic=Critic(_ScriptedClient([_response(_verdict_json())]), store),
        store=store,
        database=database,
        run_id="fixed",
    )

    assert result.run_id == "fixed"
    assert result.status is RunStatus.DONE
    assert {event.run_id for event in store.read("fixed")} == {"fixed"}
    terminal_state = database.load_state("fixed")
    assert terminal_state is not None
    assert terminal_state.status is RunStatus.DONE
    assert database.get_run("fixed") is not None


def test_loop__retry_verdict_reexecutes_same_step_and_then_finishes(
    mini_repo: Path,
    tmp_path: Path,
) -> None:
    store = TraceStore(tmp_path / "traces")
    database = _database(tmp_path)
    planner_client = _ScriptedClient([_response(_plan_json())])
    executor_client = _ScriptedClient(
        [
            _response(_result_json("First attempt lacks enough evidence.")),
            _response(_result_json("Second attempt has enough evidence.")),
        ]
    )
    critic_client = _ScriptedClient(
        [
            _response(
                _verdict_json(
                    "retry",
                    reason="Evidence is not enough yet.",
                    hint="Gather direct file evidence.",
                )
            ),
            _response(_verdict_json("proceed")),
        ]
    )

    result = run_agent_loop(
        _task(),
        planner=Planner(planner_client, store),
        executor=_executor(executor_client, mini_repo, store),
        critic=Critic(critic_client, store),
        store=store,
        database=database,
    )

    events = store.read(result.run_id)
    tool_results = [event for event in events if event.kind is TraceEventKind.tool_result]
    assert result.status is RunStatus.DONE
    assert result.steps_used == 2
    assert result.fix_cycles_used == 1
    assert [event.payload["step_index"] for event in tool_results] == [0, 0]
    assert _verdict_decisions(events) == ["retry", "proceed"]
    assert len(_report_events(events)) == 1


def test_loop__replan_verdict_calls_planner_with_hint_and_finishes(
    mini_repo: Path,
    tmp_path: Path,
) -> None:
    store = TraceStore(tmp_path / "traces")
    database = _database(tmp_path)
    replan_hint = "The original lookup path was wrong; choose a broader search."
    planner_client = _ScriptedClient(
        [
            _response(_plan_json(intent_prefix="Original lookup")),
            _response(_plan_json(intent_prefix="Replacement lookup")),
        ]
    )
    executor_client = _ScriptedClient(
        [
            _response(_result_json("The original path was not useful.")),
            _response(_result_json("The replacement path found the implementation.")),
        ]
    )
    critic_client = _ScriptedClient(
        [
            _response(
                _verdict_json(
                    "replan",
                    reason="The step is no longer useful.",
                    hint=replan_hint,
                )
            ),
            _response(_verdict_json("proceed")),
        ]
    )

    result = run_agent_loop(
        _task(),
        planner=Planner(planner_client, store),
        executor=_executor(executor_client, mini_repo, store),
        critic=Critic(critic_client, store),
        store=store,
        database=database,
    )

    events = store.read(result.run_id)
    assert result.status is RunStatus.DONE
    assert result.replans_used == 1
    assert TraceEventKind.replan in _trace_kinds(events)
    assert _verdict_decisions(events) == ["replan", "proceed"]
    assert replan_hint in planner_client.calls[1].messages[0].content
    terminal_state = database.load_state(result.run_id)
    assert terminal_state is not None
    assert terminal_state.plan[0].intent == "Replacement lookup 0."


def test_loop__always_retrying_critic_hits_budget_and_reports(
    mini_repo: Path,
    tmp_path: Path,
) -> None:
    store = TraceStore(tmp_path / "traces")
    database = _database(tmp_path)
    planner_client = _ScriptedClient(
        [
            _response(_plan_json()),
            _response(_plan_json(intent_prefix="Budget-triggered replan")),
        ]
    )
    executor_client = _ScriptedClient(
        [
            _response(_result_json("Attempt 1 still lacks proof.")),
            _response(_result_json("Attempt 2 still lacks proof.")),
            _response(_result_json("Attempt 3 still lacks proof.")),
        ]
    )
    critic_client = _ScriptedClient(
        [
            _response(_verdict_json("retry", hint="Try again.")),
            _response(_verdict_json("retry", hint="Try again.")),
            _response(_verdict_json("retry", hint="Try again.")),
        ]
    )

    result = run_agent_loop(
        _task(),
        planner=Planner(planner_client, store),
        executor=_executor(executor_client, mini_repo, store),
        critic=Critic(critic_client, store),
        store=store,
        database=database,
        budgets=Budgets(max_steps=3),
    )

    events = store.read(result.run_id)
    assert result.status is RunStatus.FAILED
    assert result.steps_used == 3
    assert result.fix_cycles_used == 2
    assert result.replans_used == 1
    assert len(executor_client.calls) == 3
    assert len(critic_client.calls) == 3
    assert len(planner_client.calls) == 2
    assert _verdict_decisions(events) == ["retry", "retry", "retry"]
    assert TraceEventKind.replan in _trace_kinds(events)
    assert len(_report_events(events)) == 1
    terminal_state = database.load_state(result.run_id)
    assert terminal_state is not None
    assert terminal_state.status is RunStatus.FAILED


def test_loop__planner_failure_reports_after_role_error(
    mini_repo: Path,
    tmp_path: Path,
) -> None:
    store = TraceStore(tmp_path / "traces")
    database = _database(tmp_path)
    planner_client = _ScriptedClient(
        [
            _response(
                "I cannot plan that.",
                stop_reason=StopReason.refusal,
                tokens_in=9,
                tokens_out=2,
                cost_usd=0.09,
            )
        ]
    )

    result = run_agent_loop(
        _task(),
        planner=Planner(planner_client, store),
        executor=_executor(_ScriptedClient([]), mini_repo, store),
        critic=Critic(_ScriptedClient([]), store),
        store=store,
        database=database,
    )

    events = store.read(result.run_id)
    assert result.status is RunStatus.FAILED
    assert result.usage == Usage(tokens_in=9, tokens_out=2, cost_usd=0.09)
    assert _trace_kinds(events) == [TraceEventKind.error, TraceEventKind.report]
    assert events[0].payload["reason"] == "refusal"
    assert len(_report_events(events)) == 1
    terminal_state = database.load_state(result.run_id)
    assert terminal_state is not None
    assert terminal_state.status is RunStatus.FAILED


def test_loop__critic_failure_reports_after_role_error(
    mini_repo: Path,
    tmp_path: Path,
) -> None:
    store = TraceStore(tmp_path / "traces")
    database = _database(tmp_path)
    planner_client = _ScriptedClient([_response(_plan_json())])
    executor_client = _ScriptedClient([_response(_result_json())])
    critic_client = _ScriptedClient(
        [
            _response(
                "I cannot judge that.",
                stop_reason=StopReason.refusal,
                tokens_in=4,
                tokens_out=5,
                cost_usd=0.04,
            )
        ]
    )

    result = run_agent_loop(
        _task(),
        planner=Planner(planner_client, store),
        executor=_executor(executor_client, mini_repo, store),
        critic=Critic(critic_client, store),
        store=store,
        database=database,
    )

    events = store.read(result.run_id)
    assert result.status is RunStatus.FAILED
    assert _trace_kinds(events) == [
        TraceEventKind.plan,
        TraceEventKind.tool_result,
        TraceEventKind.error,
        TraceEventKind.report,
    ]
    assert events[2].payload["reason"] == "refusal"
    assert len(_report_events(events)) == 1
    terminal_state = database.load_state(result.run_id)
    assert terminal_state is not None
    assert terminal_state.status is RunStatus.FAILED


def test_loop__reporter_enriches_single_report_event_and_summary(
    mini_repo: Path,
    tmp_path: Path,
) -> None:
    store = TraceStore(tmp_path / "traces")
    database = _database(tmp_path)
    planner_client = _ScriptedClient([_response(_plan_json())])
    executor_client = _ScriptedClient([_response(_result_json("The parser location is known."))])
    critic_client = _ScriptedClient([_response(_verdict_json("proceed"))])
    reporter_client = _ScriptedClient(
        [
            _response(
                _report_json(
                    headline="parse_date was located",
                    analysis="The trace shows the planned lookup completed and was verified.",
                    confidence="high",
                    open_questions=["Confirm whether callers need a public wrapper."],
                    suspects=[
                        {
                            "path": "src/sample_pkg/dates.py",
                            "reason": "The trace verified parse_date evidence in this file.",
                        }
                    ],
                    citations=["src/sample_pkg/dates.py:6"],
                ),
                tokens_in=19,
                tokens_out=20,
                cost_usd=0.19,
            )
        ]
    )

    result = run_agent_loop(
        _task(),
        planner=Planner(planner_client, store),
        executor=_executor(executor_client, mini_repo, store),
        critic=Critic(critic_client, store),
        store=store,
        database=database,
        reporter=Reporter(reporter_client),
    )

    events = store.read(result.run_id)
    [report_event] = _report_events(events)
    assert result.status is RunStatus.DONE
    assert result.summary == (
        "parse_date was located\n\nThe trace shows the planned lookup completed and was verified."
    )
    assert result.report is not None
    assert result.report.headline == "parse_date was located"
    assert [suspect.path for suspect in result.report.suspects] == ["src/sample_pkg/dates.py"]
    assert result.report.citations == ["src/sample_pkg/dates.py:6"]
    assert result.grounding is None
    assert not result.summary.startswith("Run ")
    assert report_event.payload["headline"] == "parse_date was located"
    assert report_event.payload["analysis"] == (
        "The trace shows the planned lookup completed and was verified."
    )
    assert report_event.payload["confidence"] == "high"
    assert report_event.payload["open_questions"] == [
        "Confirm whether callers need a public wrapper."
    ]
    assert report_event.payload["suspects"] == [
        {
            "path": "src/sample_pkg/dates.py",
            "reason": "The trace verified parse_date evidence in this file.",
        }
    ]
    assert report_event.payload["citations"] == ["src/sample_pkg/dates.py:6"]
    assert "grounding" not in report_event.payload
    assert "final_findings" in report_event.payload
    assert len(_report_events(events)) == 1
    assert result.usage == Usage(tokens_in=22, tokens_out=23, cost_usd=0.22)

    terminal_state = database.load_state(result.run_id)
    assert terminal_state is not None
    assert terminal_state.status is RunStatus.DONE


def test_loop__grounding_validates_ordered_unique_report_citations(
    mini_repo: Path,
    tmp_path: Path,
) -> None:
    store = TraceStore(tmp_path / "traces")
    database = _database(tmp_path)
    planner_client = _ScriptedClient([_response(_plan_json())])
    executor_client = _ScriptedClient([_response(_result_json("The parser location is known."))])
    critic_client = _ScriptedClient([_response(_verdict_json("proceed"))])
    reporter_client = _ScriptedClient(
        [
            _response(
                _report_json(
                    headline="parse_date was located",
                    analysis="The trace points to parse_date evidence.",
                    confidence="high",
                    suspects=[
                        {
                            "path": "src/sample_pkg/dates.py",
                            "reason": "The parser implementation lives here.",
                        },
                        {
                            "path": "missing.py",
                            "reason": "The model also mentioned a missing file.",
                        },
                        {
                            "path": "../secret.txt",
                            "reason": "The model proposed an escaping path.",
                        },
                    ],
                    citations=[
                        "src/sample_pkg/dates.py:6",
                        "src/sample_pkg/dates.py:999",
                        "src/sample_pkg/dates.py:6",
                    ],
                )
            )
        ]
    )

    result = run_agent_loop(
        _task(),
        planner=Planner(planner_client, store),
        executor=_executor(executor_client, mini_repo, store),
        critic=Critic(critic_client, store),
        store=store,
        database=database,
        reporter=Reporter(reporter_client),
        jail=PathJail(mini_repo),
    )

    events = store.read(result.run_id)
    [report_event] = _report_events(events)
    assert result.status is RunStatus.DONE
    assert result.report is not None
    assert result.report.citations == [
        "src/sample_pkg/dates.py:6",
        "src/sample_pkg/dates.py:999",
        "src/sample_pkg/dates.py:6",
    ]
    assert result.grounding is not None
    assert [check.citation for check in result.grounding.checks] == [
        "src/sample_pkg/dates.py:6",
        "src/sample_pkg/dates.py:999",
        "src/sample_pkg/dates.py",
        "missing.py",
        "../secret.txt",
    ]
    assert [check.status for check in result.grounding.checks] == [
        "valid",
        "line_out_of_range",
        "valid",
        "path_not_found",
        "path_escapes",
    ]
    assert [check.grounded for check in result.grounding.checks] == [
        True,
        False,
        True,
        False,
        False,
    ]
    assert result.grounding.grounded_count == 2
    assert [check.citation for check in result.grounding.ungrounded] == [
        "src/sample_pkg/dates.py:999",
        "missing.py",
        "../secret.txt",
    ]
    assert report_event.payload["grounding"] == [
        check.model_dump() for check in result.grounding.checks
    ]
    assert len(_report_events(events)) == 1


def test_loop__reporter_refusal_falls_back_to_mechanical_report_and_counts_usage(
    mini_repo: Path,
    tmp_path: Path,
) -> None:
    store = TraceStore(tmp_path / "traces")
    database = _database(tmp_path)
    planner_client = _ScriptedClient([_response(_plan_json())])
    executor_client = _ScriptedClient([_response(_result_json())])
    critic_client = _ScriptedClient([_response(_verdict_json("proceed"))])
    reporter_client = _ScriptedClient(
        [
            _response(
                "I cannot report on that.",
                stop_reason=StopReason.refusal,
                tokens_in=20,
                tokens_out=21,
                cost_usd=0.2,
            )
        ]
    )

    result = run_agent_loop(
        _task(),
        planner=Planner(planner_client, store),
        executor=_executor(executor_client, mini_repo, store),
        critic=Critic(critic_client, store),
        store=store,
        database=database,
        reporter=Reporter(reporter_client),
        jail=PathJail(mini_repo),
    )

    events = store.read(result.run_id)
    [report_event] = _report_events(events)
    assert result.status is RunStatus.DONE
    assert result.report is None
    assert result.summary.startswith(f"Run {result.run_id} succeeded.")
    assert "headline" not in report_event.payload
    assert "suspects" not in report_event.payload
    assert "citations" not in report_event.payload
    assert "grounding" not in report_event.payload
    assert "report_generation_error" in report_event.payload
    assert "refusal" in str(report_event.payload["report_generation_error"])
    assert len(_report_events(events)) == 1
    assert result.grounding is None
    assert result.usage == Usage(tokens_in=23, tokens_out=24, cost_usd=0.23)

    terminal_state = database.load_state(result.run_id)
    assert terminal_state is not None
    assert terminal_state.status is RunStatus.DONE


def test_loop__reporter_client_error_falls_back_without_escaping(
    mini_repo: Path,
    tmp_path: Path,
) -> None:
    store = TraceStore(tmp_path / "traces")
    database = _database(tmp_path)
    planner_client = _ScriptedClient([_response(_plan_json())])
    executor_client = _ScriptedClient([_response(_result_json())])
    critic_client = _ScriptedClient([_response(_verdict_json("proceed"))])

    result = run_agent_loop(
        _task(),
        planner=Planner(planner_client, store),
        executor=_executor(executor_client, mini_repo, store),
        critic=Critic(critic_client, store),
        store=store,
        database=database,
        reporter=Reporter(_ScriptedClient([])),
    )

    events = store.read(result.run_id)
    [report_event] = _report_events(events)
    assert result.status is RunStatus.DONE
    assert result.summary.startswith(f"Run {result.run_id} succeeded.")
    assert report_event.payload["report_generation_error"] == "No scripted LLM response remains."
    assert len(_report_events(events)) == 1

    terminal_state = database.load_state(result.run_id)
    assert terminal_state is not None
    assert terminal_state.status is RunStatus.DONE


def test_loop__denial_replans_with_denied_diff_then_alternative_succeeds(
    tmp_path: Path,
) -> None:
    repo, target = _init_fix_repo(tmp_path)
    store = TraceStore(tmp_path / "traces")
    denial_reason = "Keep the public API unchanged."
    gate = _SequencedGate(
        [
            ApprovalOutcome(approved=True),
            ApprovalOutcome(approved=False, reason=denial_reason),
            ApprovalOutcome(approved=True),
        ],
        target,
    )
    planner_client = _ScriptedClient(
        [
            _response(_fix_plan_json(include_branch=True, approach="initial")),
            _response(_fix_plan_json(include_branch=False, approach="alternative")),
        ]
    )
    executor_client = _ScriptedClient(
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
            _response(_result_json("The run work branch is current.")),
            _response(
                stop_reason=StopReason.tool_use,
                tool_calls=[
                    _tool_call(
                        "apply-initial",
                        "apply_patch",
                        {
                            "diff": _FIRST_DIFF,
                            "rationale": "Apply the initial correction.",
                        },
                    )
                ],
            ),
            _response(
                stop_reason=StopReason.tool_use,
                tool_calls=[
                    _tool_call(
                        "apply-alternative",
                        "apply_patch",
                        {
                            "diff": _SECOND_DIFF,
                            "rationale": "Apply a different correction.",
                        },
                    )
                ],
            ),
            _response(_result_json("The alternative patch was applied.")),
        ]
    )
    critic_client = _ScriptedClient([_response(_verdict_json()), _response(_verdict_json())])

    result = run_agent_loop(
        TaskSpec(task_type="fix", prompt="Replace old safely.", repo=str(repo)),
        planner=Planner(planner_client, store),
        executor=Executor(
            executor_client,
            _fix_registry(store, gate),
            PathJail(repo),
            store,
        ),
        critic=Critic(critic_client, store),
        store=store,
        database=_database(tmp_path),
        budgets=Budgets(max_steps=3),
        jail=PathJail(repo),
    )

    assert result.status is RunStatus.DONE
    assert result.steps_used == 3
    assert result.replans_used == 1
    assert target.read_bytes() == _SECOND_APPROACH
    assert gate.snapshots == [_ORIGINAL, _ORIGINAL, _ORIGINAL]
    assert [spec.name for spec, _args, _context in gate.calls] == [
        "git_create_branch",
        "apply_patch",
        "apply_patch",
    ]
    replan_prompt = planner_client.calls[1].messages[0].content
    assert _FIRST_DIFF in replan_prompt
    assert "different approach" in replan_prompt
    assert "do not resubmit" in replan_prompt
    assert denial_reason not in replan_prompt

    events = store.read(result.run_id)
    apply_events = [
        event
        for event in events
        if event.kind is TraceEventKind.tool_call
        and event.payload.get("tool_name") == "apply_patch"
    ]
    assert [event.payload.get("ok") for event in apply_events] == [False, True]
    assert _trace_kinds(events).count(TraceEventKind.replan) == 1
    assert len(_report_events(events)) == 1


def test_loop__denial_budget_reports_failure_without_modifying_file(
    tmp_path: Path,
) -> None:
    repo, target = _init_fix_repo(tmp_path)
    store = TraceStore(tmp_path / "traces")
    gate = _SequencedGate(
        [
            ApprovalOutcome(approved=True),
            ApprovalOutcome(approved=False, reason="Reject the initial patch."),
            ApprovalOutcome(approved=False, reason="Reject the alternative patch."),
        ],
        target,
    )
    planner_client = _ScriptedClient(
        [
            _response(_fix_plan_json(include_branch=True, approach="initial")),
            _response(_fix_plan_json(include_branch=False, approach="alternative")),
        ]
    )
    executor_client = _ScriptedClient(
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
            _response(_result_json("The run work branch is current.")),
            _response(
                stop_reason=StopReason.tool_use,
                tool_calls=[
                    _tool_call(
                        "apply-initial",
                        "apply_patch",
                        {"diff": _FIRST_DIFF, "rationale": "Apply the initial correction."},
                    )
                ],
            ),
            _response(
                stop_reason=StopReason.tool_use,
                tool_calls=[
                    _tool_call(
                        "apply-alternative",
                        "apply_patch",
                        {
                            "diff": _SECOND_DIFF,
                            "rationale": "Apply a different correction.",
                        },
                    )
                ],
            ),
        ]
    )

    result = run_agent_loop(
        TaskSpec(task_type="fix", prompt="Replace old safely.", repo=str(repo)),
        planner=Planner(planner_client, store),
        executor=Executor(
            executor_client,
            _fix_registry(store, gate),
            PathJail(repo),
            store,
        ),
        critic=Critic(_ScriptedClient([_response(_verdict_json())]), store),
        store=store,
        database=_database(tmp_path),
        budgets=Budgets(max_steps=3, max_denials=2),
        jail=PathJail(repo),
    )

    assert result.status is RunStatus.FAILED
    assert result.steps_used == 3
    assert result.replans_used == 1
    assert target.read_bytes() == _ORIGINAL
    assert gate.snapshots == [_ORIGINAL, _ORIGINAL, _ORIGINAL]
    assert "2 denial(s)" in result.summary
    assert "human reviewer denied" in result.summary
    events = store.read(result.run_id)
    [report_event] = _report_events(events)
    assert "2 denial(s)" in str(report_event.payload["failure_summary"])
    assert len(executor_client.calls) == 4
