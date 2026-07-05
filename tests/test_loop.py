import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from pydantic import JsonValue

from app.agent.critic import Critic
from app.agent.executor import Executor
from app.agent.loop import run_agent_loop
from app.agent.planner import Planner
from app.agent.state import Budgets, RunStatus, TaskSpec
from app.safety.path_jail import PathJail
from app.schemas.llm_io import LLMMessage, LLMResponse, Role, StopReason, ToolCall, Usage
from app.schemas.trace import TraceEvent, TraceEventKind
from app.services.llm_client import ToolSchema
from app.storage.db import Database
from app.storage.trace_store import RegistryTraceSink, TraceStore, render_timeline
from app.tools.get_file_tree import register as register_get_file_tree
from app.tools.read_file import register as register_read_file
from app.tools.registry import ToolRegistry
from app.tools.search_code import register as register_search_code


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


def _verdict_json(
    decision: str = "proceed",
    *,
    reason: str = "The raw evidence satisfies the success check.",
    hint: str = "",
) -> str:
    return json.dumps({"decision": decision, "reason": reason, "hint": hint})


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


def _executor(client: _ScriptedClient, mini_repo: Path, store: TraceStore) -> Executor:
    return Executor(client, _registry(store), PathJail(mini_repo), store)


def _database(tmp_path: Path) -> Database:
    return Database(tmp_path / "runs.sqlite")


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
