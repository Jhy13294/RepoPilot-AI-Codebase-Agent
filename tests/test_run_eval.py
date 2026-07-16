import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

import eval.run_eval as run_eval_module
from app.schemas.llm_io import LLMMessage, LLMResponse, Role, StopReason, Usage
from app.services.llm_client import ToolSchema
from app.storage.trace_store import TraceStore
from eval.run_eval import load_eval_tasks, run_suite, write_metrics_report, write_suite_report


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


def _response(
    content: str,
    *,
    tokens_in: int = 1,
    tokens_out: int = 1,
    cost_usd: float | None = 0.01,
) -> LLMResponse:
    return LLMResponse(
        message=LLMMessage(role=Role.assistant, content=content),
        stop_reason=StopReason.end_turn,
        usage=Usage(tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=cost_usd),
        model="fake-model",
        raw_finish_reason=StopReason.end_turn.value,
    )


def _tool_response(name: str, arguments: dict[str, object]) -> LLMResponse:
    return LLMResponse.model_validate(
        {
            "message": {
                "role": Role.assistant,
                "tool_calls": [{"id": "tool-call-1", "name": name, "arguments": arguments}],
            },
            "stop_reason": StopReason.tool_use,
            "usage": {"tokens_in": 1, "tokens_out": 1, "cost_usd": 0.02},
            "model": "fake-model",
            "raw_finish_reason": StopReason.tool_use.value,
        }
    )


def _plan_json() -> str:
    return json.dumps(
        {
            "steps": [
                {
                    "intent": "Inspect the likely source file.",
                    "suggested_tools": ["read_file"],
                    "success_check": "The relevant file is identified with evidence.",
                }
            ]
        }
    )


def _result_json(findings: str) -> str:
    return json.dumps({"findings": findings, "evidence": [findings]})


def _verdict_json() -> str:
    return json.dumps(
        {
            "decision": "proceed",
            "reason": "The scripted evidence satisfies the success check.",
            "hint": "",
        }
    )


def _report_json(
    *,
    headline: str,
    analysis: str,
    suspect_path: str,
    citation: str,
) -> str:
    return json.dumps(
        {
            "headline": headline,
            "analysis": analysis,
            "confidence": "high",
            "open_questions": [],
            "suspects": [{"path": suspect_path, "reason": "The scripted trace points here."}],
            "citations": [citation],
        }
    )


def _tasks_json(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "tasks": [
                    {
                        "id": "IGNORED-PATCH",
                        "type": "patch",
                        "fixture": "buggy-calculator",
                        "issue": "This task should not be loaded by the issue runner.",
                        "expected": {"tests_green": True},
                    },
                    {
                        "id": "EV-LOC-TEST",
                        "type": "bug_localization",
                        "fixture": "buggy-calculator",
                        "issue": "Dividing two negative numbers returns a negative result.",
                        "expected": {
                            "gold_file": "calculator/ops.py",
                            "gold_line_range": [21, 24],
                        },
                        "budgets": {"max_steps": 5},
                    },
                    {
                        "id": "EV-EXP-TEST",
                        "type": "bug_explanation",
                        "fixture": "buggy-calculator",
                        "issue": "Percentage formatting shows 0.5% instead of 50%.",
                        "expected": {
                            "gold_file": "calculator/format.py",
                            "gold_line_range": [8, 9],
                            "rubric_keywords": ["multiply by 100", "format_percent"],
                            "requires_valid_citation": True,
                        },
                        "budgets": {"max_steps": 5},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_load_eval_tasks__filters_to_issue_analysis_tasks(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.json"
    _tasks_json(tasks_path)

    tasks = load_eval_tasks(tasks_path)

    assert [task.id for task in tasks] == ["EV-LOC-TEST", "EV-EXP-TEST"]
    assert [task.type for task in tasks] == ["bug_localization", "bug_explanation"]


def test_load_eval_tasks__actual_issue_suite_has_valid_gold_references() -> None:
    tasks = load_eval_tasks(Path("eval/tasks.json"))

    assert [task.id for task in tasks] == ["EV-LOC-001", "EV-EXP-001", "EV-LOC-002"]
    for task in tasks:
        gold_path = Path("eval/fixtures") / task.fixture / task.expected.gold_file
        assert gold_path.is_file()
        if task.expected.gold_line_range is None:
            continue
        start, end = task.expected.gold_line_range
        line_count = len(gold_path.read_text(encoding="utf-8").splitlines())
        assert 1 <= start <= end <= line_count


def test_run_suite__scripts_agent_loop_scores_and_aggregates(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.json"
    _tasks_json(tasks_path)
    tasks = load_eval_tasks(tasks_path)
    client = _ScriptedClient(
        [
            _response(_plan_json(), cost_usd=0.01),
            _response(
                _result_json("calculator/ops.py contains the divide sign bug."),
                cost_usd=0.02,
            ),
            _response(_verdict_json(), cost_usd=0.03),
            _response(
                _report_json(
                    headline="divide bug localized",
                    analysis="The likely root cause is calculator/ops.py.",
                    suspect_path="calculator/ops.py",
                    citation="calculator/ops.py:21",
                ),
                cost_usd=0.04,
            ),
            _response(_plan_json(), cost_usd=0.05),
            _response(
                _result_json("calculator/format.py contains format_percent."),
                cost_usd=0.06,
            ),
            _response(_verdict_json(), cost_usd=0.07),
            _response(
                _report_json(
                    headline="format_percent misses scaling",
                    analysis="format_percent should multiply by 100 before displaying.",
                    suspect_path="calculator/format.py",
                    citation="calculator/format.py:8",
                ),
                cost_usd=0.08,
            ),
        ]
    )

    report = run_suite(
        tasks,
        Path("eval/fixtures"),
        client=client,
        repeat=1,
        work_dir=tmp_path / "state",
    )
    results_path = write_suite_report(report, tmp_path / "reports")
    persisted = json.loads(results_path.read_text(encoding="utf-8"))

    assert report.task_count == 2
    assert report.run_count == 2
    assert report.top3_hit_rate == 1.0
    assert report.citation_valid_rate == 1.0
    assert report.mean_steps == 1.0
    assert report.total_cost_usd == pytest.approx(0.36)
    assert persisted["top3_hit_rate"] == 1.0
    assert persisted["citation_valid_rate"] == 1.0

    localization_result = report.results[0]
    assert localization_result.localization is not None
    assert localization_result.localization.hit is True
    assert localization_result.localization.rank == 1
    assert [check.citation for check in localization_result.grounding_checks] == [
        "calculator/ops.py:21",
        "calculator/ops.py",
    ]
    assert [check.status for check in localization_result.grounding_checks] == ["valid", "valid"]

    explanation_result = report.results[1]
    assert explanation_result.explanation is not None
    assert explanation_result.explanation.passed is True
    assert explanation_result.explanation.rubric_hits == ["multiply by 100", "format_percent"]
    assert [check.citation for check in explanation_result.grounding_checks] == [
        "calculator/format.py:8",
        "calculator/format.py",
    ]
    assert [check.status for check in explanation_result.grounding_checks] == ["valid", "valid"]

    assert '"task_type": "issue"' in client.calls[0].messages[0].content
    assert client.calls[1].tools is not None
    assert any(schema["function"]["name"] == "read_file" for schema in client.calls[1].tools)


def test_write_metrics_report__reads_real_run_trace_and_renders_tool_metrics(
    tmp_path: Path,
) -> None:
    tasks_path = tmp_path / "tasks.json"
    _tasks_json(tasks_path)
    task = load_eval_tasks(tasks_path)[0]
    client = _ScriptedClient(
        [
            _response(_plan_json(), cost_usd=0.01),
            _tool_response(
                "read_file",
                {"path": "calculator/ops.py", "start_line": 21, "end_line": 24},
            ),
            _response(
                _result_json("calculator/ops.py:21 contains the divide sign bug."),
                cost_usd=0.03,
            ),
            _response(_verdict_json(), cost_usd=0.04),
            _response(
                _report_json(
                    headline="divide bug localized",
                    analysis="The likely root cause is calculator/ops.py.",
                    suspect_path="calculator/ops.py",
                    citation="calculator/ops.py:21",
                ),
                cost_usd=0.05,
            ),
        ]
    )
    state_dir = tmp_path / "state"
    suite = run_suite(
        [task],
        Path("eval/fixtures"),
        client=client,
        work_dir=state_dir,
    )

    report_path = write_metrics_report(
        suite,
        TraceStore(state_dir / "traces"),
        tmp_path / "nested" / "reports",
    )
    markdown = report_path.read_text(encoding="utf-8")

    assert report_path.name == "report.md"
    assert "Runs: 1" in markdown
    assert "| Task Success Rate | 1.000 |" in markdown
    assert "| bug_localization | 1.000 |" in markdown
    assert "| Tool Call Accuracy | 1.000 |" in markdown
    assert "| Invalid Tool Call Rate | 0.000 |" in markdown
    assert "| Hallucinated-path Rate | 0.000 |" in markdown
    assert "| Average Steps | 1.00 |" in markdown
    assert "| Recovery Success Rate | n/a |" in markdown
    assert "| Graceful Failure Rate | n/a |" in markdown
    assert "| Human Approval Trigger Rate | n/a |" in markdown
    assert "| Approval Ungated Count | 0 |" in markdown
    assert "| read_file |" in markdown


def test_main__writes_report_md_without_changing_summary_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tasks_path = tmp_path / "tasks.json"
    _tasks_json(tasks_path)
    suite_data = json.loads(tasks_path.read_text(encoding="utf-8"))
    suite_data["tasks"] = [
        task for task in suite_data["tasks"] if task.get("type") == "bug_localization"
    ]
    tasks_path.write_text(json.dumps(suite_data), encoding="utf-8")
    shutil.copytree(
        Path("eval/fixtures/buggy-calculator"),
        tmp_path / "fixtures" / "buggy-calculator",
    )
    client = _ScriptedClient(
        [
            _response(_plan_json(), cost_usd=0.01),
            _response(
                _result_json("calculator/ops.py contains the divide sign bug."),
                cost_usd=0.02,
            ),
            _response(_verdict_json(), cost_usd=0.03),
            _response(
                _report_json(
                    headline="divide bug localized",
                    analysis="The likely root cause is calculator/ops.py.",
                    suspect_path="calculator/ops.py",
                    citation="calculator/ops.py:21",
                ),
                cost_usd=0.04,
            ),
        ]
    )
    timestamp = "20260716T080000000000Z"
    output_root = tmp_path / "reports"
    monkeypatch.setattr(run_eval_module, "load_settings", lambda: object())
    monkeypatch.setattr(run_eval_module, "build_llm_client", lambda _settings: client)
    monkeypatch.setattr(run_eval_module, "_timestamp", lambda: timestamp)

    run_eval_module.main(
        [
            "--tasks",
            str(tasks_path),
            "--out",
            str(output_root),
        ]
    )

    report_dir = output_root / timestamp
    assert (report_dir / "results.json").is_file()
    assert "| Task Success Rate | 1.000 |" in (report_dir / "report.md").read_text(encoding="utf-8")
    summary_lines = capsys.readouterr().out.splitlines()
    assert len(summary_lines) == 7
    assert summary_lines[0] == "Issue eval summary"
    assert summary_lines[-1] == f"results={report_dir / 'results.json'}"
