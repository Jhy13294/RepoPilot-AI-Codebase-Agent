import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

import app.tools.run_tests as run_tests_module
from app.safety.path_jail import PathJail
from app.schemas.tool_io import ErrorType, ToolResult
from app.tools.base import ToolContext
from app.tools.registry import ApprovalOutcome, ToolRegistry, ToolSpec
from app.tools.run_tests import register as register_run_tests

_RUN_ID = "test-run"


class _FakeGate:
    def __init__(self, approved: bool, reason: str | None = None) -> None:
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


@pytest.fixture(autouse=True)
def _disable_external_pytest_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    monkeypatch.delenv("PYTEST_PLUGINS", raising=False)


def _python_command(*args: str) -> str:
    return shlex.join((Path(sys.executable).as_posix(), *args))


def _pytest_command() -> str:
    return _python_command("-m", "pytest", "-q", "-p", "no:cacheprovider")


def _repo_with_test(tmp_path: Path, source: str) -> Path:
    repo = tmp_path / "target-repo"
    repo.mkdir()
    (repo / "test_sample.py").write_text(source, encoding="utf-8")
    return repo


def _context(repo: Path) -> ToolContext:
    return ToolContext(run_id=_RUN_ID, jail=PathJail(repo))


def _registry(
    gate: _FakeGate | None,
    *,
    test_command: str,
    test_timeout_s: int = 120,
) -> ToolRegistry:
    registry = ToolRegistry(approval_gate=gate)
    register_run_tests(
        registry,
        test_command=test_command,
        test_timeout_s=test_timeout_s,
    )
    return registry


def _dispatch(
    repo: Path,
    raw_args: dict[str, object],
    gate: _FakeGate | None,
    *,
    test_command: str,
    test_timeout_s: int = 120,
) -> ToolResult:
    return _registry(
        gate,
        test_command=test_command,
        test_timeout_s=test_timeout_s,
    ).dispatch("run_tests", raw_args, _context(repo))


def _payload(result: ToolResult) -> dict[str, object]:
    assert result.ok is True
    assert result.error is None
    assert result.data is not None
    payload = result.data.model_dump()
    assert isinstance(payload, dict)
    return payload


def _assert_error(result: ToolResult, error_type: ErrorType) -> None:
    assert result.ok is False
    assert result.data is None
    assert result.error is not None
    assert result.error.type is error_type


def test_run_tests__approved_passing_suite_returns_structured_counts(tmp_path: Path) -> None:
    repo = _repo_with_test(
        tmp_path,
        "def test_passes():\n    assert 2 + 2 == 4\n",
    )
    gate = _FakeGate(approved=True)

    result = _dispatch(
        repo,
        {"rationale": "Verify the proposed fix."},
        gate,
        test_command=_pytest_command(),
    )

    payload = _payload(result)
    assert payload["passed"] == 1
    assert payload["failed"] == 0
    assert payload["errors"] == 0
    assert payload["skipped"] == 0
    assert payload["total"] == 1
    assert payload["exit_code"] == 0
    assert isinstance(payload["duration_ms"], int)
    assert payload["duration_ms"] >= 0
    assert payload["failures"] == []
    assert payload["failures_truncated"] is False
    assert result.meta.tool_name == "run_tests"
    assert len(gate.calls) == 1
    spec, args, context = gate.calls[0]
    assert spec.name == "run_tests"
    assert spec.risk_level == "high"
    assert spec.timeout_s > 120
    assert args.model_dump() == {"rationale": "Verify the proposed fix."}
    assert context.jail.root == repo.resolve()


def test_run_tests__failing_tests_are_successful_tool_data(tmp_path: Path) -> None:
    repo = _repo_with_test(
        tmp_path,
        (
            "def test_passes():\n"
            "    assert True\n\n"
            "def test_fails():\n"
            '    assert False, "intentional run_tests failure"\n'
        ),
    )
    gate = _FakeGate(approved=True)

    result = _dispatch(
        repo,
        {"rationale": "Check whether the patch fixes the regression."},
        gate,
        test_command=_pytest_command(),
    )

    payload = _payload(result)
    assert payload["passed"] == 1
    assert payload["failed"] == 1
    assert payload["errors"] == 0
    assert payload["skipped"] == 0
    assert payload["total"] == 2
    assert payload["exit_code"] != 0
    failures = payload["failures"]
    assert isinstance(failures, list)
    assert len(failures) == 1
    failure = failures[0]
    assert isinstance(failure, dict)
    assert "test_fails" in failure["test_id"]
    assert "intentional run_tests failure" in failure["message"]
    assert payload["failures_truncated"] is False
    assert len(gate.calls) == 1


def test_run_tests__call_failure_and_teardown_error_preserve_both_details(
    tmp_path: Path,
) -> None:
    repo = _repo_with_test(
        tmp_path,
        (
            "import pytest\n\n"
            "@pytest.fixture\n"
            "def broken_teardown():\n"
            "    yield\n"
            '    assert False, "intentional teardown failure"\n\n'
            "def test_double_failure(broken_teardown):\n"
            '    assert False, "intentional call failure"\n'
        ),
    )

    result = _dispatch(
        repo,
        {"rationale": "Capture every failing test phase."},
        _FakeGate(approved=True),
        test_command=_pytest_command(),
    )

    payload = _payload(result)
    assert payload["passed"] == 0
    assert payload["failed"] == 1
    assert payload["errors"] == 1
    assert payload["skipped"] == 0
    assert payload["total"] == 2
    failures = payload["failures"]
    assert isinstance(failures, list)
    assert len(failures) == 2
    assert all(failure["test_id"].endswith("::test_double_failure") for failure in failures)
    messages = [failure["message"] for failure in failures]
    assert any("intentional call failure" in message for message in messages)
    assert any("intentional teardown failure" in message for message in messages)
    assert payload["failures_truncated"] is False


def test_run_tests__denial_prevents_subprocess_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fail_if_called(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        nonlocal called
        called = True
        raise AssertionError("subprocess must not run after approval denial")

    monkeypatch.setattr(run_tests_module.subprocess, "run", fail_if_called)
    gate = _FakeGate(approved=False, reason="Tests are not approved on this host.")

    result = _dispatch(
        tmp_path,
        {"rationale": "Verify the proposed fix."},
        gate,
        test_command=_pytest_command(),
    )

    _assert_error(result, ErrorType.ApprovalDeniedError)
    assert result.error is not None
    assert result.error.message == "Tests are not approved on this host."
    assert called is False
    assert len(gate.calls) == 1


def test_run_tests__missing_gate_fails_closed_without_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fail_if_called(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        nonlocal called
        called = True
        raise AssertionError("subprocess must not run without an approval gate")

    monkeypatch.setattr(run_tests_module.subprocess, "run", fail_if_called)

    result = _dispatch(
        tmp_path,
        {"rationale": "Verify the proposed fix."},
        None,
        test_command=_pytest_command(),
    )

    _assert_error(result, ErrorType.ApprovalDeniedError)
    assert result.error is not None
    assert "no approval gate configured" in result.error.message
    assert called is False


def test_run_tests__missing_runner_returns_typed_execution_error(tmp_path: Path) -> None:
    gate = _FakeGate(approved=True)

    result = _dispatch(
        tmp_path,
        {"rationale": "Verify the proposed fix."},
        gate,
        test_command="repopilot-run-tests-nonexistent-executable",
    )

    _assert_error(result, ErrorType.TestExecutionError)
    assert result.error is not None
    assert result.error.details is not None
    assert result.error.details["reason"] == "runner_not_started"
    assert len(gate.calls) == 1


def test_run_tests__timeout_returns_typed_execution_error(tmp_path: Path) -> None:
    repo = _repo_with_test(
        tmp_path,
        "import time\n\ndef test_slow():\n    time.sleep(5)\n",
    )
    gate = _FakeGate(approved=True)

    result = _dispatch(
        repo,
        {"rationale": "Verify the proposed fix."},
        gate,
        test_command=_pytest_command(),
        test_timeout_s=1,
    )

    _assert_error(result, ErrorType.TestExecutionError)
    assert result.error is not None
    assert result.error.details is not None
    assert result.error.details["reason"] == "timeout"
    assert result.error.details["timeout_s"] == 1
    assert len(gate.calls) == 1


def test_run_tests__missing_junit_results_returns_typed_execution_error(tmp_path: Path) -> None:
    gate = _FakeGate(approved=True)

    result = _dispatch(
        tmp_path,
        {"rationale": "Verify the proposed fix."},
        gate,
        test_command=_python_command("-c", "pass"),
    )

    _assert_error(result, ErrorType.TestExecutionError)
    assert result.error is not None
    assert result.error.details is not None
    assert result.error.details["reason"] == "no_results"
    assert result.error.details["exit_code"] == 0
    assert len(gate.calls) == 1


def test_run_tests__invalid_junit_results_returns_typed_execution_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_path: Path | None = None

    def write_invalid_results(
        argv: tuple[str, ...],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        nonlocal results_path
        results_path = Path(argv[-1])
        results_path.write_text("<testsuites>", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(run_tests_module.subprocess, "run", write_invalid_results)

    result = _dispatch(
        tmp_path,
        {"rationale": "Verify the proposed fix."},
        _FakeGate(approved=True),
        test_command="trusted-pytest",
    )

    _assert_error(result, ErrorType.TestExecutionError)
    assert result.error is not None
    assert result.error.details is not None
    assert result.error.details["reason"] == "no_results"
    assert results_path is not None
    assert results_path.exists() is False


@pytest.mark.parametrize(
    "raw_args",
    (
        {"rationale": ""},
        {"rationale": "Verify the proposed fix.", "command": "pytest -q"},
    ),
)
def test_run_tests__invalid_args_are_rejected_before_approval(
    tmp_path: Path,
    raw_args: dict[str, object],
) -> None:
    gate = _FakeGate(approved=True)

    result = _dispatch(
        tmp_path,
        raw_args,
        gate,
        test_command=_pytest_command(),
    )

    _assert_error(result, ErrorType.InvalidArgsError)
    assert gate.calls == []


def test_run_tests__subprocess_is_locked_to_operator_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []
    results_path: Path | None = None

    def write_results(
        argv: tuple[str, ...],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        nonlocal results_path
        calls.append((argv, kwargs))
        results_path = Path(argv[-1])
        results_path.write_text(
            (
                '<testsuites><testsuite tests="1" failures="1" errors="0" skipped="0">'
                '<testcase classname="test_sample" name="test_failure">'
                '<failure message="assert locked down" />'
                "</testcase></testsuite></testsuites>"
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 1, stdout=b"", stderr=b"")

    monkeypatch.setattr(run_tests_module.subprocess, "run", write_results)
    gate = _FakeGate(approved=True)
    rationale = "Model-controlled rationale --must-never-enter-argv"

    result = _dispatch(
        tmp_path,
        {"rationale": rationale},
        gate,
        test_command="trusted-pytest --fixed-option",
        test_timeout_s=7,
    )

    payload = _payload(result)
    assert payload["failed"] == 1
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv[:2] == ("trusted-pytest", "--fixed-option")
    assert argv[-2] == "--junit-xml"
    assert rationale not in argv
    assert kwargs == {
        "cwd": tmp_path.resolve(),
        "stdin": subprocess.DEVNULL,
        "capture_output": True,
        "check": False,
        "shell": False,
        "timeout": 7,
    }
    assert results_path is not None
    assert results_path.is_relative_to(tmp_path.resolve()) is False
    assert results_path.exists() is False
    assert len(gate.calls) == 1


def test_run_tests__errors_and_skips_are_counted_from_junit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def write_results(
        argv: tuple[str, ...],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        Path(argv[-1]).write_text(
            (
                '<testsuite tests="4" failures="1" errors="1" skipped="1">'
                '<testcase classname="test_sample" name="test_passes" />'
                '<testcase classname="test_sample" name="test_double_outcome">'
                '<failure message="assert call failed" />'
                "</testcase>"
                '<testcase classname="test_sample" name="test_double_outcome">'
                '<error message="setup exploded" />'
                "</testcase>"
                '<testcase classname="test_sample" name="test_skips">'
                '<skipped message="not supported" />'
                "</testcase></testsuite>"
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 1, stdout=b"", stderr=b"")

    monkeypatch.setattr(run_tests_module.subprocess, "run", write_results)

    result = _dispatch(
        tmp_path,
        {"rationale": "Verify the proposed fix."},
        _FakeGate(approved=True),
        test_command="trusted-pytest",
    )

    payload = _payload(result)
    assert payload["passed"] == 1
    assert payload["failed"] == 1
    assert payload["errors"] == 1
    assert payload["skipped"] == 1
    assert payload["total"] == 4
    assert payload["failures"] == [
        {"test_id": "test_sample::test_double_outcome", "message": "assert call failed"},
        {"test_id": "test_sample::test_double_outcome", "message": "setup exploded"},
    ]
    assert payload["failures_truncated"] is False


def test_run_tests__failure_details_are_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_cases = "".join(
        (
            f'<testcase classname="test_sample" name="test_failure_{index}">'
            f'<failure message="failure {index}" />'
            "</testcase>"
        )
        for index in range(51)
    )

    def write_results(
        argv: tuple[str, ...],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        Path(argv[-1]).write_text(
            f'<testsuite tests="51" failures="51" errors="0" skipped="0">{test_cases}</testsuite>',
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 1, stdout=b"", stderr=b"")

    monkeypatch.setattr(run_tests_module.subprocess, "run", write_results)

    result = _dispatch(
        tmp_path,
        {"rationale": "Verify the proposed fix."},
        _FakeGate(approved=True),
        test_command="trusted-pytest",
    )

    payload = _payload(result)
    assert payload["failed"] == 51
    failures = payload["failures"]
    assert isinstance(failures, list)
    assert len(failures) == 50
    assert payload["failures_truncated"] is True


def test_run_tests__registry_result_round_trips_as_an_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def write_results(
        argv: tuple[str, ...],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        Path(argv[-1]).write_text(
            (
                '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0">'
                '<testcase classname="test_sample" name="test_passes" />'
                "</testsuite></testsuites>"
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(run_tests_module.subprocess, "run", write_results)

    result = _dispatch(
        tmp_path,
        {"rationale": "Verify the proposed fix."},
        _FakeGate(approved=True),
        test_command="trusted-pytest",
    )

    original_payload = _payload(result)
    decoded = ToolResult.model_validate_json(result.model_dump_json())
    assert decoded.ok is True
    assert decoded.error is None
    assert decoded.data is not None
    assert decoded.data.model_dump() == {
        "passed": 1,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "total": 1,
        "exit_code": 0,
        "duration_ms": original_payload["duration_ms"],
        "failures": [],
        "failures_truncated": False,
    }
    assert decoded.meta.tool_name == "run_tests"


def test_run_tests__llm_schema_exposes_only_rationale() -> None:
    registry = _registry(
        _FakeGate(approved=True),
        test_command="trusted-pytest --fixed-option",
    )

    function = registry.to_llm_schema()[0]["function"]

    assert isinstance(function, dict)
    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    assert parameters["required"] == ["rationale"]
    assert parameters["additionalProperties"] is False
    properties = parameters["properties"]
    assert isinstance(properties, dict)
    assert tuple(properties) == ("rationale",)


def test_run_tests__register_defaults_are_operator_owned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def write_results(
        argv: tuple[str, ...],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append((argv, kwargs))
        Path(argv[-1]).write_text(
            (
                '<testsuites><testsuite tests="0" failures="0" errors="0" skipped="0">'
                "</testsuite></testsuites>"
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 5, stdout=b"", stderr=b"")

    monkeypatch.setattr(run_tests_module.subprocess, "run", write_results)
    gate = _FakeGate(approved=True)
    registry = ToolRegistry(approval_gate=gate)
    register_run_tests(registry)

    result = registry.dispatch(
        "run_tests",
        {"rationale": "Run the configured default suite."},
        _context(tmp_path),
    )

    payload = _payload(result)
    assert payload["total"] == 0
    assert payload["exit_code"] == 5
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv[:2] == ("pytest", "-q")
    assert argv[-2] == "--junit-xml"
    assert kwargs["timeout"] == 120
    assert len(gate.calls) == 1
    assert gate.calls[0][0].timeout_s == 125
