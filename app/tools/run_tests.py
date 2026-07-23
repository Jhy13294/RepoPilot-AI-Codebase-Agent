"""Approval-gated execution of the operator-configured pytest command."""

import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter
from xml.etree import ElementTree

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from app.schemas.tool_io import ErrorType
from app.tools.base import ToolContext, ToolFailure
from app.tools.registry import ToolRegistry, ToolSpec

__all__ = ["register", "resolve_pytest_argv"]

_MAX_FAILURES = 50
_PYTEST_PROGRAMS = frozenset({"pytest", "py.test"})
_REGISTRY_TIMEOUT_GRACE_S = 5


class _RunTestsArgs(BaseModel):
    """Arguments for run_tests."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rationale: str = Field(min_length=1)


class _TestFailure(BaseModel):
    """One failed or errored pytest test case."""

    model_config = ConfigDict(frozen=True)

    test_id: str
    message: str


class _RunTestsPayload(BaseModel):
    """Structured pytest outcome returned by run_tests."""

    model_config = ConfigDict(frozen=True)

    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    errors: int = Field(ge=0)
    skipped: int = Field(ge=0)
    total: int = Field(ge=0)
    exit_code: int
    duration_ms: int = Field(ge=0)
    failures: list[_TestFailure]
    failures_truncated: bool

    def evidence_digest(self) -> dict[str, JsonValue]:
        """Return bounded objective evidence for downstream verification."""
        failing_test_ids: list[JsonValue] = [failure.test_id for failure in self.failures]
        return {
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "skipped": self.skipped,
            "total": self.total,
            "failing_test_ids": failing_test_ids,
        }


def resolve_pytest_argv(test_command: str) -> list[str]:
    """Split a test command, running a bare pytest via the current interpreter.

    A bare ``pytest``/``py.test`` program is rewritten to
    ``[sys.executable, "-m", "pytest", ...]`` so it resolves without the
    virtualenv console-scripts directory on PATH. Any other program (absolute
    path, ``python``, a custom runner) is returned as split.
    """
    argv = shlex.split(test_command)
    if not argv:
        raise ValueError("test_command must not be empty.")
    if argv[0] in _PYTEST_PROGRAMS:
        return [sys.executable, "-m", "pytest", *argv[1:]]
    return argv


def register(
    registry: ToolRegistry,
    *,
    test_command: str = "pytest -q",
    test_timeout_s: int = 120,
) -> None:
    """Register run_tests with an operator-configured command and timeout."""
    if not test_command.strip():
        raise ValueError("test_command must not be empty.")
    if test_timeout_s < 1:
        raise ValueError("test_timeout_s must be at least 1.")

    def handler(args: BaseModel, context: ToolContext) -> BaseModel:
        return _handle(
            args,
            context,
            test_command=test_command,
            test_timeout_s=test_timeout_s,
        )

    registry.register(
        ToolSpec(
            name="run_tests",
            description=(
                "Run the operator-configured pytest command in the workspace after explicit "
                "human approval. Supply only a rationale; the command and target cannot be "
                "provided by the model. A completed run reports failing tests as successful "
                "tool data rather than a tool error."
            ),
            args_schema=_RunTestsArgs,
            returns_schema=_RunTestsPayload,
            risk_level="high",
            timeout_s=test_timeout_s + _REGISTRY_TIMEOUT_GRACE_S,
        ),
        handler,
    )


def _handle(
    args: BaseModel,
    context: ToolContext,
    *,
    test_command: str,
    test_timeout_s: int,
) -> BaseModel:
    _RunTestsArgs.model_validate(args)

    with tempfile.TemporaryDirectory(
        prefix="repopilot-run-tests-",
        ignore_cleanup_errors=True,
    ) as temp_dir:
        results_path = Path(temp_dir) / "results.xml"
        try:
            argv = (*resolve_pytest_argv(test_command), "--junit-xml", str(results_path))
            started_at = perf_counter()
            completed = subprocess.run(
                argv,
                cwd=context.jail.root,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                check=False,
                shell=False,
                timeout=test_timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolFailure(
                ErrorType.TestExecutionError,
                "The configured test command timed out; report the timeout to the operator.",
                {
                    "reason": "timeout",
                    "timeout_s": test_timeout_s,
                    "stderr": _decode_output(exc.stderr),
                },
            ) from exc
        except (OSError, ValueError) as exc:
            raise ToolFailure(
                ErrorType.TestExecutionError,
                (
                    "The configured test runner could not be started; report the test "
                    "configuration problem to the operator."
                ),
                {"reason": "runner_not_started", "stderr": str(exc)},
            ) from exc

        duration_ms = max(0, int((perf_counter() - started_at) * 1000))
        try:
            return _parse_junit(
                results_path,
                exit_code=completed.returncode,
                duration_ms=duration_ms,
            )
        except (ElementTree.ParseError, OSError, ValueError) as exc:
            raise ToolFailure(
                ErrorType.TestExecutionError,
                (
                    "The configured test command produced no parseable pytest JUnit results; "
                    "report the test configuration problem to the operator."
                ),
                {
                    "reason": "no_results",
                    "exit_code": completed.returncode,
                    "stderr": _decode_output(completed.stderr),
                },
            ) from exc


def _parse_junit(
    results_path: Path,
    *,
    exit_code: int,
    duration_ms: int,
) -> _RunTestsPayload:
    root = ElementTree.parse(results_path).getroot()
    if _local_name(root.tag) not in {"testsuite", "testsuites"}:
        raise ValueError("JUnit root must be testsuite or testsuites.")

    passed, failed, errors, skipped, total = _junit_counts(root)
    failures: list[_TestFailure] = []
    failure_details = 0

    for test_case in (element for element in root.iter() if _local_name(element.tag) == "testcase"):
        for outcome in test_case:
            if _local_name(outcome.tag) not in {"failure", "error"}:
                continue
            failure_details += 1
            _append_failure(failures, test_case, outcome)

    return _RunTestsPayload(
        passed=passed,
        failed=failed,
        errors=errors,
        skipped=skipped,
        total=total,
        exit_code=exit_code,
        duration_ms=duration_ms,
        failures=failures,
        failures_truncated=failure_details > len(failures),
    )


def _junit_counts(root: ElementTree.Element) -> tuple[int, int, int, int, int]:
    if _local_name(root.tag) == "testsuite":
        suites = [root]
    else:
        suites = [element for element in root if _local_name(element.tag) == "testsuite"]
    if not suites:
        raise ValueError("JUnit results contain no test suites.")

    passed = 0
    failed = 0
    errors = 0
    skipped = 0
    total = 0
    for suite in suites:
        suite_total = _count_attribute(suite, "tests", required=True)
        suite_failed = _count_attribute(suite, "failures")
        suite_errors = _count_attribute(suite, "errors")
        suite_skipped = _count_attribute(suite, "skipped")
        suite_passed = suite_total - suite_failed - suite_errors - suite_skipped
        if suite_passed < 0:
            raise ValueError("JUnit result counts are inconsistent.")

        passed += suite_passed
        failed += suite_failed
        errors += suite_errors
        skipped += suite_skipped
        total += suite_total

    return passed, failed, errors, skipped, total


def _count_attribute(
    suite: ElementTree.Element,
    name: str,
    *,
    required: bool = False,
) -> int:
    raw_value = suite.get(name)
    if raw_value is None:
        if required:
            raise ValueError(f"JUnit test suite is missing its {name} count.")
        return 0
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"JUnit test suite has an invalid {name} count.") from exc
    if value < 0:
        raise ValueError(f"JUnit test suite has a negative {name} count.")
    return value


def _append_failure(
    failures: list[_TestFailure],
    test_case: ElementTree.Element,
    outcome: ElementTree.Element,
) -> None:
    if len(failures) >= _MAX_FAILURES:
        return

    class_name = (test_case.get("classname") or "").strip()
    test_name = (test_case.get("name") or "").strip()
    test_id = "::".join(part for part in (class_name, test_name) if part) or "unknown"
    attribute_message = (outcome.get("message") or "").strip()
    text_message = (outcome.text or "").strip()
    raw_message = attribute_message or text_message or _local_name(outcome.tag)
    message = " ".join(raw_message.split())
    failures.append(_TestFailure(test_id=test_id, message=message))


def _local_name(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1]


def _decode_output(output: bytes | str | None) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace").strip()
    return output.strip()
