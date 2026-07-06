"""Reporter role with constrained JSON output and no trace side effects."""

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.agent.state import TaskSpec
from app.agent.usage import UsageAccumulator
from app.schemas.agent_io import AnalysisReport, ReportConfidence
from app.schemas.llm_io import LLMMessage, Role, StopReason, Usage
from app.services.llm_client import LLMClient

_JSON_FENCE_RE = re.compile(r"\A```json\s*(?P<body>.*?)\s*```\Z", flags=re.DOTALL)


class ReporterErrorReason(StrEnum):
    """Terminal reporter failure categories."""

    refusal = "refusal"
    repair_exhausted = "repair_exhausted"


class ReporterError(Exception):
    """Raised when the reporter cannot produce a valid analysis report."""

    def __init__(
        self,
        reason: ReporterErrorReason,
        message: str,
        usage: Usage,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.usage = usage


class _ReportDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    headline: str
    analysis: str
    confidence: ReportConfidence
    open_questions: list[str] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _ParseFailure:
    reason: ReporterErrorReason
    message: str


class Reporter:
    """Synthesize the run trace into a model-authored report."""

    def __init__(self, client: LLMClient, *, max_repairs: int = 2) -> None:
        if max_repairs < 0:
            raise ValueError("max_repairs must be non-negative.")

        self._client = client
        self._system_prompt = _reporter_system_prompt()
        self._max_repairs = max_repairs

    def report(
        self,
        task: TaskSpec,
        *,
        outcome: Literal["succeeded", "failed"],
        final_findings: str,
        timeline_digest: str,
        failure_summary: str | None = None,
    ) -> AnalysisReport:
        """Produce a validated analysis report without mutating run state."""
        messages = [
            LLMMessage(
                role=Role.user,
                content=_reporter_user_prompt(
                    task,
                    outcome=outcome,
                    final_findings=final_findings,
                    timeline_digest=timeline_digest,
                    failure_summary=failure_summary,
                ),
            )
        ]
        usage = UsageAccumulator()
        attempts = 0
        max_attempts = self._max_repairs + 1
        last_failure: _ParseFailure | None = None

        while attempts < max_attempts:
            response = self._client.complete(
                messages,
                tools=None,
                system=self._system_prompt,
                temperature=None,
            )
            attempts += 1
            usage.add(response.usage)

            if response.stop_reason is StopReason.refusal:
                raise ReporterError(
                    ReporterErrorReason.refusal,
                    _refusal_message(response.message.content),
                    usage.snapshot(),
                )

            content = response.message.content
            draft, failure = _parse_report_draft(content)
            if draft is not None:
                return AnalysisReport(
                    headline=draft.headline,
                    analysis=draft.analysis,
                    confidence=draft.confidence,
                    open_questions=draft.open_questions,
                    usage=usage.snapshot(),
                )

            last_failure = failure
            if attempts < max_attempts:
                messages.append(LLMMessage(role=Role.assistant, content=content))
                messages.append(_repair_message(content, failure))

        assert last_failure is not None
        raise ReporterError(
            last_failure.reason,
            f"Reporter failed after {attempts} attempt(s): {last_failure.message}",
            usage.snapshot(),
        )


def _reporter_system_prompt() -> str:
    return "\n\n".join(
        [
            (
                "Role & mission\n"
                "You are RepoPilot's Reporter. Turn the supplied run trace into a concise "
                "AnalysisReport for a codebase task."
            ),
            (
                "Evidence boundary\n"
                "Synthesize only from the supplied task, final findings, timeline digest, and "
                "failure summary. Do not invent files, line numbers, tool results, or outcomes."
            ),
            (
                "Report policy\n"
                "Write a useful headline and analysis for the actual outcome. Use open_questions "
                "only for unresolved facts or follow-up checks that remain after the run."
            ),
            (
                "Output contract\n"
                "Return only a JSON object shaped exactly as "
                '{"headline":str,"analysis":str,"confidence":"high|medium|low",'
                '"open_questions":[str]}. Do not include markdown, prose, usage, comments, or '
                "extra fields."
            ),
        ]
    )


def _reporter_user_prompt(
    task: TaskSpec,
    *,
    outcome: Literal["succeeded", "failed"],
    final_findings: str,
    timeline_digest: str,
    failure_summary: str | None,
) -> str:
    return "\n\n".join(
        [
            f"Task JSON:\n{task.model_dump_json(indent=2)}",
            f"Outcome:\n{outcome}",
            f"Final findings:\n{final_findings if final_findings else 'None.'}",
            f"Timeline digest:\n{timeline_digest if timeline_digest else 'No trace events.'}",
            f"Failure summary:\n{failure_summary if failure_summary is not None else 'None.'}",
        ]
    )


def _parse_report_draft(content: str) -> tuple[_ReportDraft | None, _ParseFailure]:
    candidate = _strip_json_fence(content.strip())
    try:
        parsed: object = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, _ParseFailure(
            reason=ReporterErrorReason.repair_exhausted,
            message=f"json.loads error: {exc}",
        )

    try:
        return _ReportDraft.model_validate(parsed), _ParseFailure(
            reason=ReporterErrorReason.repair_exhausted,
            message="",
        )
    except ValidationError as exc:
        return None, _ParseFailure(
            reason=ReporterErrorReason.repair_exhausted,
            message=str(exc),
        )


def _strip_json_fence(content: str) -> str:
    match = _JSON_FENCE_RE.fullmatch(content)
    if match is None:
        return content
    return match.group("body").strip()


def _repair_message(content: str, failure: _ParseFailure) -> LLMMessage:
    return LLMMessage(
        role=Role.user,
        content=(
            "Your previous reporter output was invalid.\n"
            "Bad reply:\n"
            f"{content}\n\n"
            "Validation error:\n"
            f"{failure.message}\n\n"
            "Return only corrected JSON with the exact reporter schema: "
            '{"headline":str,"analysis":str,"confidence":"high|medium|low",'
            '"open_questions":[str]}.'
        ),
    )


def _refusal_message(content: str) -> str:
    if content:
        return f"Reporter refused: {content}"
    return "Reporter refused to produce an analysis report."
