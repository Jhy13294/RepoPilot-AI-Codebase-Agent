"""Interactive CLI approval gate for high-risk tool calls."""

from collections.abc import Callable

from pydantic import BaseModel
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.pretty import Pretty
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from app.tools.base import ToolContext
from app.tools.registry import ApprovalOutcome, ToolSpec

_APPROVE_RESPONSES = frozenset({"y", "yes"})
_DENY_RESPONSES = frozenset({"n", "no"})


class CliApprovalGate:
    """Present high-risk requests in a terminal and require explicit approval."""

    def __init__(
        self,
        *,
        console: Console | None = None,
        prompt: Callable[[str], str] | None = None,
    ) -> None:
        self._console = console if console is not None else Console()
        self._prompt = prompt if prompt is not None else input

    def check(
        self,
        spec: ToolSpec,
        args: BaseModel,
        context: ToolContext,
    ) -> ApprovalOutcome:
        """Render one request and return a fail-closed interactive decision."""
        self._console.print(_render_request(spec, args, context))
        try:
            response = self._prompt(
                "Approve this high-risk action? "
                "[y/yes to approve; n/no optionally followed by a reason to deny]: "
            )
        except EOFError:
            return ApprovalOutcome(
                approved=False,
                reason="Approval denied because input ended before an explicit decision.",
            )
        return _parse_decision(response)


def _render_request(spec: ToolSpec, args: BaseModel, context: ToolContext) -> Group:
    """Build the rich approval preview without performing any input or output."""
    rendered_args = args.model_dump(mode="json")
    rationale = rendered_args.get("rationale")
    diff = rendered_args.get("diff")

    heading = Text("Approval required ", style="bold")
    heading.append(" HIGH RISK ", style="bold white on red")

    metadata = Table.grid(padding=(0, 1))
    metadata.add_column(style="bold", no_wrap=True)
    metadata.add_column()
    metadata.add_row("Tool", spec.name)
    metadata.add_row("Run ID", context.run_id)
    if isinstance(rationale, str):
        metadata.add_row("Rationale", rationale)

    sections: list[RenderableType] = [heading, metadata]
    if isinstance(diff, str):
        sections.append(
            Panel(
                Syntax(
                    diff,
                    "diff",
                    background_color="default",
                    word_wrap=True,
                ),
                title="Diff",
                border_style="red",
            )
        )

    generic_args = {
        key: value for key, value in rendered_args.items() if key not in {"diff", "rationale"}
    }
    if generic_args:
        sections.append(
            Panel(
                Pretty(generic_args, expand_all=True),
                title="Other arguments" if isinstance(diff, str) else "Arguments",
                border_style="yellow",
            )
        )

    return Group(*sections)


def _parse_decision(response: str) -> ApprovalOutcome:
    """Parse a CLI decision, approving only an exact y or yes response."""
    decision = response.strip()
    normalized = decision.casefold()

    if normalized in _APPROVE_RESPONSES:
        return ApprovalOutcome(approved=True)

    if not decision:
        return ApprovalOutcome(
            approved=False,
            reason="Approval denied because no explicit approval was provided.",
        )

    denial_note = _extract_denial_note(decision)
    if denial_note is not None:
        reason = "Approval denied by the user."
        if denial_note:
            reason = f"Approval denied by the user: {denial_note}"
        return ApprovalOutcome(approved=False, reason=reason)

    return ApprovalOutcome(
        approved=False,
        reason=(
            f"Approval denied because response {decision!r} was ambiguous; "
            "only an explicit y or yes approves."
        ),
    )


def _extract_denial_note(decision: str) -> str | None:
    normalized = decision.casefold()
    if normalized in _DENY_RESPONSES:
        return ""

    for prefix in ("n:", "no:"):
        if normalized.startswith(prefix):
            return decision[len(prefix) :].strip()

    command, separator, note = decision.partition(" ")
    if separator and command.casefold() in _DENY_RESPONSES:
        return note.strip()
    return None
