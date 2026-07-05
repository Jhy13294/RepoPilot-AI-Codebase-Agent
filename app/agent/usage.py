"""Shared usage accounting helpers for agent roles."""

from app.schemas.llm_io import Usage


class UsageAccumulator:
    """Accumulate provider usage across multiple completion calls."""

    def __init__(self) -> None:
        self.tokens_in = 0
        self.tokens_out = 0
        self.cost_usd = 0.0
        self.cost_known = True

    def add(self, usage: Usage) -> None:
        """Add one provider usage record."""
        self.tokens_in += usage.tokens_in
        self.tokens_out += usage.tokens_out
        if usage.cost_usd is None:
            self.cost_known = False
            return
        self.cost_usd += usage.cost_usd

    def snapshot(self) -> Usage:
        """Return an immutable usage snapshot."""
        return Usage(
            tokens_in=self.tokens_in,
            tokens_out=self.tokens_out,
            cost_usd=self.cost_usd if self.cost_known else None,
        )
