"""Formatting helpers for calculator output."""


def format_currency(value: float) -> str:
    return f"${value:.2f}"


def format_percent(value: float) -> str:
    return f"{value:g}%"
