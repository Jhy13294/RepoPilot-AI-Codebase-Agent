"""Validation helpers for incoming todo payloads."""

from datetime import date


def parse_date(raw: object) -> date:
    """Parse an ISO-formatted date or raise ValueError."""
    if not isinstance(raw, str):
        raise ValueError("due_date must be an ISO-formatted string")

    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError("due_date must be a valid ISO date") from exc


def validate_payload(payload: object) -> dict[str, object]:
    """Validate and normalize a payload used to create a todo."""
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")

    title = payload.get("title")
    if not isinstance(title, str):
        raise ValueError("title must be a string")

    return {
        "title": title,
        "due_date": parse_date(payload.get("due_date")),
    }
