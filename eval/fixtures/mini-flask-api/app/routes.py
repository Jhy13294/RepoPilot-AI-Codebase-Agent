"""Request handlers for the todo API."""

import sqlite3
from datetime import date
from typing import cast

from app.store import TodoStore
from app.validation import validate_payload

_STORE = TodoStore()


def create_todo(payload: object) -> tuple[int, dict[str, object]]:
    """Validate an incoming payload and persist a new todo."""
    try:
        validated = validate_payload(payload)
        todo = _STORE.insert(
            title=cast(str, validated["title"]),
            due_date=cast(date, validated["due_date"]),
        )
    except ValueError as exc:
        return 422, {"error": str(exc)}
    except sqlite3.DatabaseError:
        return 500, {"error": "database write failed"}

    return 201, todo
