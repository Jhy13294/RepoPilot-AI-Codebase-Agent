"""SQLite persistence for todo items."""

import sqlite3
from datetime import date


class TodoStore:
    """Store todo items in a SQLite database."""

    def __init__(self, connection: sqlite3.Connection | None = None) -> None:
        self._connection = connection or sqlite3.connect(":memory:")
        self._create_table()

    def _create_table(self) -> None:
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS todos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL CHECK (length(trim(title)) > 0),
                due_date TEXT NOT NULL
            )
            """
        )
        self.commit()

    def insert(self, title: str, due_date: date) -> dict[str, object]:
        """Insert a todo and return its public representation."""
        cursor = self._connection.execute(
            "INSERT INTO todos (title, due_date) VALUES (?, ?)",
            (title, due_date.isoformat()),
        )
        self.commit()
        return {
            "id": cursor.lastrowid,
            "title": title,
            "due_date": due_date.isoformat(),
        }

    def commit(self) -> None:
        """Commit the current SQLite transaction."""
        self._connection.commit()
