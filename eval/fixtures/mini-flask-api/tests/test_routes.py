from app.routes import create_todo


def test_create_todo_persists_valid_payload() -> None:
    status, body = create_todo(
        {"title": "Ship release", "due_date": "2026-07-18"}
    )

    assert status == 201
    assert body["id"] == 1
    assert body["title"] == "Ship release"
    assert body["due_date"] == "2026-07-18"


def test_create_todo_returns_422_for_invalid_date() -> None:
    status, body = create_todo(
        {"title": "Ship release", "due_date": "not-a-date"}
    )

    assert status == 422
    assert body == {"error": "due_date must be a valid ISO date"}


def test_create_todo_returns_422_for_non_object_payload() -> None:
    status, body = create_todo("not an object")

    assert status == 422
    assert body == {"error": "payload must be an object"}


def test_create_todo__empty_title_returns_422() -> None:
    status, body = create_todo({"title": "", "due_date": "2026-07-18"})

    assert status == 422
    assert body == {"error": "title must not be empty"}
