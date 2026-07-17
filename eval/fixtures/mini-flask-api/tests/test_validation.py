from datetime import date

import pytest

from app.validation import parse_date, validate_payload


def test_parse_date_accepts_iso_date() -> None:
    assert parse_date("2026-07-17") == date(2026, 7, 17)


def test_parse_date_rejects_impossible_date() -> None:
    with pytest.raises(ValueError, match="valid ISO date"):
        parse_date("2026-02-30")


def test_parse_date_rejects_non_string_value() -> None:
    with pytest.raises(ValueError, match="ISO-formatted string"):
        parse_date(None)


def test_validate_payload_normalizes_due_date() -> None:
    validated = validate_payload({"title": "Ship release", "due_date": "2026-07-18"})

    assert validated == {
        "title": "Ship release",
        "due_date": date(2026, 7, 18),
    }


def test_validate_payload_rejects_non_object() -> None:
    with pytest.raises(ValueError, match="payload must be an object"):
        validate_payload(["not", "an", "object"])


def test_validate_payload_rejects_non_string_title() -> None:
    with pytest.raises(ValueError, match="title must be a string"):
        validate_payload({"title": 42, "due_date": "2026-07-18"})
