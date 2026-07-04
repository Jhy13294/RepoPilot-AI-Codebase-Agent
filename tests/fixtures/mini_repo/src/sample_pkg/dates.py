from datetime import date, datetime

_INPUT_FORMATS = ("%Y-%m-%d", "%d/%m/%Y")


def parse_date(value: str) -> date:
    stripped = value.strip()
    for fmt in _INPUT_FORMATS:
        try:
            return datetime.strptime(stripped, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unsupported date format: {value!r}")
