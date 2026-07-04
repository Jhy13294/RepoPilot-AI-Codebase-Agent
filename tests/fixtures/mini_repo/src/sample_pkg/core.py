from sample_pkg.dates import parse_date

DEFAULT_INPUT = "2026-07-04"


def normalize_due_date(value: str = DEFAULT_INPUT) -> str:
    return parse_date(value).isoformat()
