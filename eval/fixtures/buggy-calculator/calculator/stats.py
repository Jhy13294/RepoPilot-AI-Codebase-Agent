"""Statistical helpers for calculator output."""


def median(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("median requires at least one value")

    middle = len(ordered) // 2
    if len(ordered) % 2 == 0:
        return float(ordered[middle])

    return float(ordered[middle])
