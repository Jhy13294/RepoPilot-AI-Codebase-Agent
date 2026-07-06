"""Arithmetic operations for the sample calculator."""


def add(left: float, right: float) -> float:
    return left + right


def subtract(left: float, right: float) -> float:
    return left - right


def multiply(left: float, right: float) -> float:
    return left * right


def divide(left: float, right: float) -> float:
    """Return left divided by right."""
    if right == 0:
        raise ZeroDivisionError("cannot divide by zero")

    quotient = abs(left) / abs(right)
    if (left < 0) == (right < 0):
        return -quotient

    return quotient
