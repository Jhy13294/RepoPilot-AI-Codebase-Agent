"""Small calculator package with intentionally seeded bugs."""

from calculator.format import format_percent
from calculator.ops import add, divide, multiply, subtract
from calculator.stats import median

__all__ = ["add", "divide", "format_percent", "median", "multiply", "subtract"]
