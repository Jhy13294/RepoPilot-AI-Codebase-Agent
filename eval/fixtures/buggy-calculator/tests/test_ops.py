from calculator.ops import divide


def test_divide_two_negative_numbers_is_positive() -> None:
    assert divide(-8, -2) == 4.0
