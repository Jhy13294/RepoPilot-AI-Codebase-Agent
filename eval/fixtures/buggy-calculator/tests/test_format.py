from calculator.format import format_percent


def test_format_percent_scales_fractional_value() -> None:
    assert format_percent(0.5) == "50%"
