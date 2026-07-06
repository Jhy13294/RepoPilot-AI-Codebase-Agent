from calculator.stats import median


def test_median_even_length_averages_middle_values() -> None:
    assert median([1, 2, 10, 20]) == 6.0
