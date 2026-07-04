def test_fixture_tests_are_ignored_by_root_pytest() -> None:
    raise AssertionError("Fixture tests must not be collected by the root test suite.")
