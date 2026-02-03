"""learn_vrc_tst tests."""

import learn_vrc_tst


def test_stub() -> None:
    """Stub test to ensure the test suite runs."""
    print(learn_vrc_tst.__version__)  # noqa: T201


def test__greet() -> None:
    """Test the greet function."""
    assert learn_vrc_tst.greet("World") == "Hello, World!"


def test__greet_jim() -> None:
    """Test the greet_jim function."""
    assert learn_vrc_tst.greet_jim() == "Hello, Jim!"
