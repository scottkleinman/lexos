"""Test the public API for lexos.rolling_windows."""

from lexos.rolling_windows import Windows


def test_rolling_windows_exports():
    """Verify that rolling_windows components are exported correctly."""
    assert Windows is not None


def test_rolling_windows_all():
    """Verify that __all__ is correctly set for rolling_windows."""
    import lexos.rolling_windows

    assert set(lexos.rolling_windows.__all__) == {"Windows"}
