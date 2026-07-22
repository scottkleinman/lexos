"""test_init.py.

Test the public API for lexos.cutter.

Coverage: 100%
Last Updated: 22 July, 2026
"""

from lexos.cutter import TextCutter, TokenCutter


def test_cutter_exports():
    """Verify that cutter components are exported correctly."""
    assert TextCutter is not None
    assert TokenCutter is not None


def test_cutter_all():
    """Verify that __all__ is correctly set for cutter."""
    import lexos.cutter

    assert set(lexos.cutter.__all__) == {"TextCutter", "TokenCutter"}
