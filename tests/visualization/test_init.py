"""test_init.py.

Test the public API for lexos.visualization.

Coverage: 100%
Last Updated: 22 July, 2026
"""

from lexos.visualization import BubbleViz, WordCloud


def test_visualization_exports():
    """Verify that visualization components are exported correctly."""
    assert WordCloud is not None
    assert BubbleViz is not None


def test_visualization_all():
    """Verify that __all__ is correctly set for visualization."""
    import lexos.visualization

    assert set(lexos.visualization.__all__) == {"WordCloud", "BubbleViz"}
