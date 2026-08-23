"""Tests for the seetrees comparison submodule.

Coverage: 100%

Last Updated: August 23, 2026
"""

from unittest.mock import patch

import pandas as pd
import pytest

from lexos.cluster.seetrees import (
    DifferencePlot,
    DistinctiveFeaturePlot,
    OverlayPlot,
    SeeTrees,
)
from lexos.cluster.seetrees.comparison import ComparisonPlot


def test_get_difference_plot_returns_difference_plot():
    """get_difference_plot should return a DifferencePlot object."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 0.5], "word2": [0.2, 0.8]},
        index=["doc1", "doc2"],
    )
    st = SeeTrees(frequencies=frequencies)

    plotter = st.get_difference_plot("doc1", "doc2", top_diff=2)

    assert isinstance(plotter, DifferencePlot)
    assert plotter.fig is not None


def test_get_difference_plot_sanitizes_whitespace_and_linebreaks():
    """get_difference_plot should sanitize whitespace and linebreaks in labels."""
    frequencies = pd.DataFrame(
        {
            "word1": [1.0, 0.5],
            "with space": [0.2, 0.8],
            "with\nline": [0.3, 0.7],
        },
        index=["doc1", "doc2"],
    )
    st = SeeTrees(frequencies=frequencies)

    plotter = st.get_difference_plot("doc1", "doc2", top_diff=3)
    texts = [t.get_text() for t in plotter.fig.axes[0].texts]

    assert "word1" in texts
    assert "with<whitespace>space" in texts
    assert "with<linebreak>line" in texts


def test_get_difference_plot_missing_text_raises_error():
    """get_difference_plot should raise if a source or target label is missing."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 0.5], "word2": [0.2, 0.8]},
        index=["doc1", "doc2"],
    )
    st = SeeTrees(frequencies=frequencies)

    with pytest.raises(ValueError, match="Source text 'docX' not found"):
        st.get_difference_plot("docX", "doc2")


def test_get_difference_plot_accepts_custom_title_and_colors():
    """get_difference_plot should accept a custom title and bar colors."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 0.5], "word2": [0.2, 0.8]},
        index=["doc1", "doc2"],
    )
    st = SeeTrees(frequencies=frequencies)

    plotter = st.get_difference_plot(
        "doc1",
        "doc2",
        top_diff=1,
        title="Custom Difference Title",
        base_color="blue",
        highlight_color="orange",
    )

    assert plotter.title == "Custom Difference Title"
    assert plotter.base_color == "blue"
    assert plotter.highlight_color == "orange"
    assert plotter.fig.axes[0].get_title() == "Custom Difference Title"


def test_get_overlay_plot_returns_overlay_plot():
    """get_overlay_plot should return an OverlayPlot object."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 0.5], "word2": [0.2, 0.8]},
        index=["doc1", "doc2"],
    )
    st = SeeTrees(frequencies=frequencies)

    plotter = st.get_overlay_plot("doc1", "doc2", top_diff=2)

    assert isinstance(plotter, OverlayPlot)
    assert plotter.fig is not None


def test_get_overlay_plot_accepts_custom_title_and_colors():
    """get_overlay_plot should accept a custom title and line colors."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 0.5], "word2": [0.2, 0.8]},
        index=["doc1", "doc2"],
    )
    st = SeeTrees(frequencies=frequencies)

    plotter = st.get_overlay_plot(
        "doc1",
        "doc2",
        top_diff=1,
        title="Custom Overlay Title",
        source_color="purple",
        target_color="green",
    )

    assert plotter.title == "Custom Overlay Title"
    assert plotter.source_color == "purple"
    assert plotter.target_color == "green"
    assert plotter.fig.axes[0].get_title() == "Custom Overlay Title"


def test_disable_canvas_bbox_inches_returns_when_canvas_missing():
    """_disable_canvas_bbox_inches should safely return when fig has no canvas."""
    plotter = ComparisonPlot()

    class DummyFig:
        pass

    plotter._disable_canvas_bbox_inches(DummyFig())


def test_disable_canvas_bbox_inches_strips_bbox_inches_from_print_figure():
    """_disable_canvas_bbox_inches should remove bbox_inches from print_figure kwargs."""
    plotter = ComparisonPlot()

    class DummyCanvas:
        def __init__(self):
            self.kwargs_passed = None

        def print_figure(self, *args, **kwargs):
            self.kwargs_passed = kwargs
            return "ok"

    class DummyFig:
        pass

    fig = DummyFig()
    fig.canvas = DummyCanvas()
    plotter._disable_canvas_bbox_inches(fig)

    result = fig.canvas.print_figure("out.pdf", bbox_inches="tight", dpi=100)
    assert result == "ok"
    assert fig.canvas.kwargs_passed == {"dpi": 100}


def test_overlay_plot_show_recreates_figure_when_missing():
    """OverlayPlot.show should recreate the figure if it was cleared."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 0.5], "word2": [0.2, 0.8]},
        index=["doc1", "doc2"],
    )
    st = SeeTrees(frequencies=frequencies)
    plotter = st.get_overlay_plot("doc1", "doc2", top_diff=2)
    plotter.fig = None

    with patch("matplotlib.pyplot.show") as mock_show:
        plotter.show()

    assert plotter.fig is not None
    mock_show.assert_called_once()


def test_difference_plot_show_recreates_figure_when_missing():
    """DifferencePlot.show should recreate the figure if it was cleared."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 0.5], "word2": [0.2, 0.8]},
        index=["doc1", "doc2"],
    )
    st = SeeTrees(frequencies=frequencies)
    plotter = st.get_difference_plot("doc1", "doc2", top_diff=1)
    plotter.fig = None

    with patch("matplotlib.pyplot.show") as mock_show:
        plotter.show()

    assert plotter.fig is not None
    mock_show.assert_called_once()
