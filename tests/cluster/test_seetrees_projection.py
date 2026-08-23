"""Tests for the seetrees projection submodule.

Coverage: 98%. Missing: 133, 163, 196

Last Updated: August 23, 2026
"""

from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from lexos.cluster.seetrees import MDS, PCA, SeeTrees
from lexos.cluster.seetrees.projection_plot import ProjectionPlot


def test_get_mds_plot_can_be_created_from_precomputed_distance_table():
    """get_mds_plot should create an MDS plot object with a precomputed distance table."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 0.5], "word2": [0.2, 0.8]},
        index=["doc1", "doc2"],
    )
    distance_table = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=["doc1", "doc2"],
        columns=["doc1", "doc2"],
    )
    st = SeeTrees(frequencies=frequencies, distance_table=distance_table)

    plotter = st.get_mds_plot()

    assert isinstance(plotter, MDS)
    assert plotter.fig is None


def test_get_mds_plot_requires_distance_table():
    """get_mds_plot should require a precomputed distance table."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 0.2], "word2": [0.3, 0.8]},
        index=["doc1", "doc2"],
    )
    st = SeeTrees(frequencies=frequencies)

    with pytest.raises(
        ValueError, match="Distance table is required to create an MDS plot"
    ):
        st.get_mds_plot()


def test_get_mds_plot_returns_plotter_without_showing():
    """get_mds_plot should create the plot object without displaying."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 0.2], "word2": [0.3, 0.8]},
        index=["doc1", "doc2"],
    )
    distance_table = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=["doc1", "doc2"],
        columns=["doc1", "doc2"],
    )
    st = SeeTrees(frequencies=frequencies, distance_table=distance_table)

    plotter = st.get_mds_plot()

    assert plotter is not None
    assert isinstance(plotter, MDS)
    assert plotter.fig is None


def test_get_mds_plot_show_calls_matplotlib_show():
    """Calling show() on an MDS plot should display the plot."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 0.2], "word2": [0.3, 0.8]},
        index=["doc1", "doc2"],
    )
    distance_table = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=["doc1", "doc2"],
        columns=["doc1", "doc2"],
    )
    st = SeeTrees(frequencies=frequencies, distance_table=distance_table)

    plotter = st.get_mds_plot()

    with patch("matplotlib.pyplot.show") as mock_show:
        plotter.show()

    mock_show.assert_called_once()


def test_get_pca_plot_can_be_created_without_distance_table():
    """get_pca_plot should create a PCA plot object even without a distance table."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 0.5], "word2": [0.2, 0.8]},
        index=["doc1", "doc2"],
    )
    st = SeeTrees(frequencies=frequencies)

    plotter = st.get_pca_plot()

    assert isinstance(plotter, PCA)
    assert plotter.fig is None


def test_get_pca_plot_returns_plotter_without_showing():
    """get_pca_plot should create the plot object without displaying."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 0.2], "word2": [0.3, 0.8]},
        index=["doc1", "doc2"],
    )
    st = SeeTrees(frequencies=frequencies)

    plotter = st.get_pca_plot()

    assert plotter is not None
    assert isinstance(plotter, PCA)
    assert plotter.fig is None


def test_get_pca_plot_show_calls_matplotlib_show():
    """Calling show() on a PCA plot should display the plot."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 0.2], "word2": [0.3, 0.8]},
        index=["doc1", "doc2"],
    )
    st = SeeTrees(frequencies=frequencies)

    plotter = st.get_pca_plot()

    with patch("matplotlib.pyplot.show") as mock_show:
        plotter.show()

    mock_show.assert_called_once()


def test_projection_init_missing_author_raises_value_error():
    """ProjectionPlot should validate an explicit author against labels."""
    distance_table = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=["Poe_1", "Austen_1"],
        columns=["Poe_1", "Austen_1"],
    )

    with pytest.raises(ValueError, match="Author 'Shakespeare' is not present"):
        ProjectionPlot(
            distance_table=distance_table,
            labels=["Poe_1", "Austen_1"],
            author="Shakespeare",
        )


def test_mds_show_on_init_calls_plot_and_show():
    """show_on_init should plot the MDS and call matplotlib.show."""
    distance_table = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=["doc1", "doc2"],
        columns=["doc1", "doc2"],
    )
    with patch("matplotlib.pyplot.show") as mock_show:
        plotter = MDS(
            distance_table=distance_table, labels=["doc1", "doc2"], show_on_init=True
        )

    assert plotter.fig is not None
    mock_show.assert_called_once()


def test_build_density_frame_returns_expected_dataframe():
    """_build_density_frame should extract distances and author groups."""
    distance_table = pd.DataFrame(
        [[0.0, 2.0, 4.0], [2.0, 0.0, 3.0], [4.0, 3.0, 0.0]],
        index=["Poe_1", "Poe_2", "Austen_1"],
        columns=["Poe_1", "Poe_2", "Austen_1"],
    )
    plotter = ProjectionPlot(
        distance_table=distance_table, labels=["Poe_1", "Poe_2", "Austen_1"]
    )
    df = plotter._build_density_frame(pattern=r"^.*?(?=[_\s-])")

    assert list(df["class"]) == ["Poe", "Austen", "Austen"]
    assert set(df["same_author"]) == {"True", "False"}


def test_projection_compute_coords_not_implemented():
    """Base ProjectionPlot._compute_coords should raise NotImplementedError."""
    distance_table = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=["a", "b"],
        columns=["a", "b"],
    )
    plotter = ProjectionPlot(distance_table=distance_table, labels=["a", "b"])

    with pytest.raises(NotImplementedError):
        plotter._compute_coords()


def test_disable_canvas_bbox_inches_return_path():
    """_disable_canvas_bbox_inches should return cleanly for a missing canvas."""
    distance_table = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=["a", "b"],
        columns=["a", "b"],
    )
    plotter = ProjectionPlot(distance_table=distance_table, labels=["a", "b"])
    plotter._disable_canvas_bbox_inches(plt.Figure())


def test_disable_canvas_bbox_inches_strips_bbox_inches():
    """_disable_canvas_bbox_inches should remove bbox_inches from print_figure kwargs."""
    distance_table = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=["a", "b"],
        columns=["a", "b"],
    )
    plotter = ProjectionPlot(distance_table=distance_table, labels=["a", "b"])

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


def test_extract_label_class_fallbacks():
    """_extract_label_class should use separators and digit fallback when regex misses."""
    distance_table = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=["a", "b"],
        columns=["a", "b"],
    )
    plotter = ProjectionPlot(distance_table=distance_table, labels=["a", "b"])

    assert plotter._extract_label_class("Poe_1", r"^notmatch$") == "Poe"
    assert plotter._extract_label_class("Austen 2", r"^notmatch$") == "Austen"
    assert plotter._extract_label_class("Shakespeare-3", r"^notmatch$") == "Shakespeare"
    assert plotter._extract_label_class("UniqueLabel", r"^notmatch$") == "UniqueLabel"


def test_highlight_author_returns_when_none():
    """_highlight_author should do nothing when author is not set."""
    distance_table = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=["Poe_1", "Austen_1"],
        columns=["Poe_1", "Austen_1"],
    )
    plotter = MDS(
        distance_table=distance_table, labels=["Poe_1", "Austen_1"], author=None
    )
    fig, ax = plt.subplots()
    coords = np.array([[0.0, 1.0], [2.0, 3.0]])

    plotter._highlight_author(ax, coords)
    assert ax.get_title() == ""


def test_highlight_author_plots_points_and_sets_title():
    """_highlight_author should plot author points and set a highlight title."""
    distance_table = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=["Poe_1", "Austen_1"],
        columns=["Poe_1", "Austen_1"],
    )
    plotter = MDS(
        distance_table=distance_table, labels=["Poe_1", "Austen_1"], author="Poe"
    )
    fig, ax = plt.subplots()
    coords = np.array([[0.0, 1.0], [2.0, 3.0]])

    plotter._highlight_author(ax, coords)
    assert "Highlights for Poe" in ax.get_title()


def test_offset_ratio_not_implemented_in_base():
    """Base ProjectionPlot._offset_ratio should raise NotImplementedError."""
    distance_table = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=["a", "b"],
        columns=["a", "b"],
    )
    plotter = ProjectionPlot(distance_table=distance_table, labels=["a", "b"])

    with pytest.raises(NotImplementedError):
        plotter._offset_ratio()


def test_title_suffix_not_implemented_in_base():
    """Base ProjectionPlot._title_suffix should raise NotImplementedError."""
    distance_table = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=["a", "b"],
        columns=["a", "b"],
    )
    plotter = ProjectionPlot(distance_table=distance_table, labels=["a", "b"])

    with pytest.raises(NotImplementedError):
        plotter._title_suffix()


def test_plot_base_zero_offset_falls_back_for_projection():
    """_plot_base should use fallback offsets when coords have zero range."""
    distance_table = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=["Poe_1", "Austen_1"],
        columns=["Poe_1", "Austen_1"],
    )
    plotter = MDS(distance_table=distance_table, labels=["Poe_1", "Austen_1"])
    fig, ax = plt.subplots()
    coords = np.zeros((2, 2))

    plotter._plot_base(ax, coords, "MDS")
    texts = [text.get_position() for text in ax.texts]

    assert all(pos[0] == 0.5 for pos in texts)
    assert all(pos[1] == 0.5 for pos in texts)


def test_pca_plot_base_zero_offset_falls_back():
    """PCA._plot_base should use fallback offsets when coords have zero range."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 1.0], "word2": [1.0, 1.0]},
        index=["doc1", "doc2"],
    )
    plotter = PCA(frequencies=frequencies, labels=["doc1", "doc2"], random_state=42)
    fig, ax = plt.subplots()
    coords = np.zeros((2, 2))

    plotter._plot_base(ax, coords, "PCA")
    texts = [text.get_position() for text in ax.texts]

    assert all(pos[0] == 0.5 for pos in texts)
    assert all(pos[1] == 0.5 for pos in texts)
