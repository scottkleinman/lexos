"""Core SeeTrees tests.

This file contains tests for SeeTrees initialization, data loading, distance
computation, and high-level feature summary behavior.

Coverage: 98%. Missing: 87, 509

Last Updated: August 23, 2026
"""

import numpy as np
import pandas as pd
import pytest

from lexos.cluster.seetrees import (
    MDS,
    PCA,
    DensityPlot,
    DifferencePlot,
    DistinctiveFeaturePlot,
    OverlayPlot,
    SeeTrees,
    Tree,
)
from lexos.dtm import DTM


def test_seetrees_initializes_from_dtm():
    """SeeTrees should accept a Lexos DTM and convert it to a frequency table."""
    dtm = DTM()
    dtm(docs=[["apple", "banana"], ["apple", "cherry"]], labels=["doc1", "doc2"])

    st = SeeTrees(dtm=dtm)

    assert not st.frequencies.empty
    assert set(st.frequencies.index) == {"doc1", "doc2"}
    assert "apple" in st.frequencies.columns


def test_seetrees_initializes_from_dtm_with_sparse_data():
    """SeeTrees should compute distances when DTM frequencies are sparse."""
    dtm = DTM()
    dtm(
        docs=[
            ["apple", "banana"],
            ["apple", "cherry"],
            ["banana", "cherry"],
            ["apple", "banana", "cherry"],
        ],
        labels=["doc1", "doc2", "doc3", "doc4"],
    )

    st = SeeTrees(dtm=dtm)
    distance_table = st.compute_distances(metric="delta")

    assert distance_table.shape == (4, 4)
    assert np.allclose(np.diag(distance_table), np.zeros(4))


def test_seetrees_initializes_from_frequency_dataframe():
    """SeeTrees should preserve a supplied frequency DataFrame."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 0.5], "word2": [0.2, 0.8]},
        index=["doc1", "doc2"],
    )

    st = SeeTrees(frequencies=frequencies)

    assert st.frequencies.equals(frequencies)


def test_compute_distances_delta_returns_matrix():
    """compute_distances should return a valid distance matrix for delta."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 0.5], "word2": [0.2, 0.8]},
        index=["doc1", "doc2"],
    )
    st = SeeTrees(frequencies=frequencies)

    distance_table = st.compute_distances(metric="delta")

    assert distance_table.shape == (2, 2)
    assert np.allclose(np.diag(distance_table), np.zeros(2))


def test_compute_distances_invalid_metric_raises_error():
    """An unknown metric should raise a ValueError."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 0.5], "word2": [0.2, 0.8]},
        index=["doc1", "doc2"],
    )
    st = SeeTrees(frequencies=frequencies)

    with pytest.raises(ValueError, match="Unknown metric"):
        st.compute_distances(metric="invalid_metric")


def test_seetrees_initializes_from_stylo_res():
    """SeeTrees should initialize from a stylo_res dictionary."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 2.0], "word2": [0.5, 0.5]},
        index=["doc1", "doc2"],
    )
    distance_table = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]], index=["docA", "docB"], columns=["docA", "docB"]
    )
    stylo_res = {
        "frequencies": frequencies,
        "distance_table": distance_table,
        "features": ["word1", "word2"],
    }

    st = SeeTrees(stylo_res=stylo_res)

    assert st.features == ["word1", "word2"]
    assert st.labels == ["docA", "docB"]
    assert st.frequencies.equals(frequencies)


def test_init_labels_prefers_distance_table_index():
    """Labels should be derived from the distance table when available."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 2.0]},
        index=["doc1", "doc2"],
    )
    distance_table = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]], index=["d1", "d2"], columns=["d1", "d2"]
    )

    st = SeeTrees(frequencies=frequencies, distance_table=distance_table)

    assert st.labels == ["d1", "d2"]


def test_get_feature_summary_runs_without_error():
    """get_feature_summary should execute without error for a valid document."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 0.5], "word2": [0.2, 0.8]},
        index=["doc1", "doc2"],
    )
    st = SeeTrees(frequencies=frequencies)

    df = st.get_feature_summary(target_text="doc1", top=2)
    assert isinstance(df, pd.DataFrame)


def test_compute_distances_other_metrics():
    """compute_distances should support additional metric values."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 0.5, 1.0], "word2": [0.2, 0.8, 0.5]},
        index=["doc1", "doc2", "doc3"],
    )
    st = SeeTrees(frequencies=frequencies)

    for metric in ["eder_delta", "cosine_delta", "manhattan", "cosine"]:
        distance_table = st.compute_distances(metric=metric)
        assert distance_table.shape == (3, 3)
        assert np.allclose(np.diag(distance_table), np.zeros(3))


def test_apply_figure_layout_disables_canvas_bbox_inches():
    """_apply_figure_layout should adjust the figure and wrap print_figure."""
    frequencies = pd.DataFrame({"word1": [1.0], "word2": [2.0]}, index=["doc1"])
    st = SeeTrees(frequencies=frequencies)

    class DummyCanvas:
        def __init__(self):
            self.kwargs_passed = None

        def print_figure(self, *args, **kwargs):
            self.kwargs_passed = kwargs
            return "printed"

    class DummyFigure:
        def __init__(self):
            self.subplots_adjust_called = False
            self.canvas = DummyCanvas()

        def subplots_adjust(self, left, right, top, bottom):
            self.subplots_adjust_called = True
            self.left = left
            self.right = right
            self.top = top
            self.bottom = bottom

    fig = DummyFigure()
    original_print_figure = fig.canvas.print_figure

    st._apply_figure_layout(fig, left=0.1, right=0.9, top=0.85, bottom=0.15)

    assert fig.subplots_adjust_called is True
    assert fig.canvas.print_figure != original_print_figure

    result = fig.canvas.print_figure("out.pdf", bbox_inches="tight", dpi=100)
    assert result == "printed"
    assert fig.canvas.kwargs_passed == {"dpi": 100}


def test_get_difference_plot_returns_difference_plot():
    """get_difference_plot should return a DifferencePlot when valid."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 2.0], "word2": [0.5, 0.5]}, index=["doc1", "doc2"]
    )
    st = SeeTrees(frequencies=frequencies)

    plotter = st.get_difference_plot("doc1", "doc2")

    assert isinstance(plotter, DifferencePlot)


def test_get_difference_plot_validates_missing_data():
    """get_difference_plot should validate input data and labels."""
    st_empty = SeeTrees(frequencies=pd.DataFrame())

    with pytest.raises(
        ValueError, match="Frequency data is required for get_difference_plot"
    ):
        st_empty.get_difference_plot("doc1", "doc2")

    frequencies = pd.DataFrame({"word1": [1.0, 2.0]}, index=["doc1", "doc2"])
    st = SeeTrees(frequencies=frequencies)

    with pytest.raises(ValueError, match="Source text 'docX' not found"):
        st.get_difference_plot("docX", "doc2")

    with pytest.raises(ValueError, match="Target text 'docX' not found"):
        st.get_difference_plot("doc1", "docX")


def test_get_overlay_plot_returns_overlay_plot():
    """get_overlay_plot should return an OverlayPlot when valid."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 2.0], "word2": [0.5, 0.5]}, index=["doc1", "doc2"]
    )
    st = SeeTrees(frequencies=frequencies)

    plotter = st.get_overlay_plot("doc1", "doc2")

    assert isinstance(plotter, OverlayPlot)


def test_get_overlay_plot_validates_missing_data():
    """get_overlay_plot should validate input data and labels."""
    st_empty = SeeTrees(frequencies=pd.DataFrame())

    with pytest.raises(
        ValueError, match="Frequency data is required for get_overlay_plot"
    ):
        st_empty.get_overlay_plot("doc1", "doc2")

    frequencies = pd.DataFrame({"word1": [1.0, 2.0]}, index=["doc1", "doc2"])
    st = SeeTrees(frequencies=frequencies)

    with pytest.raises(ValueError, match="Source text 'docX' not found"):
        st.get_overlay_plot("docX", "doc2")

    with pytest.raises(ValueError, match="Target text 'docX' not found"):
        st.get_overlay_plot("doc1", "docX")


def test_compute_distances_requires_non_empty_frequencies():
    """compute_distances should raise when frequency data is empty."""
    st = SeeTrees(frequencies=pd.DataFrame())

    with pytest.raises(
        ValueError, match="Frequency table is required to compute distance metrics"
    ):
        st.compute_distances(metric="delta")


def test_get_density_plot_requires_distance_table():
    """get_density_plot should require a non-empty distance table."""
    frequencies = pd.DataFrame({"word1": [1.0, 2.0]}, index=["doc1", "doc2"])
    st = SeeTrees(frequencies=frequencies)

    with pytest.raises(
        ValueError, match="Distance table is required to create a DensityPlot"
    ):
        st.get_density_plot()


def test_get_density_plot_returns_density_plot():
    """get_density_plot should return a DensityPlot with computed distances."""
    frequencies = pd.DataFrame({"word1": [1.0, 2.0]}, index=["doc1", "doc2"])
    distance_table = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]], index=["doc1", "doc2"], columns=["doc1", "doc2"]
    )
    st = SeeTrees(frequencies=frequencies, distance_table=distance_table)

    plotter = st.get_density_plot(group=False, title="Density Test")

    assert isinstance(plotter, DensityPlot)
    assert plotter.group is False
    assert plotter.title == "Density Test"


def test_get_mds_plot_requires_distance_table():
    """get_mds_plot should require a computed distance table."""
    st = SeeTrees(frequencies=pd.DataFrame())

    with pytest.raises(
        ValueError, match="Distance table is required to create an MDS plot"
    ):
        st.get_mds_plot()


def test_get_mds_plot_returns_mds():
    """get_mds_plot should return an MDS object when the distance table exists."""
    frequencies = pd.DataFrame({"word1": [1.0, 2.0]}, index=["doc1", "doc2"])
    distance_table = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]], index=["doc1", "doc2"], columns=["doc1", "doc2"]
    )
    st = SeeTrees(frequencies=frequencies, distance_table=distance_table)

    plotter = st.get_mds_plot(group=True, title="MDS Test")

    assert isinstance(plotter, MDS)
    assert plotter.title == "MDS Test"


def test_get_pca_plot_returns_pca():
    """get_pca_plot should return a PCA object from the current distance table."""
    frequencies = pd.DataFrame({"word1": [1.0, 2.0]}, index=["doc1", "doc2"])
    distance_table = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]], index=["doc1", "doc2"], columns=["doc1", "doc2"]
    )
    st = SeeTrees(frequencies=frequencies, distance_table=distance_table)

    plotter = st.get_pca_plot(title="PCA Test")

    assert isinstance(plotter, PCA)
    assert plotter.title == "PCA Test"


def test_get_feature_summary_validates_missing_data():
    """get_feature_summary should validate empty frequencies and missing targets."""
    st_empty = SeeTrees(frequencies=pd.DataFrame())

    with pytest.raises(
        ValueError, match="Frequency data is required for get_feature_summary"
    ):
        st_empty.get_feature_summary(target_text="doc1")

    frequencies = pd.DataFrame({"word1": [1.0, 2.0]}, index=["doc1", "doc2"])
    st = SeeTrees(frequencies=frequencies)

    with pytest.raises(ValueError, match="Target text 'docX' not found"):
        st.get_feature_summary(target_text="docX")


def test_get_tree_calls_plot_tree_and_returns_tree(monkeypatch):
    """get_tree should construct a Tree and call its plot_tree method."""
    frequencies = pd.DataFrame({"word1": [1.0, 2.0]}, index=["doc1", "doc2"])
    distance_table = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]], index=["doc1", "doc2"], columns=["doc1", "doc2"]
    )
    st = SeeTrees(frequencies=frequencies, distance_table=distance_table)
    called = {}

    def fake_plot_tree(self, **kwargs):
        called.update(kwargs)

    monkeypatch.setattr(Tree, "plot_tree", fake_plot_tree)

    tree = st.get_tree(
        k=3,
        method="single",
        title="Tree Test",
        top_n_words=5,
        orientation="left",
        label_buffer=0.2,
        outline_y_pad=0.4,
        outline_axis_y_pad=0.2,
        outline_tip_pad_ratio=0.01,
        outline_root_pad_ratio=0.1,
    )

    assert isinstance(tree, Tree)
    assert called["k"] == 3
    assert called["method"] == "single"
    assert tree.title == "Tree Test"
    assert called["orientation"] == "left"


def test_get_feature_score_plot_validates_missing_data():
    """get_feature_score_plot should validate empty frequencies and missing targets."""
    st_empty = SeeTrees(frequencies=pd.DataFrame())

    with pytest.raises(
        ValueError, match="Frequency data is required for get_feature_score_plot"
    ):
        st_empty.get_feature_score_plot(target_text="doc1")

    frequencies = pd.DataFrame({"word1": [1.0, 2.0]}, index=["doc1", "doc2"])
    st = SeeTrees(frequencies=frequencies)

    with pytest.raises(ValueError, match="Target text 'docX' not found"):
        st.get_feature_score_plot(target_text="docX")
