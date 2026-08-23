"""Tests for the seetrees tree submodule.

Coverage: 98%. Missing: 199, 283-285, 364, 666

Last Updated: August 23, 2026
"""

import warnings
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import ClusterWarning

from lexos.cluster.seetrees import SeeTrees, Tree


def test_get_tree_returns_tree_without_error():
    """get_tree should execute without error after computing distances."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 0.5], "word2": [0.2, 0.8]},
        index=["doc1", "doc2"],
    )
    st = SeeTrees(frequencies=frequencies)
    st.compute_distances(metric="cosine")

    tree = st.get_tree(k=2)

    assert tree.fig is not None


def test_get_tree_does_not_warn_for_distance_matrix():
    """get_tree should not warn when given a precomputed distance matrix."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 0.5], "word2": [0.2, 0.8]},
        index=["doc1", "doc2"],
    )
    st = SeeTrees(frequencies=frequencies)
    st.compute_distances(metric="cosine")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        _ = st.get_tree(k=2)

    assert not any(issubclass(w.category, ClusterWarning) for w in caught)


def test_get_tree_raises_without_distance_table():
    """get_tree should raise if no distance table has been computed."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 0.5], "word2": [0.2, 0.8]},
        index=["doc1", "doc2"],
    )
    st = SeeTrees(frequencies=frequencies)

    with pytest.raises(ValueError, match="Distance table is required"):
        st.get_tree(k=2)


def test_get_tree_skips_constant_features():
    """get_tree should skip constant features when computing eta-squared."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 1.0, 1.0], "word2": [1.0, 2.0, 3.0]},
        index=["doc1", "doc2", "doc3"],
    )
    st = SeeTrees(frequencies=frequencies)
    st.compute_distances(metric="delta")

    tree = st.get_tree(k=2)

    assert tree.fig is not None


def test_tree_show_on_init_plots_and_shows():
    """Tree created with show_on_init should plot and call plt.show."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 0.5], "word2": [0.2, 0.8]},
        index=["doc1", "doc2"],
    )
    distance_table = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=["doc1", "doc2"],
        columns=["doc1", "doc2"],
    )

    with patch("matplotlib.pyplot.show") as mock_show:
        tree = Tree(
            distance_table=distance_table,
            labels=["doc1", "doc2"],
            frequencies=frequencies,
            show_on_init=True,
        )

    assert tree.fig is not None
    mock_show.assert_called_once()


def test_ensure_plot_ready_requires_labels():
    """_ensure_plot_ready should raise when labels are missing."""
    distance_table = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=["doc1", "doc2"],
        columns=["doc1", "doc2"],
    )
    tree = Tree(distance_table=distance_table, labels=[])

    with pytest.raises(ValueError, match="Labels are required to build the tree plot"):
        tree._ensure_plot_ready()


def test_sanitize_k_clamps_values():
    """_sanitize_k should clamp values to the valid document range."""
    distance_table = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=["doc1", "doc2"],
        columns=["doc1", "doc2"],
    )
    tree = Tree(distance_table=distance_table, labels=["doc1", "doc2"])

    assert tree._sanitize_k(0, 2) == 1
    assert tree._sanitize_k(3, 2) == 2


def test_validate_method_rejects_invalid():
    """_validate_method should raise for an unsupported linkage method."""
    distance_table = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=["doc1", "doc2"],
        columns=["doc1", "doc2"],
    )
    tree = Tree(distance_table=distance_table, labels=["doc1", "doc2"])

    with pytest.raises(ValueError, match="method must be one of"):
        tree._validate_method("invalid_method")


def test_compute_color_threshold_edge_cases():
    """_compute_color_threshold should handle k>=n_docs and k==1."""
    distance_table = pd.DataFrame(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]],
        index=["doc1", "doc2", "doc3"],
        columns=["doc1", "doc2", "doc3"],
    )
    tree = Tree(distance_table=distance_table, labels=["doc1", "doc2", "doc3"])

    z = np.array([[0, 1, 1.0, 0], [0, 2, 2.0, 0]])
    assert tree._compute_color_threshold(z, 3, 3) == 0.0
    assert tree._compute_color_threshold(z, 1, 3) == pytest.approx(2.0 + 1e-12)
    assert tree._compute_color_threshold(z, 2, 3) == pytest.approx(1.5)


def test_apply_orientation_axis_style_top_and_bottom():
    """_apply_orientation_axis_style should style top and bottom orientations."""
    distance_table = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=["doc1", "doc2"],
        columns=["doc1", "doc2"],
    )
    tree = Tree(distance_table=distance_table, labels=["doc1", "doc2"])

    fig, ax = plt.subplots()
    tree._apply_orientation_axis_style(ax, "top")
    assert not ax.spines["bottom"].get_visible()

    fig, ax = plt.subplots()
    tree._apply_orientation_axis_style(ax, "bottom")
    assert not ax.spines["top"].get_visible()


def test_show_calls_plot_tree_when_fig_missing():
    """Show should call plot_tree when no figure exists."""
    distance_table = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=["doc1", "doc2"],
        columns=["doc1", "doc2"],
    )
    tree = Tree(distance_table=distance_table, labels=["doc1", "doc2"])
    tree.fig = None

    def set_fig(self):
        self.fig = plt.figure()

    with patch.object(
        Tree, "plot_tree", autospec=True, side_effect=set_fig
    ) as mock_plot:
        with patch("matplotlib.pyplot.show") as mock_show:
            tree.show()

    mock_plot.assert_called_once_with(tree)
    mock_show.assert_called_once()


def test_apply_figure_layout_orientations():
    """_apply_figure_layout should adjust subplot layout for each orientation."""
    distance_table = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=["doc1", "doc2"],
        columns=["doc1", "doc2"],
    )
    tree = Tree(distance_table=distance_table, labels=["doc1", "doc2"])

    fig = plt.figure()
    tree._apply_figure_layout(fig, orientation="left", label_buffer=0.5)
    assert fig.subplotpars.right <= 0.9

    fig = plt.figure()
    tree._apply_figure_layout(fig, orientation="right", label_buffer=0.5)
    assert fig.subplotpars.left >= 0.1

    fig = plt.figure()
    tree._apply_figure_layout(fig, orientation="top", label_buffer=0.5)
    assert fig.subplotpars.bottom >= 0.12

    fig = plt.figure()
    tree._apply_figure_layout(fig, orientation="bottom", label_buffer=0.5)
    assert fig.subplotpars.top <= 0.94


def test_disable_canvas_bbox_inches_strips_bbox_inches():
    """_disable_canvas_bbox_inches should remove bbox_inches from kwargs."""
    distance_table = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=["doc1", "doc2"],
        columns=["doc1", "doc2"],
    )
    tree = Tree(distance_table=distance_table, labels=["doc1", "doc2"])

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
    tree._disable_canvas_bbox_inches(fig)

    result = fig.canvas.print_figure("out.pdf", bbox_inches="tight", dpi=100)
    assert result == "ok"
    assert fig.canvas.kwargs_passed == {"dpi": 100}


def test_cluster_top_words_handles_missing_clusters(monkeypatch):
    """_cluster_top_words should include an empty cluster when np.unique is patched."""
    distance_table = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=["doc1", "doc2"],
        columns=["doc1", "doc2"],
    )
    frequencies = pd.DataFrame(
        {"word1": [1.0, 2.0], "word2": [0.5, 0.6]},
        index=["doc1", "doc2"],
    )
    tree = Tree(
        distance_table=distance_table, labels=["doc1", "doc2"], frequencies=frequencies
    )
    clusters = np.array([1, 1])

    monkeypatch.setattr(
        "lexos.cluster.seetrees.tree.np.unique", lambda x: np.array([1, 2])
    )
    result = tree._cluster_top_words(clusters, top_n=2)

    assert result[2] == []


def test_dendrogram_color_func_returns_gray_for_mixed_link():
    """_dendrogram_color_func should return gray when leaf clusters differ."""
    distance_table = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=["doc1", "doc2"],
        columns=["doc1", "doc2"],
    )
    tree = Tree(distance_table=distance_table, labels=["doc1", "doc2"])

    z = np.array([[0, 1, 1.0, 0]])
    clusters = np.array([1, 2])
    palette = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    func = tree._dendrogram_color_func(z, clusters, palette)

    assert func(2) == "gray"
    assert func(3) == "gray"


def test_dendrogram_color_func_returns_palette_color_for_homogeneous_link():
    """_dendrogram_color_func should return a palette color when both leaves share a cluster."""
    distance_table = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=["doc1", "doc2"],
        columns=["doc1", "doc2"],
    )
    tree = Tree(distance_table=distance_table, labels=["doc1", "doc2"])

    z = np.array([[0, 1, 1.0, 0]])
    clusters = np.array([1, 1])
    palette = [(1.0, 0.0, 0.0)]
    func = tree._dendrogram_color_func(z, clusters, palette)

    assert func(2) == "#ff0000"


def test_style_cluster_regions_returns_with_empty_leaves():
    """_style_cluster_regions should return cleanly when dendro leaves are empty."""
    distance_table = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=["doc1", "doc2"],
        columns=["doc1", "doc2"],
    )
    tree = Tree(distance_table=distance_table, labels=["doc1", "doc2"])
    fig, ax = plt.subplots()

    tree._style_cluster_regions(
        ax,
        {"leaves": []},
        np.array([1, 2]),
        [(1.0, 0.0, 0.0)],
        "right",
        threshold=1.0,
        tip_x_pad=0.1,
        root_x_pad=0.1,
        y_pad=0.1,
        axis_y_pad=0.1,
    )

    assert len(ax.patches) == 0


def test_compute_region_layout_falls_back_for_left_right():
    """_compute_region_layout should fallback when x_box_right <= x_box_left."""
    distance_table = pd.DataFrame(
        [[0.0, 0.0], [0.0, 0.0]],
        index=["doc1", "doc2"],
        columns=["doc1", "doc2"],
    )
    tree = Tree(distance_table=distance_table, labels=["doc1", "doc2"])
    fig, ax = plt.subplots()
    ax.set_xlim(1.0, 1.0)

    layout, _ = tree._compute_region_layout(ax, "left", tip_x_pad=1.0, root_x_pad=1.0)
    assert layout["x_box_right"] > layout["x_box_left"]


def test_compute_region_layout_falls_back_for_top_bottom():
    """_compute_region_layout should fallback when y_box_top <= y_box_bottom."""
    distance_table = pd.DataFrame(
        [[0.0, 0.0], [0.0, 0.0]],
        index=["doc1", "doc2"],
        columns=["doc1", "doc2"],
    )
    tree = Tree(distance_table=distance_table, labels=["doc1", "doc2"])
    fig, ax = plt.subplots()
    ax.set_ylim(1.0, 1.0)

    layout, _ = tree._compute_region_layout(ax, "top", tip_x_pad=1.0, root_x_pad=1.0)
    assert layout["y_box_top"] > layout["y_box_bottom"]


def test_color_cluster_tick_labels_breaks_when_extra_labels():
    """_color_cluster_tick_labels should break when there are more labels than clusters."""
    distance_table = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=["doc1", "doc2"],
        columns=["doc1", "doc2"],
    )
    tree = Tree(distance_table=distance_table, labels=["doc1", "doc2"])
    fig, ax = plt.subplots()
    labels = [
        ax.xaxis.get_ticklabels()[0],
        ax.xaxis.get_ticklabels()[0],
        ax.xaxis.get_ticklabels()[0],
    ]

    tree._color_cluster_tick_labels(labels, [1], [(1.0, 0.0, 0.0)])
    assert labels[0].get_color() == (1.0, 0.0, 0.0)


def test_build_cluster_rectangle_top_bottom():
    """_build_cluster_rectangle should handle top/bottom orientations."""
    distance_table = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=["doc1", "doc2"],
        columns=["doc1", "doc2"],
    )
    tree = Tree(distance_table=distance_table, labels=["doc1", "doc2"])
    rect = tree._build_cluster_rectangle(
        "top",
        {"y_box_bottom": 0.0, "y_box_top": 10.0},
        0,
        1,
        0.5,
        "blue",
    )

    assert rect.get_width() == pytest.approx(19.0)
    assert rect.get_height() == pytest.approx(10.0)


def test_apply_cluster_axis_padding_inverted_y():
    """_apply_cluster_axis_padding should handle inverted y limits."""
    distance_table = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=["doc1", "doc2"],
        columns=["doc1", "doc2"],
    )
    tree = Tree(distance_table=distance_table, labels=["doc1", "doc2"])
    fig, ax = plt.subplots()
    ax.set_ylim(2.0, 1.0)

    tree._apply_cluster_axis_padding(ax, "left", axis_y_pad=0.5)
    assert ax.get_ylim()[0] > ax.get_ylim()[1]


def test_apply_cluster_axis_padding_inverted_x():
    """_apply_cluster_axis_padding should handle inverted x limits."""
    distance_table = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=["doc1", "doc2"],
        columns=["doc1", "doc2"],
    )
    tree = Tree(distance_table=distance_table, labels=["doc1", "doc2"])
    fig, ax = plt.subplots()
    ax.set_xlim(2.0, 1.0)

    tree._apply_cluster_axis_padding(ax, "top", axis_y_pad=0.5)
    assert ax.get_xlim()[0] > ax.get_xlim()[1]


def test_label_axes_top_sets_ylabel():
    """_label_axes should set y-axis label for top/bottom orientation."""
    distance_table = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=["doc1", "doc2"],
        columns=["doc1", "doc2"],
    )
    tree = Tree(distance_table=distance_table, labels=["doc1", "doc2"])
    fig, ax = plt.subplots()

    tree._label_axes(ax, orientation="top", k=2)
    assert ax.get_ylabel() == "Distance"


def test_compute_padding_returns_y_padding_for_vertical_orientation():
    """_compute_padding should compute y padding for top/bottom orientation."""
    distance_table = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=["doc1", "doc2"],
        columns=["doc1", "doc2"],
    )
    tree = Tree(distance_table=distance_table, labels=["doc1", "doc2"])
    fig, ax = plt.subplots()
    ax.set_ylim(0.0, 2.0)

    tip_pad, root_pad = tree._compute_padding(
        ax, "top", outline_tip_pad_ratio=0.1, outline_root_pad_ratio=0.05
    )
    assert tip_pad == pytest.approx(0.2)
    assert root_pad == pytest.approx(0.1)
