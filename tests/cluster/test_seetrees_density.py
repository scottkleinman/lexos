"""Tests for the seetrees density plot submodule.

Coverage: 98%. Missing: 145, 167

Last Updated: August 23, 2026
"""

import warnings
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from lexos.cluster.seetrees import DensityPlot, SeeTrees


def test_get_density_plot_requires_distance_table():
    """get_density_plot should require a precomputed distance table."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 0.5, 0.2], "word2": [0.2, 0.8, 0.4]},
        index=["Poe_Usher", "Poe_Bel", "Henry_Pirate"],
    )
    st = SeeTrees(frequencies=frequencies)

    with pytest.raises(
        ValueError, match="Distance table is required to create a DensityPlot"
    ):
        st.get_density_plot(group=True)


def test_get_density_plot_grouped_warns_on_insufficient_pairs():
    """get_density_plot grouping should warn when same-author data is too sparse."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 2.0, 3.0]},
        index=["Poe_1", "Poe_2", "Austen_1"],
    )
    distance_table = pd.DataFrame(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 2.0], [2.0, 2.0, 0.0]],
        index=["Poe_1", "Poe_2", "Austen_1"],
        columns=["Poe_1", "Poe_2", "Austen_1"],
    )
    st = SeeTrees(frequencies=frequencies, distance_table=distance_table)
    plotter = st.get_density_plot(group=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        with patch("matplotlib.pyplot.show"):
            plotter.show()

    assert any(
        "Grouped density is unlikely to be meaningful" in str(w.message)
        and "1 same-author distance pair(s)" in str(w.message)
        and "2 different-author distance pair(s)" in str(w.message)
        for w in caught
    )


def test_get_density_plot_invalid_author_raises_error():
    """get_density_plot should raise if the requested author is not present."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 0.5], "word2": [0.2, 0.8]},
        index=["Poe_1", "Austen_1"],
    )
    distance_table = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=["Poe_1", "Austen_1"],
        columns=["Poe_1", "Austen_1"],
    )
    st = SeeTrees(frequencies=frequencies, distance_table=distance_table)

    with pytest.raises(ValueError, match="Author 'Shakespeare' is not present"):
        st.get_density_plot(author="Shakespeare")


def test_get_density_plot_returns_plotter_without_showing():
    """get_density_plot should create the plot object without displaying."""
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

    with patch("matplotlib.pyplot.show") as mock_show:
        plotter = st.get_density_plot(group=True)

    assert plotter is not None
    assert isinstance(plotter, DensityPlot)
    assert plotter.fig is None
    mock_show.assert_not_called()


def test_get_density_plot_show_calls_matplotlib_show():
    """Calling show() on a DensityPlot should display the plot."""
    frequencies = pd.DataFrame(
        {
            "word1": [1.0, 0.2, 0.5, 0.7],
            "word2": [0.3, 0.8, 0.4, 0.6],
        },
        index=["Poe_1", "Poe_2", "Austen_1", "Austen_2"],
    )
    distance_table = pd.DataFrame(
        [
            [0.0, 1.0, 2.0, 2.0],
            [1.0, 0.0, 2.0, 2.0],
            [2.0, 2.0, 0.0, 1.0],
            [2.0, 2.0, 1.0, 0.0],
        ],
        index=["Poe_1", "Poe_2", "Austen_1", "Austen_2"],
        columns=["Poe_1", "Poe_2", "Austen_1", "Austen_2"],
    )
    st = SeeTrees(frequencies=frequencies, distance_table=distance_table)

    plotter = st.get_density_plot(group=True)

    with patch("matplotlib.pyplot.show") as mock_show:
        plotter.show()

    mock_show.assert_called_once()


def test_density_frame_supports_space_separator_pattern():
    """Density grouping should extract class labels before spaces or hyphens."""
    labels = ["Poe 1", "Poe 2", "Austen 1"]
    values = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 2.5], [2.0, 2.5, 0.0]])
    distance_table = pd.DataFrame(values, index=labels, columns=labels)
    plotter = DensityPlot(distance_table=distance_table, labels=labels)

    df = plotter._build_density_frame(pattern=r"^.*?(?=[_\s-])")

    assert df["class"].tolist() == ["Poe", "Austen", "Austen"]
    assert df["same_author"].tolist() == ["True", "False", "False"]


def test_density_plot_show_on_init_calls_show():
    """show_on_init should call plot_density and matplotlib.show."""
    labels = ["Poe_1", "Poe_2", "Austen_1", "Austen_2"]
    values = np.array(
        [
            [0.0, 1.0, 2.0, 2.0],
            [1.0, 0.0, 2.0, 2.0],
            [2.0, 2.0, 0.0, 1.0],
            [2.0, 2.0, 1.0, 0.0],
        ]
    )
    distance_table = pd.DataFrame(values, index=labels, columns=labels)

    with patch("matplotlib.pyplot.show") as mock_show:
        plotter = DensityPlot(
            distance_table=distance_table,
            labels=labels,
            show_on_init=True,
        )

    assert plotter.fig is not None
    mock_show.assert_called_once()


def test_add_legend_handles_only_named_items():
    """_add_legend should create a legend only for named handles."""
    plotter = DensityPlot(
        distance_table=pd.DataFrame(
            [[0.0, 1.0], [1.0, 0.0]], index=["a", "b"], columns=["a", "b"]
        ),
        labels=["a", "b"],
    )
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="foo")

    plotter._add_legend(ax)
    assert ax.get_legend() is not None
    assert any(text.get_text() == "foo" for text in ax.get_legend().get_texts())


def test_disable_canvas_bbox_inches_removes_bbox_inches():
    """_disable_canvas_bbox_inches should strip bbox_inches from print_figure kwargs."""
    plotter = DensityPlot(
        distance_table=pd.DataFrame(
            [[0.0, 1.0], [1.0, 0.0]], index=["a", "b"], columns=["a", "b"]
        ),
        labels=["a", "b"],
    )

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


def test_extract_label_class_fallbacks_to_delimiters_and_digits():
    """_extract_label_class should return the expected fallback class labels."""
    plotter = DensityPlot(
        distance_table=pd.DataFrame(
            [[0.0, 1.0], [1.0, 0.0]], index=["a", "b"], columns=["a", "b"]
        ),
        labels=["a", "b"],
    )

    assert plotter._extract_label_class("Poe_1", r"^unmatched$") == "Poe"
    assert plotter._extract_label_class("Austen 2", r"^unmatched$") == "Austen"
    assert (
        plotter._extract_label_class("Shakespeare-3", r"^unmatched$") == "Shakespeare"
    )
    assert plotter._extract_label_class("UniqueLabel", r"^unmatched$") == "UniqueLabel"


def test_finalize_axes_sets_bottom_when_ylim_nonnegative_and_title():
    """_finalize_axes should set a negative bottom when y-limits are nonnegative."""
    plotter = DensityPlot(
        distance_table=pd.DataFrame(
            [[0.0, 1.0], [1.0, 0.0]], index=["a", "b"], columns=["a", "b"]
        ),
        labels=["a", "b"],
        title="Custom Title",
    )
    fig, ax = plt.subplots()
    ax.set_ylim(0, 1)
    df = pd.DataFrame({"d": [1.0], "same_author": ["True"], "class": ["a"]})

    plotter._finalize_axes(ax, df)

    assert ax.get_title() == "Custom Title"
    assert ax.get_ylim()[0] < 0


def test_finalize_axes_highlights_author_when_no_title():
    """_finalize_axes should call _highlight_author when author is set and title is missing."""
    plotter = DensityPlot(
        distance_table=pd.DataFrame(
            [[0.0, 1.0], [1.0, 0.0]],
            index=["Poe_1", "Poe_2"],
            columns=["Poe_1", "Poe_2"],
        ),
        labels=["Poe_1", "Poe_2"],
        author="Poe",
    )
    fig, ax = plt.subplots()
    ax.set_ylim(0, 1)
    df = pd.DataFrame({"d": [1.0], "same_author": ["True"], "class": ["Poe"]})

    plotter._finalize_axes(ax, df)

    assert "Points: distances between works of Poe" in ax.get_title()


def test_get_palette_uses_custom_palette():
    """_get_palette should return overridden palette values."""
    plotter = DensityPlot(
        distance_table=pd.DataFrame(
            [[0.0, 1.0], [1.0, 0.0]], index=["a", "b"], columns=["a", "b"]
        ),
        labels=["a", "b"],
        palette={"True": "red", "False": "blue"},
    )

    palette = plotter._get_palette()
    assert palette["True"] == "red"
    assert palette["False"] == "blue"


def test_highlight_author_plots_author_distances_and_sets_title():
    """_highlight_author should plot points and update the title."""
    plotter = DensityPlot(
        distance_table=pd.DataFrame(
            [[0.0, 1.0], [1.0, 0.0]],
            index=["Poe_1", "Austen_1"],
            columns=["Poe_1", "Austen_1"],
        ),
        labels=["Poe_1", "Austen_1"],
        author="Poe",
    )
    fig, ax = plt.subplots()
    df = pd.DataFrame({"d": [1.0], "same_author": ["True"], "class": ["Poe"]})

    plotter._highlight_author(ax, df)

    assert "Points: distances between works of Poe" in ax.get_title()


def test_plot_ungrouped_density_creates_plot_without_error():
    """plot_density should create an ungrouped density plot when group is False."""
    labels = ["a", "b", "c"]
    values = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]])
    distance_table = pd.DataFrame(values, index=labels, columns=labels)

    plotter = DensityPlot(
        distance_table=distance_table,
        labels=labels,
        group=False,
        color="#ff0000",
    )
    plotter.plot_density()

    assert plotter.fig is not None
    assert len(plotter.fig.axes) == 1


def test_plot_grouped_density_with_custom_palette_and_no_warning():
    """plot_density should render grouped density without warnings when counts are sufficient."""
    labels = ["Poe_1", "Poe_2", "Austen_1", "Austen_2"]
    values = np.array(
        [
            [0.0, 1.0, 2.0, 2.0],
            [1.0, 0.0, 2.0, 2.0],
            [2.0, 2.0, 0.0, 1.0],
            [2.0, 2.0, 1.0, 0.0],
        ]
    )
    distance_table = pd.DataFrame(values, index=labels, columns=labels)

    plotter = DensityPlot(
        distance_table=distance_table,
        labels=labels,
        group=True,
        palette={"True": "red", "False": "blue"},
        title="Grouped Density",
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("error")
        plotter.plot_density()

    assert plotter.fig is not None
    assert not caught
