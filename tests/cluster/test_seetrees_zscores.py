"""Tests for the seetrees zscores submodule.

Coverage: 100%.

Last Updated: August 23, 2026
"""

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import pytest

from lexos.cluster.seetrees import (
    DistinctiveFeaturePlot,
    FeatureSummary,
    SeeTrees,
    ZscorePlot,
)


def test_get_feature_score_plot_returns_plotter():
    """get_feature_score_plot should return a DistinctiveFeaturePlot object."""
    frequencies = pd.DataFrame(
        {
            "w1": [10.0, 1.0, 1.0],
            "w2": [1.0, 5.0, 6.0],
            "w3": [1.0, 6.0, 5.0],
            "w4": [8.0, 2.0, 2.0],
        },
        index=["doc1", "doc2", "doc3"],
    )
    st = SeeTrees(frequencies=frequencies)

    plotter = st.get_feature_score_plot("doc1", top=4)

    assert isinstance(plotter, DistinctiveFeaturePlot)
    assert plotter.fig is None


def test_get_feature_score_plot_applies_required_style_elements():
    """get_feature_score_plot should return a Plotly figure with correct layout settings."""
    frequencies = pd.DataFrame(
        {
            "w1": [10.0, 1.0, 1.0],
            "w2": [1.0, 5.0, 6.0],
            "w3": [1.0, 6.0, 5.0],
            "w4": [8.0, 2.0, 2.0],
        },
        index=["doc1", "doc2", "doc3"],
    )
    st = SeeTrees(frequencies=frequencies)

    plotter = st.get_feature_score_plot(
        "doc1",
        top=4,
        title="Custom Zscore Title",
        positive_color="#ffc0cb",
        negative_color="#87ceeb",
        guide_color="#bbbbbb",
        zero_line_color="red",
    )
    fig = plotter.plot()

    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Custom Zscore Title"
    assert fig.layout.xaxis.title.text == "Standard deviation from the corpus mean"
    assert fig.layout.xaxis.dtick == 2
    assert fig.layout.width == plotter.width
    assert fig.layout.height == plotter.height

    bar = fig.data[0]
    assert set(bar.marker.color) == {"#ffc0cb", "#87ceeb"}
    assert len(bar.marker.color) == 4
    assert bar.orientation == "h"
    assert any(x > 0 for x in bar.x)
    assert any(x < 0 for x in bar.x)

    assert len(bar.hovertemplate) > 0
    assert len(fig.layout.shapes) >= 1

    zero_shapes = [
        shape
        for shape in fig.layout.shapes
        if shape.x0 == 0 and shape.x1 == 0 and shape.line.color == "red"
    ]
    assert zero_shapes
    assert fig.layout.yaxis.autorange == "reversed"


def test_get_feature_summary_plot_runs_without_error():
    """get_feature_summary should run without error for valid data."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 0.5], "word2": [0.2, 0.8]},
        index=["doc1", "doc2"],
    )
    st = SeeTrees(frequencies=frequencies)

    df = st.get_feature_summary("doc2", top=2)
    assert isinstance(df, pd.DataFrame)


def test_get_feature_summary_missing_target_raises_error():
    """get_feature_summary should raise if the target document is not found."""
    frequencies = pd.DataFrame(
        {"word1": [1.0, 0.5], "word2": [0.2, 0.8]},
        index=["doc1", "doc2"],
    )
    st = SeeTrees(frequencies=frequencies)

    import pytest

    with pytest.raises(ValueError, match="Target text 'docX' not found"):
        st.get_feature_summary(target_text="docX", top=2)


def test_zscoreplot_z_scores_fills_nan_values():
    """_z_scores should replace NaN z-scores with zero."""
    frequencies = pd.DataFrame(
        {
            "w1": [1.0, 1.0],
            "w2": [float("nan"), 2.0],
        },
        index=["doc1", "doc2"],
    )
    plotter = ZscorePlot(frequencies=frequencies, target_text="doc1", top=2)

    z_scores = plotter._z_scores()

    assert z_scores.isna().sum().sum() == 0
    assert z_scores.loc["doc1", "w2"] == pytest.approx(0.0)


def test_zscoreplot_top_series_orders_positive_and_negative_features():
    """_top_series should sort positive and negative scores separately."""
    frequencies = pd.DataFrame(
        {
            "a": [10.0, 1.0],
            "b": [1.0, 8.0],
            "c": [1.0, 0.5],
            "d": [2.0, 9.0],
        },
        index=["doc1", "doc2"],
    )
    plotter = ZscorePlot(frequencies=frequencies, target_text="doc1", top=4)

    series = plotter._top_series()

    assert list(series.index)[:2] == ["a", "c"]
    assert all(series.loc[["a", "c"]].values >= 0)
    assert list(series.index)[-2:] == ["b", "d"]
    assert all(series.loc[["b", "d"]].values <= 0)


def test_zscoreplot_plot_renders_figure_with_expected_elements():
    """Plot should produce a Matplotlib figure with bars and guide lines."""
    frequencies = pd.DataFrame(
        {
            "plus": [10.0, 1.0],
            "minus": [1.0, 20.0],
            "zero": [1.0, 1.0],
        },
        index=["doc1", "doc2"],
    )
    plotter = ZscorePlot(
        frequencies=frequencies,
        target_text="doc1",
        top=3,
        title="Zscore Test",
        positive_color="#00ff00",
        negative_color="#ff0000",
        guide_color="#cccccc",
        zero_line_color="black",
    )
    fig = plotter.plot()
    ax = fig.axes[0]

    assert isinstance(fig, plt.Figure)
    assert ax.get_title() == "Zscore Test"
    assert ax.get_xlabel() == "Standard deviation from the corpus mean"
    assert len(ax.patches) == 3
    assert any(line.get_color() == "black" for line in ax.lines)
    assert any(line.get_color() == "#cccccc" for line in ax.lines)
    assert ax.get_xlim()[0] <= 0 <= ax.get_xlim()[1]


def test_zscoreplot_show_calls_plot_when_figure_missing(monkeypatch):
    """Show should call plot when no figure is available."""
    frequencies = pd.DataFrame({"w1": [1.0, 2.0]}, index=["doc1", "doc2"])
    plotter = ZscorePlot(frequencies=frequencies, target_text="doc1", top=1)
    plot_called = {"called": False}
    show_called = {"called": False}

    def fake_plot(self):
        plot_called["called"] = True
        self.fig = plt.figure()
        return self.fig

    def fake_show(*args, **kwargs):
        show_called["called"] = True

    monkeypatch.setattr(ZscorePlot, "plot", fake_plot)
    monkeypatch.setattr(plt, "show", fake_show)

    plotter.fig = None
    plotter.show()

    assert plot_called["called"] is True
    assert show_called["called"] is True


def test_distinctive_feature_plot_show_calls_fig_show_with_config(monkeypatch):
    """DistinctiveFeaturePlot.show should display the figure without the logo."""
    frequencies = pd.DataFrame({"w1": [1.0, 2.0]}, index=["doc1", "doc2"])
    plotter = DistinctiveFeaturePlot(
        frequencies=frequencies,
        target_text="doc1",
        top=1,
    )
    show_args = {}

    class DummyFig(go.Figure):
        def show(self, config=None):
            show_args["config"] = config

    def fake_plot(self):
        self.fig = DummyFig()
        return self.fig

    monkeypatch.setattr(DistinctiveFeaturePlot, "plot", fake_plot)

    plotter.fig = None
    plotter.show()

    assert show_args["config"] == {"displaylogo": False}


def test_feature_summary_feature_order_and_render_bar_chart():
    """FeatureSummary should order features by mean and render a bar chart."""
    frequencies = pd.DataFrame(
        {
            "x": [10.0, 1.0],
            "y": [5.0, 2.0],
            "z": [2.0, 1.0],
        },
        index=["doc1", "doc2"],
    )
    summary = FeatureSummary(frequencies=frequencies, target_text="doc1", top=2)

    assert summary._feature_order() == ["x", "y"]
    fig = summary.render_bar_chart()

    assert isinstance(fig, plt.Figure)
    assert fig.axes[0].get_xlabel() == "Standard deviation from the corpus mean"
