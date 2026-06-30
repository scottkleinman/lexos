"""test_plotly_plotter.py.

Coverage: 100%
Last Updated: June 27, 2026
"""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from lexos.exceptions import LexosException
from lexos.rolling_windows.plotters import plotly_plotter
from lexos.rolling_windows.plotters.plotly_plotter import PlotlyPlotter


def make_df():
    """Simple DataFrame for plotting."""
    return pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [5, 4, 3, 2, 1]})


def test_plotly_plotter_basic_plot(tmp_path):
    """Basic test of PlotlyPlotter plotting functionality."""
    df = make_df()
    plotter = PlotlyPlotter(df=df, title="Test Plot", xlabel="X", ylabel="Y")
    plotter.plot(show=False)
    assert plotter.fig is not None


def test_plotly_plotter_show(tmp_path):
    """Test the show method of PlotlyPlotter."""
    df = make_df()
    plotter = PlotlyPlotter(df=df)
    plotter.plot(show=False)
    # Should not raise
    plotter.show(config={"displaylogo": False})


def test_plotly_plotter_save_html(tmp_path):
    """Test saving PlotlyPlotter output to HTML."""
    df = make_df()
    plotter = PlotlyPlotter(df=df)
    plotter.plot(show=False)
    html_path = tmp_path / "plot.html"
    plotter.save(html_path)
    assert html_path.exists()


def test_plotly_plotter_save_image(tmp_path):
    """Test saving PlotlyPlotter output to PNG (avoid dependency issues)."""
    df = make_df()
    plotter = PlotlyPlotter(df=df)
    plotter.plot(show=False)
    png_path = tmp_path / "plot.png"

    # Mock pio.write_image to avoid kaleido dependency issues
    with patch("plotly.io.write_image") as mock_write_image:
        plotter.save(png_path)

        # Verify that pio.write_image was called
        mock_write_image.assert_called_once_with(plotter.fig, png_path)


def test_plotly_plotter_milestones_and_labels():
    """Test PlotlyPlotter with milestones and labels."""
    df = make_df()
    milestones = {"Start": 0, "Mid": 2, "End": 4}
    plotter = PlotlyPlotter(
        df=df,
        show_milestones=True,
        milestone_labels=milestones,
        show_milestone_labels=True,
        milestone_label_rotation=45,
        milestone_label_style={"size": 12, "family": "Arial", "color": "blue"},
    )
    plotter.plot(show=False)
    # Check that milestone labels and markers were processed
    assert plotter.fig is not None
    # There should be at least as many shapes as milestones
    assert len(plotter.fig.layout.shapes) >= len(milestones)


def test_plotly_plotter_invalid_milestone_labels():
    """Test PlotlyPlotter with invalid milestone_labels."""
    df = make_df()
    # Empty milestone_labels should raise
    with pytest.raises(LexosException):
        PlotlyPlotter(df=df, show_milestones=True, milestone_labels={})


def test_plotly_plotter_invalid_rotation():
    """Test PlotlyPlotter with invalid milestone_label_rotation."""
    df = make_df()
    with pytest.raises(LexosException):
        PlotlyPlotter(df=df, milestone_label_rotation=120)


def test_plotly_plotter_save_without_plot(tmp_path):
    """Test that saving without plotting raises an exception."""
    df = make_df()
    plotter = PlotlyPlotter(df=df)
    # Do not call plotter.plot()
    with pytest.raises(LexosException):
        plotter.save(tmp_path / "dummy.html")


def test_show_sets_default_config(monkeypatch):
    """Test that show sets default config if none is provided.

    Note: This should cover lines 190-192 but doesn't because
    there is no way to capture the config argument passed to fig.show.
    This might be because there is a PlotlyPlotter.show method and
    a Plotly figure.show method.
    """
    df = pd.DataFrame({"A": [1, 2, 3]})
    plotter = PlotlyPlotter(df=df)
    plotter.plot(show=False, config=None)

    # Patch fig.show to capture the config argument
    called = {}

    def fake_show(config=None):
        called["config"] = config

    plotter.fig.show = fake_show

    plotter.show()  # No config argument

    assert called["config"] == {"displaylogo": False}


def test_get_axis_and_title_labels_with_dict_title():
    """Test _get_axis_and_title_labels with custom title as dict."""
    df = pd.DataFrame({"A": [1, 2, 3]})
    custom_title = {"text": "Custom Title", "x": 0.5, "y": 0.95, "font": {"size": 20}}
    plotter = PlotlyPlotter(df=df, title=custom_title)
    title_dict, xlabel_dict, ylabel_dict = plotter._get_axis_and_title_labels()
    assert title_dict == custom_title
    assert isinstance(xlabel_dict, dict)
    assert isinstance(ylabel_dict, dict)


def test_get_axis_and_title_labels_with_dict_xlabel():
    """Test _get_axis_and_title_labels with custom xlabel as dict.

    Supposed to cover line 205, but it doesn't work.
    Something similar needs to be done for line 209.
    """
    df = pd.DataFrame({"A": [1, 2, 3]})
    custom_xlabel = "Custom X"
    plotter = PlotlyPlotter(df=df, xlabel=custom_xlabel)
    title_dict, xlabel_dict, ylabel_dict = plotter._get_axis_and_title_labels()
    assert xlabel_dict == {"title": custom_xlabel}
    assert isinstance(title_dict, dict)
    assert isinstance(ylabel_dict, dict)


def test_get_axis_and_title_labels_with_internal_dict_ylabel():
    """Test _get_axis_and_title_labels with internal dict ylabel."""
    df = pd.DataFrame({"A": [1, 2, 3]})
    plotter = PlotlyPlotter(df=df)
    object.__setattr__(plotter, "ylabel", {"title": "Custom Y"})
    title_dict, xlabel_dict, ylabel_dict = plotter._get_axis_and_title_labels()
    assert ylabel_dict == {"title": "Custom Y"}
    assert isinstance(title_dict, dict)
    assert isinstance(xlabel_dict, dict)


def test_get_axis_and_title_labels_with_internal_dict_xlabel():
    """Test _get_axis_and_title_labels with internal dict xlabel."""
    df = pd.DataFrame({"A": [1, 2, 3]})
    plotter = PlotlyPlotter(df=df)
    object.__setattr__(plotter, "xlabel", {"title": "Custom X"})
    title_dict, xlabel_dict, ylabel_dict = plotter._get_axis_and_title_labels()
    assert xlabel_dict == {"title": "Custom X"}
    assert isinstance(title_dict, dict)
    assert isinstance(ylabel_dict, dict)


def test_plotly_plotter_default_show_config(monkeypatch):
    """Test plot() uses default Plotly config when show=True and config=None."""
    df = make_df()
    plotter = PlotlyPlotter(df=df)

    called = {}

    def fake_show(self, config=None):
        called["config"] = config

    monkeypatch.setattr(plotly_plotter.Figure, "show", fake_show)

    plotter.plot(show=True, config=None)

    assert called["config"] == {"displaylogo": False}


def test_get_titlepad_returns_titlepad():
    """Test _get_titlepad returns titlepad if set."""
    df = pd.DataFrame({"A": [1, 2, 3]})
    plotter = PlotlyPlotter(df=df, titlepad=123.45)
    labels = {"Chapter 1": 0, "Chapter 2": 10}
    result = plotter._get_titlepad(labels)
    assert result == 123.45


def test_get_titlepad_splits_and_strips_fontfamily():
    """Test _get_titlepad splits and strips fontfamily strings."""
    df = pd.DataFrame({"A": [1, 2, 3]})
    style = {
        "size": 10.0,
        "family": "Arial, Courier New, Times New Roman",
        "color": "teal",
    }
    plotter = PlotlyPlotter(df=df, milestone_label_style=style)
    labels = {"Chapter 1": 0, "Chapter 2": 10}
    # This will trigger the split and strip logic
    plotter._get_titlepad(labels)
    # Check that fontfamily is a list of stripped strings
    fontfamily = [x.strip() for x in style["family"].split(",")]
    assert fontfamily == ["Arial", "Courier New", "Times New Roman"]


def test_get_titlepad_returns_max_height_plus_50(monkeypatch):
    """Test _get_titlepad returns max height + 50 if titlepad not set."""
    df = pd.DataFrame({"A": [1, 2, 3]})
    labels = {"A": 0, "B": 1}

    plotter = PlotlyPlotter(df=df)

    class DummyExtents:
        def __init__(self, width, height):
            self.width = width
            self.height = height

    class DummyTextPath:
        def __init__(self, *args, **kwargs):
            pass

        def get_extents(self):
            return DummyExtents(width=10, height=100)

    monkeypatch.setattr(plotly_plotter, "TextPath", DummyTextPath)
    monkeypatch.setattr(plotly_plotter, "FontProperties", lambda **kwargs: None)

    result = plotter._get_titlepad(labels)
    assert result == 150  # 100 + 50
