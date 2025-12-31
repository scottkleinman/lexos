"""test_simple_plotter.py.

Coverage: 100%
Last Updated: September 11, 2025
"""

import tempfile

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing

from lexos.exceptions import LexosException
from lexos.rolling_windows.plotters.simple_plotter import SimplePlotter, interpolate


def make_df():
    """Create a simple DataFrame for plotting."""
    data = {"A": np.arange(10), "B": np.arange(10, 20)}
    return pd.DataFrame(data)


def test_simple_plotter_init_and_basic_plot():
    """Test initialization and basic plotting functionality."""
    df = make_df()
    plotter = SimplePlotter(df=df, title="Test Plot", xlabel="X", ylabel="Y")
    assert plotter.fig is not None
    assert plotter.ax is not None
    plotter.plot(show=False)  # Should not raise


def test_simple_plotter_interpolation():
    """Test interpolation functionality."""
    df = make_df()
    plotter = SimplePlotter(
        df=df, use_interpolation=True, interpolation_num=100, interpolation_kind="pchip"
    )
    plotter.plot(show=False)  # Should not raise


def test_simple_plotter_milestones_and_labels():
    """Test milestones and milestone labels functionality."""
    df = make_df()
    milestones = {"Start": 0, "Mid": 5, "End": 9}
    plotter = SimplePlotter(
        df=df,
        show_milestones=True,
        show_milestone_labels=True,
        milestone_labels=milestones,
        milestone_colors="red",
        milestone_style="--",
        milestone_labels_rotation=30,
        milestone_labels_offset=(-10, 10),
    )
    plotter.plot(show=False)  # Should not raise


def test_simple_plotter_save_and_show(tmp_path):
    """Test saving and showing the plot."""
    df = make_df()
    plotter = SimplePlotter(df=df)
    plotter.plot(show=False)
    save_path = tmp_path / "test_plot.png"
    plotter.save(save_path)
    assert save_path.exists()
    # Show returns a Figure object
    fig = plotter.show()
    assert fig is not None


def test_simple_plotter_missing_fig_save_error():
    """Test error handling when saving without a figure."""
    df = make_df()
    plotter = SimplePlotter(df=df)
    plotter.fig = None  # Simulate missing figure
    try:
        plotter.save("dummy.png")
    except LexosException as e:
        assert "There is no plot to save" in str(e)


def test_simple_plotter_missing_fig_show_error():
    """Test error handling when showing without a figure."""
    df = make_df()
    plotter = SimplePlotter(df=df)
    plotter.fig = None  # Simulate missing figure
    try:
        plotter.show()
    except LexosException as e:
        assert "There is no plot to show" in str(e)


def test_simple_plotter_edge_case_milestone_validation():
    """Test edge case validation for milestones and labels."""
    df = make_df()
    # show_milestones True but no milestone_labels
    try:
        SimplePlotter(df=df, show_milestones=True)
    except LexosException as e:
        assert "require a value for `milestone_labels`" in str(e)


def test_interpolate_legacy_and_default():
    """Test interpolation with legacy and default settings."""
    # Prepare simple data
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([0, 1, 4, 9, 16])
    xx = np.linspace(0, 4, 10)

    # Test legacy interp1d kind
    result_linear = interpolate(x, y, xx, interpolation_kind="linear")
    assert isinstance(result_linear, np.ndarray)
    assert np.allclose(
        result_linear, np.interp(xx, x, y), atol=1e-6
    )  # Should match np.interp for linear

    # Test default (not in legacy list)
    result_default = interpolate(x, y, xx, interpolation_kind="not_a_kind")
    assert isinstance(result_default, np.ndarray)
    assert np.allclose(result_default, np.interp(xx, x, y), atol=1e-6)

    # Test another legacy kind
    result_cubic = interpolate(x, y, xx, interpolation_kind="cubic")
    assert isinstance(result_cubic, np.ndarray)
    # Cubic interpolation should be smooth and pass through all points
    assert np.isclose(result_cubic[0], y[0])
    assert np.isclose(result_cubic[-1], y[-1])


def test_title_position_bottom_sets_title_y():
    """Test that setting title_position to 'bottom' sets the title y position correctly."""
    df = pd.DataFrame({"A": [1, 2, 3]})
    plotter = SimplePlotter(df=df, title="My Bottom Title", title_position="bottom")
    plotter.plot(show=False)
    if plotter.fig is None:
        print("Figure is None")
    if plotter.ax is None:
        print("Axes is None")
    title_val = plotter.ax.get_title()
    # The title should be set with y=-0.25
    assert title_val == "My Bottom Title"
    # Check the y position
    title_obj = plotter.ax.title
    assert abs(title_obj.get_position()[1] - (-0.25)) < 1e-6


def test_plotter_grid_enabled():
    """Test that enabling grid works correctly."""
    df = pd.DataFrame({"A": [1, 2, 3]})
    plotter = SimplePlotter(df=df, show_grid=True)
    plotter.plot(show=False)  # This will execute plt.grid(visible=True)
    # You can also check gridlines are visible
    gridlines = plotter.ax.get_xgridlines() + plotter.ax.get_ygridlines()
    assert any(line.get_visible() for line in gridlines)


###


def test_get_width_height_with_figsize():
    """Test that _get_width_height returns correct values when figsize is set."""
    df = pd.DataFrame({"A": [1, 2, 3]})
    plotter = SimplePlotter(df=df, figsize=(8, 5))
    width, height = plotter._get_width_height()
    assert width == 8
    assert height == 5
