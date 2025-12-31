"""simple_plotter.py.

Last Update: December 4, 2025
Last Tested: September 13, 2025
"""

from pathlib import Path
from typing import Any, ClassVar, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes._axes import Axes
from pydantic import BaseModel, ConfigDict, Field, validate_call
from scipy.interpolate import interp1d, pchip

from lexos.exceptions import LexosException
from lexos.rolling_windows.plotters.base_plotter import BasePlotter


def interpolate(
    x: np.ndarray, y: np.ndarray, xx: np.ndarray, interpolation_kind: str = None
) -> np.ndarray:
    """Get interpolated points for plotting.

    Args:
        x (np.ndarray): The x values
        y (np.ndarray): The y values
        xx (np.ndarray): The projected interpolation range
        interpolation_kind (str): The interpolation function to use.

    Returns:
        The interpolated points.

    Note:
        The interpolation function may be either
        [scipy.interpolate.pchip_interpolate](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.pchip_interpolate.html#scipy.interpolate.pchip_interpolate),
        [numpy.interp](https://numpy.org/devdocs/reference/generated/numpy.interp.html#numpy.interp),
        or one of the options for [scipy.interpolate.interp1d](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html).
        Note however, that `scipy.interpolate.interp1d` is [deprecated](https://docs.scipy.org/doc/scipy/tutorial/interpolate/1D.html#piecewise-linear-interpolation).
    """
    legacy_interp1d = [
        "linear",
        "nearest",
        "nearest-up",
        "zero",
        "slinear",
        "quadratic",
        "cubic",
        "previous",
        "next",
    ]
    # Return the values interpolated with the specified function
    if interpolation_kind == "pchip":
        interpolator = pchip(x, y)
        return interpolator(xx)
    elif interpolation_kind in legacy_interp1d:
        interpolator = interp1d(x, y, kind=interpolation_kind)
        return interpolator(xx)
    else:
        return np.interp(xx, x, y)


class MilestonesModel(BaseModel):
    """Model for the milestone labels and their positions on the x axis.

    Ensures that milestone labels exist, are properly structured, and valid.
    """

    milestone_labels: dict[str, int]


class SimplePlotter(BasePlotter):
    """Simple plotter using pyplot."""

    id: ClassVar[str] = "rw_simple_plotter"
    df: pd.DataFrame = Field(
        ..., description="A dataframe containing the data to plot."
    )
    width: Optional[float | int] = Field(
        default=6.4, description="The width in inches."
    )
    height: Optional[float | int] = Field(
        default=4.8, description="The height in inches."
    )
    figsize: Optional[tuple[float | int, float | int]] = Field(
        default=None,
        description="A tuple containing the width and height in inches (overrides the previous keywords).",
    )
    hide_spines: Optional[list[str]] = Field(
        default=["top", "right"],
        description="A list of ['top', 'right', 'bottom', 'left'] indicating which spines to hide.",
    )
    title: Optional[str] = Field(
        default="Rolling Windows Plot",
        description="The title to use for the plot.",
    )
    titlepad: Optional[float | int] = Field(
        default=6.0,
        description="The padding in points to place between the title and the plot. May need to be increased if you are showing milestone labels.",
    )
    title_position: Optional[str] = Field(
        default="top",
        description="Show the title on the 'bottom' or the 'top' of the figure.",
    )
    show_legend: Optional[bool] = Field(
        default=True, description="Whether to show the legend."
    )
    show_grid: Optional[bool] = Field(
        default=False, description="Whether to show the grid."
    )
    xlabel: Optional[str] = Field(
        default="Token Count",
        description="The text to display along the x axis.",
    )
    ylabel: Optional[str] = Field(
        default="Average Frequency",
        description="The text to display along the y axis.",
    )
    show_milestones: Optional[bool] = Field(
        default=False,
        description="Whether to show the milestone markers.",
    )
    milestone_colors: Optional[list[str] | str] = Field(
        default="teal",
        description="The colour or colours to use for milestone markers. See pyplot.vlines().",
    )
    milestone_style: Optional[str] = Field(
        default="--",
        description="The style of the milestone markers. See pyplot.vlines().",
    )
    milestone_width: Optional[int] = Field(
        default=1,
        description="The width of the milestone markers. See pyplot.vlines().",
    )
    show_milestone_labels: Optional[bool] = Field(
        default=False, description="Whether to show the milestone labels."
    )
    milestone_labels: Optional[dict] = Field(
        default=None,
        description="A dict with keys as milestone labels and values as token indexes.",
    )
    milestone_labels_ha: Optional[str] = Field(
        default="left",
        description="The horizontal alignment of the milestone labels. See pyplot.annotate().",
    )
    milestone_labels_va: Optional[str] = Field(
        default="baseline",
        description="The vertical alignment of the milestone labels. See pyplot.annotate().",
    )
    milestone_label_rotation: Optional[int] = Field(
        default=45,
        description="The rotation of the milestone labels. See pyplot.annotate().",
    )
    milestone_labels_offset: Optional[tuple] = Field(
        default=(-8, 4),
        description="A tuple containing the number of pixels along the x and y axes to offset the milestone labels. See pyplot.annotate().",
    )
    milestone_labels_textcoords: Optional[str] = Field(
        default="offset pixels",
        description="Whether to offset milestone labels by pixels or points. See pyplot.annotate(str).",
    )
    use_interpolation: Optional[bool] = Field(
        default=False, description="Whether to use interpolation on values."
    )
    interpolation_num: Optional[int] = Field(
        default=500, description="Number of values to add between points."
    )
    interpolation_kind: Optional[str] = Field(
        default="pchip", description="Algorithm to use for interpolation."
    )
    fig: Optional[plt.Figure] = None
    ax: Optional[plt.Axes] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _validate_edge_cases(self) -> None:
        """Validate edge cases for the PlotlyPlotter."""
        if self.show_milestones or self.show_milestone_labels:
            try:
                MilestonesModel(milestone_labels=self.milestone_labels)
            except ValueError:
                raise LexosException(
                    "The `show_milestones` and `show_milestone_labels` parameters require a value for `milestone_labels`. It should be a list of dicts where the keys are labels and the values are points on the x axis."
                )

    def __init__(self, **kwargs) -> None:
        """Initialise the instance with arbitrary keywords."""
        super().__init__(**kwargs)
        self._validate_edge_cases()

        # Drop the id column if it exists
        self.df.drop("id", axis=1, inplace=True, errors="ignore")

        # Get the plot dimensions and title position
        width, height = self._get_width_height()
        titlepad = self.titlepad
        titlepad = self._adjust_titlepad(titlepad, width, height)

        # Generate the plot
        self.fig, self.ax = plt.subplots(figsize=(width, height))

        # Set the spines
        for spine in self.hide_spines:
            self.ax.spines[spine].set_visible(False)

        # Labels and title
        plt.margins(x=0, y=0)
        plt.ticklabel_format(axis="both", style="plain")
        if self.title_position == "bottom":
            plt.title(self.title, y=-0.25)
        else:
            plt.title(self.title, pad=titlepad)
        # TODO: plt.xlabel(self.xlabel, fontsize=10)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)

    def _adjust_titlepad(self, titlepad: float, width: float, height: float) -> None:
        """Hack to move the title above the labels.

        Args:
            titlepad (float): The padding in points to place between the title and the plot.
            width (float): The width of the plot.
            height (float): The height of the plot.
        """
        fig, ax = plt.subplots(figsize=(width, height))
        plt.close()
        if self.show_milestone_labels and self.title_position == "top":
            # Only override self.titlepad if it is the default value
            if self.titlepad == 6.0:
                titlepad = self._get_label_height(
                    self.milestone_labels, self.milestone_label_rotation
                )
        return titlepad

    def _get_label_height(
        self, milestone_labels: dict, milestone_label_rotation: int
    ) -> float:
        """Calculate the height of the longest milestone label.

        Args:
            milestone_labels (dict): A dict containing milestone labels and x-axis positions.
            milestone_label_rotation (int): The rotation in degrees of the labels

        Returns:
            float: The height of the longest label.

        Note:
            This method is a hack to calculate the label height using a separate plot.
        """
        tmp_fig, tmp_ax = plt.subplots()
        r = tmp_fig.canvas.get_renderer()
        heights = set()
        for x in list(milestone_labels.keys()):
            t = tmp_ax.annotate(
                x,
                xy=(0, 0),
                xytext=(0, 0),
                textcoords="offset points",
                rotation=milestone_label_rotation,
            )
            bb = t.get_window_extent(renderer=r)
            heights.add(bb.height)
        plt.close()
        return max(list(heights))

    def _get_width_height(self) -> tuple[float, float]:
        """Set the figure size for the plot.

        Returns:
            tuple[float, float]: A tuple containing the width and height in inches.
        """
        if self.figsize:
            width = self.figsize[0]
            height = self.figsize[1]
        else:
            width = self.width
            height = self.height
        return (width, height)

    def _plot_interpolated(self, df: pd.DataFrame, **kwargs) -> None:
        """Plot with interpolate dvalues between points.

        Args:
            df (pd.DataFrame): A dataframe containing the data to plot.
        """
        x = np.arange(df.shape[0])
        xx = np.linspace(x[0], x[-1], self.interpolation_num)
        for term in df.columns:
            y = np.array(df[term].values.tolist())
            interpolated = interpolate(x, y, xx, self.interpolation_kind)
            plt.plot(xx, interpolated, label=term, **kwargs)

    def _show_milestones(self, df: pd.DataFrame, ax: Axes) -> None:
        """Plot the milestone markers and labels.

        Args:
            df (pd.DataFrame): A dataframe containing the data to plot.
            ax (Axes): The axes object to plot on.
        """
        # Plot the milestones with adjustments to the margin and spines
        # This looks like it is the highest value
        ymax = df.to_numpy().max()
        for k, v in self.milestone_labels.items():
            if self.show_milestones:
                plt.vlines(
                    x=v,
                    ymin=0,
                    ymax=ymax,
                    colors=self.milestone_colors,
                    ls=self.milestone_style,
                    lw=self.milestone_width,
                )
            if self.show_milestone_labels:
                ax.annotate(
                    k,
                    xy=(v, ymax),
                    ha=self.milestone_labels_ha,
                    va=self.milestone_labels_va,
                    rotation=self.milestone_label_rotation,
                    xytext=self.milestone_labels_offset,
                    textcoords=self.milestone_labels_textcoords,
                )

    @validate_call(config=model_config)
    def plot(self, show: Optional[bool] = True, **kwargs: Any) -> None:
        """Call the plotter.

        Args:
            show (Optional[bool]): Whether to show the plot after generating it.
            **kwargs (Any): Additional keyword arguments accepted by matplotlib.pyplot.plot().
        """
        # Grid
        if self.show_grid:
            plt.grid(visible=True)

        # Interpolation
        if self.use_interpolation:
            self._plot_interpolated(self.df, **kwargs)
        else:
            for term in self.df.columns:
                plt.plot(self.df[term].values.tolist(), label=term, **kwargs)  # self.ax
        if self.show_legend:
            plt.legend()

        # If milestones have been set, plot them
        if self.show_milestones or self.show_milestone_labels:
            self._show_milestones(self.df, self.ax)

        if not show:
            plt.close()

    @validate_call
    def save(self, path: Path | str, **kwargs) -> None:
        """Save the plot to a file (wrapper for `pyplot.savefig()`).

        Args:
            path (Path | str): The path to the file to save.

        Returns:
            None
        """
        if not self.fig:
            raise LexosException(
                "There is no plot to save. You must first calling `plotter(data)`."
            )
        self.fig.savefig(path, **kwargs)

    def show(self) -> None:
        """Display a plot.

        Note:
            Calling pyplot.show() doesn't work with an inline backend like Jupyter notebooks, so we need to detect this via a UserWarning and then call the `fig` attribute.
        """
        if not self.fig:
            raise LexosException(
                "There is no plot to show. You must first call `plotter(data)`."
            )
        return self.fig
