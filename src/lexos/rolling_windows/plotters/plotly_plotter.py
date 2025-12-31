"""plotly_plotter.py.

Last Update: December 4, 2025
Last Tested: September 13, 2025
"""

from pathlib import Path
from typing import Any, ClassVar, Optional

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.io as pio
from plotly.graph_objects import Figure
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
    validate_call,
)

from lexos.exceptions import LexosException
from lexos.rolling_windows.plotters.base_plotter import BasePlotter
from lexos.util import ensure_list


class MilestonesModel(BaseModel):
    """Model for the milestone labels and their positions on the x axis.

    Ensures that milestone labels exist, are properly structured, and valid.
    """

    milestone_labels: dict[str, int]

    @model_validator(mode="after")
    def check_empty_dict(self):
        """Check that the milestone_labels dict is not empty."""
        if not self.milestone_labels or len(self.milestone_labels) == 0:
            raise ValueError("`milestone_labels` dictionary is empty.")
        return self


class PlotlyPlotter(BasePlotter):
    """Plotter using Plotly."""

    id: ClassVar[str] = "rw_plotly_plotter"

    df: pd.DataFrame = Field(
        ..., description="A dataframe containing the data to plot."
    )
    width: Optional[int] = Field(
        default=700, description="The width of the plot in pixels."
    )
    height: Optional[int] = Field(
        default=450,
        description="The height of the plot in pixels. Note that if you change the height, you will need to adjust the `titelpad` manually to show the title above milestone labels.",
    )
    title: Optional[dict | str] = Field(
        default="Rolling Windows Plot",
        description="The title to use for the plot. It can be styled with a dict containing any of the keywords listed at https://plotly.com/python/reference/layout/#layout-title.",
    )
    xlabel: Optional[str] = Field(
        default="Token Count", description="The text to display along the x axis."
    )
    ylabel: Optional[str] = Field(
        default="Average Frequency", description="The text to display along the y axis."
    )
    line_color: Optional[list[str] | str] = Field(
        default=px.colors.qualitative.Plotly,
        description="The colour pattern to use for lines on the plot.",
    )
    showlegend: Optional[bool] = Field(default=True, description="Show the legend.")
    titlepad: Optional[float] = Field(
        default=None,
        description="The margin in pixels between the title and the top of the plot.",
    )
    show_milestones: Optional[bool] = Field(
        default=False, description="Whether to show the milestone markers."
    )
    milestone_marker_style: Optional[dict] = Field(
        default={"width": 1, "color": "teal"},
        description="A dict containing the styles to apply to the milestone marker. For valid properties, see https://plotly.com/python-api-reference/generated/plotly.graph_objects.layout.shape.html#plotly.graph_objects.layout.shape.Line.",
    )
    show_milestone_labels: Optional[bool] = Field(
        default=False, description="Whether to show the milestone labels."
    )
    milestone_labels: Optional[dict[str, int]] = Field(
        default=None,
        description="A dict containing the milestone labels and their values on the x-axis.",
    )
    milestone_label_rotation: Optional[float] = Field(
        default=0.0,
        description="The number of degrees clockwise to rotate the milestone labels (maximum 90).",
    )
    milestone_label_style: Optional[dict] = Field(
        default={
            "size": 10.0,
            "family": "Open Sans, verdana, arial, sans-serif",
            "color": "teal",
        },
        description="A dict containing the styling information for the milestone labels. For valid properties, see https://plotly.com/python/reference/layout/annotations/#layout-annotations-items-annotation-font.",
    )
    fig: Optional[Figure] = None

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    @field_validator("milestone_label_rotation", mode="after")
    @classmethod
    def is_valid_rotation(cls, value: float) -> float:
        """Ensure that the milestone label rotation is between 0 and 90 degrees."""
        if value > 90:
            raise LexosException(
                "Milestone labels can only be rotated clockwise a maximum of 90 degrees."
            )
        return value

    def _validate_edge_cases(self) -> None:
        """Validate edge cases for the PlotlyPlotter."""
        if self.show_milestones is True or self.show_milestone_labels is True:
            try:
                MilestonesModel(milestone_labels=self.milestone_labels)
            except ValidationError:
                if not self.milestone_labels or len(self.milestone_labels) == 0:
                    raise LexosException("`milestone_labels` dictionary is empty.")

    def __init__(self, **kwargs) -> None:
        """Initialise the instance with arbitrary keywords."""
        super().__init__(**kwargs)
        self._validate_edge_cases()

        # Massage the DataFrame for Plotly Express
        self.df["id"] = self.df.index

    @validate_call(config=model_config)
    def plot(
        self, show: Optional[bool] = True, config: Optional[dict] = None, **kwargs: Any
    ) -> None:
        """Initialise object.

        Args:
            show (Optional[bool]): Whether to display the plot immediately.
            config (Optional[dict]): A dictionary supply Plotly configuration values.
            **kwargs (Any): Additional keyword arguments accepted by plotly.express.line.

        """
        # Massage the DataFrame for Plotly Express
        df = self.df.copy()
        df["id"] = df.index
        df = pd.melt(df, id_vars="id", value_vars=df.columns[:-1])

        # Create plotly line figure
        self.fig = px.line(
            df,
            x="id",
            y="value",
            color="variable",
            color_discrete_sequence=ensure_list(self.line_color),
            width=self.width,
            height=self.height,
        )

        title_dict, xlabel_dict, ylabel_dict = self._get_axis_and_title_labels()
        self.fig.update_layout(
            title=title_dict,
            xaxis=xlabel_dict,
            yaxis=ylabel_dict,
            showlegend=self.showlegend,
            **kwargs,
        )

        # Show milestones
        if self.show_milestones:
            # Add milestones using absolute references
            for label, x in self.milestone_labels.items():
                df_val_min = df["value"].min() * 1.2
                df_val_max = df["value"].max() * 1.2
                self._plot_milestone_marker(x, df_val_min, df_val_max)
                if self.show_milestone_labels:
                    self._plot_milestone_label(label, x)

        # Increase the margin from the top to accommodate the milestone labels
        if self.show_milestone_labels:
            titlepad = self._get_titlepad(self.milestone_labels)
            self.fig.update_layout(margin=dict(t=titlepad))

        if show:
            if not config:
                config = {"displaylogo": False}
            self.fig.show(config=config)

    def _get_axis_and_title_labels(self) -> tuple[bool, str]:
        """Ensure that the title, xlabel, and ylabel values are dicts."""
        if isinstance(self.title, str):
            title_dict = dict(
                text=self.title, y=0.9, x=0.5, xanchor="center", yanchor="top"
            )
        else:
            title_dict = self.title
        if isinstance(self.xlabel, str):
            xlabel_dict = dict(title=self.xlabel)
        else:
            xlabel_dict = self.xlabel
        if isinstance(self.ylabel, str):
            ylabel_dict = dict(title=self.ylabel)
        else:
            ylabel_dict = self.ylabel
        return title_dict, xlabel_dict, ylabel_dict

    def _get_titlepad(self, labels: dict[str, int]) -> float:
        """Get a titlepad value based on the height of the longest milestone label.

        Args:
            labels (dict[str, int]): A dict with the labels as keys.

        Returns:
            A float.
        """
        if self.titlepad:
            return self.titlepad
        fontfamily = self.milestone_label_style["family"]
        if "," in self.milestone_label_style["family"]:
            fontfamily = self.milestone_label_style["family"].split(",")
            fontfamily = [x.strip() for x in fontfamily]
        tmp_fig, tmp_ax = plt.subplots()
        r = tmp_fig.canvas.get_renderer()
        heights = []
        for x in list(labels.keys()):
            t = tmp_ax.annotate(
                x,
                xy=(0, 0),
                xytext=(0, 0),
                textcoords="offset points",
                rotation=self.milestone_label_rotation,
                fontfamily=fontfamily,
                fontsize=self.milestone_label_style["size"],
            )
            bb = t.get_window_extent(renderer=r)
            heights.append(bb.height)
        plt.close()
        if max(heights) < 50:
            return max(heights) + 75
        else:
            return max(heights) + 50

    def _plot_milestone_label(self, label: str, x: int) -> None:
        """Add a milestone label to the Plotly figure.

        Args:
            label (str): The label text.
            x (int): The location on the x axis.
        """
        self.fig.add_annotation(
            x=x,
            y=1,
            xanchor="left",
            yanchor="bottom",
            xshift=-10,
            yref="paper",
            showarrow=False,
            text=label,
            textangle=-self.milestone_label_rotation,
            font=self.milestone_label_style,
        )

    def _plot_milestone_marker(
        self, x: int, df_val_min: float | int, df_val_max: float | int
    ) -> None:
        """Add a milestone marker to the Plotly figure.

        Args:
            x (int): The location on the x axis.
            df_val_min (float | int): The minimum value in the pandas DataFrame.
            df_val_max (float | int): The maximum value in the pandas DataFrame.
        """
        self.fig.add_shape(
            type="line",
            yref="y",
            xref="x",
            x0=x,
            y0=0,  # df_val_min,
            x1=x,
            y1=df_val_max,
            line=self.milestone_marker_style,
        )

    @validate_call(config=model_config)
    def save(self, path: Path | str, **kwargs: Any) -> None:
        """Save the plot to a file.

        Args:
            path (Path | str): The path to the file to save.
            **kwargs (Any): Additional keyword arguments accepted by plotly.io.write_html or plotly.io.write_image.
        """
        if not self.fig:
            raise LexosException(
                "There is no plot to save, try calling `plotter(data)`."
            )
        # Try first to save as HTML; if that doesn't work, try to save as a static image
        if Path(path).suffix == ".html":
            self.fig.write_html(path, **kwargs)
        else:
            pio.write_image(self.fig, path)

    @validate_call(config=model_config)
    def show(self, config: Optional[dict] = None) -> None:
        """Display a plot.

        Args:
            config (Optional[dict]): A dictionary supply Plotly configuration values.
        """
        if not config:
            config = {"displaylogo": False}
        self.fig.show(config=config)
