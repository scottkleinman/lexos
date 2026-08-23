"""density_plot.py.

Last Updated: August 23, 2026
Last Tested: August 23, 2026
"""

import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pydantic import BaseModel, ConfigDict, Field


class DensityPlot(BaseModel):
    """Encapsulate view_distances plotting logic as a density plot."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    distance_table: pd.DataFrame = Field(
        ...,
        description="Distance table containing pairwise distances between items.",
    )
    labels: list[str] = Field(
        ...,
        description="List of labels corresponding to the items in the distance table.",
    )
    frequencies: pd.DataFrame = Field(
        default_factory=pd.DataFrame,
        description="Optional frequency table for the items.",
    )
    author: str | None = Field(
        default=None, description="Specific author to highlight in the plot."
    )
    group: bool = Field(
        default=True,
        description="Whether to group distances by same author.",
    )
    pattern: str = Field(
        default=r"^.*?(?=[_\s-]|\d)",
        description="Regular expression pattern to extract author classes from labels.",
    )
    title: str | None = Field(
        default=None,
        description="Optional title for the density plot.",
    )
    palette: dict[str, str] | None = Field(
        default=None,
        description="Optional color palette for grouped density curves. Keys should be 'True' and 'False'.",
    )
    color: str = Field(
        default="#cccccc",
        description="Fill color for ungrouped density plots.",
    )
    left: float = Field(default=0.14, description="Left margin for the figure layout.")
    show_on_init: bool = Field(
        default=False,
        description="Whether to display the plot immediately when the object is created.",
    )
    right: float = Field(
        default=0.95, description="Right margin for the figure layout."
    )
    top: float = Field(default=0.92, description="Top margin for the figure layout.")
    bottom: float = Field(
        default=0.15, description="Bottom margin for the figure layout."
    )
    fig: plt.Figure | None = Field(
        default=None, description="Matplotlib figure containing the density plot."
    )

    def __init__(self, **data):
        """Initialize the DensityPlot object."""
        super().__init__(**data)
        self.labels = list(self.labels)
        self.frequencies = (
            self.frequencies if self.frequencies is not None else pd.DataFrame()
        )
        if self.author is not None and self.pattern is not None:
            classes = [
                self._extract_label_class(label, self.pattern) for label in self.labels
            ]
            if self.author not in classes:
                raise ValueError(
                    f"Author '{self.author}' is not present in the distance table."
                )
        if self.show_on_init:
            self.plot_density()
            plt.show()

    def _add_legend(self, ax):
        handles, labels = ax.get_legend_handles_labels()
        legend_items = [
            (h, l) for h, l in zip(handles, labels) if l and not l.startswith("_")
        ]
        if legend_items:
            handles, labels = zip(*legend_items)
            legend = ax.legend(
                handles,
                labels,
                title="",
                loc="upper center",
                bbox_to_anchor=(0.5, 1.08),
                ncol=2,
            )
            if legend is not None:
                legend.set_title("")

    def _build_density_frame(self, pattern: str) -> pd.DataFrame:
        """Build a DataFrame suitable for density plotting.

        Args:
            pattern (str): Regular expression pattern to extract author classes from labels.

        Returns:
            pd.DataFrame: DataFrame containing distances, same_author flags, and classes.
        """
        labels = list(self.distance_table.index)
        classes = [self._extract_label_class(label, pattern) for label in labels]

        matrix = self.distance_table.to_numpy()
        lower = np.tril_indices_from(matrix, k=-1)
        dvals = matrix[lower]
        pairs = list(zip(lower[0], lower[1]))
        same_author = [classes[i] == classes[j] for i, j in pairs]
        pair_class = [classes[i] for i, _ in pairs]

        return pd.DataFrame(
            {
                "d": dvals,
                "same_author": np.array(same_author).astype(str),
                "class": pair_class,
            }
        )

    def _create_figure(self):
        """Create the base figure and axes for the density plot."""
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.grid(False)
        return fig, ax

    def _disable_canvas_bbox_inches(self, fig):
        """Disable bbox_inches overrides on the figure's canvas print method."""
        canvas = getattr(fig, "canvas", None)
        if canvas is None or not hasattr(canvas, "print_figure"):
            return

        original_print_figure = canvas.print_figure

        def _print_figure_no_bbox_inches(*args, **kwargs):
            kwargs.pop("bbox_inches", None)
            return original_print_figure(*args, **kwargs)

        canvas.print_figure = _print_figure_no_bbox_inches

    def _extract_label_class(self, label: str, pattern: str) -> str:
        """Extract a class label from a document label using a fallback strategy."""
        match = re.search(pattern, label)
        if match:
            return match.group(0)

        for sep in ["_", " ", "-"]:
            if sep in label:
                return label.split(sep, 1)[0]

        digit_match = re.match(r"^(.+?)(?:\d+)$", label)
        if digit_match:
            return digit_match.group(1)

        return label

    def _finalize_axes(self, ax, df: pd.DataFrame):
        ax.set_xlim(left=0)
        ax.margins(x=0.05, y=0.05)
        ymin, ymax = ax.get_ylim()
        if ymin >= 0:
            ax.set_ylim(bottom=-0.05 * max(ymax, 1.0))
        ax.set_xlabel("Distance")
        ax.set_ylabel("Density")
        if self.title is not None:
            ax.set_title(self.title)
        elif self.author is not None:
            self._highlight_author(ax, df)
        sns.despine(ax=ax)

    def _finalize_figure(self, fig):
        fig.subplots_adjust(
            left=self.left, right=self.right, top=self.top, bottom=self.bottom
        )
        self._disable_canvas_bbox_inches(fig)

    def _get_palette(self) -> dict[str, str]:
        if self.palette is None:
            return {"True": "pink", "False": "lightblue"}
        return {
            "True": self.palette.get("True", "pink"),
            "False": self.palette.get("False", "lightblue"),
        }

    def _highlight_author(self, ax, df: pd.DataFrame):
        author_distances = df[df["class"] == self.author]
        ax.scatter(
            author_distances["d"],
            np.zeros(len(author_distances)),
            s=40,
            color="black",
            zorder=10,
        )
        ax.set_title(f"Points: distances between works of {self.author}")

    def _plot_grouped_density(self, ax, df: pd.DataFrame):
        counts = df["same_author"].value_counts().to_dict()
        true_count = int(counts.get("True", 0))
        false_count = int(counts.get("False", 0))
        if true_count < 2 or false_count < 2:
            warnings.warn(
                "Grouped density is unlikely to be meaningful because "
                f"there are only {true_count} same-author distance pair(s) "
                f"and {false_count} different-author distance pair(s). "
                "If you want a meaningful grouped density, add more same-author "
                "documents, verify that label grouping is correct, or use "
                "group=False for the overall distance density.",
                UserWarning,
                stacklevel=2,
            )
        palette = self._get_palette()
        sns.kdeplot(
            data=df,
            x="d",
            hue="same_author",
            fill=True,
            multiple="layer",
            alpha=0.55,
            palette=palette,
            common_norm=False,
            warn_singular=False,
            ax=ax,
            linewidth=0,
            legend=False,
        )
        meds = df.groupby("same_author")["d"].median()
        for dval in meds:
            ax.axvline(dval, linestyle="--", color="white", linewidth=1)
        self._add_legend(ax)

    def _plot_ungrouped_density(self, ax, df: pd.DataFrame):
        sns.kdeplot(
            data=df,
            x="d",
            fill=True,
            color=self.color,
            alpha=0.5,
            warn_singular=False,
            ax=ax,
            linewidth=0,
        )
        ax.axvline(df["d"].median(), linestyle="--", color="white", linewidth=1)

    def plot_density(self):
        """Plot the density of distances, optionally grouped by author."""
        df = self._build_density_frame(self.pattern)
        with plt.ioff():
            fig, ax = self._create_figure()
            if self.group:
                self._plot_grouped_density(ax, df)
            else:
                self._plot_ungrouped_density(ax, df)
            self._finalize_axes(ax, df)
            self._finalize_figure(fig)
        self.fig = fig

    def show(self):
        """Display the density plot."""
        if self.fig is None:
            self.plot_density()
        plt.show()
