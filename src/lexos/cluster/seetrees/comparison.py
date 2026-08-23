"""comparison.py.

Last Updated: August 23, 2026
Last Tested: August 23, 2026
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .util import sanitize_label_text


class ComparisonPlot(BaseModel):
    """Base class for stylometric comparison plots."""

    frequencies: pd.DataFrame = Field(
        default_factory=pd.DataFrame, description="Term frequency table."
    )
    source_text: str = Field(default="", description="Label for the source text.")
    target_text: str = Field(default="", description="Label for the target text.")
    top_diff: int = Field(
        default=10, description="Number of top differences to highlight."
    )
    max_rank: int = Field(
        default=100, description="Limit for the number of features to rank."
    )
    title: str | None = Field(
        default=None, description="Optional title for the comparison plot."
    )
    fig: plt.Figure | None = Field(
        default=None, description="Matplotlib figure object."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    def _apply_figure_layout(
        self,
        fig,
        left: float = 0.06,
        right: float = 0.96,
        top: float = 0.94,
        bottom: float = 0.12,
    ):
        """Apply a compact layout to a Matplotlib Figure."""
        fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
        self._disable_canvas_bbox_inches(fig)

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

    def _z_scores(self) -> pd.DataFrame:
        """Compute z-scores for the frequency table.

        Returns:
            pd.DataFrame: Z-scores for the frequency table.
        """
        z_scores = (self.frequencies - self.frequencies.mean()) / self.frequencies.std()
        return z_scores.fillna(0)

    def _feature_order(self) -> list[str]:
        """Determine the order of features based on mean frequency.

        Returns:
            list[str]: Ordered list of feature names based on mean frequency.
        """
        return (
            self.frequencies.mean()
            .sort_values(ascending=False)
            .index.tolist()[: self.max_rank]
        )


class OverlayPlot(ComparisonPlot):
    """Plot the stylometric overlay of two texts."""

    source_color: str = Field(
        default="#ff9999", description="Color used for the source text overlay line."
    )
    target_color: str = Field(
        default="#99c2ff", description="Color used for the target text overlay line."
    )
    frequencies: pd.DataFrame = Field(
        default_factory=pd.DataFrame, description="Term frequency table."
    )
    source_text: str = Field(default="", description="Label for the source text.")
    target_text: str = Field(default="", description="Label for the target text.")
    top_diff: int = Field(
        default=10, description="Number of top differences to highlight."
    )
    max_rank: int = Field(
        default=100, description="Limit for the number of features to rank."
    )
    title: str | None = Field(
        default=None, description="Optional title for the comparison plot."
    )
    fig: plt.Figure | None = Field(
        default=None, description="Matplotlib figure object."
    )

    def __init__(self, **data):
        """Initialize the OverlayPlot with given data."""
        super().__init__(**data)
        self.fig = self.plot_overlay()
        self._apply_figure_layout(self.fig)

    def plot_overlay(self) -> plt.Figure:
        """Plot the stylometric overlays of the source and target texts.

        Returns:
            plt.Figure: Matplotlib figure containing the overlay plot.
        """
        z_scores = self._z_scores()
        src_profile = z_scores.loc[self.source_text]
        tgt_profile = z_scores.loc[self.target_text]
        feature_order = self._feature_order()
        x = np.arange(1, len(feature_order) + 1)
        src_values = src_profile.loc[feature_order].astype(float).to_numpy()
        tgt_values = tgt_profile.loc[feature_order].astype(float).to_numpy()

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.grid(False)
        ax.plot(
            x,
            src_values,
            color=self.source_color,
            linewidth=1.8,
            label=self.source_text,
            alpha=0.85,
        )
        ax.plot(
            x,
            tgt_values,
            color=self.target_color,
            linewidth=1.8,
            label=self.target_text,
            alpha=0.85,
        )

        ax.axhline(0, color="black", linestyle="-", linewidth=1)
        for sd in [-2, -1, 1, 2]:
            ax.axhline(sd, color="gray", linestyle="--", alpha=0.45, linewidth=0.8)

        combined_scores = np.maximum(np.abs(src_values), np.abs(tgt_values))
        peak_indices = np.argsort(combined_scores)[-self.top_diff :][::-1]

        for idx in peak_indices:
            label = feature_order[idx]
            x_pos = x[idx]
            y_value = (
                src_values[idx]
                if abs(src_values[idx]) >= abs(tgt_values[idx])
                else tgt_values[idx]
            )
            y_offset = 0.08 if y_value >= 0 else -0.08
            ax.plot(
                [x_pos, x_pos],
                [0, y_value],
                color="gray",
                linestyle=":",
                linewidth=0.8,
                alpha=0.7,
            )
            ax.scatter([x_pos], [y_value], color="black", s=20)
            ax.text(
                x_pos,
                y_value + y_offset,
                sanitize_label_text(label),
                ha="center",
                va="bottom" if y_value >= 0 else "top",
                fontsize=9,
                color="black",
            )

        ax.set_xlim(0.5, len(feature_order) + 0.5)
        if not self.title:
            ax.set_title(
                f"Stylometric Overlay: {self.source_text} vs {self.target_text}"
            )
        else:
            ax.set_title(self.title)
        ax.set_xlabel("Feature frequency rank")
        ax.set_ylabel("Standard deviation from the corpus mean")
        ax.tick_params(axis="y", labelleft=True)
        ax.yaxis.set_ticks_position("left")
        return fig

    def show(self) -> None:
        """Display the overlay plot, creating it if necessary."""
        if self.fig is None:
            self.fig = self.plot_overlay()
        plt.show()


class DifferencePlot(ComparisonPlot):
    """Plot the z-score difference between two texts."""

    base_color: str = Field(
        default="gray", description="Color used for non-highlighted difference bars."
    )
    highlight_color: str = Field(
        default="red", description="Color used for highlighted difference bars."
    )
    frequencies: pd.DataFrame = Field(
        default_factory=pd.DataFrame, description="Term frequency table."
    )
    source_text: str = Field(default="", description="Label for the source text.")
    target_text: str = Field(default="", description="Label for the target text.")
    top_diff: int = Field(
        default=10, description="Number of top differences to highlight."
    )
    max_rank: int = Field(
        default=100, description="Limit for the number of features to rank."
    )
    title: str | None = Field(
        default=None, description="Optional title for the comparison plot."
    )
    fig: plt.Figure | None = Field(
        default=None, description="Matplotlib figure object."
    )

    def __init__(self, **data):
        """Initialize the DifferencePlot with given data."""
        super().__init__(**data)
        self.fig = self.plot_difference()
        self._apply_figure_layout(self.fig)

    def plot_difference(self) -> plt.Figure:
        """Plot the difference in z-scores between the source and target texts.

        Returns:
            plt.Figure: Matplotlib figure containing the difference plot.
        """
        z_scores = self._z_scores()
        src_profile = z_scores.loc[self.source_text]
        tgt_profile = z_scores.loc[self.target_text]
        feature_order = self._feature_order()
        diff_series = (tgt_profile - src_profile).loc[feature_order].astype(float)

        df_comp = pd.DataFrame(
            {
                "Feature": feature_order,
                "Rank": np.arange(1, len(feature_order) + 1),
                "Difference": diff_series.values,
            }
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.subplots_adjust(left=0.12, right=0.96, top=0.94, bottom=0.12)
        diff_mask = df_comp["Difference"].abs().nlargest(self.top_diff).index
        fill_colors = np.where(
            df_comp.index.isin(diff_mask),
            self.highlight_color,
            self.base_color,
        )

        ax.bar(
            df_comp["Rank"],
            df_comp["Difference"],
            color=fill_colors,
            edgecolor="grey",
            width=0.8,
        )
        ax.axhline(0, color="black", linestyle="-", alpha=0.7)

        if not self.title:
            ax.set_title(
                f"Z-Score Differences: {self.target_text} minus {self.source_text}"
            )
        else:
            ax.set_title(self.title)
        ax.set_xlabel("Feature frequency rank")
        ax.set_ylabel("Difference between z-scores")

        ax.set_xlim(0.0, len(feature_order) + 1)
        ax.set_xticks(np.arange(0, len(feature_order) + 1, 25))
        ax.set_xticklabels(
            [str(int(tick)) for tick in np.arange(0, len(feature_order) + 1, 25)],
            rotation=45,
            ha="right",
            fontsize=9,
        )
        ax.set_yticks(
            np.arange(
                int(np.floor(ax.get_ylim()[0])),
                int(np.ceil(ax.get_ylim()[1])) + 1,
                1,
            )
        )

        ax.set_facecolor("white")
        ax.grid(False)

        top_feature_set = set(df_comp.loc[diff_mask, "Feature"])
        y_min, y_max = ax.get_ylim()
        label_offset = (y_max - y_min) * 0.03
        for bar, feature, diff in zip(
            ax.patches, df_comp["Feature"], df_comp["Difference"]
        ):
            if feature not in top_feature_set:
                continue
            rank = bar.get_x() + bar.get_width() / 2
            y = diff + (label_offset if diff >= 0 else -label_offset)
            va = "bottom" if diff >= 0 else "top"
            ax.text(
                rank,
                y,
                sanitize_label_text(feature),
                ha="center",
                va=va,
                fontsize=8,
                color="black",
                clip_on=False,
            )

        supp_text = [
            (0.01, 0.95, f"more in {self.source_text}"),
            (0.01, 0.05, f"more in {self.target_text}"),
        ]
        ax.text(
            supp_text[0][0],
            supp_text[0][1],
            supp_text[0][2],
            transform=ax.transAxes,
            rotation=90,
            va="top",
            ha="left",
            color="pink",
            fontsize=9,
        )
        ax.text(
            supp_text[1][0],
            supp_text[1][1],
            supp_text[1][2],
            transform=ax.transAxes,
            rotation=90,
            va="bottom",
            ha="left",
            color="lightblue",
            fontsize=9,
        )

        return fig

    def show(self) -> None:
        """Display the difference plot, creating it if necessary."""
        if self.fig is None:
            self.fig = self.plot_difference()
        plt.show()
