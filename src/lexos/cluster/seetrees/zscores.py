"""zscores.py.

Last Updated: August 23, 2026
Last Tested: August 23, 2026
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pydantic import BaseModel, ConfigDict, Field

from .util import sanitize_label_text


class ZscorePlot(BaseModel):
    """Render a ranked z-score bar chart for a target text.

    The current SeeTrees class only creates DistinctiveFeaturePlot objects, since the plot can be quite cluttered. This class is provided as a basis for further development for users who prefer Matplotlib.
    """

    frequencies: pd.DataFrame = Field(
        default_factory=pd.DataFrame, description="Term frequency table."
    )
    target_text: str = Field(default="", description="Label for the target text.")
    top: int = Field(default=20, description="Number of features to display.")
    title: str | None = Field(default=None, description="Optional plot title.")
    positive_color: str = Field(
        default="#f6c1cc", description="Bar color for positive z-scores."
    )
    negative_color: str = Field(
        default="#b9dff1", description="Bar color for negative z-scores."
    )
    guide_color: str = Field(
        default="#c9ced6", description="Guide line color for non-zero SD lines."
    )
    zero_line_color: str = Field(
        default="red", description="Guide line color for the zero SD line."
    )
    fig: plt.Figure | None = Field(
        default=None, description="Matplotlib Figure object for plotting."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    def _z_scores(self) -> pd.DataFrame:
        z_scores = (self.frequencies - self.frequencies.mean()) / self.frequencies.std()
        return z_scores.fillna(0)

    def _top_series(self) -> pd.Series:
        z_scores = self._z_scores()
        text_profile = z_scores.loc[self.target_text]
        top_series = text_profile.abs().sort_values(ascending=False).head(self.top)
        selected = text_profile.loc[top_series.index]

        positive = selected[selected >= 0].sort_values(ascending=False)
        negative = selected[selected < 0].sort_values(ascending=True)
        return pd.concat([positive, negative])

    def plot(self) -> plt.Figure:
        """Render the z-score chart and return the figure."""
        ranked = self._top_series()
        features = ranked.index.tolist()
        values = ranked.to_numpy(dtype=float)
        colors = np.where(values >= 0, self.positive_color, self.negative_color)
        y_pos = np.arange(len(features))

        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        bars = ax.barh(y_pos, values, color=colors, edgecolor="white", height=0.95)
        ax.invert_yaxis()
        ax.set_yticks([])

        min_value = float(np.floor(min(values.min(), -1.0)))
        max_value = float(np.ceil(max(values.max(), 1.0)))

        min_tick = int(2 * np.floor(min_value / 2))
        max_tick = int(2 * np.ceil(max_value / 2))
        ax.set_xlim(min_tick, max_tick)
        ax.set_xticks(np.arange(min_tick, max_tick + 1, 2))

        for sd in range(int(np.floor(min_value)), int(np.ceil(max_value)) + 1):
            if sd == 0:
                continue
            ax.axvline(
                sd, color=self.guide_color, linestyle=":", linewidth=1.6, zorder=0
            )

        ax.axvline(
            0,
            color=self.zero_line_color,
            linestyle=":",
            linewidth=2.0,
            zorder=1,
        )

        # Keep feature labels close to the zero guide line for readability.
        label_offset = max(0.012 * (max_tick - min_tick), 0.04)
        for i, (feature, value, bar) in enumerate(zip(features, values, bars)):
            y_center = bar.get_y() + bar.get_height() / 2
            ax.text(
                value / 2,
                y_center,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white",
                clip_on=False,
            )
            if value >= 0:
                ax.text(
                    -label_offset,
                    y_center,
                    sanitize_label_text(feature),
                    ha="right",
                    va="center",
                    fontsize=10,
                    color="#4d4d4d",
                )
            else:
                ax.text(
                    label_offset,
                    y_center,
                    sanitize_label_text(feature),
                    ha="left",
                    va="center",
                    fontsize=10,
                    color="#4d4d4d",
                )

        ax.tick_params(axis="x", colors="#4d4d4d", labelsize=11)
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        if self.title:
            ax.set_title(self.title)
        else:
            ax.set_title(f"Top {self.top} z-scores in {self.target_text}")
        ax.set_xlabel("Standard deviation from the corpus mean")

        self.fig = fig
        return fig

    def show(self) -> None:
        """Display the z-score chart, creating it if needed."""
        if self.fig is None:
            self.plot()
        plt.show()


class DistinctiveFeaturePlot(BaseModel):
    """Render a ranked z-score bar chart for a target text using Plotly."""

    frequencies: pd.DataFrame = Field(
        default_factory=pd.DataFrame, description="Term frequency table."
    )
    target_text: str = Field(default="", description="Label for the target text.")
    top: int = Field(default=20, description="Number of features to display.")
    title: str | None = Field(default=None, description="Optional plot title.")
    positive_color: str = Field(
        default="#f6c1cc", description="Bar color for positive z-scores."
    )
    negative_color: str = Field(
        default="#b9dff1", description="Bar color for negative z-scores."
    )
    guide_color: str = Field(
        default="#c9ced6", description="Guide line color for non-zero SD lines."
    )
    zero_line_color: str = Field(
        default="red", description="Guide line color for the zero SD line."
    )
    width: int = Field(default=800, description="Figure width in pixels.")
    height: int = Field(default=600, description="Figure height in pixels.")
    fig: go.Figure | None = Field(
        default=None, description="Plotly Figure object for plotting."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    def _z_scores(self) -> pd.DataFrame:
        z_scores = (self.frequencies - self.frequencies.mean()) / self.frequencies.std()
        return z_scores.fillna(0)

    def _top_series(self) -> pd.Series:
        z_scores = self._z_scores()
        text_profile = z_scores.loc[self.target_text]
        top_series = text_profile.abs().sort_values(ascending=False).head(self.top)
        selected = text_profile.loc[top_series.index]

        positive = selected[selected >= 0].sort_values(ascending=False)
        negative = selected[selected < 0].sort_values(ascending=False)
        return pd.concat([positive, negative])

    def plot(self) -> go.Figure:
        """Render the z-score chart and return the Plotly figure."""
        ranked = self._top_series()
        features = ranked.index.tolist()
        values = ranked.to_numpy(dtype=float)
        colors = [
            self.positive_color if v >= 0 else self.negative_color for v in values
        ]

        max_abs = max(abs(values).max(), 1.0)
        min_value = float(np.floor(min(values.min(), -1.0)))
        max_value = float(np.ceil(max(values.max(), 1.0)))

        fig = go.Figure(
            data=go.Bar(
                x=values,
                y=features,
                orientation="h",
                marker_color=colors,
                text=[f"{v:.2f}" for v in values],
                textposition="inside",
                insidetextanchor="middle",
                hovertemplate="%{y}<br>Z-score: %{x:.2f}<extra></extra>",
            )
        )

        shapes = []
        for sd in np.arange(int(np.floor(min_value)), int(np.ceil(max_value)) + 1):
            if sd == 0:
                shapes.append(
                    dict(
                        type="line",
                        x0=0,
                        x1=0,
                        y0=-0.5,
                        y1=len(features) - 0.5,
                        line=dict(color=self.zero_line_color, dash="dot", width=2),
                    )
                )
            else:
                shapes.append(
                    dict(
                        type="line",
                        x0=sd,
                        x1=sd,
                        y0=-0.5,
                        y1=len(features) - 0.5,
                        line=dict(color=self.guide_color, dash="dot", width=1),
                    )
                )

        annotations = []
        small_offset = max_abs * 0.02
        for feature, value in zip(features, values):
            x = -small_offset if value >= 0 else small_offset
            anchor = "right" if value >= 0 else "left"
            annotations.append(
                dict(
                    x=x,
                    y=feature,
                    xanchor=anchor,
                    yanchor="middle",
                    text=sanitize_label_text(feature),
                    showarrow=False,
                    font=dict(color="#4d4d4d", size=10),
                )
            )

        fig.update_layout(
            title=self.title or f"Top {self.top} z-scores in {self.target_text}",
            xaxis=dict(
                title="Standard deviation from the corpus mean",
                tickmode="linear",
                dtick=2,
                zeroline=False,
            ),
            yaxis=dict(autorange="reversed", showticklabels=False),
            plot_bgcolor="white",
            shapes=shapes,
            annotations=annotations,
            margin=dict(l=140, r=40, t=80, b=40),
            width=self.width,
            height=self.height,
        )

        self.fig = fig
        return fig

    def show(self) -> None:
        """Display the Plotly z-score chart without the Plotly logo."""
        if self.fig is None:
            self.plot()
        self.fig.show(config={"displaylogo": False})


class FeatureSummary(BaseModel):
    """Encapsulate view_scores summary rendering.

    Suggestions for improving the chart:
      - switch selection from raw z-score cutoff to rank-based or feature-importance ranking
      - split positive and negative features into separate views instead of mixing both directions
      - use a dot/lollipop plot rather than bars to reduce visual clutter when many values tie
      - annotate only the most distinctive features and omit low-variance ties
      - collapse ties into grouped rank buckets when many values are identical
    """

    frequencies: pd.DataFrame = Field(
        default_factory=pd.DataFrame, description="Term frequency table."
    )
    target_text: str = Field(default="", description="Label for the target text.")
    top: int = Field(default=20, description="Number of features to display.")

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    def _z_scores(self) -> pd.DataFrame:
        z_scores = (self.frequencies - self.frequencies.mean()) / self.frequencies.std()
        return z_scores.fillna(0)

    def _feature_order(self) -> list[str]:
        return (
            self.frequencies.mean()
            .sort_values(ascending=False)
            .index.tolist()[: self.top]
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the top distinctive features and their z-scores into a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing the top features and their z-scores.
        """
        z_scores = self._z_scores()
        text_profile = z_scores.loc[self.target_text]
        top_features = (
            text_profile.abs()
            .sort_values(ascending=False)
            .head(self.top)
            .index.tolist()
        )

        return pd.DataFrame(
            {
                "Feature": top_features,
                "Z-score": text_profile.loc[top_features].astype(float).to_numpy(),
            }
        )

    def render_bar_chart(self) -> plt.Figure:
        """Render a horizontal bar chart of the top distinctive features for the target text.

        Returns:
            plt.Figure: Matplotlib figure containing the bar chart.
        """
        plotter = ZscorePlot(
            frequencies=self.frequencies,
            target_text=self.target_text,
            top=self.top,
        )
        return plotter.plot()
