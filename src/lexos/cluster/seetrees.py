"""seetrees.py.

This module is adapted from the R code from the 'see' package by Artjoms Šeļa (https://github.com/perechen/seetrees).

Last Updated: August 18, 2026
Last Tested: August 18, 2026
"""

import re
import warnings
from typing import Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from pydantic import BaseModel, ConfigDict, Field


def sanitize_label_text(label: str) -> str:
    """Sanitize data-driven plot labels for whitespace and linebreak visibility."""
    if label is None:
        return ""
    if "\r\n" in label or "\r" in label or "\n" in label:
        label = label.replace("\r\n", "<linebreak>")
        label = label.replace("\r", "<linebreak>").replace("\n", "<linebreak>")
    if " " in label or "\t" in label:
        label = label.replace(" ", "<whitespace>").replace("\t", "<whitespace>")
    return label


from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

from lexos.dtm import DTM


class Tree(BaseModel):
    """Render a stylometric dendrogram and cluster word summary."""

    labels: list[str] = Field(
        default_factory=list, description="List of document labels."
    )
    distance_table: pd.DataFrame = Field(
        default_factory=pd.DataFrame, description="Pairwise distance table."
    )
    frequencies: pd.DataFrame = Field(
        default_factory=pd.DataFrame, description="Term frequency table."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    def __init__(self, **data):
        """Initialize the Tree object."""
        super().__init__(**data)
        self.labels = list(self.labels)
        self.frequencies = (
            self.frequencies if self.frequencies is not None else pd.DataFrame()
        )

    def view(
        self,
        k: int = 2,
        top_n_words: int = 10,
        outline_y_pad: float = 0.3,
        outline_axis_y_pad: float = 0.1,
        outline_tip_pad_ratio: float = 0.002,
        outline_root_pad_ratio: float = 0.1,
    ):
        """Render a dendrogram and show cluster-specific top words.

        Args:
            k (int): Number of clusters to display in the dendrogram.
            top_n_words (int): Number of top words to display for each cluster.
            outline_y_pad (float): Vertical padding for cluster outlines.
            outline_axis_y_pad (float): Additional vertical padding for the axis.
            outline_tip_pad_ratio (float): Horizontal padding ratio for dendrogram tips.
            outline_root_pad_ratio (float): Horizontal padding ratio for dendrogram root.

        Raises:
            ValueError: If the distance table is empty.
        """
        if self.distance_table.empty:
            raise ValueError(
                "Distance table is required. Run compute_distances() first."
            )

        condensed = squareform(self.distance_table.to_numpy())
        z = linkage(condensed, method="ward")
        clusters = fcluster(z, k, criterion="maxclust")

        threshold = z[-k, 2] if 1 < k < len(self.labels) else 0.0
        palette = sns.color_palette("tab10", max(3, k))
        color_func = self._dendrogram_color_func(z, clusters, palette)
        cluster_words = self._cluster_top_words(clusters, top_n_words)

        fig = plt.figure(figsize=(14, 10))
        fig.patch.set_facecolor("white")
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.24)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax1.set_facecolor("white")
        ax2.set_facecolor("white")

        dendro = dendrogram(
            z,
            labels=self.labels,
            orientation="right",
            color_threshold=threshold,
            link_color_func=color_func,
            ax=ax1,
        )

        x0, x1 = ax1.get_xlim()
        if x0 < x1:
            ax1.invert_xaxis()

        ax1.margins(x=0)

        ax1.yaxis.tick_right()
        ax1.tick_params(axis="y", labelright=True, labelleft=False, pad=1)
        ax1.spines["right"].set_visible(False)
        ax1.grid(False)

        x0, x1 = ax1.get_xlim()
        tip_x_pad = abs(x1 - x0) * max(0.0, outline_tip_pad_ratio)
        root_x_pad = abs(x1 - x0) * max(0.0, outline_root_pad_ratio)
        self._style_cluster_regions(
            ax1,
            dendro,
            clusters,
            palette,
            threshold,
            tip_x_pad,
            root_x_pad,
            outline_y_pad,
            outline_axis_y_pad,
        )

        ax1.set_title(f"Dendrogram Cut into k={k} Groups")
        ax1.set_xlabel("Distance")
        ax1.tick_params(axis="y", labelsize=7)

        ax2.axis("off")
        x_positions = np.linspace(0.15, 0.85, k)
        for idx, cluster_id in enumerate(sorted(cluster_words)):
            words = cluster_words[cluster_id]
            color = palette[idx % len(palette)]
            x = x_positions[idx]

            ax2.text(
                x,
                0.95,
                f"Cluster {idx + 1}",
                ha="center",
                va="top",
                fontsize=12,
                fontweight="bold",
                color=color,
            )
            for word_index, word in enumerate(words):
                ax2.text(
                    x,
                    0.82 - word_index * 0.08,
                    sanitize_label_text(word),
                    ha="center",
                    va="top",
                    fontsize=10,
                    color=color,
                )

        self._apply_figure_layout(fig)
        plt.show()

    def _apply_figure_layout(self, fig: plt.Figure):
        """Apply a compact layout to a Matplotlib Figure.

        Args:
            fig (plt.Figure): The Matplotlib figure to adjust.
        """
        fig.subplots_adjust(left=0.0, right=0.96, top=0.94, bottom=0.12)
        self._disable_canvas_bbox_inches(fig)

    def _disable_canvas_bbox_inches(self, fig: plt.Figure):
        """Disable bbox_inches overrides on the figure's canvas print method.

        Args:
            fig (plt.Figure): The Matplotlib figure whose canvas will be modified.
        """
        canvas = getattr(fig, "canvas", None)
        if canvas is None or not hasattr(canvas, "print_figure"):
            return

        original_print_figure = canvas.print_figure

        def _print_figure_no_bbox_inches(*args, **kwargs):
            """Override print_figure to ignore bbox_inches argument."""
            kwargs.pop("bbox_inches", None)
            return original_print_figure(*args, **kwargs)

        canvas.print_figure = _print_figure_no_bbox_inches

    def _cluster_top_words(
        self, clusters: np.ndarray, top_n: int = 10
    ) -> dict[int, list[str]]:
        """Identify the top N words for each cluster based on z-scores.

        Args:
            clusters (np.ndarray): Array of cluster assignments for each document.
            top_n (int): Number of top words to return for each cluster.

        Returns:
            dict[int, list[str]]: A dictionary mapping cluster IDs to their top N words.
        """
        z_scores = (self.frequencies - self.frequencies.mean()) / self.frequencies.std()
        z_scores = z_scores.fillna(0)

        cluster_top_words: dict[int, list[str]] = {}
        for cluster_id in np.unique(clusters):
            members = self.frequencies.index[clusters == cluster_id]
            if len(members) == 0:
                cluster_top_words[cluster_id] = []
                continue

            cluster_mean = z_scores.loc[members].mean(axis=0)
            cluster_top_words[cluster_id] = list(
                cluster_mean.sort_values(ascending=False).head(top_n).index
            )

        return cluster_top_words

    def _dendrogram_color_func(
        self,
        z: np.ndarray,
        clusters: np.ndarray,
        palette: list[tuple[float, float, float]],
    ) -> Callable[[int], str]:
        """Generate a color function for dendrogram links based on cluster membership.

        Args:
            z (np.ndarray): Linkage matrix from hierarchical clustering.
            clusters (np.ndarray): Array of cluster assignments for each document.
            palette (list[tuple[float, float, float]]): List of RGB colors for
            each cluster.

        Returns:
            Callable[[int], str]: A function that maps a link ID to a color.
        """
        n_leaves = z.shape[0] + 1
        link_to_cluster: dict[int, list[int]] = {}

        def leaf_clusters(node: int) -> list[int]:
            """Recursively find the cluster IDs of all leaves under a given node.

            Args:
                node (int): The node ID in the linkage matrix.

            Returns:
                list[int]: List of cluster IDs for the leaves under the node.
            """
            if node < n_leaves:
                return [int(clusters[node])]
            return link_to_cluster[node]

        for i, row in enumerate(z):
            left = int(row[0])
            right = int(row[1])
            members = leaf_clusters(left) + leaf_clusters(right)
            link_to_cluster[n_leaves + i] = members

        def color_func(link_id: int) -> str:
            """Determine the color for a given link in the dendrogram.

            Args:
                link_id (int): The link ID in the dendrogram.

            Returns:
                str: Hex color code for the link.
            """
            members = link_to_cluster.get(link_id, [])
            if members and all(m == members[0] for m in members):
                color = palette[(members[0] - 1) % len(palette)]
                return matplotlib.colors.to_hex(color)
            return "gray"

        return color_func

    def _style_cluster_regions(
        self,
        ax,
        dendro: dict,
        clusters: np.ndarray,
        palette: list[tuple[float, float, float]],
        threshold: float,
        tip_x_pad: float,
        root_x_pad: float,
        y_pad: float,
        axis_y_pad: float,
    ):
        """Draw colored rectangles around clusters in the dendrogram.

        Args:
            ax: Matplotlib Axes object for the dendrogram.
            dendro (dict): Dendrogram data structure from scipy's dendrogram function.
            clusters (np.ndarray): Array of cluster assignments for each document.
            palette (list[tuple[float, float, float]]): List of RGB colors for each cluster.
            threshold (float): Distance threshold used to define clusters.
            tip_x_pad (float): Horizontal padding for dendrogram tips.
            root_x_pad (float): Horizontal padding for dendrogram root.
            y_pad (float): Vertical padding for cluster rectangles.
            axis_y_pad (float): Additional vertical padding for the axis.
        """
        leaves = dendro.get("leaves", [])
        if not leaves:
            return

        ordered_clusters = [int(clusters[i]) for i in leaves]
        x0, x1 = ax.get_xlim()
        x_leaf, x_root = min(x0, x1), max(x0, x1)

        x_box_left = x_leaf + max(0.0, tip_x_pad)
        x_box_right = x_root - max(0.0, root_x_pad)
        if x_box_right <= x_box_left:
            x_box_left = x_leaf + (x_root - x_leaf) * 0.1
            x_box_right = x_root - (x_root - x_leaf) * 0.1

        for tick_index, label in enumerate(ax.get_yticklabels()):
            if tick_index >= len(ordered_clusters):
                break
            cid = ordered_clusters[tick_index]
            color = palette[(cid - 1) % len(palette)]
            label.set_color(color)

        start = 0
        while start < len(ordered_clusters):
            cid = ordered_clusters[start]
            end = start
            while end + 1 < len(ordered_clusters) and ordered_clusters[end + 1] == cid:
                end += 1

            y_bottom = 10 * start + y_pad
            y_top = 10 * (end + 1) - y_pad
            color = palette[(cid - 1) % len(palette)]
            rect = Rectangle(
                (x_box_left, y_bottom),
                x_box_right - x_box_left,
                y_top - y_bottom,
                fill=False,
                edgecolor=color,
                linestyle=(0, (6, 3)),
                linewidth=1.8,
                alpha=0.8,
            )
            ax.add_patch(rect)

            start = end + 1

        y0, y1 = ax.get_ylim()
        if y0 < y1:
            ax.set_ylim(y0 - axis_y_pad, y1 + axis_y_pad)
        else:
            ax.set_ylim(y0 + axis_y_pad, y1 - axis_y_pad)


class ScoreSummary(BaseModel):
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
        z_scores = self._z_scores()
        text_profile = z_scores.loc[self.target_text]
        top_features = (
            text_profile.abs()
            .sort_values(ascending=False)
            .head(self.top)
            .index.tolist()
        )
        values = text_profile.loc[top_features].astype(float).to_numpy()
        colors = np.where(values > 0, "#ffb3ba", "#bae1ff")

        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        bars = ax.barh(top_features, values, color=colors, edgecolor="none", height=0.8)
        ax.invert_yaxis()
        ax.set_yticks([])
        ax.tick_params(left=False, labelleft=False)

        max_abs_value = max(abs(values).max(), 0.1)
        ax.set_xlim(-max_abs_value * 1.25, max_abs_value * 1.25)

        label_padding = max_abs_value * 0.03
        ax.axvline(0, color="black", linestyle="-", linewidth=1)
        for sd in [-2, -1, 1, 2]:
            ax.axvline(sd, color="gray", linestyle="--", alpha=0.6, linewidth=0.8)

        for bar, feature in zip(bars, top_features):
            value = bar.get_width()
            y = bar.get_y() + bar.get_height() / 2
            ax.text(
                value / 2,
                y,
                f"{value:.2f}",
                va="center",
                ha="center",
                fontsize=9,
                color="white" if abs(value) > 0.35 else "black",
                clip_on=False,
            )
            label_x = value + (label_padding if value >= 0 else -label_padding)
            ax.text(
                label_x,
                y,
                sanitize_label_text(feature),
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=10,
                color="black",
                clip_on=False,
            )

        ax.set_title(f"Most Distinctive Features in: {self.target_text}")
        return fig


class ComparisonPlot(BaseModel):
    """Encapsulate compare_scores plotting logic."""

    frequencies: pd.DataFrame = Field(
        default_factory=pd.DataFrame, description="Term frequency table."
    )
    source_text: str = Field(default="", description="Label for the source text.")
    target_text: str = Field(default="", description="Label for the target text.")
    top_diff: int = Field(
        default=10, description="Number of top differences to highlight."
    )
    rank_limit: int = Field(
        default=100, description="Limit for the number of features to rank."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

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
            .index.tolist()[: self.rank_limit]
        )

    def plot_profile(self) -> plt.Figure:
        """Plot the stylometric profiles of the source and target texts.

        Returns:
            plt.Figure: Matplotlib figure containing the profile plot.
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
            color="#ff9999",
            linewidth=1.8,
            label=self.source_text,
            alpha=0.85,
        )
        ax.plot(
            x,
            tgt_values,
            color="#99c2ff",
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
        ax.set_title(f"Stylometric Profile: {self.source_text} vs {self.target_text}")
        ax.set_xlabel("Feature frequency rank")
        ax.set_ylabel("Standard deviation from the corpus mean")
        ax.tick_params(axis="y", labelleft=True)
        ax.yaxis.set_ticks_position("left")
        return fig

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
        fill_colors = np.where(df_comp.index.isin(diff_mask), "red", "gray")

        ax.bar(
            df_comp["Rank"],
            df_comp["Difference"],
            color=fill_colors,
            edgecolor="grey",
            width=0.8,
        )
        ax.axhline(0, color="black", linestyle="-", alpha=0.7)

        ax.set_title(
            f"Z-Score Differences ({self.target_text} minus {self.source_text})"
        )
        ax.set_xlabel("Feature frequency rank")
        ax.set_ylabel("Difference between z-scores")

        ax.set_xlim(0.0, len(feature_order) + 0.5)
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


class DistancePlot(BaseModel):
    """Encapsulate view_distances plotting logic."""

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

    def __init__(self, **data):
        """Initialize the DistancePlot object."""
        super().__init__(**data)
        self.labels = list(self.labels)
        self.frequencies = (
            self.frequencies if self.frequencies is not None else pd.DataFrame()
        )

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

    def plot_density(
        self,
        group: bool = True,
        author: str | None = None,
        pattern: str = r"^.*?(?=[_\s-]|\d)",
    ) -> plt.Figure:
        """Plot the density of distances, optionally grouped by author.

        Args:
            group (bool): Whether to group distances by same author.
            author (str | None): Specific author to highlight in the plot.
            pattern (str): Regular expression pattern to extract author classes from labels.

        Returns:
            plt.Figure: Matplotlib figure containing the density plot.
        """
        df = self._build_density_frame(pattern)
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.grid(False)

        if group:
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
            palette = {"True": "pink", "False": "lightblue"}
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
        else:
            sns.kdeplot(
                data=df,
                x="d",
                fill=True,
                color="#cccccc",
                alpha=0.5,
                warn_singular=False,
                ax=ax,
                linewidth=0,
            )
            ax.axvline(df["d"].median(), linestyle="--", color="white", linewidth=1)

        ax.set_xlim(left=0)
        ax.margins(x=0.05, y=0.05)
        ymin, ymax = ax.get_ylim()
        if ymin >= 0:
            ax.set_ylim(bottom=-0.05 * max(ymax, 1.0))
        ax.set_xlabel("Distance")
        ax.set_ylabel("Density")
        sns.despine(ax=ax)

        if author is not None:
            author_distances = df[df["class"] == author]
            ax.scatter(
                author_distances["d"],
                np.zeros(len(author_distances)),
                s=40,
                color="black",
                zorder=10,
            )
            ax.set_title(f"Points: distances between works of {author}")

        return fig

    def plot_projection(
        self,
        method: str,
        random_state: int = 42,
        metric: str | None = None,
        author: str | None = None,
        pattern: str = r"^.*?(?=[_\s-]|\d)",
    ) -> plt.Figure:
        """Plot a 2D projection of the distance matrix using MDS or PCA.

        Args:
            method (str): Projection method, either 'MDS' or 'PCA'.
            random_state (int): Random seed for reproducibility.
            metric (str | None): Distance metric for MDS; ignored for PCA.
            author (str | None): Specific author to highlight in the plot.
            pattern (str): Regular expression pattern to extract author classes from labels.

        Returns:
            plt.Figure: Matplotlib figure containing the projection plot.
        """
        if method.upper() == "MDS":
            coords = MDS(
                n_components=2,
                dissimilarity="precomputed",
                random_state=random_state,
            ).fit_transform(self.distance_table.to_numpy())
            title_suffix = f"MDS (Distance Matrix via '{metric or 'precomputed'}')"
        else:
            z_scores = (
                self.frequencies - self.frequencies.mean()
            ) / self.frequencies.std()
            coords = PCA(n_components=2, random_state=random_state).fit_transform(
                z_scores.fillna(0).to_numpy()
            )
            title_suffix = "PCA (Z-scored Profiles)"

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.grid(False)
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            color="#e7a",
            edgecolors="black",
            s=100,
            alpha=0.8,
        )
        x_offset = np.ptp(coords[:, 0]) * (0.03 if method.upper() == "MDS" else 0.015)
        y_offset = np.ptp(coords[:, 1]) * (0.03 if method.upper() == "MDS" else 0.015)
        if x_offset == 0:
            x_offset = 0.5
        if y_offset == 0:
            y_offset = 0.5

        for i, label in enumerate(self.labels):
            ax.text(
                coords[i, 0] + x_offset,
                coords[i, 1] + y_offset,
                sanitize_label_text(label),
                fontsize=9,
                alpha=0.8,
            )

        ax.set_title(f"Stylometric Distribution via {title_suffix}")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")

        if author is not None:
            labels = list(self.distance_table.index)
            classes = []
            for label in labels:
                match = re.search(pattern, label)
                classes.append(match.group(0) if match else "")
            author_distances = self._build_density_frame(pattern)
            author_distances = author_distances[author_distances["class"] == author]
            ax.scatter(
                author_distances["d"],
                np.zeros(len(author_distances)),
                s=40,
                color="black",
                zorder=10,
            )
            ax.set_title(f"Points: distances between works of {author}")

        return fig

    def render(
        self,
        method: str = "MDS",
        metric: str | None = None,
        random_state: int = 42,
        group: bool = True,
        author: str | None = None,
        pattern: str = r"^.*?(?=_)",
    ) -> plt.Figure:
        """Render the appropriate plot based on the specified method.

        Args:
            method (str): Plotting method, either 'MDS', 'PCA', or 'DENSITY'.
            metric (str | None): Distance metric for MDS; ignored for PCA and density plots.
            random_state (int): Random seed for reproducibility.
            group (bool): Whether to group distances by same author for density plots.
            author (str | None): Specific author to highlight in the plot.
            pattern (str): Regular expression pattern to extract author classes from labels.

        Returns:
            plt.Figure: Matplotlib figure containing the rendered plot.
        """
        if method.upper() == "DENSITY":
            return self.plot_density(group=group, author=author, pattern=pattern)
        if method.upper() in {"MDS", "PCA"}:
            return self.plot_projection(
                method=method, metric=metric, random_state=random_state
            )
        raise ValueError("Method must be either 'MDS', 'PCA', or 'DENSITY'.")


class SeeTrees(BaseModel):
    """SeeTrees class for stylometric analysis and visualization."""

    distance_table: pd.DataFrame = Field(
        default_factory=pd.DataFrame, description="Optional distance matrix."
    )
    dtm: DTM | None = Field(
        default=None, description="Optional Lexos DTM to initialize frequencies from."
    )
    features: list[str] = Field(
        default_factory=list, description="Optional list of feature names."
    )
    frequencies: pd.DataFrame = Field(
        default_factory=pd.DataFrame, description="Optional frequency table."
    )
    labels: list[str] = Field(
        default_factory=list,
        description="Document labels derived from the distance matrix or frequency table.",
    )
    stylo_res: dict | None = Field(
        default=None,
        description="Optional dictionary containing stylo output keys such as `frequencies`, `distance_table`, and `features`.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    def __init__(self, **data):
        """Initialize a SeeTrees instance."""
        super().__init__(**data)
        if self.dtm is not None:
            self._init_from_dtm()
        elif self.stylo_res is not None:
            self._init_from_stylo_res()
        else:
            self._init_from_raw()

        self._ensure_dense_frequencies()
        self._init_labels()

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

    def _init_from_dtm(self):
        """Initialize frequencies from a Lexos DTM."""
        self.frequencies = self.dtm.to_df(transpose=True)
        self.distance_table = (
            pd.DataFrame(self.distance_table)
            if self.distance_table is not None
            else pd.DataFrame()
        )
        self.features = list(self.features) if self.features is not None else []

    def _init_from_stylo_res(self):
        """Initialize frequencies, distance table, and features from a stylo result dictionary."""
        self.frequencies = pd.DataFrame(self.stylo_res.get("frequencies", {}))
        self.distance_table = pd.DataFrame(self.stylo_res.get("distance_table", {}))
        self.features = list(self.stylo_res.get("features", []))

    def _init_from_raw(self):
        self.frequencies = (
            pd.DataFrame(self.frequencies)
            if self.frequencies is not None
            else pd.DataFrame()
        )
        self.distance_table = (
            pd.DataFrame(self.distance_table)
            if self.distance_table is not None
            else pd.DataFrame()
        )
        self.features = list(self.features) if self.features is not None else []

    def _ensure_dense_frequencies(self):
        """Ensure frequencies are stored as a dense float DataFrame."""
        if hasattr(self.frequencies, "sparse"):
            self.frequencies = self.frequencies.sparse.to_dense()
        self.frequencies = self.frequencies.astype(float)

    def _init_labels(self):
        if not self.distance_table.empty:
            self.labels = list(self.distance_table.index)
        else:
            self.labels = (
                list(self.frequencies.index) if not self.frequencies.empty else []
            )

    def compare_scores(
        self,
        source_text: str,
        target_text: str,
        top_diff: int = 10,
        view_type: str = "profile",
        rank_limit: int = 100,
    ):
        """Compare two documents using profile or difference plotting.

        Args:
            source_text (str): The reference text label.
            target_text (str): The text label to compare.
            top_diff (int): Number of top differing features to label.
            view_type (str): One of 'profile' or 'difference'.
            rank_limit (int): Maximum feature frequency rank to display.

        Raises:
            ValueError: If the frequency table is empty or either label is missing.
        """
        if self.frequencies.empty:
            raise ValueError("Frequency data is required for compare_scores.")
        if source_text not in self.frequencies.index:
            raise ValueError(f"Source text '{source_text}' not found in corpus.")
        if target_text not in self.frequencies.index:
            raise ValueError(f"Target text '{target_text}' not found in corpus.")

        plotter = ComparisonPlot(
            frequencies=self.frequencies,
            source_text=source_text,
            target_text=target_text,
            top_diff=top_diff,
            rank_limit=rank_limit,
        )

        if view_type == "profile":
            fig = plotter.plot_profile()
        elif view_type == "difference":
            fig = plotter.plot_difference()
        else:
            raise ValueError("view_type must be either 'profile' or 'difference'.")

        self._apply_figure_layout(fig)
        plt.show()

    def compute_distances(self, metric: str = "delta") -> pd.DataFrame:
        """Compute a stylometric distance matrix from frequency data.

        Supports multiple stylometric metrics including Burrows' Delta,
        Eder's Delta, and cosine variants.

        Args:
            metric (str): Distance metric to compute. Valid values are:
                `'delta'`, `'eder_delta'`, `'cosine_delta'`, `'manhattan'`,
                and `'cosine'`.

        Returns:
            pd.DataFrame: Pairwise distance matrix indexed by the original labels.

        Raises:
            ValueError: If the frequency table is empty or the metric is unknown.
        """
        if self.frequencies.empty:
            raise ValueError("Frequency table is required to compute distance metrics.")

        # Features must be ordered from most frequent to least frequent for Eder's Delta
        # Calculate mean frequency across corpus to determine exact rank
        mean_freqs = self.frequencies.mean().sort_values(ascending=False)
        ordered_freqs = self.frequencies[mean_freqs.index]

        # Calculate standard Z-scores
        z_scores = (ordered_freqs - ordered_freqs.mean()) / ordered_freqs.std()
        z_scores = z_scores.fillna(0)
        n_features = z_scores.shape[1]

        if metric.lower() == "delta":
            distances = pdist(z_scores.to_numpy(), metric="cityblock") / n_features

        elif metric.lower() == "eder_delta":
            # Assign 1-based ranks for the features
            ranks = np.arange(1, n_features + 1)
            # Apply Eder's descending linear weight formula
            eder_weights = -(ranks / n_features) + 1 + (1 / n_features)

            # Multiply scaled Z-scores by the weights
            weighted_z = z_scores.to_numpy() * eder_weights
            distances = pdist(weighted_z, metric="cityblock")

        elif metric.lower() == "cosine_delta":
            distances = pdist(z_scores.to_numpy(), metric="cosine")
        elif metric.lower() == "manhattan":
            distances = pdist(ordered_freqs.to_numpy(), metric="cityblock")
        elif metric.lower() == "cosine":
            distances = pdist(ordered_freqs.to_numpy(), metric="cosine")
        else:
            raise ValueError(
                f"Unknown metric '{metric}'. Choose from: 'delta', 'eder_delta', 'cosine_delta', 'manhattan', 'cosine'."
            )

        # Update the module's distance table matrix
        self.distance_table = pd.DataFrame(
            squareform(distances),
            index=ordered_freqs.index,
            columns=ordered_freqs.index,
        )
        self.labels = list(self.distance_table.index)
        return self.distance_table

    def view_distances(
        self,
        method: str = "MDS",
        metric: str | None = None,
        random_state: int = 42,
        group: bool = True,
        author: str | None = None,
        pattern: str = r"^.*?(?=[_\s-]|\d)",
    ):
        """Visualize pairwise distance density or projection depending on arguments.

        Args:
            method (str): Projection method for legacy support. If set to 'MDS'
                or 'PCA', performs the older projection plot. If set to
                'density', performs the R-style distance density plot.
            metric (str | None): Optional metric name to compute a distance
                matrix before projecting with MDS. Ignored for density mode.
            random_state (int): Random seed for reproducible MDS/PCA results.
            group (bool): For density mode, plot intra- and inter-class distances separately.
            author (str | None): For density mode, highlight distances for this author/class.
            pattern (str): Regex to extract the class label from document labels.

        Raises:
            ValueError: If required data is missing or invalid.
        """
        if method.upper() == "DENSITY" and self.distance_table.empty:
            raise ValueError(
                "Distance table is required for density mode. Run compute_distances() first."
            )

        plotter = DistancePlot(
            distance_table=self.distance_table,
            labels=self.labels,
            frequencies=self.frequencies,
        )

        if author is not None:
            classes = [
                plotter._extract_label_class(label, pattern) for label in self.labels
            ]
            if author not in classes:
                raise ValueError(
                    f"Author '{author}' is not present in the distance table."
                )

        if method.upper() == "MDS" and metric is not None:
            self.compute_distances(metric=metric)

        plotter = DistancePlot(
            distance_table=self.distance_table,
            labels=self.labels,
            frequencies=self.frequencies,
        )

        fig = plotter.render(
            method=method,
            metric=metric,
            random_state=random_state,
            group=group,
            author=author,
            pattern=pattern,
        )

        if method.upper() == "DENSITY":
            self._apply_figure_layout(
                fig,
                left=0.14,
                right=0.95,
                top=0.92,
                bottom=0.15,
            )
        else:
            self._apply_figure_layout(
                fig,
                left=0.12,
                right=0.96,
                top=0.94,
                bottom=0.12,
            )

        plt.show()

    def view_scores(
        self,
        target_text: str,
        top: int = 20,
        display: str = "table",
    ):
        """Render the most distinctive features for a target text.

        Args:
            target_text (str): The text to analyze for distinctive features.
            top (int): Number of top features to display based on absolute z-score.
            display (str): 'table' to show a summary table, 'bar' to show a bar chart.

        Raises:
            ValueError: If the frequency table is empty or if the target text is not found in the corpus.
        """
        if self.frequencies.empty:
            raise ValueError("Frequency data is required for view_scores.")
        if target_text not in self.frequencies.index:
            raise ValueError(f"Target text '{target_text}' not found in corpus.")

        if display not in {"table", "bar"}:
            raise ValueError("display must be either 'table' or 'bar'.")

        summary = ScoreSummary(
            frequencies=self.frequencies,
            target_text=target_text,
            top=top,
        )

        if display == "bar":
            fig = summary.render_bar_chart()
            self._apply_figure_layout(fig)
            plt.show()
            return fig

        return summary.to_dataframe()

    def view_tree(
        self,
        k: int = 2,
        top_n_words: int = 10,
        outline_y_pad: float = 0.3,
        outline_axis_y_pad: float = 0.1,
        outline_tip_pad_ratio: float = 0.002,
        outline_root_pad_ratio: float = 0.1,
    ):
        """Render a dendrogram and show cluster-specific top words.

        Args:
            k (int): Number of clusters to display in the dendrogram.
            top_n_words (int): Number of top words to display for each cluster.
            outline_y_pad (float): Vertical padding for cluster outlines.
            outline_axis_y_pad (float): Additional vertical padding for the axis.
            outline_tip_pad_ratio (float): Horizontal padding ratio for dendrogram tips.
            outline_root_pad_ratio (float): Horizontal padding ratio for dendrogram root.
        """
        tree = Tree(
            labels=self.labels,
            distance_table=self.distance_table,
            frequencies=self.frequencies,
        )
        tree.view(
            k=k,
            top_n_words=top_n_words,
            outline_y_pad=outline_y_pad,
            outline_axis_y_pad=outline_axis_y_pad,
            outline_tip_pad_ratio=outline_tip_pad_ratio,
            outline_root_pad_ratio=outline_root_pad_ratio,
        )
