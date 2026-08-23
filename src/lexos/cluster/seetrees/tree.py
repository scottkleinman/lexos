"""tree.py.

Last Updated: 22 August, 2026
Last Tested: 22 August, 2026
"""

from typing import Callable, Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from pydantic import BaseModel, ConfigDict, Field
from scipy.cluster.hierarchy import cut_tree, dendrogram, linkage
from scipy.spatial.distance import squareform

from .util import sanitize_label_text


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
    title: str | None = Field(
        default=None, description="Optional title for the dendrogram."
    )
    show_on_init: bool = Field(
        default=False,
        description="Whether to display the plot immediately when the object is created.",
    )
    fig: plt.Figure | None = Field(
        default=None,
        description="Matplotlib figure containing the dendrogram plot.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    def __init__(self, **data):
        """Initialize the Tree object."""
        super().__init__(**data)
        self.labels = list(self.labels)
        self.frequencies = (
            self.frequencies if self.frequencies is not None else pd.DataFrame()
        )
        if self.show_on_init:
            self.plot_tree()
            plt.show()

    def plot_tree(
        self,
        k: int = 2,
        method: Literal[
            "single",
            "complete",
            "average",
            "weighted",
            "centroid",
            "median",
            "ward",
        ] = "ward",
        top_n_words: int = 10,
        orientation: Literal["left", "right", "top", "bottom"] = "right",
        label_buffer: float = 0.0,
        outline_y_pad: float = 0.3,
        outline_axis_y_pad: float = 0.1,
        outline_tip_pad_ratio: float = 0.002,
        outline_root_pad_ratio: float = 0.1,
    ) -> plt.Figure:
        """Create the dendrogram figure without displaying it."""
        n_docs = self._ensure_plot_ready()
        k = self._sanitize_k(k, n_docs)
        self._validate_method(method)

        condensed = self._condense_distance_table()
        z = linkage(condensed, method=method)
        clusters = self._assign_clusters(z, k)

        threshold = self._compute_color_threshold(z, k, n_docs)
        palette = self._get_cluster_palette(k)
        color_func = self._dendrogram_color_func(z, clusters, palette)
        cluster_words = self._cluster_top_words(clusters, top_n_words)

        fig, ax1, ax2 = self._create_plot_axes()
        dendro = self._draw_dendrogram(ax1, z, orientation, threshold, color_func)
        self._finalize_dendrogram_axes(ax1, orientation)

        tip_x_pad, root_x_pad = self._compute_padding(
            ax1, orientation, outline_tip_pad_ratio, outline_root_pad_ratio
        )
        self._style_cluster_regions(
            ax1,
            dendro,
            clusters,
            palette,
            orientation,
            threshold,
            tip_x_pad,
            root_x_pad,
            outline_y_pad,
            outline_axis_y_pad,
        )

        self._label_axes(ax1, orientation, k)
        self._draw_cluster_word_summary(ax2, cluster_words, palette)

        self._apply_figure_layout(fig, orientation, label_buffer)
        self.fig = fig
        return fig

    def _ensure_plot_ready(self) -> int:
        if self.distance_table.empty:
            raise ValueError(
                "Distance table is required. Run compute_distances() first."
            )
        n_docs = len(self.labels)
        if n_docs == 0:
            raise ValueError("Labels are required to build the tree plot.")
        return n_docs

    def _sanitize_k(self, k: int, n_docs: int) -> int:
        return max(1, min(k, n_docs))

    def _validate_method(self, method: str) -> None:
        valid_methods = {
            "single",
            "complete",
            "average",
            "weighted",
            "centroid",
            "median",
            "ward",
        }
        if method not in valid_methods:
            raise ValueError(
                "method must be one of: " + ", ".join(sorted(valid_methods))
            )

    def _condense_distance_table(self) -> np.ndarray:
        return squareform(self.distance_table.to_numpy())

    def _assign_clusters(self, z: np.ndarray, k: int) -> np.ndarray:
        return cut_tree(z, n_clusters=[k]).reshape(-1) + 1

    def _compute_color_threshold(self, z: np.ndarray, k: int, n_docs: int) -> float:
        if k >= n_docs:
            return 0.0
        if k == 1:
            return float(z[-1, 2]) + 1e-12
        lower = float(z[-k, 2])
        upper = float(z[-k + 1, 2])
        return (lower + upper) / 2.0

    def _get_cluster_palette(self, k: int):
        return sns.color_palette("tab10", max(3, k))

    def _create_plot_axes(self):
        fig = plt.figure(figsize=(14, 10))
        fig.patch.set_facecolor("white")
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.24)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax1.set_facecolor("white")
        ax2.set_facecolor("white")
        return fig, ax1, ax2

    def _draw_dendrogram(
        self,
        ax: plt.Axes,
        z: np.ndarray,
        orientation: Literal["left", "right", "top", "bottom"],
        threshold: float,
        color_func: Callable[[int], str],
    ):
        return dendrogram(
            z,
            labels=self.labels,
            orientation=orientation,
            color_threshold=threshold,
            link_color_func=color_func,
            ax=ax,
        )

    def _finalize_dendrogram_axes(
        self, ax: plt.Axes, orientation: Literal["left", "right", "top", "bottom"]
    ):
        if orientation in ["left", "right"]:
            ax.margins(x=0)
        else:
            ax.margins(y=0)
        self._apply_orientation_axis_style(ax, orientation)
        ax.grid(False)

    def _compute_padding(
        self,
        ax: plt.Axes,
        orientation: Literal["left", "right", "top", "bottom"],
        outline_tip_pad_ratio: float,
        outline_root_pad_ratio: float,
    ):
        if orientation in ["left", "right"]:
            x0, x1 = ax.get_xlim()
            return (
                abs(x1 - x0) * max(0.0, outline_tip_pad_ratio),
                abs(x1 - x0) * max(0.0, outline_root_pad_ratio),
            )
        y0, y1 = ax.get_ylim()
        return (
            abs(y1 - y0) * max(0.0, outline_tip_pad_ratio),
            abs(y1 - y0) * max(0.0, outline_root_pad_ratio),
        )

    def _label_axes(
        self,
        ax: plt.Axes,
        orientation: Literal["left", "right", "top", "bottom"],
        k: int,
    ):
        if not self.title:
            self.title = self._default_title_for_k(k)
        ax.set_title(self.title)
        if orientation in ["left", "right"]:
            ax.set_xlabel("Distance")
            ax.tick_params(axis="y", labelsize=7)
        else:
            ax.set_ylabel("Distance")
            ax.tick_params(axis="x", labelsize=7)

    def _default_title_for_k(self, k: int) -> str:
        return f"Dendrogram Cut into k={self._sanitize_k(k, len(self.labels))} Groups"

    def _draw_cluster_word_summary(
        self, ax: plt.Axes, cluster_words: dict[int, list[str]], palette
    ):
        ax.axis("off")
        x_positions = np.linspace(0.15, 0.85, len(cluster_words))
        for idx, cluster_id in enumerate(sorted(cluster_words)):
            words = cluster_words[cluster_id]
            color = palette[idx % len(palette)]
            x = x_positions[idx]

            ax.text(
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
                ax.text(
                    x,
                    0.82 - word_index * 0.08,
                    sanitize_label_text(word),
                    ha="center",
                    va="top",
                    fontsize=10,
                    color=color,
                )

    def _apply_orientation_axis_style(
        self, ax: plt.Axes, orientation: Literal["left", "right", "top", "bottom"]
    ) -> None:
        """Apply orientation-specific tick styling so leaf labels stay visible."""
        if orientation == "right":
            # In scipy's 'right' orientation, leaves are on the left side.
            ax.yaxis.tick_left()
            ax.tick_params(axis="y", labelleft=True, labelright=False, pad=1)
            ax.spines["left"].set_visible(False)
        elif orientation == "left":
            # In scipy's 'left' orientation, leaves are on the right side.
            ax.yaxis.tick_right()
            ax.tick_params(axis="y", labelright=True, labelleft=False, pad=1)
            ax.spines["right"].set_visible(False)
        elif orientation == "top":
            # In scipy's 'top' orientation, leaves are shown at the bottom side.
            ax.xaxis.tick_bottom()
            ax.tick_params(axis="x", labelbottom=True, labeltop=False, pad=1)
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha("right")
            ax.spines["bottom"].set_visible(False)
        else:
            # In scipy's 'bottom' orientation, leaves are shown at the top side.
            ax.xaxis.tick_top()
            ax.tick_params(axis="x", labeltop=True, labelbottom=False, pad=1)
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha("left")
            ax.spines["top"].set_visible(False)

    def show(self) -> None:
        """Display the plot, creating it if necessary."""
        if self.fig is None:
            self.plot_tree()
        plt.show()

    def _apply_figure_layout(
        self,
        fig: plt.Figure,
        orientation: Literal["left", "right", "top", "bottom"] = "right",
        label_buffer: float = 0.0,
    ):
        """Apply a compact layout to a Matplotlib Figure.

        Args:
            fig (plt.Figure): The Matplotlib figure to adjust.
        """
        label_buffer = max(0.0, label_buffer)

        if orientation == "left":
            # In 'left' orientation, leaves/labels are on the right side.
            fig.subplots_adjust(
                left=0.04,
                right=max(0.55, 0.90 - label_buffer),
                top=0.94,
                bottom=0.12,
            )
        elif orientation == "right":
            # In 'right' orientation, leaves/labels are on the left side.
            fig.subplots_adjust(
                left=min(0.45, 0.10 + label_buffer),
                right=0.98,
                top=0.94,
                bottom=0.12,
            )
        elif orientation == "top":
            # In 'top' orientation, leaves/labels are on the bottom side.
            fig.subplots_adjust(
                left=0.06,
                right=0.98,
                top=0.90,
                bottom=min(0.45, 0.12 + label_buffer),
            )
        else:
            # In 'bottom' orientation, leaves/labels are on the top side.
            fig.subplots_adjust(
                left=0.06,
                right=0.98,
                top=max(0.55, 0.94 - label_buffer),
                bottom=0.18,
            )
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
        orientation: Literal["left", "right", "top", "bottom"],
        threshold: float,
        tip_x_pad: float,
        root_x_pad: float,
        y_pad: float,
        axis_y_pad: float,
    ):
        """Draw colored rectangles around clusters in the dendrogram."""
        leaves = dendro.get("leaves", [])
        if not leaves:
            return

        ordered_clusters = [int(clusters[i]) for i in leaves]
        box_layout, tick_labels = self._compute_region_layout(
            ax, orientation, tip_x_pad, root_x_pad
        )
        self._color_cluster_tick_labels(tick_labels, ordered_clusters, palette)
        self._draw_cluster_region_rectangles(
            ax,
            ordered_clusters,
            palette,
            orientation,
            box_layout,
            y_pad,
        )
        self._apply_cluster_axis_padding(ax, orientation, axis_y_pad)

    def _compute_region_layout(
        self,
        ax,
        orientation: Literal["left", "right", "top", "bottom"],
        tip_x_pad: float,
        root_x_pad: float,
    ):
        """Compute axis layout for region rectangles and tick labels.

        Args:
            ax: Matplotlib Axes object for the dendrogram.
            orientation (Literal["left", "right", "top", "bottom"]): Plot orientation.
            tip_x_pad (float): Padding near dendrogram tips.
            root_x_pad (float): Padding near the dendrogram root.

        Returns:
            tuple[dict, list]: A dictionary with rectangle layout coordinates and a list of tick labels.
        """
        if orientation in ["left", "right"]:
            x0, x1 = ax.get_xlim()
            x_leaf, x_root = min(x0, x1), max(x0, x1)
            x_box_left = x_leaf + max(0.0, tip_x_pad)
            x_box_right = x_root - max(0.0, root_x_pad)
            if x_box_right <= x_box_left:
                x_box_left = x_leaf + (x_root - x_leaf) * 0.1
                x_box_right = x_root - (x_root - x_leaf) * 0.1
            return {
                "x_box_left": x_box_left,
                "x_box_right": x_box_right,
            }, ax.get_yticklabels()

        y0, y1 = ax.get_ylim()
        y_leaf, y_root = min(y0, y1), max(y0, y1)
        y_box_bottom = y_leaf + max(0.0, tip_x_pad)
        y_box_top = y_root - max(0.0, root_x_pad)
        if y_box_top <= y_box_bottom:
            y_box_bottom = y_leaf + (y_root - y_leaf) * 0.1
            y_box_top = y_root - (y_root - y_leaf) * 0.1
        return {
            "y_box_bottom": y_box_bottom,
            "y_box_top": y_box_top,
        }, ax.get_xticklabels()

    def _color_cluster_tick_labels(
        self,
        tick_labels,
        ordered_clusters: list[int],
        palette: list[tuple[float, float, float]],
    ):
        """Color tick labels according to cluster membership.

        Args:
            tick_labels: Tick label artists from a Matplotlib Axes.
            ordered_clusters (list[int]): Cluster IDs in leaf order.
            palette (list[tuple[float, float, float]]): RGB colors for each cluster.
        """
        for tick_index, label in enumerate(tick_labels):
            if tick_index >= len(ordered_clusters):
                break
            cid = ordered_clusters[tick_index]
            color = palette[(cid - 1) % len(palette)]
            label.set_color(color)

    def _draw_cluster_region_rectangles(
        self,
        ax,
        ordered_clusters: list[int],
        palette: list[tuple[float, float, float]],
        orientation: Literal["left", "right", "top", "bottom"],
        box_layout: dict,
        y_pad: float,
    ):
        """Draw rectangular region outlines around cluster groups.

        Args:
            ax: Matplotlib Axes object for the dendrogram.
            ordered_clusters (list[int]): Cluster IDs in leaf order.
            palette (list[tuple[float, float, float]]): RGB colors for each cluster.
            orientation (Literal["left", "right", "top", "bottom"]): Plot orientation.
            box_layout (dict): Coordinates for rectangle placement.
            y_pad (float): Padding between rectangle edges and leaves.
        """
        start = 0
        while start < len(ordered_clusters):
            cid = ordered_clusters[start]
            end = start
            while end + 1 < len(ordered_clusters) and ordered_clusters[end + 1] == cid:
                end += 1

            color = palette[(cid - 1) % len(palette)]
            rect = self._build_cluster_rectangle(
                orientation,
                box_layout,
                start,
                end,
                y_pad,
                color,
            )
            ax.add_patch(rect)
            start = end + 1

    def _build_cluster_rectangle(
        self,
        orientation: Literal["left", "right", "top", "bottom"],
        box_layout: dict,
        start: int,
        end: int,
        y_pad: float,
        color,
    ):
        """Build a Matplotlib Rectangle for a cluster region.

        Args:
            orientation (Literal["left", "right", "top", "bottom"]): Plot orientation.
            box_layout (dict): Coordinates for rectangle placement.
            start (int): Index of the first leaf in the current cluster block.
            end (int): Index of the last leaf in the current cluster block.
            y_pad (float): Padding between rectangle edges and leaves.
            color: Edge color for the rectangle.

        Returns:
            Rectangle: The configured Matplotlib rectangle patch.
        """
        if orientation in ["left", "right"]:
            y_bottom = 10 * start + y_pad
            y_top = 10 * (end + 1) - y_pad
            return Rectangle(
                (box_layout["x_box_left"], y_bottom),
                box_layout["x_box_right"] - box_layout["x_box_left"],
                y_top - y_bottom,
                fill=False,
                edgecolor=color,
                linestyle=(0, (6, 3)),
                linewidth=1.8,
                alpha=0.8,
            )

        x_left = 10 * start + y_pad
        x_right = 10 * (end + 1) - y_pad
        return Rectangle(
            (x_left, box_layout["y_box_bottom"]),
            x_right - x_left,
            box_layout["y_box_top"] - box_layout["y_box_bottom"],
            fill=False,
            edgecolor=color,
            linestyle=(0, (6, 3)),
            linewidth=1.8,
            alpha=0.8,
        )

    def _apply_cluster_axis_padding(
        self,
        ax,
        orientation: Literal["left", "right", "top", "bottom"],
        axis_y_pad: float,
    ):
        """Apply additional axis padding so cluster rectangles do not clip.

        Args:
            ax: Matplotlib Axes object for the dendrogram.
            orientation (Literal["left", "right", "top", "bottom"]): Plot orientation.
            axis_y_pad (float): Amount of padding to add to the axis limits.
        """
        if orientation in ["left", "right"]:
            y0, y1 = ax.get_ylim()
            if y0 < y1:
                ax.set_ylim(y0 - axis_y_pad, y1 + axis_y_pad)
            else:
                ax.set_ylim(y0 + axis_y_pad, y1 - axis_y_pad)
            return

        x0, x1 = ax.get_xlim()
        if x0 < x1:
            ax.set_xlim(x0 - axis_y_pad, x1 + axis_y_pad)
        else:
            ax.set_xlim(x0 + axis_y_pad, x1 - axis_y_pad)
