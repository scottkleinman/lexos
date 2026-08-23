"""__init__.py.

This module is adapted from the R code from the 'see' package by Artjoms Šeļa (https://github.com/perechen/seetrees).

Last Updated: August 23, 2026
Last Tested: August 23, 2026
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from scipy.spatial.distance import pdist, squareform

from lexos.dtm import DTM

from .comparison import DifferencePlot, OverlayPlot
from .density_plot import DensityPlot
from .projection_plot import MDS, PCA
from .tree import Tree
from .zscores import DistinctiveFeaturePlot, FeatureSummary, ZscorePlot


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
    fig: plt.Figure | None = Field(
        default=None, description="Matplotlib Figure object for plotting."
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

    def get_difference_plot(
        self,
        source_text: str,
        target_text: str,
        top_diff: int = 10,
        max_rank: int = 100,
        title: str | None = None,
        base_color: str = "gray",
        highlight_color: str = "red",
    ) -> DifferencePlot:
        """Get a difference plot for two documents.

        Args:
            source_text (str): The reference text label.
            target_text (str): The text label to compare.
            top_diff (int): Number of top differing features to label.
            max_rank (int): Maximum feature frequency rank to display.

        Returns:
            DifferencePlot: An object for visualizing the difference in z-scores.

        Raises:
            ValueError: If the frequency table is empty or either label is missing.
        """
        if self.frequencies.empty:
            raise ValueError("Frequency data is required for get_difference_plot.")
        if source_text not in self.frequencies.index:
            raise ValueError(f"Source text '{source_text}' not found in corpus.")
        if target_text not in self.frequencies.index:
            raise ValueError(f"Target text '{target_text}' not found in corpus.")

        return DifferencePlot(
            frequencies=self.frequencies,
            source_text=source_text,
            target_text=target_text,
            top_diff=top_diff,
            max_rank=max_rank,
            title=title,
            base_color=base_color,
            highlight_color=highlight_color,
        )

    def get_overlay_plot(
        self,
        source_text: str,
        target_text: str,
        top_diff: int = 10,
        max_rank: int = 100,
        title: str | None = None,
        source_color: str = "#ff9999",
        target_color: str = "#99c2ff",
    ) -> OverlayPlot:
        """Compare two documents using overlay plotting.

        Args:
            source_text (str): The reference text label.
            target_text (str): The text label to compare.
            top_diff (int): Number of top differing features to label.
            max_rank (int): Maximum feature frequency rank to display.

        Returns:
            OverlayPlot: An OverlayPlot object for the specified documents.

        Raises:
            ValueError: If the frequency table is empty or either label is missing.
        """
        if self.frequencies.empty:
            raise ValueError("Frequency data is required for get_overlay_plot.")
        if source_text not in self.frequencies.index:
            raise ValueError(f"Source text '{source_text}' not found in corpus.")
        if target_text not in self.frequencies.index:
            raise ValueError(f"Target text '{target_text}' not found in corpus.")

        return OverlayPlot(
            frequencies=self.frequencies,
            source_text=source_text,
            target_text=target_text,
            top_diff=top_diff,
            max_rank=max_rank,
            title=title,
            source_color=source_color,
            target_color=target_color,
        )

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

    def get_density_plot(
        self,
        group: bool = True,
        author: str | None = None,
        pattern: str = r"^.*?(?=[_\s-]|\d)",
        title: str | None = None,
        palette: dict[str, str] | None = None,
        color: str = "#cccccc",
        left: float = 0.14,
        right: float = 0.95,
        top: float = 0.92,
        bottom: float = 0.15,
    ) -> DensityPlot:
        """Return a DensityPlot object for viewing distances.

        Args:
            group (bool): Whether to group distances by the same author/class.
            author (str | None): Specific author/class to highlight in the plot.
            pattern (str): Regex pattern to extract author classes from labels.
            title (str | None): Optional title for the density plot.
            palette (dict[str, str] | None): Optional color palette for grouped density curves.
            color (str): Fill color for ungrouped density plots.
            left (float): Left margin for the figure layout.
            right (float): Right margin for the figure layout.
            top (float): Top margin for the figure layout.
            bottom (float): Bottom margin for the figure layout.

        Returns:
            DensityPlot: Configured DensityPlot object.
        """
        if self.distance_table is None or self.distance_table.empty:
            raise ValueError(
                "Distance table is required to create a DensityPlot. Run compute_distances() first."
            )
        plotter = DensityPlot(
            distance_table=self.distance_table,
            labels=self.labels,
            frequencies=self.frequencies,
            author=author,
            group=group,
            pattern=pattern,
            title=title,
            palette=palette,
            color=color,
            left=left,
            right=right,
            top=top,
            bottom=bottom,
        )
        return plotter

    def get_mds_plot(
        self,
        group: bool = True,
        author: str | None = None,
        pattern: str = r"^.*?(?=[_\s-]|\d)",
        title: str | None = None,
        left: float = 0.12,
        right: float = 0.96,
        top: float = 0.94,
        bottom: float = 0.12,
    ) -> MDS:
        """Return an MDS object for viewing distances.

        Args:
            group (bool): Whether to group distances by the same author/class.
            author (str | None): Specific author/class to highlight in the plot.
            pattern (str): Regex pattern to extract author classes from labels.
            title (str | None): Optional title for the MDS plot.
            left (float): Left margin for the figure layout.
            right (float): Right margin for the figure layout.
            top (float): Top margin for the figure layout.
            bottom (float): Bottom margin for the figure layout.

        Returns:
            MDS: Configured MDS object.
        """
        if self.distance_table is None or self.distance_table.empty:
            raise ValueError(
                "Distance table is required to create an MDS plot. Run compute_distances() first."
            )
        plotter = MDS(
            distance_table=self.distance_table,
            labels=self.labels,
            frequencies=self.frequencies,
            author=author,
            group=group,
            pattern=pattern,
            title=title,
            left=left,
            right=right,
            top=top,
            bottom=bottom,
        )
        return plotter

    def get_pca_plot(
        self,
        author: str | None = None,
        pattern: str = r"^.*?(?=[_\s-]|\d)",
        title: str | None = None,
        left: float = 0.12,
        right: float = 0.96,
        top: float = 0.94,
        bottom: float = 0.12,
    ) -> PCA:
        """Return a PCA object for viewing distances."""
        plotter = PCA(
            distance_table=self.distance_table,
            labels=self.labels,
            frequencies=self.frequencies,
            author=author,
            pattern=pattern,
            title=title,
            left=left,
            right=right,
            top=top,
            bottom=bottom,
        )
        return plotter

    def get_feature_summary(
        self,
        target_text: str,
        top: int = 20,
    ):
        """Return the most distinctive features for a target text."""
        if self.frequencies.empty:
            raise ValueError("Frequency data is required for get_feature_summary.")
        if target_text not in self.frequencies.index:
            raise ValueError(f"Target text '{target_text}' not found in corpus.")

        summary = FeatureSummary(
            frequencies=self.frequencies,
            target_text=target_text,
            top=top,
        )

        return summary.to_dataframe()

    def get_tree(
        self,
        k: int = 2,
        method: str = "ward",
        title: str | None = None,
        top_n_words: int = 10,
        orientation: str = "right",
        label_buffer: float = 0.0,
        outline_y_pad: float = 0.3,
        outline_axis_y_pad: float = 0.1,
        outline_tip_pad_ratio: float = 0.002,
        outline_root_pad_ratio: float = 0.1,
    ) -> Tree:
        """Return a Tree object for further customization or saving.

        Args:
            k (int): Number of clusters to display in the dendrogram.
            method (str): Linkage method for hierarchical clustering.
            title (str | None): Optional title for the dendrogram.
            top_n_words (int): Number of top words to display for each cluster.
            orientation (str): Dendrogram orientation. One of 'left', 'right', 'top', or 'bottom'.
            label_buffer (float): Extra subplot margin reserved for leaf labels on the active label side.
            outline_y_pad (float): Vertical padding for cluster outlines.
            outline_axis_y_pad (float): Additional vertical padding for the axis.
            outline_tip_pad_ratio (float): Horizontal padding ratio for dendrogram tips.
            outline_root_pad_ratio (float): Horizontal padding ratio for dendrogram root.
        """
        tree = Tree(
            labels=self.labels,
            distance_table=self.distance_table,
            frequencies=self.frequencies,
            title=title,
        )
        tree.plot_tree(
            k=k,
            method=method,
            top_n_words=top_n_words,
            orientation=orientation,
            label_buffer=label_buffer,
            outline_y_pad=outline_y_pad,
            outline_axis_y_pad=outline_axis_y_pad,
            outline_tip_pad_ratio=outline_tip_pad_ratio,
            outline_root_pad_ratio=outline_root_pad_ratio,
        )
        return tree

    def get_feature_score_plot(
        self,
        target_text: str,
        top: int = 20,
        title: str | None = None,
        positive_color: str = "#f6c1cc",
        negative_color: str = "#b9dff1",
        guide_color: str = "#c9ced6",
        zero_line_color: str = "red",
        height: int = 600,
        width: int = 800,
    ) -> DistinctiveFeaturePlot:
        """Return a DistinctiveFeaturePlot object for top feature score visualization.

        Args:
            target_text (str): The text to analyze for distinctive features.
            top (int): Number of top features to display.
            title (str | None): Optional chart title.
            positive_color (str): Bar color for positive z-scores.
            negative_color (str): Bar color for negative z-scores.
            guide_color (str): Dotted guide-line color for non-zero SD lines.
            zero_line_color (str): Dotted guide-line color for the zero SD line.
            height (int): Figure height in pixels.
            width (int): Figure width in pixels.

        Returns:
            DistinctiveFeaturePlot: Configured Plotly plot object.

        Raises:
            ValueError: If the frequency table is empty or if the target text is not found.
        """
        if self.frequencies.empty:
            raise ValueError("Frequency data is required for get_feature_score_plot.")
        if target_text not in self.frequencies.index:
            raise ValueError(f"Target text '{target_text}' not found in corpus.")

        return DistinctiveFeaturePlot(
            frequencies=self.frequencies,
            target_text=target_text,
            top=top,
            title=title,
            positive_color=positive_color,
            negative_color=negative_color,
            guide_color=guide_color,
            zero_line_color=zero_line_color,
            width=width,
            height=height,
        )
