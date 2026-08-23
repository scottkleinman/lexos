"""projection_plot.py.

Last Updated: August 23, 2026
Last Tested: August 23, 2026
"""

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.decomposition import PCA as SKLearnPCA
from sklearn.manifold import MDS as SKLearnMDS

from .util import sanitize_label_text


class ProjectionPlot(BaseModel):
    """Base class for projection plots with lazy figure creation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    author: str | None = Field(
        default=None, description="Specific author to highlight in the plot."
    )
    pattern: str = Field(
        default=r"^.*?(?=[_\s-]|\d)",
        description="Regular expression pattern to extract author classes from labels.",
    )
    distance_table: pd.DataFrame = Field(
        ...,
        description="Distance table containing pairwise distances between items.",
    )
    frequencies: pd.DataFrame = Field(
        default_factory=pd.DataFrame,
        description="Optional frequency table for the items.",
    )
    labels: list[str] = Field(
        ...,
        description="List of labels corresponding to the items in the distance table.",
    )
    title: str | None = Field(
        default=None,
        description="Optional title for the projection plot.",
    )
    left: float = Field(default=0.12, description="Left margin for the figure layout.")
    right: float = Field(
        default=0.96, description="Right margin for the figure layout."
    )
    top: float = Field(default=0.94, description="Top margin for the figure layout.")
    bottom: float = Field(
        default=0.12, description="Bottom margin for the figure layout."
    )
    show_on_init: bool = Field(
        default=False,
        description="Whether to display the plot immediately when the object is created.",
    )
    fig: plt.Figure | None = Field(
        default=None, description="Matplotlib figure containing the projection plot."
    )

    def __init__(self, **data):
        """Initialize the ProjectionPlot object."""
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
            self.plot_projection()
            plt.show()

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

    def _compute_coords(self) -> np.ndarray:
        """Compute the coordinates for the projection plot.

        Returns:
            np.ndarray: The computed coordinates for the projection plot.
        """
        raise NotImplementedError

    def _create_figure(self) -> tuple[plt.Figure, plt.Axes]:
        """Create a new matplotlib figure and axes for the projection plot.

        Returns:
            tuple: A tuple containing the matplotlib figure and axes.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.grid(False)
        return fig, ax

    def _disable_canvas_bbox_inches(self, fig: plt.Figure) -> None:
        """Disable bbox_inches overrides on the figure's canvas print method.

        Args:
            fig (plt.Figure): The matplotlib figure whose canvas print method will be modified.
        """
        canvas = getattr(fig, "canvas", None)
        if canvas is None or not hasattr(canvas, "print_figure"):
            return

        original_print_figure = canvas.print_figure

        def _print_figure_no_bbox_inches(*args, **kwargs):
            kwargs.pop("bbox_inches", None)
            return original_print_figure(*args, **kwargs)

        canvas.print_figure = _print_figure_no_bbox_inches

    def _extract_label_class(self, label: str, pattern: str) -> str:
        """Extract a class label from a document label using a fallback strategy.

        Args:
            label (str): The document label from which to extract the class.
            pattern (str): Regular expression pattern to extract author classes from labels.

        Returns:
            str: The extracted class label.
        """
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

    def _finalize_figure(self, fig: plt.Figure) -> None:
        """Finalize the figure layout and disable bbox_inches overrides.

        Args:
            fig (plt.Figure): The matplotlib figure to finalize.

        Returns:
            None
        """
        fig.subplots_adjust(
            left=self.left, right=self.right, top=self.top, bottom=self.bottom
        )
        self._disable_canvas_bbox_inches(fig)

    def _highlight_author(self, ax: plt.Axes, coords: np.ndarray) -> None:
        """Highlight points corresponding to a specific author on the plot.

        Args:
            ax (plt.Axes): The matplotlib axes on which to highlight the author.
            coords (np.ndarray): The coordinates of the plotted points.
        """
        if self.author is None:
            return

        classes = [
            self._extract_label_class(label, self.pattern) for label in self.labels
        ]
        author_mask = np.array([class_name == self.author for class_name in classes])
        if not author_mask.any():
            return

        ax.scatter(
            coords[author_mask, 0],
            coords[author_mask, 1],
            facecolors="none",
            edgecolors="black",
            linewidths=2,
            s=200,
            zorder=12,
        )
        if self.title is None:
            ax.set_title(f"Highlights for {self.author}")

    def _offset_ratio(self) -> float:
        """Return the offset ratio for label placement in the plot.

        Returns:
            float: The offset ratio for label placement.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    def _plot_base(self, ax: plt.Axes, coords: np.ndarray, title_suffix: str) -> None:
        """Plot the base scatter plot and labels on the provided axes.

        Args:
            ax (plt.Axes): The matplotlib axes on which to plot.
            coords (np.ndarray): The coordinates of the points to plot.
            title_suffix (str): Suffix for the plot title indicating the projection method.

        Returns:
            None
        """
        classes = [
            self._extract_label_class(label, self.pattern) for label in self.labels
        ]
        unique_classes = sorted(set(classes))
        palette = plt.cm.get_cmap("tab10", len(unique_classes))
        class_colors = {cls: palette(i) for i, cls in enumerate(unique_classes)}

        for i, label in enumerate(self.labels):
            cls = classes[i]
            ax.scatter(
                coords[i, 0],
                coords[i, 1],
                color=class_colors[cls],
                edgecolors="black",
                s=100,
                alpha=0.8,
            )
            x_offset = np.ptp(coords[:, 0]) * self._offset_ratio()
            y_offset = np.ptp(coords[:, 1]) * self._offset_ratio()
            if x_offset == 0:
                x_offset = 0.5
            if y_offset == 0:
                y_offset = 0.5
            ax.text(
                coords[i, 0] + x_offset,
                coords[i, 1] + y_offset,
                sanitize_label_text(label),
                fontsize=9,
                color=class_colors[cls],
                alpha=0.8,
            )

        ax.set_title(self.title or f"Stylometric Distribution via {title_suffix}")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")

    def _title_suffix(self) -> str:
        """Return a suffix for the plot title indicating the projection method.

        Returns:
            str: The title suffix for the projection plot.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    def plot_projection(self) -> None:
        """Compute coordinates, create the figure, plot the base scatter, and finalize the figure."""
        coords = self._compute_coords()
        fig, ax = self._create_figure()
        self._plot_base(ax, coords, self._title_suffix())
        self._finalize_figure(fig)
        self.fig = fig

    def show(self) -> None:
        """Display the projection plot, creating it if necessary."""
        if self.fig is None:
            self.plot_projection()
        plt.show()


class MDS(ProjectionPlot):
    """Encapsulate view_distances plotting logic as an MDS projection plot."""

    metric: str | None = Field(default=None, description="Distance metric for MDS.")
    random_state: int = Field(
        default=42, description="Random seed for reproducibility."
    )

    def _compute_coords(self) -> np.ndarray:
        """Compute the coordinates for the MDS projection plot.

        Returns:
            np.ndarray: The computed coordinates for the MDS projection plot.
        """
        return SKLearnMDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=self.random_state,
        ).fit_transform(self.distance_table.to_numpy())

    def _title_suffix(self) -> str:
        """Return a suffix for the plot title indicating the MDS projection method.

        Returns:
            str: The title suffix for the MDS projection plot.
        """
        return f"MDS (Distance Matrix via '{self.metric or 'precomputed'}')"

    def _offset_ratio(self) -> float:
        """Return the offset ratio for label placement in the MDS plot.

        Returns:
            float: The offset ratio for label placement in the MDS plot.
        """
        return 0.03


class PCA(ProjectionPlot):
    """Encapsulate view_distances plotting logic as a PCA projection plot."""

    distance_table: pd.DataFrame = Field(
        default_factory=pd.DataFrame,
        description="Optional distance table for compatibility; PCA uses frequencies.",
    )
    random_state: int = Field(
        default=42, description="Random seed for reproducibility."
    )

    def _compute_coords(self) -> np.ndarray:
        """Compute the coordinates for the PCA projection plot.

        Returns:
            np.ndarray: The computed coordinates for the PCA projection plot.
        """
        z_scores = (self.frequencies - self.frequencies.mean()) / self.frequencies.std()
        return SKLearnPCA(n_components=2, random_state=self.random_state).fit_transform(
            z_scores.fillna(0).to_numpy()
        )

    def _offset_ratio(self) -> float:
        """Return the offset ratio for label placement in the PCA plot.

        Returns:
            float: The offset ratio for label placement in the PCA plot.
        """
        return 0.015

    def _plot_base(self, ax: plt.Axes, coords: np.ndarray, title_suffix: str) -> None:
        """Plot the PCA scatter plot and labels with PCA-specific axis styling."""
        classes = [
            self._extract_label_class(label, self.pattern) for label in self.labels
        ]
        unique_classes = sorted(set(classes))
        palette = plt.cm.get_cmap("tab10", len(unique_classes))
        class_colors = {cls: palette(i) for i, cls in enumerate(unique_classes)}

        ax.axvline(0, color="lightgrey", linestyle="--", linewidth=1, zorder=0)
        ax.axhline(0, color="lightgrey", linestyle="--", linewidth=1, zorder=0)

        for i, label in enumerate(self.labels):
            cls = classes[i]
            ax.scatter(
                coords[i, 0],
                coords[i, 1],
                color=class_colors[cls],
                edgecolors="black",
                s=100,
                alpha=0.8,
            )
            x_offset = np.ptp(coords[:, 0]) * self._offset_ratio()
            y_offset = np.ptp(coords[:, 1]) * self._offset_ratio()
            if x_offset == 0:
                x_offset = 0.5
            if y_offset == 0:
                y_offset = 0.5
            ax.text(
                coords[i, 0] + x_offset,
                coords[i, 1] + y_offset,
                sanitize_label_text(label),
                fontsize=9,
                color=class_colors[cls],
                alpha=0.8,
            )

        ax.set_title(self.title or f"Stylometric Distribution via {title_suffix}")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")

    def _title_suffix(self) -> str:
        """Return a suffix for the plot title indicating the PCA projection method.

        Returns:
            str: The title suffix for the PCA projection plot.
        """
        return "PCA (Z-scored Profiles)"
