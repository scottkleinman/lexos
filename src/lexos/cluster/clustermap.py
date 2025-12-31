"""clustermap.py.

Last Updated: December 3, 2025
Last Tested: December 5, 2025

Note: These clustermap classes are highly experimental and may change in the future.
They may require fiddling with size and layout to be readable. The clustermap may
also not be the best way to visualize textual data, so please use with caution.
For other possibilities see Stylo's seetrees plugin: https://github.com/perechen/seetrees.
"""

from pathlib import Path
from typing import Any, Optional

import fastcluster
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.colors import ListedColormap
from numpy.typing import ArrayLike
from plotly.subplots import make_subplots
from pydantic import BaseModel, ConfigDict, Field, validate_call
from scipy.cluster import hierarchy

from lexos.cluster.sync_script import SYNC_SCRIPT
from lexos.dtm import DTM
from lexos.exceptions import LexosException

sns.set_theme()


def _get_matrix(matrix: ArrayLike | DTM | pd.DataFrame) -> ArrayLike | pd.DataFrame:
    """Get a valid matrix from the input.

    Args:
        matrix (ArrayLike | DTM | pd.DataFrame): The input matrix, which can be an ArrayLike object, a DTM, or a pandas DataFrame.

    Returns:
        ArrayLike | pd.DataFrame: A valid matrix that is not a sparse array and has more than one document.
    """
    if isinstance(matrix, DTM):
        matrix = matrix.to_df()
        matrix.index.name = "terms"

    # Ensure that a DataFrame matrix is not a sparse array
    # Let fastcluster make up for the processing time
    if isinstance(matrix, pd.DataFrame) and hasattr(matrix, "sparse"):
        matrix = matrix.sparse.to_dense()

    if isinstance(matrix, list) and len(matrix) == 0:
        raise LexosException("The document-term matrix cannot be empty.")

    if isinstance(matrix, list):
        first_row = len(matrix[0])
        first_row = len(matrix)
    else:
        first_row = matrix.shape[0]
    if first_row < 2:
        raise LexosException(
            "The document-term matrix must have more than one document."
        )

    return matrix


# Public alias for testing and API
get_matrix = _get_matrix


class Clustermap(BaseModel):
    """Clustermap."""

    dtm: ArrayLike | DTM | pd.DataFrame = Field(
        ..., description="The document-term matrix."
    )
    labels: Optional[list[str]] = Field(
        None, description="The labels for the clustermap."
    )
    metric: Optional[str] = Field(
        "euclidean",
        description="The metric to use for the dendrograms.",
    )
    method: Optional[str] = Field(
        "average",
        description="The method to use for the dendrograms.",
    )
    hide_upper: Optional[bool] = Field(False, description="Hide the upper dendrogram.")
    hide_side: Optional[bool] = Field(False, description="Hide the side dendrogram.")
    title: Optional[str] = Field(None, description="The title for the dendrogram.")
    fig: Optional[matplotlib.figure.Figure] = Field(
        None, description="The figure for the dendrogram."
    )
    z_score: Optional[int] = Field(1, description="The z-score for the clustermap.")
    pivot_kws: Optional[dict[str, str]] = Field(
        None, description="The pivot kwargs for the clustermap."
    )
    standard_scale: Optional[int] = Field(
        None,
        description="The standard scale for the clustermap.",
    )
    figsize: Optional[tuple[int, int]] = Field(
        (8, 8), description="The figure size for the clustermap."
    )
    cbar_kws: Optional[dict] = Field(
        None, description="The cbar kwargs for the clustermap."
    )
    row_cluster: Optional[bool] = Field(
        True, description="Whether to cluster the rows."
    )
    col_cluster: Optional[bool] = Field(
        True, description="Whether to cluster the columns."
    )
    row_linkage: Optional[np.ndarray] = Field(
        None,
        description="Precomputed linkage matrix for the rows. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage for specific formats.",
    )
    col_linkage: Optional[np.ndarray] = Field(
        None,
        description="Precomputed linkage matrix for the columns. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage for specific formats.",
    )
    row_colors: Optional[list | pd.DataFrame | pd.Series | str | ListedColormap] = (
        Field(None, description="The row colors.")
    )
    col_colors: Optional[list | pd.DataFrame | pd.Series | str | ListedColormap] = (
        Field(None, description="The column colors.")
    )
    mask: Optional[np.ndarray | pd.DataFrame] = Field(
        None, description="The mask for the clustermap."
    )
    dendrogram_ratio: Optional[float | tuple[float, float]] = Field(
        (0.1, 0.2),
        description="The dendrogram ratio for the clustermap.",
    )
    colors_ratio: Optional[float] = Field(
        0.03, description="The colors ratio for the clustermap."
    )
    cbar_pos: Optional[tuple[str | float]] = Field(
        (0.02, 0.32, 0.03, 0.2),
        description="The cbar position for the clustermap.",
    )
    tree_kws: Optional[dict] = Field(
        None, description="The tree kwargs for the dendrograms."
    )
    center: Optional[float | int] = Field(
        0, description="The center for the clustermap."
    )
    cmap: Optional[str] = Field("vlag", description="The cmap for the clustermap.")
    linewidths: Optional[float] = Field(
        0.75, description="The linewidths for the dendrograms."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        **data,
    ) -> None:
        """Initialize the Clustermap instance."""
        super().__init__(**data)

        # Set the labels
        self._set_labels()

        # Get the matrix based on the data type
        matrix = _get_matrix(self.dtm)

        # Get colour palettes for the dendrograms
        # Ensure that lists of colours are longer than the number of labels
        # Not sure if this is necessary for column colours
        # if isinstance(self.col_colors, list) and len(self.dtm.labels) >= len(self.col_colors):
        #     raise LexosException("The length of `col_colors` must have be greater than the number of labels.")
        if isinstance(self.row_colors, list) and len(self.labels) >= len(
            self.row_colors
        ):
            raise LexosException(
                "The length of `row_colors` must be greater than the number of labels."
            )
        col_colors, row_colors = self._get_colors()

        # Validate the linkage matrices
        self._validate_linkage_matrices()

        # Perform the clustering
        g = sns.clustermap(
            matrix,
            cmap=self.cmap,
            method=self.method,
            metric=self.metric,
            figsize=self.figsize,
            col_colors=col_colors,
            row_colors=row_colors,
            center=self.center,
            linewidths=self.linewidths,
            z_score=self.z_score,
            pivot_kws=self.pivot_kws,
            standard_scale=self.standard_scale,
            cbar_kws=self.cbar_kws,
            row_linkage=self.row_linkage,
            col_linkage=self.col_linkage,
            mask=self.mask,
            dendrogram_ratio=self.dendrogram_ratio,
            colors_ratio=self.colors_ratio,
            cbar_pos=self.cbar_pos,
            tree_kws=self.tree_kws,
        )

        # Remove the dendrogram on the top
        if self.hide_upper:
            g.ax_col_dendrogram.remove()

        # Remove the dendrogram on the left
        if self.hide_side:
            g.ax_row_dendrogram.remove()

        # Add the title
        if self.title:
            if self.hide_upper:
                y = 0.95
            else:
                y = 1.05
            g.figure.suptitle(self.title, y=y)

        # Save the fig variable
        self.fig = g.figure

        # Do not automatically display -- require fig.show()
        plt.close(self.fig)

    def _get_colors(self) -> ListedColormap | None:
        """Get the row and column colors for the clustermap.

        Notes:
        - For valid palettes, see https://seaborn.pydata.org/generated/seaborn.color_palette.html.
        - The value "default" will use the husl palette with 8 colours.

        Returns:
            A matplotlib ListedColormap or None.
        """
        # Convert palette to vectors drawn on the side of the matrix
        # None means no colours, "default" means use the husl palette
        if self.col_colors is None:
            col_colors = None
        elif isinstance(self.col_colors, (pd.DataFrame, pd.Series)):
            col_colors = self.col_colors
        elif self.col_colors == "default":
            col_colors = sns.husl_palette(8, s=0.45)
        else:
            try:
                col_colors = sns.color_palette(self.col_colors, len(self.col_colors))
            except ValueError:
                raise LexosException("Invalid column palette.")

        if self.row_colors is None:
            row_colors = None
        elif isinstance(self.row_colors, (pd.DataFrame, pd.Series)):
            row_colors = self.row_colors
        elif self.row_colors == "default":
            row_colors = sns.husl_palette(8, s=0.45)
        else:
            try:
                row_colors = sns.color_palette(self.row_colors, len(self.row_colors))
            except ValueError:
                raise LexosException("Invalid row palette.")

        return col_colors, row_colors

    def _set_attrs(self, **kwargs: Any):
        """Set the attributes of the class.

        Args:
            **kwargs: The attributes to set.
        """
        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)

    def _set_labels(self):
        """Set the labels for the clustermap."""
        if not self.labels:
            if isinstance(self.dtm, DTM):
                self.labels = self.dtm.labels
            elif isinstance(self.dtm, pd.DataFrame):
                self.labels = self.dtm.columns.values.tolist()[1:]
            else:
                self.labels = [f"Doc{i + 1}" for i, _ in enumerate(self.dtm)]

    def _validate_linkage_matrices(self):
        """Validate the linkage matrices."""
        # TODO: raise a LexosException if hierarchy.is_valid_linkage fails
        if self.row_linkage is not None:
            try:
                hierarchy.is_valid_linkage(self.row_linkage, throw=True)
            except (TypeError, ValueError) as e:
                raise LexosException(f"Invalid `row_linkage` value: {e}")
        if self.col_linkage is not None:
            try:
                hierarchy.is_valid_linkage(self.col_linkage, throw=True)
            except (TypeError, ValueError) as e:
                raise LexosException(f"Invalid `col_linkage` value: {e}")

    def save(self, path: Path | str, **kwargs: Any):
        """Save the figure to a file.

        Args:
            path (Path | str): The path of the file to save.
            **kwargs (Any): Additional keyword arguments for pyplot.savefig. See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html.
        """
        self.fig.savefig(path, **kwargs)

    def show(self):
        """Show the figure if it is hidden.

        This is a helper method. You can also reference the figure
        using `ClusterMap.fig`. This will generally display in a
        Jupyter notebook.
        """
        return self.fig


def _create_dendrogram_traces(
    linkage_matrix: np.ndarray,
    labels: Optional[list[str]] = None,
    orientation: str = "bottom",
    color: str = "rgb(50,50,50)",
    line_width: float = 1.0,
) -> list[go.Scatter]:
    """Create dendrogram traces from linkage matrix.

    Args:
        linkage_matrix (array-like): Linkage matrix from scipy.cluster.hierarchy.linkage
        labels (list, optional): Labels for the leaves
        orientation (str): Orientation of dendrogram ('top', 'bottom', 'left', 'right')
        color (str): Color for dendrogram lines
        line_width (float): Width of dendrogram lines

    Returns:
        traces (list): List of plotly scatter traces for dendrogram
    """
    dendro_data = hierarchy.dendrogram(
        linkage_matrix, labels=labels, no_plot=True, color_threshold=-np.inf
    )

    traces = []

    # Extract coordinates
    icoord = np.array(dendro_data["icoord"])
    dcoord = np.array(dendro_data["dcoord"])

    # Access the line objects to identify and remove the baseline (where all y-coords are 0)
    mask = ~(dcoord == 0).all(axis=1)  # Create boolean mask for non-baseline segments
    icoord = icoord[mask]
    dcoord = dcoord[mask]

    # Create line traces for each dendrogram segment
    for i in range(len(icoord)):
        x_coords = icoord[i]
        y_coords = dcoord[i]

        if orientation in ["top", "bottom"]:
            # Standard orientation
            if orientation == "bottom":
                y_coords = -y_coords + max(dcoord.flatten())
        else:
            # Swap coordinates for left/right orientation
            x_coords, y_coords = y_coords, x_coords
            if orientation == "left":
                x_coords = -x_coords + max(dcoord.flatten())
                # Shift dendrogram to touch the right edge
                x_coords = x_coords + (max(x_coords) - min(x_coords)) * 0.03

        # Create scatter trace for this segment
        trace = go.Scatter(
            x=x_coords,
            y=y_coords,
            mode="lines",
            line=dict(color=color, width=line_width),
            showlegend=False,
            hoverinfo="skip",
        )
        traces.append(trace)

    return traces, dendro_data


class PlotlyClusterGrid:
    """Plotly implementation of clustered heatmap with dendrograms."""

    def __init__(
        self,
        data: pd.DataFrame | np.ndarray,
        z_score: Optional[int] = None,
        standard_scale: Optional[int] = None,
        mask: Optional[np.ndarray | pd.DataFrame] = None,
        figsize: tuple[int, int] = (800, 600),
        dendrogram_ratio: float | tuple[float, float] = 0.2,
    ) -> None:
        """Initialize the cluster grid.

        Args:
            data (DataFrame or array-like): Rectangular data for clustering
            z_score (int, optional): Whether to z-score rows (0) or columns (1)
            standard_scale (int, optional): Whether to standard scale rows (0) or columns (1)
            mask (bool array or DataFrame, optional): Mask for data visualization
            figsize (tuple[int, int]): Figure size (width, height)
            dendrogram_ratio (float | tuple[float, float]): Ratio of dendrogram size to heatmap size
        """
        # Convert data to DataFrame if needed
        if isinstance(data, pd.DataFrame):
            self.data = data.copy()
        else:
            self.data = pd.DataFrame(data)

        # Process data
        self.data2d = self._format_data(z_score, standard_scale)
        self.mask = self._process_mask(mask)

        # Store configuration
        self.figsize = figsize
        self.dendrogram_ratio = dendrogram_ratio

    def _format_data(
        self, z_score: Optional[int] = None, standard_scale: Optional[int] = None
    ) -> pd.DataFrame:
        """Format and normalize data.

        Args:
            z_score (int, optional): Whether to z-score rows (0) or columns (1)
            standard_scale (int, optional): Whether to standard scale rows (0) or columns (1)

        Returns:
            pd.DataFrame: Formatted data
        """
        data2d = self.data.copy()

        if z_score is not None and standard_scale is not None:
            raise ValueError(
                "Cannot perform both z-scoring and standard-scaling on data"
            )

        if z_score is not None:
            data2d = self._z_score(data2d, z_score)
        if standard_scale is not None:
            data2d = self._standard_scale(data2d, standard_scale)

        return data2d

    @staticmethod
    def _z_score(data2d: pd.DataFrame, axis: int = 1) -> pd.DataFrame:
        """Standardize the mean and variance of the data axis.

        Args:
            data2d (pd.DataFrame): Data to z-score
        Returns:
            pd.DataFrame: Z-scored data
        """
        if axis == 1:
            z_scored = data2d
        else:
            z_scored = data2d.T

        z_scored = (z_scored - z_scored.mean()) / z_scored.std()

        if axis == 1:
            return z_scored
        else:
            return z_scored.T

    @staticmethod
    def _standard_scale(data2d: pd.DataFrame, axis: int = 1) -> pd.DataFrame:
        """Divide the data by the difference between the max and min.

        Args:
            data2d (pd.DataFrame): Data to standard scale
            axis (int, optional): Axis along which to scale (0 for rows, 1 for columns)

        Returns:
            pd.DataFrame: Standard scaled data
        """
        if axis == 1:
            standardized = data2d
        else:
            standardized = data2d.T

        subtract = standardized.min()
        standardized = (standardized - subtract) / (
            standardized.max() - standardized.min()
        )

        if axis == 1:
            return standardized
        else:
            return standardized.T

    def _process_mask(
        self, mask: Optional[np.ndarray | pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        """Process mask for data visualization.

        Args:
            mask (np.ndarray | pd.DataFrame, optional): Mask to apply to the data

        Returns:
            pd.DataFrame: Processed mask
        """
        if mask is None:
            return None

        if isinstance(mask, pd.DataFrame):
            if not (
                mask.index.equals(self.data2d.index)
                and mask.columns.equals(self.data2d.columns)
            ):
                raise ValueError("Mask must have the same index and columns as data.")
        else:
            mask = np.asarray(mask)
            if mask.shape != self.data2d.shape:
                raise ValueError("Mask must have the same shape as data.")
            mask = pd.DataFrame(
                mask, index=self.data2d.index, columns=self.data2d.columns, dtype=bool
            )

        # Add missing data to mask
        mask = mask | pd.isnull(self.data2d)
        return mask

    def _calculate_linkage(
        self, data: np.ndarray, method: str = "average", metric: str = "euclidean"
    ) -> np.ndarray:
        """Calculate linkage matrix.

        Args:
            data (np.ndarray): Data to cluster
            method (str): Linkage method
            metric (str): Distance metric

        Returns:
            np.ndarray: Linkage matrix
        """
        euclidean_methods = ("centroid", "median", "ward")
        euclidean = metric == "euclidean" and method in euclidean_methods
        if euclidean or method == "single":
            return fastcluster.linkage_vector(data, method=method, metric=metric)
        else:
            return fastcluster.linkage(data, method=method, metric=metric)


class PlotlyClustermap(BaseModel):
    """Plotly version of the Clustermap."""

    dtm: Optional[ArrayLike | DTM | pd.DataFrame] = Field(
        ..., description="The document-term matrix."
    )
    labels: Optional[list[str]] = Field(
        None, description="The labels for the clustermap."
    )
    metric: Optional[str] = Field(
        "euclidean",
        description="The metric to use for the dendrograms.",
    )
    method: Optional[str] = Field(
        "average",
        description="The method to use for the dendrograms.",
    )
    hide_upper: Optional[bool] = Field(False, description="Hide the upper dendrogram.")
    hide_side: Optional[bool] = Field(False, description="Hide the side dendrogram.")
    title: Optional[str] = Field(None, description="The title for the dendrogram.")
    fig: Optional[go.Figure] = Field(None, description="The figure for the clustermap.")
    z_score: Optional[int] = Field(1, description="The z-score for the clustermap.")
    pivot_kws: Optional[dict[str, str]] = Field(
        None, description="The pivot kwargs for the clustermap."
    )
    standard_scale: Optional[int] = Field(
        None,
        description="The standard scale for the clustermap.",
    )
    figsize: Optional[tuple[int, int]] = Field(
        (700, 700), description="The figure size for the clustermap in pixels."
    )
    cbar_kws: Optional[dict] = Field(
        None, description="The cbar kwargs for the clustermap."
    )
    row_cluster: Optional[bool] = Field(
        True, description="Whether to cluster the rows."
    )
    col_cluster: Optional[bool] = Field(
        True, description="Whether to cluster the columns."
    )
    row_linkage: Optional[np.ndarray] = Field(
        None,
        description="Precomputed linkage matrix for the rows. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage for specific formats.",
    )
    col_linkage: Optional[np.ndarray] = Field(
        None,
        description="Precomputed linkage matrix for the columns. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage for specific formats.",
    )
    row_colors: Optional[list | pd.DataFrame | pd.Series | str | ListedColormap] = (
        Field(None, description="The row colors.")
    )
    col_colors: Optional[list | pd.DataFrame | pd.Series | str | ListedColormap] = (
        Field(None, description="The column colors.")
    )
    mask: Optional[np.ndarray | pd.DataFrame] = Field(
        None, description="The mask for the clustermap."
    )
    dendrogram_ratio: Optional[float | tuple[float, float]] = Field(
        (0.8, 0.2),
        description="The dendrogram ratio for the clustermap.",
    )
    colors_ratio: Optional[float] = Field(
        0.03, description="The colors ratio for the clustermap."
    )
    cbar_pos: Optional[tuple[str]] = Field(
        (0.02, 0.32, 0.03, 0.2),
        description="The cbar position for the clustermap.",
    )
    colorbar: Optional[dict[str, Any]] = Field(
        dict(x=0.11, y=0.5, xref="container", yref="container", len=0.6),
        description="The colorbar position for the clustermap. This is a more generic version than `cbar_pos` and can be used to set the position of the colorbar in a more flexible way.",
    )
    tree_kws: Optional[dict] = Field(
        None, description="The tree kwargs for the dendrograms."
    )
    center: Optional[float | int] = Field(
        0, description="The center for the clustermap. Default could be None."
    )
    cmap: Optional[str] = Field("viridis", description="The cmap for the clustermap.")
    linewidths: Optional[float] = Field(
        0.75, description="The linewidths for the dendrograms. Default could be 0."
    )
    annot: Optional[bool] = Field(
        False, description="Whether to annotate the clustermap."
    )
    fmt: Optional[str] = Field(
        ".2g", description="The format for the annotations in the clustermap."
    )
    show_dendrogram_labels: Optional[bool] = Field(
        False, description="Whether to show the labels on the dendrograms."
    )
    show_heatmap_labels: Optional[bool] = Field(
        True, description="Whether to show the labels on the heatmap."
    )
    kwargs: Any = Field(
        {}, description="Additional keyword arguments for the clustermap."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        **data,
    ) -> None:
        """Initialize the PlotlyClustermap instance."""
        super().__init__(**data)

        # Set the labels
        self._set_labels()

        # Get the matrix based on the data type
        matrix = _get_matrix(self.dtm)

        # Extract our custom parameters from kwargs to prevent them being passed to plotly components
        filtered_kwargs = self.kwargs.copy()
        filtered_kwargs.pop(
            "show_dendrogram_labels", None
        )  # This is already a function parameter
        filtered_kwargs.pop(
            "show_heatmap_labels", None
        )  # This is already a function parameter
        filtered_kwargs.pop(
            "title", None
        )  # Title should go to layout, not heatmap trace

        # Determine whether to show heatmap labels
        if self.show_heatmap_labels is None:
            # Auto-mode: hide y-axis (left) labels when row dendrogram is present,
            # but always show x-axis (bottom) labels for readability
            show_heatmap_x_labels = (
                True  # Always show bottom labels unless explicitly disabled
            )
            show_heatmap_y_labels = (
                not self.row_cluster
            )  # Hide left labels only if row dendrogram present
        else:
            # Manual mode: use the same setting for both axes
            show_heatmap_x_labels = self.show_heatmap_labels
            show_heatmap_y_labels = self.show_heatmap_labels

        # Initialize cluster grid
        grid = PlotlyClusterGrid(
            data=matrix,
            z_score=self.z_score,
            standard_scale=self.standard_scale,
            mask=self.mask,
            figsize=self.figsize,
            dendrogram_ratio=self.dendrogram_ratio,
        )

        # Handle dendrogram ratios
        if isinstance(self.dendrogram_ratio, (list, tuple)):
            row_dendrogram_ratio, col_dendrogram_ratio = self.dendrogram_ratio
        else:
            row_dendrogram_ratio = col_dendrogram_ratio = self.dendrogram_ratio

        # Handle tree styling
        if self.tree_kws is None:
            self.tree_kws = {}
        tree_color = self.tree_kws.get("color", "rgb(50,50,50)")
        tree_width = self.tree_kws.get("linewidth", 1.0)

        # Calculate clustering
        data_array = grid.data2d.values

        # Row clustering
        row_linkage = data.get("row_linkage", None)
        if self.row_cluster:
            if row_linkage is None:
                row_linkage = grid._calculate_linkage(
                    data_array, self.method, self.metric
                )
            row_dendro_traces, row_dendro_data = _create_dendrogram_traces(
                row_linkage,
                labels=[str(x) for x in grid.data2d.index]
                if self.show_dendrogram_labels
                else None,
                orientation="left",
                color=tree_color,
                line_width=tree_width,
            )
            row_order = row_dendro_data["leaves"]
        else:
            row_order = list(range(len(grid.data2d.index)))
            row_dendro_traces = []
            row_dendro_data = None

        # Column clustering
        col_linkage = data.get("col_linkage", None)
        if self.col_cluster:
            if col_linkage is None:
                col_linkage = grid._calculate_linkage(
                    data_array.T, self.method, self.metric
                )
            col_dendro_traces, col_dendro_data = _create_dendrogram_traces(
                col_linkage,
                labels=[str(x) for x in grid.data2d.columns]
                if self.show_dendrogram_labels
                else None,
                orientation="top",
                color=tree_color,
                line_width=tree_width,
            )
            col_order = col_dendro_data["leaves"]
        else:
            col_order = list(range(len(grid.data2d.columns)))
            col_dendro_traces = []
            col_dendro_data = None

        # Reorder data
        ordered_data = grid.data2d.iloc[row_order, col_order]

        # Create subplot layout
        n_rows = 2 if self.col_cluster else 1
        n_cols = 2 if self.row_cluster else 1

        # Calculate subplot dimensions
        if self.row_cluster and self.col_cluster:
            row_heights = [col_dendrogram_ratio, 1 - col_dendrogram_ratio]
            col_widths = [1 - row_dendrogram_ratio, row_dendrogram_ratio]
            # subplot_titles = ["", "Column Dendrogram", "Row Dendrogram", "Heatmap"]
        elif self.col_cluster:
            row_heights = [col_dendrogram_ratio, 1 - col_dendrogram_ratio]
            col_widths = [1.0]
            # subplot_titles = ["Column Dendrogram", "Heatmap"]
        elif self.row_cluster:
            row_heights = [1.0]
            col_widths = [1 - row_dendrogram_ratio, row_dendrogram_ratio]
            # subplot_titles = ["Heatmap", "Row Dendrogram"]
        else:
            row_heights = [1.0]
            col_widths = [1.0]
            # subplot_titles = ["Heatmap"]

        # Create subplots
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            row_heights=row_heights,
            column_widths=col_widths,
            horizontal_spacing=0,  # Remove padding between dendrograms and heatmap
            vertical_spacing=0,  # Remove padding between dendrograms and heatmap
            # subplot_titles=None,  # We'll add custom titles if needed
        )

        # Determine subplot positions
        heatmap_row = n_rows
        heatmap_col = 1 if not self.row_cluster else n_cols

        # Prepare heatmap data
        z_data = ordered_data.values
        x_labels = [str(x) for x in ordered_data.columns]
        y_labels = [str(y) for y in ordered_data.index]

        # Apply mask if provided
        if grid.mask is not None:
            mask_ordered = grid.mask.iloc[row_order, col_order]
            z_data = np.where(mask_ordered.values, np.nan, z_data)

        # Add heatmap
        heatmap_trace = go.Heatmap(
            z=z_data,
            x=x_labels,
            y=y_labels,
            colorscale=self.cmap,
            zmid=self.center,
            showscale=True,
            colorbar=self.colorbar,
            name="",  # Remove Trace 0 from hover
            **filtered_kwargs,
        )

        fig.add_trace(heatmap_trace, row=heatmap_row, col=heatmap_col)

        # Add column dendrogram
        if not self.hide_upper:
            if self.col_cluster and col_dendro_traces:
                for trace in col_dendro_traces:
                    fig.add_trace(trace, row=1, col=heatmap_col)

        # Add row dendrogram
        if not self.hide_side:
            if self.row_cluster and row_dendro_traces:
                for trace in row_dendro_traces:
                    fig.add_trace(trace, row=heatmap_row, col=1)

        # Reverse the traces for the row dendrogram to match the Seaborn dendrogram
        fig.update_yaxes(row=heatmap_row, col=1, autorange="reversed")

        # Add annotations if requested
        if self.annot:
            annotations = []
            for i, row in enumerate(y_labels):
                for j, col in enumerate(x_labels):
                    if not (grid.mask is not None and mask_ordered.iloc[i, j]):
                        cell_value = z_data[i, j]
                        if not np.isnan(cell_value):
                            max_abs_val = np.nanmax(np.abs(z_data))
                            text_color = (
                                "white"
                                if abs(cell_value) > max_abs_val / 2
                                else "black"
                            )

                            annotations.append(
                                dict(
                                    x=j,
                                    y=i,
                                    text=format(cell_value, self.fmt),
                                    showarrow=False,
                                    font=dict(color=text_color, size=10),
                                    xref=f"x{heatmap_col}" if heatmap_col > 1 else "x",
                                    yref=f"y{heatmap_row}" if heatmap_row > 1 else "y",
                                )
                            )

            fig.update_layout(annotations=annotations)

        # Update layout
        fig.update_layout(
            title=self.title if self.title else None,
            width=self.figsize[0],
            height=self.figsize[1],
            showlegend=False,
        )

        # Configure axes for each subplot
        for row in range(1, n_rows + 1):
            for col in range(1, n_cols + 1):
                # Generate xaxis and yaxis references
                xaxis_ref = (
                    f"xaxis{col + (row - 1) * n_cols}"
                    if col + (row - 1) * n_cols > 1
                    else "xaxis"
                )
                yaxis_ref = (
                    f"yaxis{col + (row - 1) * n_cols}"
                    if col + (row - 1) * n_cols > 1
                    else "yaxis"
                )

                # Default settings for all subplots
                fig.update_layout(
                    {
                        xaxis_ref: dict(
                            showticklabels=False,
                            showgrid=False,
                            zeroline=False,
                            showline=False,
                            ticks="",
                        ),
                        yaxis_ref: dict(
                            showticklabels=False,
                            showgrid=False,
                            zeroline=False,
                            showline=False,
                            ticks="",
                        ),
                    }
                )

        # Special configuration for heatmap
        heatmap_xaxis = (
            f"xaxis{heatmap_col + (heatmap_row - 1) * n_cols}"
            if heatmap_col + (heatmap_row - 1) * n_cols > 1
            else "xaxis"
        )
        heatmap_yaxis = (
            f"yaxis{heatmap_col + (heatmap_row - 1) * n_cols}"
            if heatmap_col + (heatmap_row - 1) * n_cols > 1
            else "yaxis"
        )

        fig.update_layout(
            {
                heatmap_xaxis: dict(
                    showticklabels=show_heatmap_x_labels,
                    tickmode="array" if show_heatmap_x_labels else "linear",
                    tickvals=list(range(len(x_labels)))
                    if show_heatmap_x_labels
                    else [],
                    ticktext=x_labels if show_heatmap_x_labels else [],
                    tickangle=45 if show_heatmap_x_labels else 0,
                    side="bottom",
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    ticks="" if not show_heatmap_x_labels else "outside",
                ),
                heatmap_yaxis: dict(
                    showticklabels=show_heatmap_y_labels,
                    tickmode="array" if show_heatmap_y_labels else "linear",
                    tickvals=list(range(len(y_labels)))
                    if show_heatmap_y_labels
                    else [],
                    ticktext=y_labels if show_heatmap_y_labels else [],
                    autorange="reversed",  # Reverse to match typical heatmap orientation
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    ticks="" if not show_heatmap_y_labels else "outside",
                    side="right",
                ),
            }
        )

        # Configure dendrogram axes ranges
        if self.col_cluster and col_dendro_data:
            col_dend_xaxis = f"xaxis{heatmap_col}" if heatmap_col > 1 else "xaxis"
            col_dend_yaxis = f"yaxis{heatmap_col}" if heatmap_col > 1 else "yaxis"

            # Set ranges for column dendrogram
            fig.update_layout(
                {
                    col_dend_xaxis: dict(
                        range=[0, len(ordered_data.columns) * 10 + 5],
                        showticklabels=self.show_dendrogram_labels,
                        showgrid=False,
                        zeroline=False,
                        showline=False,
                        ticks="" if not self.show_dendrogram_labels else "outside",
                    ),
                    col_dend_yaxis: dict(
                        range=[
                            0,
                            max(np.array(col_dendro_data["dcoord"]).flatten()) * 1.00,
                        ],
                        showticklabels=self.show_dendrogram_labels,
                        showgrid=False,
                        zeroline=False,
                        showline=False,
                        ticks="" if not self.show_dendrogram_labels else "outside",
                    ),
                }
            )

        if self.row_cluster and row_dendro_data:
            row_dend_xaxis = (
                f"xaxis{1 + (heatmap_row - 1) * n_cols}"
                if 1 + (heatmap_row - 1) * n_cols > 1
                else "xaxis"
            )

            row_dend_yaxis = (
                f"yaxis{1 + (heatmap_row - 1) * n_cols}"
                if 1 + (heatmap_row - 1) * n_cols > 1
                else "yaxis"
            )

            # Set ranges for row dendrogram
            fig.update_layout(
                {
                    row_dend_xaxis: dict(
                        range=[
                            0,
                            max(np.array(row_dendro_data["dcoord"]).flatten()) * 1.01,
                        ],
                        showticklabels=self.show_dendrogram_labels,
                        showgrid=False,
                        zeroline=False,
                        showline=False,
                        ticks="" if not self.show_dendrogram_labels else "outside",
                    ),
                    row_dend_yaxis: dict(
                        range=[0, len(ordered_data.index) * 10],
                        showticklabels=self.show_dendrogram_labels,
                        showgrid=False,
                        zeroline=False,
                        showline=False,
                        ticks="" if not self.show_dendrogram_labels else "outside",
                    ),
                }
            )

        fig.update_layout(title_x=0.5)  # Automatically adjust x-axis margins

        self.fig = fig

        # Adjust layout if upper dendrogram is hidden
        self._adjust_layout_for_hidden_upper()

    def _adjust_layout_for_hidden_upper(self) -> None:
        """Adjust the layout when the upper dendrogram is hidden to move components up."""
        if not self.hide_upper:
            return

        # Get the current layout
        layout = self.fig.layout

        # Find which subplot contains the heatmap and row dendrogram
        heatmap_subplot = None
        row_dendro_subplot = None

        # Iterate through the traces to find the heatmap and row dendrogram
        for i, trace in enumerate(self.fig.data):
            if hasattr(trace, "type") and trace.type == "heatmap":
                # This is the heatmap
                heatmap_subplot = (
                    getattr(trace, "xaxis", "x"),
                    getattr(trace, "yaxis", "y"),
                )
            elif hasattr(trace, "type") and trace.type == "scatter":
                # This might be the row dendrogram - check if it's on the side
                x_axis = getattr(trace, "xaxis", "x")
                y_axis = getattr(trace, "yaxis", "y")
                if x_axis != heatmap_subplot[0] if heatmap_subplot else True:
                    row_dendro_subplot = x_axis, y_axis

        # Adjust the domains to move everything up
        # The key is to expand the y-domain of the bottom subplots to fill the top space

        updates = {}

        # Move heatmap up by expanding its y-domain
        if heatmap_subplot:
            heatmap_xaxis = (
                heatmap_subplot[0].replace("x", "xaxis")
                if "x" in heatmap_subplot[0]
                else "xaxis"
            )
            heatmap_yaxis = (
                heatmap_subplot[1].replace("y", "yaxis")
                if "y" in heatmap_subplot[1]
                else "yaxis"
            )

            # Get current domain or use defaults
            current_layout = getattr(layout, heatmap_yaxis, {})
            current_domain = getattr(current_layout, "domain", [0.0, 0.8])

            # Expand to fill the full height
            updates[f"{heatmap_yaxis}.domain"] = [0.0, 1.0]

        # Move row dendrogram up by expanding its y-domain
        if row_dendro_subplot and not self.hide_side:
            row_dendro_xaxis = (
                row_dendro_subplot[0].replace("x", "xaxis")
                if "x" in row_dendro_subplot[0]
                else "xaxis"
            )
            row_dendro_yaxis = (
                row_dendro_subplot[1].replace("y", "yaxis")
                if "y" in row_dendro_subplot[1]
                else "yaxis"
            )

            # Expand to fill the full height
            updates[f"{row_dendro_yaxis}.domain"] = [0.0, 1.0]

        # Apply the updates
        if updates:
            self.fig.update_layout(updates)

    def _set_labels(self):
        """Set the labels for the clustermap."""
        if not self.labels:
            if isinstance(self.dtm, DTM):
                self.labels = self.dtm.labels
            elif isinstance(self.dtm, pd.DataFrame):
                self.labels = self.dtm.columns.values.tolist()[1:]
            else:
                self.labels = [f"Doc{i + 1}" for i, _ in enumerate(self.dtm)]

    @validate_call(config=model_config)
    def save(self, path: Path | str, **kwargs: Any) -> None:
        """Save a static image of the figure to disk.

        Alias of `write_image()`

        Args:
            path: The file path to save the image.
            **kwargs (Any): Additional arguments to pass to the write_image method.
        """
        self.write_image(path, **kwargs)

    def show(self):
        """Show the clustermap."""
        config = dict(
            displaylogo=False,
            modeBarButtonsToRemove=["toggleSpikelines"],
            scrollZoom=True,
        )
        self.fig.show(config=config)

    def to_html(self, include_sync=False, **kwargs: Any) -> str:
        """Create an HTML representation of the figure with optional synchronization.

        Wrapper from the Plotly Figure to_html method.
        See https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html.

        Args:
            include_sync (bool): Whether to include the synchronization script.
            **kwargs (Any): Additional keyword arguments for the to_html method.

        Returns:
            str: The HTML representation of the figure.
        """
        html = self.fig.to_html(**kwargs)

        if include_sync:
            # Insert the script before the closing </body> tag
            html = html.replace("</body>", f"{SYNC_SCRIPT}</body>")

        return html

    def to_image(self, **kwargs: Any) -> bytes:
        """Create a static image of the figure.

        Args:
            **kwargs (Any): Additional keyword arguments for the to_image method.

        Returns:
            bytes: The image in bytes.

        Wrapper from the Plotly Figure to_html method.
        See https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html.
        """
        return self.fig.to_image(**kwargs)

    @validate_call(config=model_config)
    def write_html(self, path: Path | str, **kwargs: Any) -> None:
        """Save an HTML representation of the figure to disk.

        Args:
            path (Path | str): The file path to save the HTML.
            **kwargs (Any): Additional arguments to pass to the write_html method.

        Wrapper from the Plotly Figure write_html method.
        See https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html.
        """
        return self.fig.write_html(path, **kwargs)

    @validate_call(config=model_config)
    def write_image(self, path: Path | str, **kwargs: Any) -> None:
        """Save a static image of the figure to disk.

        Args:
            path (Path | str): The file path to save the image.
            **kwargs (Any): Additional arguments to pass to the write_image method.

        Wrapper from the Plotly Figure write_image method.
        See https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html.
        """
        return self.fig.write_image(path, **kwargs)
