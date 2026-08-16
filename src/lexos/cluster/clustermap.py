"""clustermap.py.

Last Updated: August 16, 2026
Last Tested: August 16, 2026

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
from lexos.util import safe_recursion_limit

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
    with safe_recursion_limit(len(linkage_matrix) + 1):
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
            mean = data2d.mean(axis=1)
            std = data2d.std(axis=1, ddof=0)
            return data2d.sub(mean, axis=0).div(std, axis=0)

        mean = data2d.mean(axis=0)
        std = data2d.std(axis=0, ddof=0)
        return data2d.sub(mean, axis=1).div(std, axis=1)

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
            min_vals = data2d.min(axis=1)
            max_vals = data2d.max(axis=1)
            return data2d.sub(min_vals, axis=0).div(max_vals - min_vals, axis=0)

        min_vals = data2d.min(axis=0)
        max_vals = data2d.max(axis=0)
        return data2d.sub(min_vals, axis=1).div(max_vals - min_vals, axis=1)

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

        self._set_labels()
        matrix = _get_matrix(self.dtm)
        show_heatmap_x_labels, show_heatmap_y_labels = (
            self._resolve_heatmap_label_settings()
        )
        filtered_kwargs = self._prepare_heatmap_kwargs()

        grid = self._create_cluster_grid(matrix)
        row_dendro_traces, row_dendro_data, row_order = self._prepare_row_dendrogram(
            grid, data.get("row_linkage", None)
        )
        col_dendro_traces, col_dendro_data, col_order = self._prepare_col_dendrogram(
            grid, data.get("col_linkage", None)
        )

        ordered_data = grid.data2d.iloc[row_order, col_order]
        z_data = ordered_data.values
        x_labels = [str(x) for x in ordered_data.columns]
        y_labels = [str(y) for y in ordered_data.index]

        mask_ordered = None
        if grid.mask is not None:
            mask_ordered = grid.mask.iloc[row_order, col_order]
            z_data = np.where(mask_ordered.values, np.nan, z_data)

        self.fig = self._build_figure(
            ordered_data=ordered_data,
            grid=grid,
            z_data=z_data,
            x_labels=x_labels,
            y_labels=y_labels,
            row_dendro_traces=row_dendro_traces,
            col_dendro_traces=col_dendro_traces,
            row_dendro_data=row_dendro_data,
            col_dendro_data=col_dendro_data,
            show_heatmap_x_labels=show_heatmap_x_labels,
            show_heatmap_y_labels=show_heatmap_y_labels,
            filtered_kwargs=filtered_kwargs,
            mask_ordered=mask_ordered,
        )

    def _resolve_heatmap_label_settings(self) -> tuple[bool, bool]:
        """Resolve final visibility states for heatmap axis labels."""
        if self.show_heatmap_labels is None:
            return True, not self.row_cluster

        return self.show_heatmap_labels, self.show_heatmap_labels

    def _prepare_heatmap_kwargs(self) -> dict[str, Any]:
        """Exclude clustermap-specific kwargs from the heatmap trace config."""
        filtered_kwargs = self.kwargs.copy()
        filtered_kwargs.pop("show_dendrogram_labels", None)
        filtered_kwargs.pop("show_heatmap_labels", None)
        filtered_kwargs.pop("title", None)
        return filtered_kwargs

    def _create_cluster_grid(
        self, matrix: ArrayLike | pd.DataFrame | DTM
    ) -> PlotlyClusterGrid:
        """Build the PlotlyClusterGrid used for clustering and masking."""
        return PlotlyClusterGrid(
            data=matrix,
            z_score=self.z_score,
            standard_scale=self.standard_scale,
            mask=self.mask,
            figsize=self.figsize,
            dendrogram_ratio=self.dendrogram_ratio,
        )

    def _prepare_row_dendrogram(
        self,
        grid: PlotlyClusterGrid,
        row_linkage: Optional[np.ndarray],
    ) -> tuple[list[go.Scatter], dict | None, list[int]]:
        """Generate row dendrogram traces and the row order."""
        if not self.row_cluster:
            return [], None, list(range(len(grid.data2d.index)))

        if row_linkage is None:
            row_linkage = grid._calculate_linkage(
                grid.data2d.values, self.method, self.metric
            )

        traces, data = _create_dendrogram_traces(
            row_linkage,
            labels=[str(x) for x in grid.data2d.index]
            if self.show_dendrogram_labels
            else None,
            orientation="left",
            color=self.tree_kws.get("color", "rgb(50,50,50)")
            if self.tree_kws
            else "rgb(50,50,50)",
            line_width=self.tree_kws.get("linewidth", 1.0) if self.tree_kws else 1.0,
        )

        return traces, data, data["leaves"]

    def _prepare_col_dendrogram(
        self,
        grid: PlotlyClusterGrid,
        col_linkage: Optional[np.ndarray],
    ) -> tuple[list[go.Scatter], dict | None, list[int]]:
        """Generate column dendrogram traces and the column order."""
        if not self.col_cluster:
            return [], None, list(range(len(grid.data2d.columns)))

        if col_linkage is None:
            col_linkage = grid._calculate_linkage(
                grid.data2d.values.T, self.method, self.metric
            )

        traces, data = _create_dendrogram_traces(
            col_linkage,
            labels=[str(x) for x in grid.data2d.columns]
            if self.show_dendrogram_labels
            else None,
            orientation="top",
            color=self.tree_kws.get("color", "rgb(50,50,50)")
            if self.tree_kws
            else "rgb(50,50,50)",
            line_width=self.tree_kws.get("linewidth", 1.0) if self.tree_kws else 1.0,
        )

        return traces, data, data["leaves"]

    def _get_subplots_config(self) -> tuple[int, int, list[float], list[float]]:
        """Compute subplot dimensions for the cluster grid and dendrograms."""
        row_dendrogram_ratio, col_dendrogram_ratio = (
            self.dendrogram_ratio
            if isinstance(self.dendrogram_ratio, (list, tuple))
            else (self.dendrogram_ratio, self.dendrogram_ratio)
        )

        if self.row_cluster and self.col_cluster:
            return (
                2,
                2,
                [col_dendrogram_ratio, 1 - col_dendrogram_ratio],
                [1 - row_dendrogram_ratio, row_dendrogram_ratio],
            )

        if self.col_cluster:
            return 2, 1, [col_dendrogram_ratio, 1 - col_dendrogram_ratio], [1.0]

        if self.row_cluster:
            return 1, 2, [1.0], [1 - row_dendrogram_ratio, row_dendrogram_ratio]

        return 1, 1, [1.0], [1.0]

    def _get_heatmap_position(self, n_rows: int, n_cols: int) -> tuple[int, int]:
        """Return the row and column indices for the heatmap subplot."""
        return n_rows, 1 if not self.row_cluster else n_cols

    def _axis_ref(self, row: int, col: int, n_cols: int, axis_type: str) -> str:
        """Return the Plotly axis reference name for a subplot cell."""
        axis_index = col + (row - 1) * n_cols
        if axis_index == 1:
            return f"{axis_type}axis"
        return f"{axis_type}axis{axis_index}"

    def _create_heatmap_trace(
        self,
        z_data: np.ndarray,
        x_labels: list[str],
        y_labels: list[str],
        filtered_kwargs: dict[str, Any],
    ) -> go.Heatmap:
        """Build the Plotly heatmap trace."""
        return go.Heatmap(
            z=z_data,
            x=x_labels,
            y=y_labels,
            colorscale=self.cmap,
            zmid=self.center,
            showscale=True,
            colorbar=self.colorbar,
            name="",
            **filtered_kwargs,
        )

    def _build_figure(
        self,
        ordered_data: pd.DataFrame,
        grid: PlotlyClusterGrid,
        z_data: np.ndarray,
        x_labels: list[str],
        y_labels: list[str],
        row_dendro_traces: list[go.Scatter],
        col_dendro_traces: list[go.Scatter],
        row_dendro_data: dict | None,
        col_dendro_data: dict | None,
        show_heatmap_x_labels: bool,
        show_heatmap_y_labels: bool,
        filtered_kwargs: dict[str, Any],
        mask_ordered: Optional[pd.DataFrame] = None,
    ) -> go.Figure:
        """Build the Plotly figure with subplots, heatmap, and dendrograms."""
        n_rows, n_cols, row_heights, col_widths = self._get_subplots_config()
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            row_heights=row_heights,
            column_widths=col_widths,
            horizontal_spacing=0,
            vertical_spacing=0,
        )

        heatmap_row, heatmap_col = self._get_heatmap_position(n_rows, n_cols)
        fig.add_trace(
            self._create_heatmap_trace(z_data, x_labels, y_labels, filtered_kwargs),
            row=heatmap_row,
            col=heatmap_col,
        )

        if not self.hide_upper and self.col_cluster and col_dendro_traces:
            for trace in col_dendro_traces:
                fig.add_trace(trace, row=1, col=heatmap_col)

        if not self.hide_side and self.row_cluster and row_dendro_traces:
            for trace in row_dendro_traces:
                fig.add_trace(trace, row=heatmap_row, col=1)

        fig.update_yaxes(row=heatmap_row, col=1, autorange="reversed")

        if self.annot:
            annotations = self._add_annotations(
                z_data, x_labels, y_labels, mask_ordered, grid, heatmap_row, heatmap_col
            )
            fig.update_layout(annotations=annotations)

        fig.update_layout(
            title=self.title if self.title else None,
            width=self.figsize[0],
            height=self.figsize[1],
            showlegend=False,
        )

        self._configure_axes(
            fig,
            n_rows,
            n_cols,
            heatmap_row,
            heatmap_col,
            x_labels,
            y_labels,
            show_heatmap_x_labels,
            show_heatmap_y_labels,
            row_dendro_data,
            col_dendro_data,
            ordered_data,
        )

        fig.update_layout(title_x=0.5)
        self.fig = fig
        return fig

    def _configure_axes(
        self,
        fig: go.Figure,
        n_rows: int,
        n_cols: int,
        heatmap_row: int,
        heatmap_col: int,
        x_labels: list[str],
        y_labels: list[str],
        show_heatmap_x_labels: bool,
        show_heatmap_y_labels: bool,
        row_dendro_data: dict | None,
        col_dendro_data: dict | None,
        ordered_data: pd.DataFrame,
    ) -> None:
        """Configure axis properties for all subplots and dendrograms."""
        for row in range(1, n_rows + 1):
            for col in range(1, n_cols + 1):
                fig.update_layout(
                    {
                        self._axis_ref(row, col, n_cols, "x"): self._base_axis_style(),
                        self._axis_ref(row, col, n_cols, "y"): self._base_axis_style(),
                    }
                )

        heatmap_xaxis = self._axis_ref(heatmap_row, heatmap_col, n_cols, "x")
        heatmap_yaxis = self._axis_ref(heatmap_row, heatmap_col, n_cols, "y")

        fig.update_layout(
            {
                heatmap_xaxis: self._heatmap_axis_style(
                    show_heatmap_x_labels,
                    x_labels,
                    tickangle=45,
                    side="bottom",
                ),
                heatmap_yaxis: self._heatmap_axis_style(
                    show_heatmap_y_labels,
                    y_labels,
                    tickangle=0,
                    side="right",
                    autorange="reversed",
                ),
            }
        )

        if self.col_cluster and col_dendro_data:
            self._configure_dendrogram_axes(
                fig,
                x_axis=self._axis_ref(1, heatmap_col, n_cols, "x"),
                y_axis=self._axis_ref(1, heatmap_col, n_cols, "y"),
                x_range=[0, len(ordered_data.columns) * 10 + 5],
                y_range=[0, max(np.array(col_dendro_data["dcoord"]).flatten()) * 1.00],
            )

        if self.row_cluster and row_dendro_data:
            self._configure_dendrogram_axes(
                fig,
                x_axis=self._axis_ref(heatmap_row, 1, n_cols, "x"),
                y_axis=self._axis_ref(heatmap_row, 1, n_cols, "y"),
                x_range=[0, max(np.array(row_dendro_data["dcoord"]).flatten()) * 1.01],
                y_range=[0, len(ordered_data.index) * 10],
            )

    def _configure_dendrogram_axes(
        self,
        fig: go.Figure,
        x_axis: str,
        y_axis: str,
        x_range: list[float],
        y_range: list[float],
    ) -> None:
        """Configure axis properties for a dendrogram subplot."""
        fig.update_layout(
            {
                x_axis: self._dendrogram_axis_style(range=x_range),
                y_axis: self._dendrogram_axis_style(range=y_range),
            }
        )

    def _base_axis_style(self) -> dict[str, Any]:
        """Return the shared base style for subplot axes."""
        return dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks="",
        )

    def _heatmap_axis_style(
        self,
        show_labels: bool,
        labels: list[str],
        tickangle: int,
        side: str,
        autorange: str | None = None,
    ) -> dict[str, Any]:
        """Return axis style for heatmap axes."""
        config = self._base_axis_style()
        config.update(
            showticklabels=show_labels,
            tickmode="array" if show_labels else "linear",
            tickvals=list(range(len(labels))) if show_labels else [],
            ticktext=labels if show_labels else [],
            tickangle=tickangle,
            side=side,
            ticks="" if not show_labels else "outside",
        )
        if autorange is not None:
            config["autorange"] = autorange
        return config

    def _dendrogram_axis_style(self, range: list[float]) -> dict[str, Any]:
        """Return axis style for dendrogram axes."""
        return dict(
            range=range,
            showticklabels=self.show_dendrogram_labels,
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks="" if not self.show_dendrogram_labels else "outside",
        )

    def _add_annotations(
        self,
        z_data: np.ndarray,
        x_labels: list[str],
        y_labels: list[str],
        mask_ordered: Optional[pd.DataFrame],
        grid: PlotlyClusterGrid,
        heatmap_row: int,
        heatmap_col: int,
    ) -> list[dict]:
        """Add annotations to the heatmap cells."""
        if z_data.size == 0:
            return []

        annotations = []
        mask = mask_ordered.values if mask_ordered is not None else None
        max_abs_val = (
            np.nanmax(np.abs(z_data[~np.isnan(z_data)]))
            if np.any(~np.isnan(z_data))
            else 0.0
        )

        for i, j in np.ndindex(z_data.shape):
            if self._skip_annotation(z_data, mask, i, j):
                continue

            annotations.append(
                self._build_annotation(
                    i,
                    j,
                    z_data[i, j],
                    heatmap_row,
                    heatmap_col,
                    max_abs_val,
                )
            )

        return annotations

    def _skip_annotation(
        self,
        z_data: np.ndarray,
        mask: Optional[np.ndarray],
        i: int,
        j: int,
    ) -> bool:
        """Return True when a cell should not be annotated."""
        if mask is not None and mask[i, j]:
            return True

        return np.isnan(z_data[i, j])

    def _build_annotation(
        self,
        row: int,
        col: int,
        value: float,
        heatmap_row: int,
        heatmap_col: int,
        max_abs_val: float,
    ) -> dict:
        """Build the annotation dictionary for a single cell."""
        return dict(
            x=col,
            y=row,
            text=format(value, self.fmt),
            showarrow=False,
            font=dict(color=self._annotation_color(value, max_abs_val), size=10),
            xref=f"x{heatmap_col}" if heatmap_col > 1 else "x",
            yref=f"y{heatmap_row}" if heatmap_row > 1 else "y",
        )

    def _annotation_color(self, value: float, max_abs_val: float) -> str:
        """Choose annotation text color based on cell value."""
        return "white" if abs(value) > max_abs_val / 2 else "black"

    def _adjust_layout_for_hidden_upper(self) -> None:
        """Adjust the layout when the upper dendrogram is hidden to move components up."""
        if not self.hide_upper:
            return

        heatmap_axes, row_dendro_axes = self._find_heatmap_and_row_dendrogram_axes()
        updates = {}

        if heatmap_axes:
            updates[f"{self._axis_name(heatmap_axes[1])}.domain"] = [0.0, 1.0]

        if row_dendro_axes and not self.hide_side:
            updates[f"{self._axis_name(row_dendro_axes[1])}.domain"] = [0.0, 1.0]

        if updates:
            self.fig.update_layout(updates)

    def _find_heatmap_and_row_dendrogram_axes(
        self,
    ) -> tuple[tuple[str, str] | None, tuple[str, str] | None]:
        """Return axes tuples for the heatmap and row dendrogram traces."""
        heatmap_axes = None
        candidate_axes = []

        for trace in self.fig.data:
            trace_type = getattr(trace, "type", None)
            x_axis = getattr(trace, "xaxis", "x")
            y_axis = getattr(trace, "yaxis", "y")

            if trace_type == "heatmap":
                heatmap_axes = (x_axis, y_axis)
            elif trace_type == "scatter":
                candidate_axes.append((x_axis, y_axis))

        row_dendro_axes = None
        if heatmap_axes:
            row_dendro_axes = next(
                (axes for axes in candidate_axes if axes[0] != heatmap_axes[0]),
                None,
            )

        return heatmap_axes, row_dendro_axes

    def _axis_name(self, axis: str) -> str:
        """Normalize a Plotly axis reference for layout updates."""
        if axis.startswith("xaxis") or axis.startswith("yaxis"):
            return axis
        if axis.startswith("x"):
            return f"xaxis{axis[1:]}" if len(axis) > 1 else "xaxis"
        if axis.startswith("y"):
            return f"yaxis{axis[1:]}" if len(axis) > 1 else "yaxis"
        return axis

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
