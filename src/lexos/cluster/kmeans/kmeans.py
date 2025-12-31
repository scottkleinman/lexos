"""kmeans.py.

Lexos KMeans clustering module for document-term matrices.

Last Updated: 2025-12-04
Last Tested: 2025-12-05

# TODO:
- Implement silhouette score? See https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
"""

from pathlib import Path
from typing import Any, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pydantic import BaseModel, ConfigDict, Field, validate_call

# Import with a different name to avoid conflicts
from sklearn.cluster import (
    KMeans as sklearn_KMeans,
)
from sklearn.decomposition import PCA
from wasabi import msg

from lexos.dtm import DTM
from lexos.exceptions import LexosException


class KMeans(BaseModel):
    """Perform and visualize KMeans clustering with optional dimensionality reduction."""

    # Configurable parameters for clustering
    dtm: DTM | pd.DataFrame | np.ndarray = Field(
        default=None, description="Input document-term matrix."
    )
    k: int = Field(default=2, description="Number of clusters to use.")
    init: Literal["k-means++", "random"] = Field(
        default="k-means++", description="Initialization method for centroids."
    )
    max_iter: int = Field(
        default=300, description="Maximum number of iterations for the algorithm."
    )
    n_init: int = Field(default=10, description="Number of initializations to perform.")
    tol: float = Field(default=1e-4, description="Relative tolerance for convergence.")
    random_state: Optional[int] = Field(
        default=42, description="Random seed for reproducibility."
    )

    # Attributes populated after clustering
    labels: Optional[list[str]] = None
    cluster_assignments: Optional[np.ndarray] = None
    fig: Optional[go.Figure] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data) -> None:
        """Initialize KMeans clustering with the provided parameters."""
        super().__init__(**data)

        # Get a valid matrix from the input DTM or DataFrame
        matrix = self._get_valid_matrix()

        if self.k is None:
            raise LexosException(
                "Number of clusters 'k' must be specified for KMeans clustering."
            )
        try:
            _kmeans = sklearn_KMeans(
                n_clusters=self.k,
                init=self.init,
                max_iter=self.max_iter,
                n_init=self.n_init,
                tol=self.tol,
                random_state=self.random_state,
            )
        except Exception as e:
            raise LexosException(f"KMeans clustering failed: {e}")
        try:
            self.cluster_assignments = _kmeans.fit_predict(matrix)
        except Exception as e:
            raise LexosException(f"KMeans clustering failed: {e}")

    def _get_valid_matrix(self) -> np.ndarray:
        """Convert the input into a valid NumPy matrix format.

        Supports DTM (Lexos), pandas DataFrame, or NumPy array.
        Raises an error for unsupported formats or too few documents.
        """
        if isinstance(self.dtm, DTM):
            df = self.dtm.to_df().T
        elif isinstance(self.dtm, pd.DataFrame):
            df = self.dtm.T
        elif isinstance(self.dtm, np.ndarray):
            df = pd.DataFrame(self.dtm)
        else:
            raise LexosException(
                "Unsupported input: must be DTM, DataFrame, or ndarray."
            )

        # Must have more than 1 document to cluster
        if df.shape[0] < 2:
            raise LexosException("Need at least 2 documents for clustering.")

        return df.values

    @validate_call(config=model_config)
    def elbow_plot(
        self,
        k_range: range = range(1, 10),
        show: bool = True,
        save_path: Optional[str] = None,
        return_knee: bool = False,
    ) -> Optional[int]:
        """Generate an elbow plot to help determine the optimal number of clusters (k).

        Args:
            k_range (range): Range of k values to evaluate.
            show (bool): Whether to display the plot interactively.
            save_path (Optional[str]): Optional file path to save the elbow plot.
            return_knee (bool): If True, return the detected elbow point (optimal k).

        Returns:
            Optional[int]: Optimal number of clusters, only if return_knee is True.
        """
        # Ensure valid matrix and k range based on document count
        matrix = self._get_valid_matrix()

        min_k = min(k_range)
        max_k = min(len(matrix), max(k_range))

        if min_k > max_k:
            raise LexosException(
                f"Invalid k range ({min_k}–{max(k_range)}) exceeds document count ({len(matrix)})."
            )

        adjusted_range = range(min_k, max_k + 1)
        msg.info(
            f"Running elbow plot for k = {min_k} to {max_k} (limited to document count)"
        )

        # Run KMeans for each k in the specified range and record inertia
        inertias = []
        for k in adjusted_range:
            try:
                model = sklearn_KMeans(
                    n_clusters=k,
                    init=self.init,
                    max_iter=self.max_iter,
                    n_init=self.n_init,
                    tol=self.tol,
                    random_state=42,
                )
                model.fit(matrix)
                inertias.append(model.inertia_)
            except Exception as e:
                raise LexosException(f"Error fitting KMeans for k={k}: {e}")

        # Use the "maximum distance to line" method to detect elbow
        point1 = np.array([adjusted_range[0], inertias[0]])
        point2 = np.array([adjusted_range[-1], inertias[-1]])

        def distance_to_line(p):
            return np.linalg.norm(
                np.cross(point2 - point1, point1 - p)
            ) / np.linalg.norm(point2 - point1)

        distances = [
            distance_to_line(np.array([k, inertia]))
            for k, inertia in zip(adjusted_range, inertias)
        ]
        optimal_k = adjusted_range[np.argmax(distances)]

        # Plot inertia vs. number of clusters and show elbow with vertical line
        plt.figure(figsize=(8, 5))
        plt.plot(list(adjusted_range), inertias, marker="o", label="Inertia")
        plt.axvline(
            optimal_k, color="red", linestyle="--", label=f"Elbow at k={optimal_k}"
        )
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Inertia (Within-cluster Sum of Squares)")
        plt.title("Elbow Method for Optimal k")
        plt.grid(True)
        plt.legend()

        if save_path:
            plt.savefig(save_path)

        if show:
            plt.show()
        else:
            plt.close()

        if return_knee:
            return optimal_k

    @validate_call(config=model_config)
    def save(self, path: str | Path, html: bool = False, **kwargs: Any) -> None:
        """Save the most recent Plotly figure to an image or HTML file.

        Args:
            path (str | Path): Path to the output image file.
            html (bool): If True, save as HTML; otherwise, save as image.
            **kwargs (Any): Additional parameters for saving the figure. See https://plotly.github.io/plotly.py-docs/generated/plotly.io.write_image.html and https://plotly.github.io/plotly.py-docs/generated/plotly.io.write_html.html.
        """
        if self.fig is None:
            raise LexosException("No figure available: run a plot method first.")
        if html:
            self.fig.write_html(path, **kwargs)
        else:
            self.fig.write_image(path, **kwargs)

    @validate_call(config=model_config)
    def scatter(
        self,
        dim: int = 2,
        title: Optional[str] = None,
        show: bool = False,
        save_path: Optional[str | Path] = None,
        **kwargs: Any,
    ) -> Optional[go.Figure]:
        """Generate a 2D or 3D PCA scatter plot of the KMeans clusters.

        Args:
            show (bool): Whether to display the plot.
            dim: (int): The number of dimensions.
            title (Optional[str]): Optional title for the plot.
            save_path (Optional[str | Path]): Optional file path to save the plot.
            **kwargs (Any): Additional parameters for saving the figure. See https://plotly.github.io/plotly.py-docs/generated/plotly.io.write_image.html.

        Returns:
            go.Figure: The Plotly 3D scatter plot.
        """
        if self.cluster_assignments is None:
            raise LexosException("You must run clustering before plotting.")

        if dim not in [2, 3]:
            raise LexosException("The number of dimensions must be either 2 or 3.")

        # Reduce dimensions for plot
        matrix = self._get_valid_matrix()
        try:
            pca = PCA(n_components=dim)
        except ValueError as e:
            raise LexosException(f"Failed to perform PCA: {e}")
        try:
            reduced = pca.fit_transform(matrix)
        except Exception as e:
            raise LexosException(f"Failed to reduce dimensions: {e}")

        # Start cluster numbering from 1 for display
        cluster_assignments = [str(i + 1) for i in self.cluster_assignments]

        # Build DataFrame for plotting
        if dim == 2:
            df = pd.DataFrame(
                {
                    "x": reduced[:, 0],
                    "y": reduced[:, 1],
                    "Cluster": cluster_assignments,
                    "Document": self.labels
                    or [f"Doc{i + 1}" for i in range(len(matrix))],
                }
            )
        else:
            df = pd.DataFrame(
                {
                    "x": reduced[:, 0],
                    "y": reduced[:, 1],
                    "z": reduced[:, 2],
                    "Cluster": cluster_assignments,
                    "Document": self.labels
                    or [f"Doc{i + 1}" for i in range(len(matrix))],
                }
            )

        # Create scatter plot
        if dim == 2:
            fig = px.scatter(
                df,
                x="x",
                y="y",
                color="Cluster",
                hover_name="Document",
                title=title,
            )
        else:
            fig = px.scatter_3d(
                df,
                x="x",
                y="y",
                z="z",
                color="Cluster",
                hover_name="Document",
                title=title,
            )

        # Update the layout
        fig.update_layout(margin=dict(l=12, r=10, t=40, b=10))

        # Assign the figure to the instance attribute
        self.fig = fig

        # Save the figure if requested
        if save_path:
            fig.write_image(save_path, **kwargs)

        # Show the figure if requested
        if show:
            config = dict(
                displaylogo=False,
                modeBarButtonsToRemove=["toggleSpikelines"],
                scrollZoom=True,
            )
            fig.show(config=config)
            return None

        # Otherwise, return the figure
        else:
            return fig

    @validate_call(config=model_config)
    def to_csv(self, path: str | Path, **kwargs: Any) -> None:
        """Export a CSV of PCA coordinates and cluster labels.

        Args:
            path (str | Path): File path to save the CSV.
            **kwargs (Any): Additional parameters for pandas DataFrame.to_csv().
        """
        if self.cluster_assignments is None:
            raise LexosException("No clustering results: run clustering first.")

        # Perform PCA to 2 components
        matrix = self._get_valid_matrix()
        pca = PCA(n_components=2)
        coords = pca.fit_transform(matrix)

        # Create output DataFrame
        df = pd.DataFrame(
            {
                "Document": self.labels or [f"Doc{i + 1}" for i in range(len(coords))],
                "Cluster": self.cluster_assignments.astype(str),
                "PC1": coords[:, 0],
                "PC2": coords[:, 1],
            }
        )

        # Export to CSV
        try:
            df.to_csv(path, index=False)
        except Exception as e:
            raise LexosException(f"Failed to export CSV: {e}")

    @validate_call(config=model_config)
    def voronoi(
        self,
        title: Optional[str] = None,
        show: bool = True,
        save_path: Optional[str | Path] = None,
        grid_step: Optional[float] = None,
        max_points: int = 200_000,
        **kwargs: Any,
    ) -> Optional[go.Figure]:
        """Plot Voronoi-like decision regions for KMeans clustering using 2D PCA.

        Args:
            title (Optional[str]): Optional title for the plot.
            show (bool): Whether to display the plot interactively.
            save_path (Optional[str | Path]): File path to save the plot.
            grid_step (Optional[float]): Grid step size; estimated if None.
            max_points (int): Maximum grid points for memory efficiency.
            **kwargs (Any): Additional parameters for saving the figure. See https://plotly.github.io/plotly.py-docs/generated/plotly.io.write_image.html.
        """
        # Reduce dimensions for 2D Voronoi visualization
        matrix = self._get_valid_matrix()
        try:
            pca = PCA(n_components=2)
        except ValueError as e:
            raise LexosException(f"Failed to perform PCA: {e}")
        try:
            reduced = pca.fit_transform(matrix)
        except Exception as e:
            raise LexosException(f"Failed to reduce dimensions: {e}")

        if self.k is None:
            raise LexosException(
                "Number of clusters 'k' must be specified for KMeans clustering."
            )

        # Fit KMeans on reduced data for plotting
        kmeans = sklearn_KMeans(
            n_clusters=self.k,
            init=self.init,
            max_iter=self.max_iter,
            n_init=self.n_init,
            tol=self.tol,
            random_state=42,
        ).fit(reduced)

        centroids = kmeans.cluster_centers_

        # Define grid boundaries with buffer
        x_min, x_max = reduced[:, 0].min() - 1, reduced[:, 0].max() + 1
        y_min, y_max = reduced[:, 1].min() - 1, reduced[:, 1].max() + 1

        # Estimate grid resolution to avoid memory overload
        if grid_step is None:
            range_area = (x_max - x_min) * (y_max - y_min)
            grid_step = (range_area / max_points) ** 0.5
            msg.info(
                f"Grid step auto-adjusted to {grid_step:.2f} to avoid memory overload."
            )

        # Create mesh grid and predict cluster for each point
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, grid_step), np.arange(y_min, y_max, grid_step)
        )
        grid = np.c_[xx.ravel(), yy.ravel()]
        z = kmeans.predict(grid).reshape(xx.shape)

        fig = go.Figure()

        # Add background colored Voronoi regions
        fig.add_trace(
            go.Heatmap(
                x=xx[0],
                y=yy[:, 0],
                z=z,
                colorscale="YlGnBu",
                showscale=False,
                opacity=0.4,
            )
        )

        # Overlay documents per cluster
        doc_labels = np.array(
            self.labels or [f"Doc{i + 1}" for i in range(len(reduced))]
        )
        for i in range(self.k):
            cluster_mask = self.cluster_assignments == i
            fig.add_trace(
                go.Scatter(
                    x=reduced[cluster_mask, 0],
                    y=reduced[cluster_mask, 1],
                    mode="markers",
                    name=f"Cluster {i + 1}",
                    text=doc_labels[cluster_mask],
                    hovertemplate="%{text}<extra></extra>",
                    marker=dict(size=8),
                )
            )

        # Add centroid markers
        fig.add_trace(
            go.Scatter(
                x=centroids[:, 0],
                y=centroids[:, 1],
                mode="markers+text",
                name="Centroids",
                text=[f"C{i + 1}" for i in range(self.k)],
                hoverinfo="text",
                textposition="top center",
                marker=dict(symbol="x", size=14, color="black"),
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="PC1",
            yaxis_title="PC2",
        )

        self.fig = fig
        if save_path:
            fig.write_image(save_path, **kwargs)
        if show:
            config = dict(
                displaylogo=False,
                modeBarButtonsToRemove=["toggleSpikelines"],
                scrollZoom=True,
            )
            fig.show(config=config)
            return None
        else:
            return fig
