"""clustermap.py."""

from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import figure

sns.set_theme()


class ClusterMap:
    """ClusterMap."""

    def __init__(
        self,
        dtm: Any,
        z_score: int = 1,
        labels: List[str] = None,
        pivot_kws: Dict[str, str] = None,
        method: str = "average",
        metric: str = "euclidean",
        standard_scale: int = None,
        figsize: tuple = (8, 8),
        cbar_kws: dict = None,
        row_cluster: bool = True,
        col_cluster: bool = True,
        row_linkage: bool = None,
        col_linkage: bool = None,
        row_colors: Union[list, pd.DataFrame, pd.Series] = None,
        col_colors: Union[list, pd.DataFrame, pd.Series] = None,
        mask: Union[np.ndarray, pd.DataFrame] = None,
        dendrogram_ratio: Union[float, Tuple[float]] = (0.1, 0.2),
        colors_ratio: float = 0.03,
        cbar_pos: Tuple[str] = (0.02, 0.32, 0.03, 0.2),
        tree_kws: dict = None,
        center: int = 0,
        cmap: str = "vlag",
        linewidths: float = 0.75,
        show: bool = False,
        title: str = None,
    ) -> figure.Figure:
        """Initialize the ClusterMap object.

        Args:
            dtm (Any): The data to cluster.
            z_score (int, optional): The z-score to use. Defaults to 1.
            labels (List[str], optional): The labels to use. Defaults to None.
            pivot_kws (Dict[str, str], optional): The pivot kwargs. Defaults to None.
            method (str, optional): The method to use. Defaults to "average".
            metric (str, optional): The metric to use. Defaults to "euclidean".
            standard_scale (int, optional): The standard scale to use. Defaults to None.
            figsize (tuple, optional): The figure size to use. Defaults to (8, 8).
            cbar_kws (dict, optional): The cbar kwargs. Defaults to None.
            row_cluster (bool, optional): Whether to cluster the rows. Defaults to True.
            col_cluster (bool, optional): Whether to cluster the columns. Defaults to True.
            row_linkage (bool, optional): Whether to use row linkage. Defaults to None.
            col_linkage (bool, optional): Whether to use column linkage. Defaults to None.
            row_colors (Union[list, pd.DataFrame, pd.Series], optional): The row colors. Defaults to None.
            col_colors (Union[list, pd.DataFrame, pd.Series], optional): The column colors. Defaults to None.
            mask (Union[np.ndarray, pd.DataFrame], optional): The mask to use. Defaults to None.
            dendrogram_ratio (Union[float, Tuple[float]], optional): The dendrogram ratio to use. Defaults to (.1, .2).
            colors_ratio (float, optional): The colors ratio to use. Defaults to 0.03.
            cbar_pos (Tuple[str], optional): The cbar position to use. Defaults to (.02, .32, .03, .2).
            tree_kws (dict, optional): The tree kwargs. Defaults to None.
            center (int, optional): The center to use. Defaults to 0.
            cmap (str, optional): The cmap to use. Defaults to "vlag".
            linewidths (float, optional): The linewidths to use. Defaults to .75.
            show (bool, optional): Whether to show the figure. Defaults to False.
            title (str, optional): The title to use. Defaults to None.

        Returns:
            matplotlib.figure.Figure: The figure.
        """
        self.dtm = dtm
        self.z_score = z_score
        self.labels = labels
        self.figsize = figsize
        self.show = show
        self.method = method
        self.metric = metric
        self.standard_scale = standard_scale
        self.title = title
        self.row_colors = row_colors
        self.col_colors = col_colors
        self.pivot_kws = pivot_kws
        self.cbar_kws = cbar_kws
        self.row_cluster = row_cluster
        self.col_cluster = col_cluster
        self.row_linkage = row_linkage
        self.col_linkage = col_linkage
        self.mask = mask
        self.dendrogram_ratio = dendrogram_ratio
        self.colors_ratio = colors_ratio
        self.cbar_pos = cbar_pos
        self.tree_kws = tree_kws
        self.center = center
        self.cmap = cmap
        self.linewidths = linewidths
        self.df = dtm.get_table()
        self.build()

    def build(self):
        """Build the clustermap."""
        # Get the labels if necessary
        if self.labels is None:
            # raise ValueError("Please provide labels.")
            self.labels = self.df.columns.values.tolist()[1:]

        # Convert palette to vectors drawn on the side of the matrix
        if self.row_colors is None or self.col_colors is None:
            column_pal = sns.husl_palette(8, s=0.45)
            column_lut = dict(zip(map(str, self.df), column_pal))
            column_colors = pd.Series(self.labels, index=self.df.columns[1:]).map(
                column_lut
            )
            if self.row_colors is None:
                self.row_colors = column_colors
            if self.col_colors is None:
                self.col_colors = column_colors

        # Perform the cluster
        g = sns.clustermap(
            self.df.corr(),
            cmap=self.cmap,
            method=self.method,
            metric=self.metric,
            figsize=self.figsize,
            row_colors=self.row_colors,
            col_colors=self.col_colors,
            row_cluster=self.row_cluster,
            col_cluster=self.col_cluster,
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

        # Remove the dendrogram on the left
        g.ax_row_dendrogram.remove()

        # Add the title
        if self.title:
            g.fig.suptitle(self.title, y=1.05)

        # Save the fig variable
        self.fig = g.fig

        # Suppress the output
        if not self.show:
            plt.close()
            return self.fig

    def savefig(self, filename: str):
        """Show the figure if it is hidden.

        Args:
            filename (str): The name of the file to save.
        """
        self.fig.savefig(filename)

    def showfig(self):
        """Show the figure if it is hidden.

        This is a helper method. You can also reference the figure
        using `ClusterMap.fig`. This will generally display in a
        Jupyter notebook.
        """
        return self.fig
