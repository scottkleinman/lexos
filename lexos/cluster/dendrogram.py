"""dendrogram.py."""

from typing import Any, Callable, List, Union

import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist


class Dendrogram():
    """Dendrogram.

    Typical usage:

    ```python
    from lexos.cluster.dendrogram import Dendrogram

    dendrogram = Dendrogram(dtm, show=True)

    or

    dendrogram = Dendrogram(dtm, show=True)
    dendrogram.fig
    ```
    """

    def __init__(
        self,
        dtm: Any,
        labels: List[str] = None,
        metric: str = "euclidean",
        method: str = "average",
        truncate_mode: str = None,
        color_threshold: str = None,
        get_leaves: bool = True,
        orientation: str = "top",
        count_sort: Union[bool, str] = None,
        distance_sort: Union[bool, str] = None,
        show_leaf_counts: bool = False,
        no_plot: bool = False,
        no_labels: bool = False,
        leaf_rotation: int = 90,
        leaf_font_size: int = None,
        leaf_label_func: Callable = None,
        show_contracted: bool = False,
        link_color_func: Callable = None,
        ax = None,
        above_threshold_color: str = "C0",
        title: str = None,
        figsize: tuple = (10, 10),
        show: bool = False
    ) -> dict:
        """Initialise the Dendrogram."""
        # Create an empty plot for matplotlib
        self.dtm = dtm
        self.labels = labels
        self.metric = metric
        self.method = method
        self.truncate_mode = truncate_mode
        self.color_threshold = color_threshold
        self.get_leaves = get_leaves
        self.orientation = orientation
        self.count_sort = count_sort
        self.distance_sort = distance_sort
        self.show_leaf_counts = show_leaf_counts
        self.no_plot = no_plot
        self.no_labels = no_labels
        self.leaf_rotation = leaf_rotation
        self.leaf_font_size = leaf_font_size
        self.leaf_label_func = leaf_label_func
        self.show_contracted = show_contracted
        self.link_color_func = link_color_func
        self.ax = ax
        self.above_threshold_color = above_threshold_color
        self.title = title
        self.figsize = figsize
        self.show = show

        # Get the dtm table
        self.dtm_table = dtm.get_table()

        # Use default labels from the DTM table
        if self.labels is None:
            self.labels = self.dtm_table.columns.values.tolist()[1:]

        # Set "terms" as the index and transpose the table
        self.dtm_table = self.dtm_table.set_index("terms").T

        # Build the dendrogram
        self.build()

    def build(self):
        """Build a dendrogram."""
        # Create the distance and linkage matrixes for matplotlib
        X = pdist(self.dtm_table, metric=self.metric)
        Z = sch.linkage(X, self.method)
        fig, ax = plt.subplots(figsize=self.figsize)
        if self.title:
            plt.title(self.title)
        sch.dendrogram(
            Z,
            labels=self.labels,
            truncate_mode=self.truncate_mode,
            color_threshold=self.color_threshold,
            get_leaves=self.get_leaves,
            orientation=self.orientation,
            count_sort=self.count_sort,
            distance_sort=self.distance_sort,
            show_leaf_counts=self.show_leaf_counts,
            no_plot=self.no_plot,
            no_labels=self.no_labels,
            leaf_rotation=self.leaf_rotation,
            leaf_font_size=self.leaf_font_size,
            leaf_label_func=self.leaf_label_func,
            show_contracted=self.show_contracted,
            link_color_func=self.link_color_func,
            ax=self.ax,
            above_threshold_color=self.above_threshold_color
        )
        self.fig = fig

        if not self.show:
            plt.close()

    def savefig(self, filename: str):
        """Show the figure if it is hidden.

        Args:
            filename (str): The name of the file to save.
        """
        self.fig.savefig(filename)

    def showfig(self):
        """Show the figure if it is hidden.

        This is a helper method. You can also reference the figure
        using `Dendrogram.fig`. This will generally display in a
        Jupyter notebook.
        """
        return self.fig