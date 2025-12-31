"""dendrogram.py.

Last Updated: July 25, 2025
Last Tested: December 5, 2025
"""

from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from matplotlib.axes import Axes
from numpy.typing import ArrayLike
from pydantic import BaseModel, ConfigDict, Field, validate_call
from scipy.spatial.distance import pdist

from lexos.dtm import DTM
from lexos.exceptions import LexosException


class Dendrogram(BaseModel):
    """Dendrogram.

    Typical usage:

    ```python
    from lexos.cluster import Dendrogram

    dendrogram = Dendrogram(dtm=dtm)
    dendrogram.show()
    ```

    The dtm parameter can be a a DTM instance or a pandas DataFrame with terms
    as columns and docs as rows (the output of `DTM.to_df(transpose=True)`).
    It can also be an equivalent numpy array or list of lists. But in most cases,
    it will be most convenient to use a DTM instance.
    """

    dtm: Optional[ArrayLike | DTM | pd.DataFrame] = Field(
        None, description="The document-term matrix."
    )
    labels: Optional[list[str]] = Field(
        None, description="The labels for the dendrogram."
    )
    metric: Optional[str] = Field(
        "euclidean", description="The metric to use for the dendrogram."
    )
    method: Optional[str] = Field(
        "average", description="The method to use for the dendrogram."
    )
    truncate_mode: Optional[str] = Field(
        None, description="The truncate mode for the dendrogram."
    )
    color_threshold: Optional[float] = Field(
        None, description="The color threshold for the dendrogram."
    )
    get_leaves: Optional[bool] = Field(
        True, description="The get leaves for the dendrogram."
    )
    orientation: Optional[str] = Field(
        "top", description="The orientation for the dendrogram."
    )
    count_sort: Optional[bool | str] = Field(
        None, description="The count sort for the dendrogram."
    )
    distance_sort: Optional[bool | str] = Field(
        None, description="The distance sort for the dendrogram."
    )
    show_leaf_counts: Optional[bool] = Field(
        False, description="The show leaf counts for the dendrogram."
    )
    no_plot: Optional[bool] = Field(
        False, description="The no plot for the dendrogram."
    )
    no_labels: Optional[bool] = Field(
        False, description="The no labels for the dendrogram."
    )
    leaf_rotation: Optional[int] = Field(
        90, description="The leaf rotation for the dendrogram."
    )
    leaf_font_size: Optional[int] = Field(
        None, description="The leaf font size for the dendrogram."
    )
    leaf_label_func: Optional[Callable] = Field(
        None, description="The leaf label function for the dendrogram."
    )
    show_contracted: Optional[bool] = Field(
        False, description="The show contracted for the dendrogram."
    )
    link_color_func: Optional[Callable] = Field(
        None, description="The link color function for the dendrogram."
    )
    ax: Optional[Axes] = Field(None, description="The ax for the dendrogram.")
    above_threshold_color: Optional[str] = Field(
        "C0", description="The above threshold color for the dendrogram."
    )
    title: Optional[str] = Field(None, description="The title for the dendrogram.")
    figsize: Optional[tuple] = Field(
        (10, 10), description="The figsize for the dendrogram."
    )
    fig: Optional[plt.Figure] = Field(
        None, description="The figure for the dendrogram."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data) -> None:
        """Initialize the Dendrogram instance."""
        super().__init__(**data)

        # Ensure there is a document-term matrix with more than one document
        if self.dtm is None:
            raise LexosException("You must provide a document-term matrix.")

        # Ensure there are labels
        if not self.labels:
            if isinstance(self.dtm, DTM):
                self.labels = self.dtm.labels
            elif isinstance(self.dtm, pd.DataFrame):
                self.labels = self.dtm.columns.values.tolist()
            else:
                self.labels = [f"Doc{i + 1}" for i, _ in enumerate(self.dtm)]

        # Get the matrix based on the data type
        matrix = self._get_valid_matrix()

        # Check to see if the number of labels matches the number of rows
        # Make sure we have a matrix length for list input
        if isinstance(matrix, list):
            matrix_length = len(matrix)
        else:
            matrix_length = matrix.shape[0]
        if len(self.labels) != matrix_length:
            raise LexosException(
                "The number of labels must match the number of documents."
            )

        # Generate the pairwise distance and linkage matrices
        X = pdist(matrix, metric=self.metric)
        Z = sch.linkage(X, self.method)

        # Generate the dendrogram
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
            above_threshold_color=self.above_threshold_color,
        )
        self.fig = fig
        plt.close()

    def _get_valid_matrix(self):
        """Get a valid matrix based on the data type of the dtm."""
        shape_error = "The document-term matrix must have more than one document."
        type_error = "The document-term matrix must contain only numeric values."
        label_error = "The number of labels must match the number of documents."

        # DTM input
        if isinstance(self.dtm, DTM):
            if len(self.dtm.labels) < 2:
                raise LexosException(shape_error)
            df = self.dtm.to_df()
            df.index.name = "terms"
            matrix = df.T

        # DataFrame input
        elif isinstance(self.dtm, pd.DataFrame):
            if self.dtm.shape[0] < 3:
                raise LexosException(shape_error)
            if not np.issubdtype(self.dtm.values.dtype, np.number):
                raise LexosException(type_error)
            matrix = self.dtm

        # Raw array/list input
        else:
            matrix = self.dtm

            # List input
            if isinstance(matrix, list):
                if len(matrix) < 2:
                    raise LexosException(shape_error)
                if not all(isinstance(x, (int, float)) for row in matrix for x in row):
                    raise LexosException(type_error)

            # NumPy array input
            elif isinstance(matrix, np.ndarray):
                # Consolidated NumPy array checks
                if matrix.shape[0] < 2:
                    raise LexosException(shape_error)
                if not np.issubdtype(matrix.dtype, np.number):
                    raise LexosException(type_error)
            # You might want an 'else' here if there are other unsupported types
            else:
                raise LexosException("Unsupported document-term matrix type.")

        # Make sure we have a matrix length for list input
        if isinstance(matrix, list):
            matrix_length = len(matrix)
        else:
            matrix_length = matrix.shape[0]

        # Check labels vs matrix row count
        if self.labels and len(self.labels) != matrix_length:
            raise LexosException(label_error)

        return matrix

    @validate_call
    def save(self, path: Path | str):
        """Save the figure as a file.

        Args:
            path (Path | str): The path to the file to save.
        """
        if not path or path == "":
            raise LexosException("You must provide a valid path.")
        self.fig.savefig(path)

    def show(self):
        """Show the figure if it is hidden.

        This is a helper method. You can also reference the figure
        using `Dendrogram.fig`. This will generally display in a
        Jupyter notebook.
        """
        return self.fig
