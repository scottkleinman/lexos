"""plotly_dendrogram.py.

Last Updated: July 7, 2025
Last Tested: December 5, 2025

Information here about how to add truncate mode: https://stackoverflow.com/questions/70801281/how-can-i-plot-a-truncated-dendrogram-plot-using-plotly
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np  # Added numpy import
import pandas as pd
import scipy.cluster.hierarchy as sch
from numpy.typing import ArrayLike
from plotly.figure_factory import create_dendrogram
from plotly.graph_objs.graph_objs import Figure, Scatter
from pydantic import BaseModel, ConfigDict, Field, validate_call
from scipy.spatial.distance import pdist

from lexos.dtm import DTM
from lexos.exceptions import LexosException


class PlotlyDendrogram(BaseModel):
    """PlotlyDendrogram.

    Typical usage:

    ```python
    from lexos.cluster import PlotlyDendrogram

    dendrogram = PlotlyDendrogram(dtm)
    dendrogram.show()

    Needs some work in returning the figure as a figure
    and html and html div.
    ```
    """

    dtm: Optional[ArrayLike | DTM | pd.DataFrame] = Field(
        None, json_schema_extra={"The document-term matrix."}
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
    get_leaves: Optional[bool] = Field(
        True, description="The get leaves for the dendrogram."
    )
    orientation: Optional[str] = Field(
        "bottom", description="The orientation for the dendrogram."
    )
    title: Optional[str] = Field(None, description="The title for the dendrogram.")
    figsize: Optional[tuple] = Field(
        (10, 10), description="The figsize for the dendrogram."
    )
    colorscale: Optional[list] = Field(
        None, description="The colorscale for the dendrogram."
    )
    hovertext: Optional[list] = Field(
        None, description="The hovertext for the dendrogram."
    )
    color_threshold: Optional[float] = Field(
        None, description="The color threshold for the dendrogram."
    )
    config: Optional[dict] = Field(
        dict(
            displaylogo=False,
            modeBarButtonsToRemove=["toggleSpikelines"],
            scrollZoom=True,
        ),
        description="The config for the dendrogram.",
    )
    x_tickangle: Optional[int] = Field(
        0, description="The x tickangle for the dendrogram."
    )
    y_tickangle: Optional[int] = Field(
        0, description="The y tickangle for the dendrogram."
    )
    fig: Optional[Figure] = Field(None, description="The figure for the dendrogram.")
    layout: Optional[dict] = Field(
        {},
        description="The layout for the dendrogram. Keywords and values to be passed to plotly.graph_objects.Figure.update_layout().",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        """Initialize the PlotlyDendrogram instance."""
        super().__init__(**data)

        def distfun(x: ArrayLike) -> ArrayLike:
            """Get the pairwise distance matrix.

            Args:
                x (ArrayLike): The distance matrix.

            Returns:
                ArrayLike: The pairwise distance matrix.
            """
            return pdist(x, metric=self.metric)

        def linkagefun(x: ArrayLike) -> ArrayLike:
            """Get the hierarchical clustering encoded as a linkage matrix.

            Args:
                x (ArrayLike): The pairwise distance matrix.

            Returns:
                ArrayLike: The linkage matrix.
            """
            return sch.linkage(x, self.method)

        # Ensure there is a document-term matrix
        if self.dtm is None:
            raise LexosException("You must provide a document-term matrix.")

        # Get the matrix based on the data type
        matrix = self._get_valid_matrix()

        # Ensure there are labels (moved after matrix validation to get correct shape)
        if not self.labels:
            if isinstance(self.dtm, DTM):
                self.labels = self.dtm.labels
            elif isinstance(self.dtm, pd.DataFrame):
                # Corrected to use index for labels, as rows are documents after transpose
                self.labels = self.dtm.index.tolist()
            else:  # If matrix is a numpy array or list (now handled by _get_valid_matrix converting to numpy)
                self.labels = [f"Doc{i + 1}" for i in range(matrix.shape[0])]

        # Create the figure
        self.fig = create_dendrogram(
            matrix,
            labels=self.labels,
            distfun=distfun,
            linkagefun=linkagefun,
            orientation=self.orientation,
            colorscale=self.colorscale,
            hovertext=self.hovertext,
            color_threshold=self.color_threshold,
        )

        # Set the standard layout
        self.fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=0, pad=0),
            hovermode="x",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            xaxis=dict(
                showline=False, ticks="", tickangle=self.x_tickangle, automargin=True
            ),
            yaxis=dict(
                showline=False, ticks="", tickangle=self.y_tickangle, automargin=True
            ),
        )

        # Set the title
        if self.title is not None:
            title = dict(
                text=self.title, x=0.5, y=0.95, xanchor="center", yanchor="top"
            )
            self.fig.update_layout(title=title, margin=dict(t=40))

        # Add user-configured layout
        self.fig.update_layout(self.layout)

        # Hack to ensure that leaves on the edge of the plot are not clipped.
        # Only works for bottom and left orientation.
        # Add an invisible scatter point to extend the margin.
        if self.orientation in ["bottom", "left"]:
            x_value = max([max(data["x"]) for data in self.fig.data])
            dummy_scatter = Scatter(
                x=[x_value], y=[0], mode="markers", opacity=0, hoverinfo="skip"
            )
            self.fig.add_trace(trace=dummy_scatter)

        # Move labels for top and right orientation
        if self.orientation == "top":
            self.fig.update_xaxes(side="top")
        if self.orientation == "right":
            self.fig.update_yaxes(side="right")

        plt.close()

    def _get_valid_matrix(self):
        """Get a valid matrix based on the data type of the dtm."""  # End of _get_valid_matrix docstring
        if isinstance(self.dtm, DTM):
            matrix = self.dtm.to_df()
            matrix.index.name = "terms"
            matrix = matrix.T
        elif isinstance(self.dtm, list):  # Added handling for list input
            matrix = np.array(self.dtm)  # Convert list to numpy array
        else:
            matrix = self.dtm

        # Now, `matrix` will always be a pandas DataFrame or a numpy array when we check `shape`
        if matrix.shape[0] < 2:
            raise LexosException(
                "The document-term matrix must have more than one document."
            )
        return matrix

    def show(self) -> None:
        """Show the figure."""  # End of show docstring
        if self.fig is None:
            raise LexosException(
                "You must call the instance before showing the figure."
            )
        self.fig.show(config=self.config)

    def to_html(self, **kwargs):
        """Create an HTML representation of the figure.

        Wrapper from the Plotly Figure to_html method.
        See https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html.
        """  # End of to_html docstring
        if self.fig is None:
            raise LexosException("You must call the instance before generating HTML.")
        return self.fig.to_html(**kwargs)

    def to_image(self, **kwargs):
        """Create a static image of the figure.

        Wrapper from the Plotly Figure to_image method.
        See https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html.
        """  # End of to_image docstring
        if self.fig is None:
            raise LexosException(
                "You must call the instance before generating an image."
            )
        return self.fig.to_image(**kwargs)

    @validate_call(config=model_config)
    def write_html(self, path: Path | str, **kwargs):
        """Save an HTML representation of the figure to disk.

        Wrapper from the Plotly Figure write_html method.
        See https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html.
        """
        if self.fig is None:
            raise LexosException("You must call the instance before saving the figure.")
        # Removed: if "file" in kwargs: kwargs["file"] = path
        self.fig.write_html(
            str(path), **kwargs
        )  # Convert path to string for write_html

    @validate_call(config=model_config)
    def write_image(self, path: Path | str, **kwargs):
        """Save a static image of the figure to disk.

        Wrapper from the Plotly Figure write_image method.
        See https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html.
        """
        if self.fig is None:
            raise LexosException("You must call the instance before saving the figure.")
        # Removed: if "file" in kwargs: kwargs["file"] = path
        self.fig.write_image(
            str(path), **kwargs
        )  # Convert path to string for write_image
