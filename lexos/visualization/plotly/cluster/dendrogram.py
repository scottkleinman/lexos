"""plotly_dendrogram.py."""

import math
from typing import Any, List

import scipy.cluster.hierarchy as sch
from plotly.figure_factory import create_dendrogram
from plotly.graph_objs.graph_objs import Figure, Scatter
from plotly.offline import plot as _plot
from scipy.spatial.distance import pdist


class PlotlyDendrogram():
    """PlotlyDendrogram.

    Typical usage:

    ```python
    from lexos.visualization.plotly.cluster.dendrogram import PlotlyDendrogram

    dendrogram = PlotlyDendrogram(dtm, show=True)

    or

    dendrogram = PlotlyDendrogram(dtm)
    dendrogram.fig


    Needs some work in returning the figure as a figure
    and html and html div.
    ```
    """

    def __init__(
        self,
        dtm: Any,
        labels: List[str] = None,
        metric: str = "euclidean",
        method: str = "average",
        truncate_mode: str = None,
        get_leaves: bool = True,
        orientation: str = "bottom",
        title: str = None,
        figsize: tuple = (10, 10),
        show: bool = False,
        colorscale: List = None,
        hovertext: List = None,
        color_threshold: float = None,
        config: dict = dict(
            displaylogo=False,
            modeBarButtonsToRemove=[
                "toImage",
                "toggleSpikelines"
            ],
            scrollZoom=True
        ),
        x_tickangle: int = 0,
        y_tickangle: int = 0,
        **layout
    ) -> dict:
        """Initialise the Dendrogram."""
        # Create an empty plot for matplotlib
        self.dtm = dtm
        self.labels = labels
        self.metric = metric
        self.method = method
        self.truncate_mode = truncate_mode
        self.get_leaves = get_leaves
        self.orientation = orientation
        self.title = title
        self.figsize = figsize
        self.show = show
        self.colorscale = colorscale
        self.hovertext = hovertext
        self.color_threshold = color_threshold
        self.config = config
        self.x_tickangle = x_tickangle
        self.y_tickangle = y_tickangle
        self.layout = layout

        # Get the dtm table
        self.df = self.dtm.get_table()

        # Use default labels from the DTM table
        if self.labels is None:
            self.labels = self.df.columns.values.tolist()[1:]

        # Set "terms" as the index and transpose the table
        self.df = self.df.set_index("terms").T

        # Build the dendrogram
        self.build()

    def build(self):
        """Build a dendrogram."""
        # Set the distance and linkage metrics
        def distfun(x):
            """Get the pairwise distance matrix.

            Args:
                x (Any): The distance matrix.

            Returns:
                Any: The pairwise distance matrix.
            """
            return pdist(x, metric=self.metric)

        def linkagefun(x):
            """Get the hierarchical clustering encoded as a linkage matrix.

            Args:
                x (Any): The pairwise distance matrix.

            Returns:
                Any: The linkage matrix.
            """
            return sch.linkage(x, self.method)

        # Create the figure
        self.fig = create_dendrogram(self.df,
                                labels=self.labels,
                                distfun=distfun,
                                linkagefun=linkagefun,
                                orientation=self.orientation,
                                colorscale=self.colorscale,
                                hovertext=self.hovertext,
                                color_threshold=self.color_threshold
                                )

        # Set the standard layout
        self.fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=0, pad=10),
            hovermode='x',
            paper_bgcolor="rgba(0, 0, 0, 0)",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            xaxis=dict(showline=False, ticks="", tickangle=self.x_tickangle),
            yaxis=dict(showline=False, ticks="", tickangle=self.y_tickangle)
        )

        # Set the title
        if self.title is not None:
            title = dict(
                text=self.title,
                x=0.5,
                y=0.95,
                xanchor="center",
                yanchor="top"
            )
            self.fig.update_layout(
                title=title,
                margin=dict(t=40)
            )

        # Add user-configured layout
        self.fig.update_layout(**self.layout)

        # Extend figure hack
        max_label_len = len(max(self.labels, key=len))
        self.fig = _extend_figure(
            self.fig,
            self.orientation,
            max_label_len
        )

        if self.show:
            self.fig.show(config=self.config)

    def showfig(self):
        """Show the figure.

        Calling `Dendrogram.fig` when the dendrogram has been set
        to `False` does not apply the config (there is no way to
        do this in Plotly. Calling `Dendrogram.showfig()` rebuilds
        the fig with the config applied.
        """
        self.show = True
        self.build()

    def to_html(self,
               show_link: bool = False,
               output_type: str = "div",
               include_plotlyjs: bool = False,
               filename: str = None,
               auto_open: bool = False,
               config: dict = None):
        """Convert the figure to HTML.
        Args:
            show_link (bool): For exporting to Plotly cloud services. Default is `False`.
            output_type (str): If `file`, then the graph is saved as a standalone HTML
                file and plot returns None. If `div`, then plot returns a string that
                just contains the HTML <div> that contains the graph and the script to
                generate the graph. Use `file` if you want to save and view a single
                graph at a time in a standalone HTML file. Use `div` if you are embedding
                these graphs in an existing HTML file. Default is `div`.
            include_plotlyjs (bool): If True, include the plotly.js source code in the
                output file or string, which is good for standalone web pages but makes
                for very large files. If you are embedding the graph in a webpage, it
                is better to import the plotly.js library and use a `div`. Default is `False`.
            filename (str): The local filename to save the outputted chart to. If the
                filename already exists, it will be overwritten. This argument only applies
                if output_type is `file`. The default is `temp-plot.html`.
            auto_open (bool): If True, open the saved file in a web browser after saving.
                This argument only applies if output_type is `file`. Default is `False`.
            config (dict): A dict of parameters in the object's configuration.

        Note:
            This method uses `plotly.offline.plot`, which no longer appears to be documented.
            It has been replaced by renderers: https://plotly.com/python/renderers/. However,
            there does not appear to be an HTML renderer, so no attempt has been made to
            use the new functionality.
        """
        if self.config:
            config = self.config

        if filename and output_type == "file":
            return _plot(
                self.fig,
                show_link=show_link,
                output_type="file",
                include_plotlyjs=include_plotlyjs,
                filename=filename,
                auto_open=auto_open,
                config=config
            )
        elif filename and output_type == "div":
            pl = _plot(
                self.fig,
                show_link=show_link,
                output_type="div",
                include_plotlyjs=include_plotlyjs,
                auto_open=auto_open,
                config=config
            )
            with open(filename, "w") as f:
                f.write(pl)
            return pl
        else:
            return _plot(
                self.fig,
                show_link=show_link,
                output_type="div",
                include_plotlyjs=include_plotlyjs,
                auto_open=auto_open,
                config=config
            )



# Plotly Figure Extension Hack
def _extend_figure(figure: Figure, orientation: str, max_label_len: int) -> Figure:
    """Extend the figure margins.

    Use this function to extend figure margins so that long label will not
    get cut off and the edging leafs will not touch the border of the plot.

    Args:
        figure (Figure): The figure to extend.
        orientation (str): The orientation of the dendrogram.
        max_label_len (int): The length of the longest label.

    Returns:
        figure (Figure): The extended figure.

    !!! note
        The extend figure function is a hack. If plotly develops better solutions,
        this method will be removed.
        In addition, the magic numbers in this function are based on some tests,
        which may not be reliable for all use cases. If possible, they should
        be replaced ASAP to adjust the figure style based on the selected
        orientation.
    """
    if orientation not in ["bottom", "left"]:
        raise ValueError("Invalid orientation.")
    # Extend the bottom margin to fit all labels.
    elif orientation == "bottom":
        figure.layout.update({'margin': {'b': max_label_len * 6}})
        # Calculate the space right most label needs.
        right_margin = len(figure.layout.xaxis.ticktext[-1]) * 5 \
            if len(figure.layout.xaxis.ticktext[-1]) * 5 > 100 else 100
        # Update right margin as well.
        figure.layout.update({"margin": {"r": right_margin}})
        # Find the max x value in the plot.
        max_x = max([max(data["x"]) for data in figure.data])
        # Calculate proper x coordinate the figure should extend to.
        x_value = max_x + 5
    # Extend the left margin to fit all labels.
    else:
        figure.layout.update({'margin': {'l': max_label_len * 11}})
        # Find the max x value in the plot.
        max_x = max([max(data["x"]) for data in figure["data"]])
        # Calculate proper x coordinate the figure should extend to.
        x_value = math.ceil(max_x * 100) / 100
    # Get the dummy scatter plot.
    dummy_scatter = _get_dummy_scatter(x_value=x_value)
    # Add dummy scatter to the figure.
    figure.add_trace(trace=dummy_scatter)
    # Return the formatted figure.
    return figure


def _get_dummy_scatter(x_value: float) -> Scatter:
    """Create a invisible scatter point at (x_value, 0).

    Use this function to help extend the margin of the dendrogram plot.

    Args:
        x_value (float): The desired x value we want to extend the margin to.

    Returns:
        tuple: An invisible scatter point at (x_value, 0).
    """
    return Scatter(
        x=[x_value],
        y=[0],
        mode="markers",
        opacity=0,
        hoverinfo="skip"
    )
