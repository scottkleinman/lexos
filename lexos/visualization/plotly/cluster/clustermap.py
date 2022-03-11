"""plotly_clustermap.py.

Typical usage:
    plotly_clustermap = PlotlyClustermap(dtm)
    plotly_clustermap.showfig()
"""

from typing import Any

import plotly.colors
import plotly.graph_objects as go
import scipy.cluster.hierarchy as sch
from plotly.figure_factory import create_dendrogram
from plotly.offline import plot as _plot
from scipy.spatial.distance import pdist, squareform


class PlotlyClustermap():
    """PlotlyClustermap."""

    def __init__(self,
                 dtm: Any,
                 metric: str = "euclidean",
                 method: str = "average",
                 hide_upper: bool = False,
                 hide_side: bool = False,
                 colorscale: str = "Viridis",
                 width: int = 600,
                 height: int = 600,
                 title: str = None,
                 config: dict = dict(
                     displaylogo=False,
                     modeBarButtonsToRemove=[
                        "toImage",
                        "toggleSpikelines"
                    ],
                    scrollZoom=True
                 ),
                 show: bool = False):
        """Initialise the Clustermap.

        Args:
            dtm (Any): The document-term-matrix
        """
        self.dtm = dtm
        table = dtm.get_table()
        self.labels = table.columns.values.tolist()[1:]
        self.df = table.set_index("terms").T
        self.metric = metric
        self.method = method
        self.hide_upper = hide_upper
        self.hide_side = hide_side
        self.colorscale = colorscale
        self.width = width
        self.height = height
        self.config = config
        self.title = title
        self.show = show
        self.build()

    def build(self) -> Any:
        """Build a clustermap."""
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

        # Initialize figure by creating upper dendrogram
        fig = create_dendrogram(self.df,
                                distfun=distfun,
                                linkagefun=linkagefun,
                                orientation="bottom",
                                labels=self.labels,
                                colorscale=self._get_colorscale(),
                                color_threshold=None)
        for i in range(len(fig["data"])):
            fig["data"][i]["yaxis"] = "y2"

        # Renders the upper dendrogram invisible
        # Also removes the labels, so you have to rely on hovertext
        if self.hide_upper:
            fig.for_each_trace(lambda trace: trace.update(visible=False))

        # Create Side Dendrogram
        dendro_side = create_dendrogram(self.df,
                                        distfun=distfun,
                                        linkagefun=linkagefun,
                                        orientation="right",
                                        colorscale=self._get_colorscale(),
                                        color_threshold=None)
        for i in range(len(dendro_side["data"])):
            dendro_side["data"][i]["xaxis"] = "x2"

        # Add Side Dendrogram Data to Figure
        if not self.hide_side:
            for data in dendro_side["data"]:
                fig.add_trace(data)

        # Create Heatmap
        dendro_leaves = dendro_side["layout"]["yaxis"]["ticktext"]
        dendro_leaves = list(map(int, dendro_leaves))
        data_dist = pdist(self.df)
        heat_data = squareform(data_dist)
        heat_data = heat_data[dendro_leaves, :]
        heat_data = heat_data[:, dendro_leaves]

        num = len(self.labels)
        heatmap = [
            go.Heatmap(
                x=dendro_leaves,
                y=dendro_leaves,
                z=heat_data,
                colorscale=self.colorscale,
                hovertemplate="X: %{x}<br>Y: %{customdata}<br>Z: %{z}<extra></extra>",
                customdata=[[label for x in range(num)] for label in self.labels]
            )
        ]

        heatmap[0]["x"] = fig["layout"]["xaxis"]["tickvals"]
        heatmap[0]["y"] = dendro_side["layout"]["yaxis"]["tickvals"]

        # Add Heatmap Data to Figure
        for data in heatmap:
            fig.add_trace(data)

        # Edit Layout
        fig.update_layout({"width": self.width, "height": self.height,
                           "showlegend": False, "hovermode": "closest",
                           })

        # Edit xaxis (dendrogram)
        if not self.hide_side:
            x = .15
        else:
            x = 0
        fig.update_layout(xaxis={"domain": [x, 1],
                                 "mirror": False,
                                 "showgrid": False,
                                 "showline": False,
                                 "zeroline": False,
                                 "ticks": ""})
        # Edit xaxis2 (heatmap)
        fig.update_layout(xaxis2={"domain": [0, .15],
                                  "mirror": False,
                                  "showgrid": False,
                                  "showline": False,
                                  "zeroline": False,
                                  "showticklabels": False,
                                  "ticks": ""})

        # Edit yaxis (heatmap)
        fig.update_layout(yaxis={"domain": [0, .85],
                                 "mirror": False,
                                 "showgrid": False,
                                 "showline": False,
                                 "zeroline": False,
                                 "showticklabels": False,
                                 "ticks": "",
                                 })
        # Edit yaxis2 (dendrogram)
        fig.update_layout(yaxis2={"domain": [.840, .975],
                                  "mirror": False,
                                  "showgrid": False,
                                  "showline": False,
                                  "zeroline": False,
                                  "showticklabels": False,
                                  "ticks": ""})

        fig.update_layout(margin=dict(l=0),
                          paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          xaxis_tickfont=dict(color="rgba(0,0,0,0)"))

        # Set the title
        if self.title:
            title = dict(
                text=self.title,
                x=0.5,
                y=0.95,
                xanchor="center",
                yanchor="top"
            )
            fig.update_layout(
                title=title,
                margin=dict(t=40)
            )

        # Save the figure variable
        self.fig = fig

        # Show the plot
        if self.show:
            self.fig.show(config=self.config)

    def _get_colorscale(self) -> list:
        """Get the colorscale as a list.

        Plotly continuous colorscales assign colors to the range [0, 1]. This function
        computes the intermediate color for any value in that range.

        Plotly doesn't make the colorscales directly accessible in a common format.
        Some are ready to use, and others are just swatche that need to be constructed
        into a colorscale.
        """
        try:
            colorscale = plotly.colors.PLOTLY_SCALES[self.colorscale]
        except ValueError:
            swatch = getattr(plotly.colors.sequential, self.colorscale)
            colors, scale = plotly.colors.convert_colors_to_same_type(swatch)
            colorscale = plotly.colors.make_colorscale(colors, scale=scale)
        return colorscale

    def savefig(self, filename: str):
        """Save the figure.

        Args:
            filename (str): The name of the file to save.
        """
        self.fig.savefig(filename)

    def showfig(self):
        """Show the figure."""
        self.fig.show(config=self.config)

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
