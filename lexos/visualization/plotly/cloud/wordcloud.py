"""wordcloud.py."""

from typing import List, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode

init_notebook_mode(connected=True)
from wordcloud import WordCloud


def make_wordcloud(data: Union[dict, list, object, str, tuple],
                   opts: dict = None,
                   round: int = None
                  ):
    """Make a word cloud.

    Accepts data from a string, list of lists or tuples, a dict with
    terms as keys and counts/frequencies as values, or a dataframe.

    Args:
        data (Union[dict, list, object, str, tuple]): The data. Accepts a text string, a list of lists or tuples,
            a dict with the terms as keys and the counts/frequencies as values, or a dataframe with "term" and
            "count" or "frequency" columns.
        opts (dict): The WordCloud() options.
            For testing, try {"background_color": "white", "max_words": 2000, "contour_width": 3, "contour_width": "steelblue"}
        round (int): An integer (generally between 100-300) to apply a mask that rounds the word cloud.

    Returns:
        object: A WordCloud object if show is set to False.

    Notes:
        - For a full list of options, see https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html#wordcloud-wordcloud.
        - If `show=False` the function expects to be called with something like `wordcloud = make_wordcloud(data, show=False)`.
            This returns WordCloud object which can be manipulated by any of its methods, such as `to_file()`. See the
            WordCloud documentation for a list of methods.
    """
    if isinstance(data, str):
        wordcloud = WordCloud(**opts).generate_from_text(data)
    else:
        if isinstance(data, list):
            data = {x[0]: x[1] for x in data}
        elif isinstance(data, pd.DataFrame):
            term_counts = data.to_dict(orient="records")
            try:
                data = {x["terms"]: x["count"] for x in term_counts}
            except KeyError:
                data = {x["terms"]: x["frequency"] for x in term_counts}
        wordcloud = WordCloud(**opts).generate_from_frequencies(data)
    return wordcloud

def plot(dtm: object,
         docs: List[str] = None,
         opts: dict = None,
         layout: dict = None,
         show: bool = True,) -> go.Figure:
    """Convert a Python word cloud to a Plotly word cloud.

    This is some prototype code for generating word clouds in Plotly.
    Based on https://github.com/PrashantSaikia/Wordcloud-in-Plotly.

    This is really a case study because Plotly does not do good
    word clouds. One of the limitations is that `WordCloud.layout_`
    always returns `None` for orientation and frequencies for
    counts. That limits the options for replicating its output.

    Run with:

    ```python
    from lexos.visualization.plotly.cloud.wordcloud import plot
    plot(dtm)

    or

    wc = plot(dtm, show=False)
    wc.show()
    ```

    Args:
        dtm (object): A lexos.DTM object.
        docs: (List[str]): A list of document names to use.
        opts: (dict): A dict of options to pass to WordCloud.
        layout: (dict): A dict of options to pass to Plotly.
        show: (bool): Whether to show the plot.

    Returns:
        object: A Plotly figure.
    """
    word_list=[]
    freq_list=[]
    fontsize_list=[]
    position_list=[]
    orientation_list=[]
    color_list=[]
    layout_opts = {
        "xaxis": {
            "showgrid": False,
            "showticklabels": False,
            "zeroline": False
        },
        "yaxis": {
            "showgrid": False,
            "showticklabels": False,
            "zeroline": False
        },
        "autosize": False,
        "width": 750,
        "height": 750,
        "margin": {
            "l": 50,
            "r": 50,
            "b": 100,
            "t": 100,
            "pad": 4
        }
    }

    if layout:
        for k, v in layout.items():
            layout_opts[k] = v

    # Get the dtm table
    data = dtm.get_table()

    # Get the counts for the desired documents
    if docs:
        docs = ["terms"] + docs
        data = data[docs].copy()
        # Create a new column with the total for each row
        data["count"] = data.sum(axis=1)
    # Get the dtm sums
    else:
        data["count"] = data.sum(axis=1)
        # data = data.rename({"terms": "term", "sum": "count"}, axis=1)

    # Ensure that the table only has terms and counts
    data = data[["terms", "count"]].copy()

    # Create the word cloud
    if opts is None:
        opts = {}
    wc = make_wordcloud(data, opts)

    # Plot the word cloud
    for (word, freq), fontsize, position, orientation, color in wc.layout_:
        word_list.append(word)
        freq_list.append(freq)
        fontsize_list.append(fontsize)
        position_list.append(position)
        orientation_list.append(orientation)
        color_list.append(color)

    # Get the positions
    x=[]
    y=[]
    for i in position_list:
        x.append(i[0])
        y.append(i[1])

    # Get the relative occurence frequencies
    new_freq_list = []
    for i in freq_list:
        new_freq_list.append(f"{round(i*100, 2)}%")
    new_freq_list
    trace = go.Scatter(x=x,
                       y=y,
                       textfont = dict(size=fontsize_list, color=color_list),
                       hoverinfo="text",
                       hovertext=[f"{w}: {f}" for w, f in zip(word_list, new_freq_list)],
                       mode="text",
                       text=word_list
                      )

    # Set the laoyt and create the figure
    layout = go.Layout(layout_opts)
    fig = go.Figure(data=[trace], layout=layout)

    # Show the plot and/or return the figure
    if show:
        fig.show()
        return fig
    else:
        return fig