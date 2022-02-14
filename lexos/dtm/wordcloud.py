"""wordcloud.py."""
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wordcloud import WordCloud


def get_rows(lst, n):
    """Yield successive n-sized rows from a list of documents.

    Args:
        lst (list): A list of documents.
        n (int): The number of columns in the row.

    Yields:
        list: A generator with the documents separated into rows.
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def make_wordcloud(data: Union[dict, list, object, str, tuple],
                   opts: dict = None,
                   show: bool = True,
                   figure_opts: dict = None,
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
        show (bool): Whether to show the plotted word cloud or return it as a WordCloud object.
        figure_opts (dict): A dict of matplotlib figure options.
        round (int): An integer (generally between 100-300) to apply a mask that rounds the word cloud.

    Returns:
        object: A WordCloud object if show is set to False.

    Notes:
        - For a full list of options, see https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html#wordcloud-wordcloud.
        - If `show=False` the function expects to be called with something like `wordcloud = make_wordcloud(data, show=False)`.
            This returns WordCloud object which can be manipulated by any of its methods, such as `to_file()`. See the
            WordCloud documentation for a list of methods.
    """
    if round:
        x, y = np.ogrid[:300, :300]
        mask = (x - 150) ** 2 + (y - 150) ** 2 > round ** 2
        mask = 255 * mask.astype(int)
        opts["mask"] = mask
    if isinstance(data, str):
        wordcloud = WordCloud(**opts).generate_from_text(data)
    else:
        if isinstance(data, list):
            data = {x[0]: x[1] for x in data}
        elif isinstance(data, pd.DataFrame):
            term_counts = data.to_dict(orient="records")
            try:
                data = {x["term"]: x["count"] for x in term_counts}
            except KeyError:
                data = {x["term"]: x["frequency"] for x in term_counts}
        wordcloud = WordCloud(**opts).generate_from_frequencies(data)
    if show:
        if figure_opts:
            plt.figure(**figure_opts)
        plt.axis("off")
        plt.imshow(wordcloud)
        plt.show()
    else:
        return wordcloud


def make_multiclouds(docs: List[Union[dict, object, str, tuple]],
                   opts: dict = None,
                   ncols: int = 3,
                   title: str = None,
                   labels: List[str] = None,
                   show: bool = True,
                   figure_opts: dict = None,
                   round: int = None
                  ):
    """Make multiclouds.

    Accepts data from a string, list of lists or tuples, a dict with
    terms as keys and counts/frequencies as values, or a dataframe.

    The best input is a dtm produced by `get_dtm_table()`.

    Args:
        docs (List[Union[dict, object, str, tuple]]): The data. Accepts a list of text strings, a list of tuples,
            or dicts with the terms as keys and the counts/frequencies as values, or a dataframe with "term" and
            "count" or "frequency" columns.
        opts (dict): The WordCloud() options.
            For testing, try {"background_color": "white", "max_words": 2000, "contour_width": 3, "contour_width": "steelblue"}
        ncols (int): The number of columns in the grid.
        title (str): The title of the grid.
        labels (List[str]): The document labels for each subplot.
        show (bool): Whether to show the plotted word cloud or return it as a WordCloud object.
        figure_opts (dict): A dict of matplotlib figure options.
        round (int): An integer (generally between 100-300) to apply a mask that rounds the word cloud.

    Returns:
        object: A WordCloud object if show is set to False.

    Notes:
        - For a full list of options, see https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html#wordcloud-wordcloud.
        - If `show=False` the function expects to be called with something like `wordcloud = make_wordcloud(data, show=False)`.
            This returns WordCloud object which can be manipulated by any of its methods, such as `to_file()`. See the
            WordCloud documentation for a list of methods.
    """
    # Process the docs data into a list
    if isinstance(docs, pd.core.frame.DataFrame):
        # Assumes a df with columns: Terms, Doc_Label, DocLabel,...
        # Transpose the df
        docs = docs.T
        # Grab the first row for the header
        new_header = docs.iloc[0]
        # Drop the first row
        docs = docs[1:]
        # Set the header row as the df header
        docs.columns = new_header
        # Return a dict
        docs = docs.to_dict(orient="records")
    # Ensure that anything that is not a list of strings is converted
    # to the appropriate format.
    elif isinstance(docs, list):
        if all(isinstance(s, str) for s in docs):
            pass
        else:
            docs = [{x[0:1]: x[1:2] for x in data} for data in docs]

    # List for multiple word clouds if they are to be returned.
    multiclouds = []
    # Create a rounded mask.
    if round:
        x, y = np.ogrid[:300, :300]
        mask = (x - 150) ** 2 + (y - 150) ** 2 > round ** 2
        mask = 255 * mask.astype(int)
        opts["mask"] = mask
    # Constrain the layout
    figure_opts["constrained_layout"] = True
    # Create the figure.
    fig = plt.figure(**figure_opts)
    # Add the title
    if title:
        fig.suptitle(title)
    # Calculate the number of rows and columns.
    nrows = int(np.ceil(len(docs) / ncols))
    spec = fig.add_gridspec(nrows, ncols)
    # Divide the data into rows.
    rows = list(get_rows(docs, ncols))
    # Set an index for labels
    i = 0
    # Loop through the rows.
    for row, doc in enumerate(rows):
        # Loop through the documents in the row.
        for col, data in enumerate(doc):
            # Create a subplot.
            ax = fig.add_subplot(spec[row, col])
            # Generate the subplot's word cloud.
            if isinstance(data, str):
                    wordcloud = WordCloud(**opts).generate_from_text(data)
            else:
                wordcloud = WordCloud(**opts).generate_from_frequencies(data)
            # If `show=True`, show the word cloud.
            if show:
                ax.imshow(wordcloud)
                ax.axis("off")
                # Set the image title from the label
                if labels:
                    ax.set_title(labels[i])
                    i += 1
            # Otherwise, add the word cloud to the multiclouds list.
            else:
                multiclouds.append(wordcloud)
    # If `show=False`, return the multiclouds list.
    if not show:
        return multiclouds

def plotly_wc(wc):
    """Convert a Python word cloud to a Plotly word cloud.

    This is some prototype code for generating word clouds in Plotly.
    Based on https://github.com/PrashantSaikia/Wordcloud-in-Plotly.

    Doesn't handle counts because `WordCloud.layout_` returns frequencies.
    `WordCloud.layout_` always seems to return `None` for orientation.
    Plotly sizing works differently and needs to be handled from options.
    To round the corners of a Plotly graph you would have to `change go.layout.Shape`.

    For Jupyter notebooks, requires:

    ```python
    import plotly.graph_objects as go
    from plotly.offline import iplot, init_notebook_mode
    init_notebook_mode(connected=True)
    ```

    Run with:

    ```python
    wc = make_wordcloud(sums, opts, show=False, figure_opts={"figsize": (15, 8)})
    plotly_wc = plotly_wc(wc)
    plotly_wc.show()
    ```
    """
    word_list=[]
    freq_list=[]
    fontsize_list=[]
    position_list=[]
    orientation_list=[]
    color_list=[]
    for (word, freq), fontsize, position, orientation, color in wc.layout_:
        word_list.append(word)
        freq_list.append(freq)
        fontsize_list.append(fontsize)
        position_list.append(position)
        orientation_list.append(orientation)
        color_list.append(color)
    # get the positions
    x=[]
    y=[]
    for i in position_list:
        x.append(i[0])
        y.append(i[1])
    # get the relative occurence frequencies
    new_freq_list = []
    for i in freq_list:
        new_freq_list.append(f"{round(i*100, 2)}%")
    new_freq_list
    trace = go.Scatter(x=x,
                       y=y,
                       textfont = dict(size=fontsize_list, color=color_list),
                       hoverinfo='text',
                       hovertext=[f'{w}: {f}' for w, f in zip(word_list, new_freq_list)],
                       mode='text',
                       text=word_list
                      )
    layout = go.Layout({'xaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
                        'yaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False}})
    fig = go.Figure(data=[trace], layout=layout)
    fig.update_layout(
        autosize=False,
        width=750,
        height=750,
        margin=dict(l=50,r=50,b=100,t=100,pad=4)
    )
    return fig