"""wordcloud.py."""
from typing import Iterator, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wordcloud import WordCloud


def get_rows(lst, n) -> Iterator[int]:
    """Yield successive n-sized rows from a list of documents.

    Args:
        lst (list): A list of documents.
        n (int): The number of columns in the row.

    Yields:
        A generator with the documents separated into rows.
    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def wordcloud(
    dtm: Union[dict, list, object, pd.DataFrame, str, tuple],
    docs: List[str] = None,
    opts: dict = None,
    show: bool = True,
    figure_opts: dict = None,
    round: int = None,
    filename: str = None,
) -> object:
    """Make a word cloud.

    Accepts data from a string, list of lists or tuples, a dict with
    terms as keys and counts/frequencies as values, or a dataframe.

    Args:
        dtm (Union[dict, list, object, pd.DataFrame, str, tuple]): The data.
            Accepts a text string, a list of lists or tuples, a dataframe,
            or a dict with the terms as keys and the counts/frequencies as
            values, or a dataframe with "term" and "count" or "frequency" columns.
        docs (list): A list of documents to be selected from the DTM.
        opts (dict): The WordCloud() options.
            For testing, try {"background_color": "white", "max_words": 2000, "contour_width": 3, "contour_color": "steelblue"}
        show (bool): Whether to show the plotted word cloud or return it as a WordCloud object.
        figure_opts (dict): A dict of matplotlib figure options.
        round (int): An integer (generally between 100-300) to apply a mask that rounds the word cloud.
        filename (str): The filename to save the word cloud to.

    Returns:
        object: A WordCloud object if show is set to False.

    Notes:
        - For a full list of options, see https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html#wordcloud-wordcloud.
        - If `show=False` the function expects to be called with something like `wordcloud = make_wordcloud(data, show=False)`.
            This returns WordCloud object which can be manipulated by any of its methods, such as `to_file()`. See the
            WordCloud documentation for a list of methods.
    """
    if opts is None:
        opts = {
            "background_color": "white",
            "max_words": 2000,
            "contour_width": 0,
            "contour_color": "steelblue",
        }
    if figure_opts is None:
        figure_opts = {}

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

    # Ensure that the table only has terms and counts
    data = data[["terms", "count"]].copy()

    # Set the mask, if using
    if round:
        x, y = np.ogrid[:300, :300]
        mask = (x - 150) ** 2 + (y - 150) ** 2 > round ** 2
        mask = 255 * mask.astype(int)
        opts["mask"] = mask

    # Generate the WordCloud
    if isinstance(data, str):
        wc = WordCloud(**opts).generate_from_text(data)
    else:
        if isinstance(data, list):
            data = {x[0]: x[1] for x in data}
        elif isinstance(data, pd.DataFrame):
            term_counts = data.to_dict(orient="records")
            try:
                data = {x["terms"]: x["count"] for x in term_counts}
            except KeyError:
                data = {x["terms"]: x["frequency"] for x in term_counts}
        wc = WordCloud(**opts).generate_from_frequencies(data)

    # Plot or return the WordCloud
    if show:
        if figure_opts:
            plt.figure(**figure_opts)
        plt.axis("off")
        # If a filename is provided, save the figure
        if filename:
            plt.to_file(filename)
        plt.imshow(wc)
        plt.show()
    else:
        return wc


def multicloud(
    dtm: List[Union[dict, pd.DataFrame, object, str, tuple]],
    docs: List[str] = None,
    opts: dict = None,
    ncols: int = 3,
    title: str = None,
    labels: List[str] = None,
    show: bool = True,
    figure_opts: dict = None,
    round: int = None,
    filename: str = None,
) -> object:
    """Make multiclouds.

    Accepts data from a string, list of lists or tuples, a dict with
    terms as keys and counts/frequencies as values, or a dataframe.

    The best input is a dtm produced by `get_dtm_table()`.

    Args:
        dtm (List[Union[dict, object, str, tuple]]): The data.
            Accepts a list of text strings, a list of tuples,
            or dicts with the terms as keys and the counts/frequencies
            as values, or a dataframe with "term" and "count" columns.
        docs: (List[str]): A list of documents to be selected from the DTM.
        opts (dict): The WordCloud() options.
            For testing, try {"background_color": "white", "max_words": 2000, "contour_width": 3, "contour_color": "steelblue"}
        ncols (int): The number of columns in the grid.
        title (str): The title of the grid.
        labels (List[str]): The document labels for each subplot.
        show (bool): Whether to show the plotted word cloud or return it as a WordCloud object.
        figure_opts (dict): A dict of matplotlib figure options.
        round (int): An integer (generally between 100-300) to apply a mask that rounds the word cloud.
        filename (str): The filename to save the figure to.

    Returns:
        object: A WordCloud object if show is set to False.

    Notes:
        - For a full list of options, see https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html#wordcloud-wordcloud.
        - If `show=False` the function expects to be called with something like `wordcloud = make_wordcloud(data, show=False)`.
            This returns WordCloud object which can be manipulated by any of its methods, such as `to_file()`. See the
            WordCloud documentation for a list of methods.
    """
    if opts is None:
        opts = {
            "background_color": "white",
            "max_words": 2000,
            "contour_width": 0,
            "contour_color": "steelblue",
        }
    if figure_opts is None:
        figure_opts = {}

    # Get table
    data = dtm.get_table()

    # Get the counts for the desired documents
    if docs:
        data = data[["terms"] + docs].copy()

    # Transposed table
    data = data.T

    # Process the data data into a list
    if isinstance(data, pd.DataFrame):
        # Grab the first row for the header
        new_header = data.iloc[0]
        # Drop the first row
        data = data[1:]
        # Set the header row as the df header
        data.columns = new_header
        # Return a dict
        data = data.to_dict(orient="records")
    # Ensure that anything that is not a list of strings is converted
    # to the appropriate format.
    elif isinstance(data, list):
        if all(isinstance(s, str) for s in data):
            pass
        else:
            data = [{x[0:1]: x[1:2] for x in doc} for doc in data]

    # List for multiple word clouds if they are to be returned.
    multiclouds = []

    # Create a rounded mask
    if round:
        x, y = np.ogrid[:300, :300]
        mask = (x - 150) ** 2 + (y - 150) ** 2 > round ** 2
        mask = 255 * mask.astype(int)
        opts["mask"] = mask

    # Constrain the layout
    figure_opts["constrained_layout"] = True

    # Create the figure
    fig = plt.figure(**figure_opts)

    # Add the title
    if title:
        fig.suptitle(title)

    # Calculate the number of rows and columns

    nrows = int(np.ceil(len(data) / ncols))
    spec = fig.add_gridspec(nrows, ncols)

    # Divide the data into rows
    rows = list(get_rows(data, ncols))

    # Set an index for labels
    i = 0

    # Loop through the rows
    for row, doc in enumerate(rows):

        # Loop through the documents in the row
        for col, data in enumerate(doc):

            # Create a subplot
            ax = fig.add_subplot(spec[row, col])

            # Generate the subplot's word cloud
            if isinstance(data, str):
                wordcloud = WordCloud(**opts).generate_from_text(data)
            else:
                wordcloud = WordCloud(**opts).generate_from_frequencies(data)

            # If `show=True`, show the word cloud
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

    # If a filename is provided, save the figure
    if filename:
        fig.savefig(filename)

    # If `show=False`, return the multiclouds list.
    if not show:
        return multiclouds
