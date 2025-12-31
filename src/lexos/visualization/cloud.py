"""cloud.py.

Last Update: December 4, 2025
Last Tested: December 5, 2025
"""

import math
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pydantic import BaseModel, ConfigDict, Field, validate_call
from spacy.schemas import DocJSONSchema
from spacy.tokens import Doc, Span, Token
from wordcloud import WordCloud as PythonWordCloud

from lexos.dtm import DTM
from lexos.exceptions import LexosException
from lexos.visualization import processors

# Valid input types
single_doc_types = dict[str, int | float] | Doc | Span | str | list[str] | list[Token]
multi_doc_types = (
    str
    | list[str]
    | list[list[str]]
    | list[Doc]
    | list[Span]
    | list[list[Token]]
    | dict[str, int | float]
    | pd.DataFrame
    | DTM
)


class WordCloud(BaseModel):
    """A Pydantic model for WordCloud options."""

    data: single_doc_types | multi_doc_types | pd.DataFrame = Field(
        ...,
        description="The data to generate the word cloud from. Accepts data from a string, list of lists or tuples, a dict with terms as keys and counts/frequencies as values, or a dataframe.",
    )
    docs: Optional[int | str | list[int] | list[str]] = Field(
        None, description="A list of documents to be selected from the DTM."
    )
    limit: Optional[int] = Field(
        None, description="The maximum number of terms to plot."
    )
    title: Optional[str] = Field(None, description="The title of the plot.")
    height: int = Field(
        200, gt=50, description="The height of the word cloud in pixels."
    )
    width: int = Field(200, gt=50, description="The width of the word cloud in pixels.")
    opts: Optional[dict[str, Any]] = Field(
        {
            "background_color": "white",
            "max_words": 2000,
            "contour_width": 0,
            "contour_color": "steelblue",
        },
        description="The WordCloud() options.",
    )
    figure_opts: Optional[dict[str, Any]] = Field(
        {}, description="A dict of matplotlib figure options."
    )
    round: Optional[int] = Field(
        0,
        description="An integer to apply a mask that rounds the word cloud. It is best to use 100 or higher for a circular mask, but it will depend on the height and width of the word cloud.",
    )
    counts: dict[str, int] = Field(None, description="A dictionary of term counts.")
    cloud: PythonWordCloud | None = Field(
        None, description="The generated WordCloud object."
    )
    fig: Optional[plt.Figure] = Field(
        None, description="The matplotlib figure object for the word cloud."
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True, json_schema_extra=DocJSONSchema.schema()
    )

    def __init__(self, **data: Any) -> None:
        """Initialize the WordCloud model."""
        super().__init__(**data)

        # Set the figure dimensions
        self.opts["height"] = self.height
        self.opts["width"] = self.width

        # Set the mask, if using
        if self.round > 0:
            x, y = np.ogrid[:300, :300]
            mask = (x - 150) ** 2 + (y - 150) ** 2 > self.round**2
            mask = 255 * mask.astype(int)
            self.opts["mask"] = mask

        # Process the data into a consistent format
        self.counts = processors.process_data(self.data, self.docs, self.limit)

        # Generate the word cloud
        self.cloud = PythonWordCloud(**self.opts).generate_from_frequencies(self.counts)

    @validate_call
    def save(self, path: Path | str, **kwargs: Any) -> None:
        """Save the WordCloud to a file.

        Args:
            path (Path | str): The file path to save the WordCloud image.
            **kwargs (Any): Additional keyword arguments for `plt.savefig`.
        """
        if self.cloud is None:
            raise LexosException("No WordCloud object to save.")
        self.fig = plt.figure(**self.figure_opts)
        if self.title:
            self.fig.suptitle(self.title)
        plt.axis("off")
        plt.imshow(self.cloud, interpolation="bilinear")
        plt.savefig(path, **kwargs)
        plt.close()

    def show(self) -> None:
        """Show the figure if it is hidden.

        This is a helper method. It will generally display in a
        Jupyter notebook.
        """
        self.fig = plt.figure(**self.figure_opts)
        if self.title:
            self.fig.suptitle(self.title)
        plt.axis("off")
        plt.imshow(self.cloud, interpolation="bilinear")


class MultiCloud(BaseModel):
    """A Pydantic model for creating multiple WordClouds arranged in a grid using the topic_clouds approach."""

    data: list[str] | list[list[str]] | list[Doc] | list[Span] | DTM | pd.DataFrame = (
        Field(
            ...,
            description="The data to generate word clouds from. Accepts list of documents, DTM, or DataFrame.",
        )
    )
    docs: Optional[int | str | list[int] | list[str]] = Field(
        None, description="A list of documents to be selected from the DTM/DataFrame."
    )
    limit: Optional[int] = Field(
        None, description="The maximum number of terms to plot per cloud."
    )
    figsize: tuple[int, int] = Field(
        (10, 10), description="The size of the overall figure."
    )
    layout: Optional[str | tuple[int, int]] = Field(
        "auto",
        description="The number of rows and columns in the figure. Default is 'auto'.",
    )
    opts: Optional[dict[str, Any]] = Field(
        {
            "background_color": "white",
            "max_words": 2000,
            "contour_width": 0,
            "contour_color": "steelblue",
        },
        description="The WordCloud() options applied to each word cloud.",
    )
    round: Optional[int] = Field(
        0,
        description="An integer to apply a mask that rounds each word cloud. It is best to use 100 or higher for a circular mask.",
    )
    title: Optional[str] = Field(None, description="Overall title for the figure.")
    labels: Optional[list[str]] = Field(
        None, description="Labels for each subplot/word cloud."
    )
    doc_data: Optional[list[dict[str, int | float]]] = Field(
        None, description="Processed document data for each word cloud."
    )
    fig: Optional[plt.Figure] = Field(
        None, description="The matplotlib figure object for the multi-cloud plot."
    )
    wordcloud: Optional[PythonWordCloud] = Field(
        None, description="The WordCloud object used for generating clouds."
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True, json_schema_extra=DocJSONSchema.schema()
    )

    def __init__(self, **data: Any) -> None:
        """Initialize the MultiCloud model."""
        super().__init__(**data)

        # Process different data types to get individual document data
        self.doc_data = self._process_data()

        # Setup the WordCloud object
        self.wordcloud = self._setup_wordcloud()

        # Render the figure
        self._render()

    def _process_data(self) -> list[dict[str, int | float]]:
        """Process the input data into individual document dictionaries."""
        if isinstance(self.data, DTM):
            # Make sure there is data
            if (
                self.data.doc_term_matrix is None
                or self.data.doc_term_matrix.shape[0] == 0
            ):
                raise LexosException("Empty DTM provided.")
            # Extract documents from DTM
            doc_data = []
            selected_docs = (
                self.docs
                if self.docs is not None
                else range(self.data.doc_term_matrix.shape[0])
            )
            if isinstance(selected_docs, (int, str)):
                selected_docs = [selected_docs]

            for doc_idx in selected_docs:
                # Get term frequencies for this document
                if isinstance(doc_idx, str):
                    doc_idx = self.data.labels.index(doc_idx)
                doc_counts = {}

                # Get the row as a 1D array and convert to list/scalar values
                doc_row = self.data.doc_term_matrix[doc_idx]
                if hasattr(doc_row, "toarray"):  # sparse matrix
                    doc_row = doc_row.toarray().flatten()

                for term_idx, count in enumerate(doc_row):
                    # Convert to scalar value before comparison
                    count_value = (
                        float(count.item()) if hasattr(count, "item") else float(count)
                    )
                    if count_value > 0:
                        doc_counts[self.data.vectorizer.terms_list[term_idx]] = (
                            count_value
                        )
                doc_data.append(doc_counts)

        elif isinstance(self.data, pd.DataFrame):
            # Make sure there is data
            if self.data.empty:
                raise LexosException("Empty DataFrame provided.")
            # Process DataFrame - assume it's a document-term matrix
            doc_data = []
            selected_docs = (
                self.docs if self.docs is not None else range(len(self.data))
            )
            if isinstance(selected_docs, (int, str)):
                selected_docs = [selected_docs]

            for doc_idx in selected_docs:
                if isinstance(doc_idx, str):
                    doc_idx = self.data.index.get_loc(doc_idx)
                doc_counts = self.data.iloc[doc_idx].to_dict()
                # Filter out zero counts and convert to float
                doc_counts = {
                    k: float(v.item() if hasattr(v, "item") else v)
                    for k, v in doc_counts.items()
                    if (float(v.item()) if hasattr(v, "item") else float(v)) > 0
                }
                doc_data.append(doc_counts)

        elif isinstance(self.data, list):
            # Make sure the data is not empty
            if not self.data or len(self.data) == 0:
                raise LexosException("No valid data provided for MultiCloud.")
            # Process list of documents using the processors module
            doc_data = [
                processors.process_data(doc, None, self.limit) for doc in self.data
            ]

        else:
            raise LexosException("Unsupported data type for MultiCloud.")

        return doc_data

    def _setup_wordcloud(self) -> PythonWordCloud:
        """Configure a single WordCloud object to be reused."""
        # Set the mask if using round
        if self.round > 0:
            x, y = np.ogrid[:300, :300]
            mask = (x - 150) ** 2 + (y - 150) ** 2 > self.round**2
            mask = 255 * mask.astype(int)
            self.opts["mask"] = mask

        # Set max_words if limit is specified
        if self.limit:
            self.opts["max_words"] = self.limit

        return PythonWordCloud(**self.opts)

    def _render(self) -> None:
        """Generate and display the multi-cloud figure."""
        # Set parameters for plotting
        sns.set_theme()
        plt.rcParams["figure.figsize"] = self.figsize

        # Calculate layout
        n = len(self.doc_data)
        if self.layout == "auto":
            columns = math.floor(math.sqrt(n))
            rows = math.ceil(n / columns)
        elif isinstance(self.layout, tuple):
            rows, columns = self.layout
        else:
            raise LexosException("Invalid layout specification.")

        # Create the figure
        self.fig = plt.figure(figsize=self.figsize)

        # Add overall title
        if self.title:
            self.fig.suptitle(self.title, fontsize=16)

        # Generate the word clouds
        for i, doc_counts in enumerate(self.doc_data):
            self.wordcloud.generate_from_frequencies(doc_counts)
            plt.subplot(rows, columns, i + 1)
            plt.imshow(self.wordcloud, interpolation="bilinear")
            plt.axis("off")

            # Add label if provided
            if self.labels and i < len(self.labels):
                plt.title(self.labels[i])
            else:
                plt.title(f"Doc {i}")

        # Get the figure and close to prevent automatic display
        self.fig = plt.gcf()
        plt.close()

    @validate_call
    def save(self, path: Path | str, **kwargs: Any) -> None:
        """Save the MultiCloud figure to a file.

        Args:
            path (Path | str): The file path to save the MultiCloud image.
            **kwargs (Any): Additional keyword arguments for `plt.savefig`.
        """
        if self.fig is None:
            raise LexosException("No figure to save.")
        self.fig.savefig(path, **kwargs)

    def show(self) -> None:
        """Display the multi-cloud figure."""
        if self.fig is None:
            raise LexosException("No figure to show.")
        # Use IPython display for Jupyter notebooks
        try:
            from IPython.display import display

            display(self.fig)
        except ImportError:
            # Fallback for non-Jupyter environments
            plt.figure(self.fig.number)
            plt.show()


class MultiCloudOld(BaseModel):
    """A Pydantic model for creating multiple WordClouds arranged in a grid.

    # NOTE: This Class is deprecated.
    """

    data: list[str] | list[list[str]] | list[Doc] | list[Span] | DTM | pd.DataFrame = (
        Field(
            ...,
            description="The data to generate word clouds from. Accepts list of documents, DTM, or DataFrame.",
        )
    )
    docs: Optional[int | str | list[int] | list[str]] = Field(
        None, description="A list of documents to be selected from the DTM/DataFrame."
    )
    limit: Optional[int] = Field(
        None, description="The maximum number of terms to plot."
    )
    ncols: int = Field(3, gt=0, description="Number of columns in the grid layout.")
    height: int = Field(
        200, gt=50, description="The height of each word cloud in pixels."
    )
    width: int = Field(
        200, gt=50, description="The width of each word cloud in pixels."
    )
    opts: Optional[dict[str, Any]] = Field(
        {
            "background_color": "white",
            "max_words": 2000,
            "contour_width": 0,
            "contour_color": "steelblue",
        },
        description="The WordCloud() options applied to each word cloud.",
    )
    figure_opts: Optional[dict[str, Any]] = Field(
        {}, description="A dict of matplotlib figure options."
    )
    round: Optional[int] = Field(
        0,
        description="An integer to apply a mask that rounds each word cloud. It is best to use 100 or higher for a circular mask.",
    )
    title: Optional[str] = Field(None, description="Overall title for the figure.")
    labels: Optional[list[str]] = Field(
        None, description="Labels for each subplot/word cloud."
    )
    padding: float = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description="Amount of padding between subplots (0.0 to 1.0).",
    )
    clouds: list[WordCloud] = Field(
        default_factory=list, description="List of generated WordCloud objects."
    )
    fig: Optional[plt.Figure] = Field(
        None, description="The matplotlib figure object for the multi-cloud plot."
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True, json_schema_extra=DocJSONSchema.schema()
    )

    def __init__(self, **data: Any) -> None:
        """Initialize the MultiCloud model."""
        super().__init__(**data)

        # Process different data types to get individual document data
        doc_data = self._process_data()

        # Create individual WordCloud objects
        self.clouds = []
        for doc in doc_data:
            try:
                # Create a WordCloud instance for each document
                wc = WordCloud(
                    data=doc,
                    limit=self.limit,
                    opts=self.opts,
                    round=self.round,
                    width=self.width,
                    height=self.height,
                )
                self.clouds.append(wc)
            except Exception as e:
                raise LexosException(f"Failed to create word cloud: {e}")

        # Render the figure
        self._render()

    def _process_data(self) -> list:
        """Process the input data into individual documents."""
        if isinstance(self.data, DTM):
            # Make sure there is data
            if (
                self.data.doc_term_matrix is None
                or self.data.doc_term_matrix.shape[0] == 0
            ):
                raise LexosException("Empty DTM provided.")
            # Extract documents from DTM
            doc_data = []
            selected_docs = (
                self.docs
                if self.docs is not None
                else range(self.data.doc_term_matrix.shape[0])
            )
            if isinstance(selected_docs, (int, str)):
                selected_docs = [selected_docs]

            for doc_idx in selected_docs:
                # Get term frequencies for this document
                if isinstance(doc_idx, str):
                    doc_idx = self.data.labels.index(doc_idx)
                doc_counts = {}

                # Get the row as a 1D array and convert to list/scalar values
                doc_row = self.data.doc_term_matrix[doc_idx]
                if hasattr(doc_row, "toarray"):  # sparse matrix
                    doc_row = doc_row.toarray().flatten()

                for term_idx, count in enumerate(doc_row):
                    # Convert to scalar value before comparison
                    count_value = (
                        float(count.item()) if hasattr(count, "item") else float(count)
                    )
                    if count_value > 0:
                        doc_counts[self.data.vectorizer.terms_list[term_idx]] = int(
                            count_value
                        )
                doc_data.append(doc_counts)

        elif isinstance(self.data, pd.DataFrame):
            # Make sure there is data
            if self.data.empty == True:
                raise LexosException("Empty DataFrame provided.")
            # Process DataFrame - assume it's a document-term matrix
            doc_data = []
            selected_docs = (
                self.docs if self.docs is not None else range(len(self.data))
            )
            if isinstance(selected_docs, (int, str)):
                selected_docs = [selected_docs]

            for doc_idx in selected_docs:
                if isinstance(doc_idx, str):
                    doc_idx = self.data.index.get_loc(doc_idx)
                doc_counts = self.data.iloc[doc_idx].to_dict()
                # Filter out zero counts
                doc_counts = {
                    k: v
                    for k, v in doc_counts.items()
                    if (float(v) if hasattr(v, "item") else v) > 0
                }
                doc_data.append(doc_counts)

        elif isinstance(self.data, list):
            # Make sure the data is not empty
            if not self.data or len(self.data) == 0:
                raise LexosException("No valid data provided for MultiCloud.")
            # Handle list of documents
            doc_data = self.data

        return doc_data

    def _render(self) -> None:
        """Generate and display the multi-cloud figure."""
        # Calculate layout
        num_clouds = len(self.clouds)
        nrows = int(np.ceil(num_clouds / self.ncols))

        # Set up figure with padding
        figure_opts = self.figure_opts.copy()
        figure_opts.setdefault("figsize", (self.ncols * 4, nrows * 3))

        # Remove constrained_layout if it exists since we're setting manual spacing
        figure_opts.pop("constrained_layout", None)

        self.fig, axes = plt.subplots(nrows, self.ncols, **figure_opts)

        # Add padding between subplots and adjust top margin for title
        if self.title:
            # More space below title when there's a suptitle
            self.fig.subplots_adjust(
                wspace=self.padding,
                hspace=self.padding,
                top=0.82,  # Leaves more space at the top for the title
            )
        else:
            # Normal spacing when no title
            self.fig.subplots_adjust(wspace=self.padding, hspace=self.padding)

        # Add padding between subplots
        self.fig.subplots_adjust(wspace=self.padding, hspace=self.padding)

        # Handle single row case
        if nrows == 1:
            axes = axes.reshape(1, -1) if self.ncols > 1 else np.array([[axes]])
        elif self.ncols == 1:
            axes = axes.reshape(-1, 1)

        # Add overall title
        if self.title:
            self.fig.suptitle(self.title, fontsize=16, y=0.90)  # Positioned lower

        # Plot each word cloud
        for i, cloud in enumerate(self.clouds):
            row = i // self.ncols
            col = i % self.ncols

            ax = axes[row, col]

            # Display the word cloud
            ax.imshow(cloud.cloud, interpolation="bilinear")
            ax.axis("off")

            # Add label if provided
            if self.labels and i < len(self.labels):
                ax.set_title(self.labels[i])
            elif hasattr(cloud.data, "__len__"):
                ax.set_title(f"Doc {i + 1}", fontdict={"fontsize": 10})

        # Hide unused subplots
        for i in range(num_clouds, nrows * self.ncols):
            row = i // self.ncols
            col = i % self.ncols
            axes[row, col].axis("off")
            axes[row, col].set_visible(False)

        # Prevent automatic display
        self.fig = plt.gcf()
        plt.close()

    @validate_call
    def save(self, path: Path | str, **kwargs: Any) -> None:
        """Save the MultiCloud figure to a file.

        Args:
            path (Path | str): The file path to save the MultiCloud image.
            **kwargs (Any): Additional keyword arguments for `plt.savefig`.
        """
        if self.fig is None:
            raise LexosException("No figure to save.")
        self.fig.savefig(path, **kwargs)

    def get_clouds(self) -> list[WordCloud]:
        """Return the list of individual WordCloud objects."""
        return self.clouds

    def show(self) -> plt.Figure:
        """Display the multi-cloud figure."""
        if self.fig is None:
            raise LexosException("No figure to show.")
        return self.fig
