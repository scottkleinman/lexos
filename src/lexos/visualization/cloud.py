"""cloud.py.

Last Updated: August 16, 2026
Last Tested: August 16, 2026
"""

import math
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    opts: dict[str, Any] = Field(
        default_factory=lambda: {
            "background_color": "white",
            "max_words": 2000,
            "contour_width": 0,
            "contour_color": "steelblue",
        },
        description="The WordCloud() options.",
    )
    figure_opts: dict[str, Any] = Field(
        default_factory=dict,
        description="A dict of matplotlib figure options.",
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
        arbitrary_types_allowed=True,
        json_schema_extra=DocJSONSchema.model_json_schema(),
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
        ax = self.fig.add_subplot(111)
        if self.title:
            self.fig.suptitle(self.title)
        ax.axis("off")
        ax.imshow(self.cloud.to_array(), interpolation="bilinear")
        self.fig.savefig(path, **kwargs)
        plt.close(self.fig)

    def show(self) -> None:
        """Show the figure if it is hidden.

        This is a helper method. It will generally display in a
        Jupyter notebook.
        """
        self.fig = plt.figure(**self.figure_opts)
        if self.title:
            self.fig.suptitle(self.title)
        plt.axis("off")
        plt.imshow(self.cloud.to_array(), interpolation="bilinear")


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
    opts: dict[str, Any] = Field(
        default_factory=lambda: {
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
        arbitrary_types_allowed=True,
        json_schema_extra=DocJSONSchema.model_json_schema(),
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
            return self._process_dtm()
        if isinstance(self.data, pd.DataFrame):
            return self._process_dataframe()
        if isinstance(self.data, list):
            return self._process_list()

        raise LexosException("Unsupported data type for MultiCloud.")

    def _process_dtm(self) -> list[dict[str, int | float]]:
        """Process a DTM into document frequency dictionaries."""
        if self.data.doc_term_matrix is None or self.data.doc_term_matrix.shape[0] == 0:
            raise LexosException("Empty DTM provided.")

        selected_docs = self._normalize_selected_docs(
            self.docs,
            range(self.data.doc_term_matrix.shape[0]),
            lambda label: self.data.labels.index(label),
        )

        return [self._process_dtm_row(doc_idx) for doc_idx in selected_docs]

    def _process_dataframe(self) -> list[dict[str, int | float]]:
        """Process a DataFrame into document frequency dictionaries."""
        if self.data.empty:
            raise LexosException("Empty DataFrame provided.")

        selected_docs = self._normalize_selected_docs(
            self.docs,
            range(len(self.data)),
            lambda label: self.data.index.get_loc(label),
        )

        return [self._process_dataframe_row(doc_idx) for doc_idx in selected_docs]

    def _process_list(self) -> list[dict[str, int | float]]:
        """Process a list of documents into word frequencies."""
        if not self.data:
            raise LexosException("No valid data provided for MultiCloud.")

        return [processors.process_data(doc, None, self.limit) for doc in self.data]

    def _normalize_selected_docs(
        self,
        docs: Optional[int | str | list[int] | list[str]],
        default_docs: range,
        label_mapper: Any,
    ) -> list[int | str]:
        """Normalize selection values into a list of document indices."""
        if docs is None:
            docs = default_docs
        if isinstance(docs, (int, str)):
            docs = [docs]

        return [label_mapper(doc) if isinstance(doc, str) else doc for doc in docs]

    def _process_dtm_row(self, doc_idx: int | str) -> dict[str, int | float]:
        if isinstance(doc_idx, str):
            doc_idx = self.data.labels.index(doc_idx)

        doc_row = self.data.doc_term_matrix[doc_idx]
        if hasattr(doc_row, "toarray"):
            doc_row = doc_row.toarray().flatten()

        return self._filter_positive_counts(
            {
                term: self._to_float(count)
                for term, count in zip(self.data.vectorizer.terms_list, doc_row)
            }
        )

    def _process_dataframe_row(self, doc_idx: int | str) -> dict[str, int | float]:
        if isinstance(doc_idx, str):
            doc_idx = self.data.index.get_loc(doc_idx)

        return self._filter_positive_counts(self.data.iloc[doc_idx].to_dict())

    def _filter_positive_counts(self, counts: dict[str, Any]) -> dict[str, int | float]:
        return {
            term: value
            for term, value in ((k, self._to_float(v)) for k, v in counts.items())
            if value > 0
        }

    def _to_float(self, value: Any) -> float:
        return float(value.item()) if hasattr(value, "item") else float(value)

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
        # Create a local figure without mutating global matplotlib style
        self.fig = plt.figure(figsize=self.figsize)

        # Calculate layout
        n = len(self.doc_data)
        if self.layout == "auto":
            columns = math.floor(math.sqrt(n))
            rows = math.ceil(n / columns)
        elif isinstance(self.layout, tuple):
            rows, columns = self.layout
        else:
            raise LexosException("Invalid layout specification.")

        # Add overall title
        if self.title:
            self.fig.suptitle(self.title, fontsize=16)

        # Generate the word clouds
        for i, doc_counts in enumerate(self.doc_data):
            self.wordcloud.generate_from_frequencies(doc_counts)
            ax = self.fig.add_subplot(rows, columns, i + 1)
            ax.imshow(self.wordcloud.to_array(), interpolation="bilinear")
            ax.axis("off")

            # Add label if provided
            if self.labels and i < len(self.labels):
                ax.set_title(self.labels[i])
            else:
                ax.set_title(f"Doc {i}")

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
