"""Public API for the `lexos.dtm` package.

Phase 1 export surface:
- DTM
- Vectorizer
"""

from typing import Callable, Iterable, Literal, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from natsort import natsorted, ns
from pydantic import BaseModel, ConfigDict, Field, validate_call
from spacy.schemas import DocJSONSchema
from spacy.tokens import Doc
from textacy.representations.vectorizers import Vectorizer as TextacyVectorizer

from lexos.constants import SORTING_ALGORITHM
from lexos.exceptions import LexosException

validation_config = ConfigDict(
    arbitrary_types_allowed=True, json_schema_extra=DocJSONSchema
)


class Vectorizer(BaseModel):
    """Wrapper class for Textacy's Vectorizer."""

    @validate_call(config=validation_config)
    def __call__(
        self,
        *,
        tf_type: Literal["linear", "sqrt", "log", "binary"] = "linear",
        idf_type: Optional[Literal["linear", "sqrt", "log"]] = None,
        dl_type: Optional[Literal["linear", "sqrt", "log"]] = None,
        norm: Optional[Literal["l1", "l2"]] = None,
        min_df: int | float = 1,
        max_df: int | float = 1.0,
        max_n_terms: Optional[int] = None,
        vocabulary_terms: Optional[dict[str, int] | Iterable[str]] = None,
    ) -> TextacyVectorizer:
        """Return a Textacy Vectorizer object.

        Args:
            tf_type (str): Term frequency type.
            idf_type (str): Inverse document frequency type.
            dl_type (str): Document length type.
            norm (str): Normalization type.
            min_df (int | float): Minimum document frequency.
            max_df (int | float): Maximum document frequency.
            max_n_terms (int): Maximum number of terms.
            vocabulary_terms (dict | Iterable[str]): Vocabulary terms.

        Returns:
            TextacyVectorizer: A Textacy Vectorizer object.
        """
        return TextacyVectorizer(
            tf_type=tf_type,
            idf_type=idf_type,
            dl_type=dl_type,
            norm=norm,
            min_df=min_df,
            max_df=max_df,
            max_n_terms=max_n_terms,
            vocabulary_terms=vocabulary_terms,
        )


__all__ = ["DTM", "Vectorizer"]


class DTM(BaseModel):
    """Class for a document-term matrix."""

    docs: Optional[list[list[str] | Doc]] = Field(
        default=None, description="A list of spaCy docs or a list of token lists."
    )
    labels: Optional[list[str]] = Field(
        default=None, description="A list of labels for the documents."
    )
    vectorizer: Optional[Callable] = Field(
        default=TextacyVectorizer,
        description="A callable Vectorizer. Must have a fit_transform() method.",
    )
    alg: Optional[ns] = Field(
        default=ns.LOCALE, description="The sorting algorithm to use."
    )
    doc_term_matrix: Optional[sp.spmatrix] = Field(
        default=None, description="The document-term matrix."
    )

    model_config = validation_config

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape of the DTM.

        Returns:
            tuple[int, int]: The shape of the DTM.
        """
        if self.doc_term_matrix is None:
            raise LexosException("DTM must be built before accessing its shape")
        return self.doc_term_matrix.shape

    @property
    def sorted_terms_list(self) -> list[str]:
        """Return a natsorted list of terms in the DTM.

        Returns:
            list[str]: A natsorted list of terms in the DTM.
        """
        if self.vectorizer is None or not hasattr(self.vectorizer, "terms_list"):
            # This handles cases where vectorizer might be None or not properly set up
            # Before attempting to access terms_list.
            raise LexosException(
                "Vectorizer or its 'terms_list' attribute is not available to get sorted terms."
            )
        return natsorted(self.vectorizer.terms_list, reverse=False, alg=self.alg)

    @property
    def sorted_term_counts(self) -> dict[str, int]:
        """Return a natsorted dict of terms and their TOTAL counts across all documents in the DTM.

        Returns:
            dict[str, int]: A natsorted dict of terms and their total counts.
        """
        # 1. Handle edge cases: DTM not built or empty
        if self.doc_term_matrix is None or self.doc_term_matrix.shape[1] == 0:
            return {}  # Return an empty dictionary if no DTM or no terms

        # 2. Get the terms (column names) from the vectorizer
        if not hasattr(self.vectorizer, "terms_list") or not self.vectorizer.terms_list:
            # Fallback for unexpected mock scenarios or custom vectorizers
            # A well-formed DTM should always have terms_list if it has a non-empty matrix
            raise LexosException(
                "Vectorizer must have 'terms_list' attribute to get sorted term counts."
            )
        terms = self.vectorizer.terms_list

        # 3. Calculate the sum of each term (column) across all documents
        # The .sum(axis=0) returns a 1xN matrix. .A1 flattens it to a 1D numpy array for sparse matrices.
        term_totals = self.doc_term_matrix.sum(axis=0).A1.astype(int)

        # 4. Create a dictionary mapping terms to their total counts
        # Ensure terms and term_totals have the same length
        if len(terms) != len(term_totals):
            raise LexosException(
                f"Mismatch between number of terms ({len(terms)}) "
                f"and calculated term totals ({len(term_totals)})."
            )
        term_counts_dict = dict(zip(terms, term_totals))

        # 5. Natsort the items (term-count pairs) by term name, then convert back to a dictionary
        # This uses the natsort library based on the instance's 'alg'
        sorted_items = natsorted(
            term_counts_dict.items(), key=lambda item: item[0], alg=self.alg
        )
        return dict(sorted_items)

    def __init__(
        self, **data: dict[str, list | str | Callable | ns | sp.spmatrix]
    ) -> None:
        """Initialize the DTM class.

        Args:
            data (dict): A dictionary of data to initialize the DTM.
                - docs: A list of spaCy docs or a list of token lists.
                - labels: A list of labels for the documents.
                - vectorizer: A callable Vectorizer. Must have a fit_transform() method.
                - alg: The sorting algorithm to use (default is ns.LOCALE).
                - doc_term_matrix: The document-term matrix (optional).
                - **kwargs: Additional keyword arguments to pass to the vectorizer.
        """
        super().__init__(**data)
        # Make sure that a vectorizer instance is called with any keyword arguments
        kwargs = {
            k: v
            for k, v in data.items()
            if k not in ["docs", "labels", "vectorizer", "alg", "doc_term_matrix"]
        }
        self.vectorizer = self.vectorizer(**kwargs)

    def __call__(
        self,
        docs: Optional[list[list[str] | Doc]],
        labels: Optional[Iterable[str]],
        **kwargs: dict[str, str | int | float | bool],
    ) -> None:
        """Call method for DTM class.

        Args:
            docs (list[list[str] | Doc]): A list of spaCy docs or a list of token lists.
            labels (list[str]): A list of labels for the documents.
            **kwargs (dict): Additional keyword arguments to pass to the vectorizer.

        Note:
            - If you want to filter the docs by token attributes, you can do so beforehand
            and pass the filtered docs to this method.
            - If you want to sort the dataframe, use pandas sort_values(), but make sure to
              pass `SORTING_ALGORITHM` or `self.alg` to the `key` parameter for natsorting.
        """
        if docs is None or len(docs) == 0:
            raise LexosException(
                "You must provide a list of docs or a list of lists of token strings."
            )

        # Ensure that the vectorizer is not None
        if self.vectorizer is None:
            self.vectorizer = TextacyVectorizer()

        # Update the vectorizer with any additional keyword arguments
        self._update_vectorizer(**kwargs)

        # Make sure the sorting algorithm is valid
        self._validate_sorting_algorithm()

        # Coerce the docs to a list of token lists
        if docs:
            self.docs = docs
        self.docs = [
            [token.text for token in doc] if isinstance(doc, Doc) else doc
            for doc in self.docs
        ]
        # Set the instance labels
        if labels:
            self.labels = labels
        elif self.labels is None:
            self.labels = [f"Doc{i + 1}" for i in range(len(self.docs))]

        # Make sure the number of docs matches the number of labels
        if len(self.docs) != len(self.labels):
            raise LexosException("The number of docs must match the number of labels.")

        # Call the vectorizer to build the DTM
        try:
            # Fit the vectorizer to the docs and build the document-term matrix
            self.doc_term_matrix = self.vectorizer.fit_transform(self.docs)

        except Exception as e:
            raise LexosException(f"Error building DTM: {e}")

    def _get_term_percentages(
        self,
        df: pd.DataFrame,
        rounding: int = 3,
        as_str: bool = False,
        sum: bool = False,
        mean: bool = False,
        median: bool = False,
    ) -> pd.DataFrame:
        """Return a dataframe with term frequencies.

        Args:
            df (pd.DataFrame): The dataframe to convert to percentages.
            rounding (int): The number of decimal places to round to.
            as_str (bool): Whether to return the terms as strings.
            sum (bool): Whether to include a column for the sum of each row.
            mean (bool): Whether to include a column for the mean of each row.
            median (bool): Whether to include a column for the median of each row.

        Returns:
            pd.DataFrame: A dataframe with term frequencies.
        """
        total_sum = df.values.sum()
        if total_sum != 0:
            df = df.div(total_sum).mul(100)
        else:
            df = df.mul(0.0)
        # Densify after arithmetic to ensure round/astype work on all backends
        if hasattr(df, "sparse"):
            df = df.sparse.to_dense()
        if sum:
            df["Total"] = df.sum(numeric_only=True, axis=1)
        if mean:
            df["Mean"] = df.mean(numeric_only=True, axis=1)
        if median:
            df["Median"] = df.median(numeric_only=True, axis=1)
        if rounding:
            df = df.round(rounding)
        if as_str:
            df = df.astype(str) + "%"
        return df

    def _update_vectorizer(self, **kwargs: dict[str, str | int | float | bool]) -> None:
        """Update the vectorizer with additional keyword arguments.

        Args:
            kwargs (dict): Additional keyword arguments to update the vectorizer.
        """
        _PARAMS = (
            "tf_type",
            "idf_type",
            "dl_type",
            "norm",
            "min_df",
            "max_df",
            "max_n_terms",
            "vocabulary_terms",
        )
        params = {k: getattr(self.vectorizer, k) for k in _PARAMS}
        params.update(kwargs)
        self.vectorizer = TextacyVectorizer(**params)

    def _validate_sorting_algorithm(self) -> bool:
        """Ensure that the specified sorting algorithm is a valid natsort locale.

        Returns:
            bool: Whether the sorting algorithm is valid.
        """
        if self.alg not in [e for e in ns]:
            locales = ", ".join([f"ns.{e.name}" for e in ns])
            err = (
                f"Invalid sorting algorithm: {self.alg}.",
                f"Valid algorithms for `alg` are: {locales}.",
                "See https://natsort.readthedocs.io/en/stable/api.html#natsort.ns.",
            )
            raise LexosException(" ".join(err))
        return True

    def to_df(
        self,
        by: Optional[list | list[str]] = None,
        ascending: Optional[bool | list[bool]] = True,
        as_percent: Optional[bool] = False,
        rounding: Optional[int] = 3,
        transpose: Optional[bool] = False,
        sum: Optional[bool] = False,
        mean: Optional[bool] = False,
        median: Optional[bool] = False,
    ) -> pd.DataFrame:
        """Return the whole DTM as a pandas dataframe.

        Args:
            by (Optional[list | list[str]]): The column(s) to sort by.
            ascending (Optional[bool | list[bool]]): Whether to sort in ascending order.
            as_percent (Optional[bool]): Whether to return the terms as percentages.
            rounding (Optional[int]): The number of decimal places to round to.
            transpose (Optional[bool]): Whether to transpose the dataframe.
            sum (Optional[bool]): Whether to include a column for the sum of each row.
            mean (Optional[bool]): Whether to include a column for the mean of each row.
            median (Optional[bool]): Whether to include a column for the median of each row.

        Returns:
            pd.DataFrame: The DTM as a pandas dataframe.
        """
        if by is None:
            by = self.labels[0]
        try:
            if sp.issparse(self.doc_term_matrix):
                df = pd.DataFrame.sparse.from_spmatrix(
                    self.doc_term_matrix,
                    columns=self.vectorizer.terms_list,
                    index=self.labels,
                ).T
            else:
                df = pd.DataFrame(
                    self.doc_term_matrix,
                    columns=self.vectorizer.terms_list,
                    index=self.labels,
                ).T
        except Exception as e:
            raise LexosException(f"Error converting DTM to DataFrame: {e}")
        if as_percent:
            df = self._get_term_percentages(
                df,
                rounding=rounding,
                # Explicitly control formatting based on the value of as_percent
                as_str=True if as_percent == "string" else False,
                sum=sum,
                mean=mean,
                median=median,
            )
            df = df.sort_values(by=by, ascending=ascending)
            if transpose:
                df = df.T
        else:
            # Support sorting by statistics columns that are added later in this branch.
            # Compute temporary sort columns when needed, then remove them after sorting.
            sort_by = by
            temp_sort_cols = {}
            sort_keys = by if isinstance(by, list) else [by]
            for key in sort_keys:
                if key == "Total" and sum:
                    temp_sort_cols[key] = "__lexos_sort_total__"
                elif key == "Mean" and mean:
                    temp_sort_cols[key] = "__lexos_sort_mean__"
                elif key == "Median" and median:
                    temp_sort_cols[key] = "__lexos_sort_median__"

            if temp_sort_cols:
                if hasattr(df, "sparse"):
                    df = df.sparse.to_dense()
                for key, temp_col in temp_sort_cols.items():
                    if key == "Total":
                        df[temp_col] = df.sum(numeric_only=True, axis=1)
                    elif key == "Mean":
                        df[temp_col] = df.mean(numeric_only=True, axis=1)
                    elif key == "Median":
                        df[temp_col] = df.median(axis=1)

                if isinstance(by, list):
                    sort_by = [temp_sort_cols.get(key, key) for key in by]
                else:
                    sort_by = temp_sort_cols.get(by, by)

            df = df.sort_values(by=sort_by, ascending=ascending)
            if temp_sort_cols:
                df = df.drop(columns=list(temp_sort_cols.values()))
            if transpose:
                df = df.T
            if sum or mean or median:
                if hasattr(df, "sparse"):
                    df = df.sparse.to_dense()
                if sum:
                    df["Total"] = df.sum(numeric_only=True, axis=1)
                if mean:
                    df["Mean"] = df.mean(numeric_only=True, axis=1)
                if median:
                    df["Median"] = df.median(axis=1)
        return df
