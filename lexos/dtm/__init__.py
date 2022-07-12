"""dtm.py."""
from typing import Any, List, Union

import pandas as pd
import spacy
from natsort import natsort_keygen, ns
from pydantic import BaseModel, ValidationError, validator

from lexos.exceptions import LexosException

# This should probably be moved to constants.py
SORTING_ALGORITHM = natsort_keygen(alg=ns.LOCALE)


class DtmData(BaseModel):
    """DtmData class.

    This model validates the input data for the DTM and, if necessary,
    coerces it to a list of token lists.
    """

    docs: List[Union[List[str], spacy.tokens.doc.Doc]]

    class Config:
        """Config class."""

        arbitrary_types_allowed = True

    @validator("docs", pre=True, always=True)
    def ensure_token_lists(cls, v):
        """Coerces input to a list of token lists where each token is a string."""
        tokens = []
        for doc in v:
            if isinstance(doc, spacy.tokens.doc.Doc):
                tokens.append([token.text for token in doc])
            elif isinstance(doc, list):
                if all(isinstance(sub, str) for sub in doc):
                    tokens.append(doc)
                else:
                    raise LexosException("Each list item must be a string.")
            else:
                raise LexosException("Could not parse the document list.")
        return tokens


class DTM:
    """Class for a document-term matrix."""

    def __init__(
        self,
        docs=List[Union[List[str], spacy.tokens.doc.Doc]],
        labels: List[str] = None,
    ) -> None:
        """Initialise the DTM.

        Args:
            docs (List[Union[List[str], spacy.tokens.doc.Doc]]): A list of spaCy docs or a list of token lists.
            labels (List[str]): A list of labels for the documents.

        Returns:
            None
        """
        self.docs = docs
        self.table = None
        if not labels:
            self.labels = ["doc" + str(i) for i in range(len(docs))]
        else:
            self.labels = labels
        self.vectorizer_settings = {}
        self.vectorizer = self.set_vectorizer(new=True)
        self.build()

    def build(self):
        """Build a new DTM matrix based on the current vectorizer."""
        doc_token_lists = DtmData(docs=self.docs).docs
        self.matrix = self.vectorizer.fit_transform(doc_token_lists)
        # Require explicit calling of get_table after each build to ensure table is up to date.
        # Ensures that the two processes can be kept separate if desired.
        self.table = None

    def get_table(self, transpose: bool = False) -> pd.DataFrame:
        """Get a Textacy document-term matrix as a pandas dataframe.

        Args:
            transpose (bool): If True, terms are columns and docs are rows.

        Returns:
                pd.Dataframe
        """
        if self.table is not None:
            return self.table
        else:
            rows = []
            for term in self.vectorizer.terms_list:
                row = [term]
                terms = self.vectorizer.vocabulary_terms[term]
                freq = self.matrix[0:, terms].toarray()
                [row.append(item[0]) for item in freq]
                rows.append(row)
            df = pd.DataFrame(rows, columns=["terms"] + self.labels)
            if transpose:
                df.rename({"terms": "docs"}, axis=1, inplace=True)
                df = df.T
            self.table = df
            return df

    def get_freq_table(
        self, rounding: int = 3, as_percent: bool = False
    ) -> pd.DataFrame:
        """Get a table with the relative frequencies of terms in each document.

        Args:
            rounding (int): The number of digits to round floats.
            as_percent (bool): Whether to return the frequencies as percentages.

        Returns:
            pd.DataFrame: A dataframe with the relative frequencies.
        """
        df = self.get_table().copy()
        df.set_index("terms", inplace=True)
        if as_percent:
            return df.apply(
                lambda row: ((row / row.sum()) * 100).round(rounding), axis=1
            ).reset_index()
        else:
            return df.apply(
                lambda row: row / row.sum().round(rounding), axis=1
            ).reset_index()

    def get_stats_table(
        self, stats: Union[List[str], str] = "sum", rounding: int = 3
    ) -> pd.DataFrame:
        """Get a table with the sum, mean, and/or median calculated for each row.

        Args:
            stats (Union[List[str], str]): One or more of "sum", "mean", and/or "median".
            rounding (int): The number of digits to round floats.

        Returns:
            pd.DataFrame: A dataframe with the calculated statistics.
        """
        df = self.get_table()
        tmp = df.copy()
        if "sum" in stats:
            tmp["sum"] = df.sum(axis=1, numeric_only=True)
        if "mean" in stats:
            tmp["mean"] = df.mean(axis=1, numeric_only=True).round(rounding)
        if "median" in stats:
            median = df.median(axis=1, numeric_only=True)
            tmp["median"] = median.round(rounding)
        return tmp

    def get_terms(self):
        """Get an alphabetical list of terms."""
        return self.vectorizer.vocabulary_terms

    def get_term_counts(
        self,
        sort_by: Union[list, List[str]] = ["terms", "sum"],
        ascending: Union[bool, List[bool]] = True,
        alg=SORTING_ALGORITHM,
    ) -> List[tuple]:
        """Get a list of term counts with optional sorting.

        Args:
            sort_by Union[list, List[str]]): The column(s) to sort by in order of preference.
            ascending (Union[bool, List[bool]]): Whether to sort values in ascending or descending order.

        Returns:
            List(tuple): A list of tuples containing terms and counts.
        """
        if alg != SORTING_ALGORITHM:
            self._validate_sorting_algorithm(alg)
        df = self.get_stats_table("sum").sort_values(
            by=sort_by, ascending=ascending, key=alg
        )
        terms = df["terms"].values.tolist()
        sums = df["sum"].values.tolist()
        return [(terms[i], sums[i]) for i, _ in enumerate(terms)]

    def least_frequent(self, max_n_terms: int = 100, start: int = 0) -> pd.DataFrame:
        """Get the most frequent terms in the DTM.

        Args:
            max_n_terms (int): The number of terms to return.
            start: int = 0: The start index in the DTM table.

        Returns:
            pd.DataFrame: The reduced DTM table.

        Note: This function should not be used if `min_df` or `max_df` is set in
        the vectorizer because the table will be cut twice.
        """
        df = self.get_stats_table("sum").sort_values(by="sum", ascending=True)
        df = df[start:]
        return df.tail(max_n_terms)

    def most_frequent(self, max_n_terms: int = 100, start: int = 0) -> pd.DataFrame:
        """Get the most frequent terms in the DTM.

        Args:
            max_n_terms (int): The number of terms to return.
            start: int = 0: The start index in the DTM table.

        Returns:
            pd.DataFrame: The reduced DTM table.

        Note: This function should not be used if `min_df` or `max_df` is set in
        the vectorizer because the table will be cut twice.
        """
        df = self.get_stats_table("sum").sort_values(by="sum", ascending=False)
        return df[start:max_n_terms]

    def set_vectorizer(
        self,
        tf_type: str = "linear",
        idf_type: str = None,
        dl_type: str = None,
        norm: Union[list, str] = None,
        min_df: Union[float, int] = 1,
        max_df: Union[float, int] = 1.0,
        max_n_terms: int = None,
        vocabulary_terms: Union[list, str] = None,
        new: bool = False,
    ):
        """Set the vectorizer.

        By default, returns a vectorizer that gets raw counts.
        """
        from textacy.representations.vectorizers import Vectorizer

        vectorizer = Vectorizer(
            tf_type=tf_type,
            idf_type=idf_type,
            dl_type=dl_type,
            norm=norm,
            min_df=min_df,
            max_df=max_df,
            max_n_terms=max_n_terms,
            vocabulary_terms=vocabulary_terms,
        )
        self.vectorizer_settings = {
            "tf_type": tf_type,
            "idf_type": idf_type,
            "norm": norm,
            "min_df": min_df,
            "max_df": max_df,
            "max_n_terms": max_n_terms,
        }
        if new:
            return vectorizer
        else:
            self.vectorizer = vectorizer

    def _validate_sorting_algorithm(self, alg: Any) -> bool:
        """Ensure that the specified sorting algorithm is a valid natsort locale.

        Args:
            alg: The sorting algorithm to validate.

        Returns:
            bool: Whether the sorting algorithm is valid.
        """
        if alg not in [e for e in ns]:
            locales = ", ".join([f"ns.{e.name}" for e in ns])
            err = (
                f"Invalid sorting algorithm: {alg}.",
                f"Valid algorithms for `alg` are: {locales}.",
                "See https://natsort.readthedocs.io/en/stable/api.html#natsort.ns.",
            )
            raise LexosException(" ".join(err))
        return True
