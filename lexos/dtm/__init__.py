"""dtm.py."""
import re
from collections import Counter
from typing import Any, Dict, List, Union

import pandas as pd
from natsort import natsort_keygen, ns

from lexos.tokenizer.lexosdoc import LexosDoc

# This should probably be moved to constants.py
SORTING_ALGORITHM = natsort_keygen(alg=ns.LOCALE)


class DTM:
    """Class for a document-term matrix."""

    def __init__(self, docs=List[Union[list, object]], labels=List[str]):
        """Initialise the DTM."""
        self.docs = docs
        self.table = None
        self.labels = labels
        self.vectorizer_settings = {}
        self.vectorizer = self.set_vectorizer(new=True)
        self.build()

    def build(self):
        """Build a new DTM matrix based on the current vectorizer."""
        # doc_tokens = [[token.text for token in doc] for doc in self.docs]
        doc_tokens = self._validate_input()
        self.matrix = self.vectorizer.fit_transform(doc_tokens)
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
            tmp["sum"] = df.sum(axis=1)
        if "mean" in stats:
            tmp["mean"] = df.mean(axis=1).round(rounding)
        if "median" in stats:
            median = df.median(axis=1)
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
        df = self.get_stats_table("sum").sort_values(
            by=sort_by, ascending=ascending, key=alg
        )
        terms = df["terms"].values.tolist()
        sums = df["sum"].values.tolist()
        return [(terms[i], sums[i]) for i, _ in enumerate(terms)]

    def least_frequent(self, max_n_terms: int = 100, start: int = -1) -> pd.DataFrame:
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
        return df[start:max_n_terms]

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

    def _validate_input(self) -> List[List[str]]:
        """Make sure that the DTM input consists of valid token lists or spaCy documents.

        Returns:
            List[List[str]]: A list of token lists.
        """
        input = self.docs
        # If all items are of the same data type
        if all(isinstance(sub, type(input[0])) for sub in input[1:]):
            # If all items are strings
            if all(isinstance(sub, str) for sub in input[1:]):
                return input
            # If all items are objects
            elif all(isinstance(sub, object) for sub in input[1:]):
                return [[token.text for token in doc] for doc in input]
            else:
                raise Exception("Input contained an unrecognized data type.")
        else:
            raise Exception("All items in the list must be of the same data type.")


# The code below is technically superseded by the DTM class,
# but, it is a nice convenience method.


def get_doc_term_counts(
    docs: List[object],
    limit: int = None,
    start: Any = 0,
    end: Any = None,
    filters: List[Union[Dict[str, str], str]] = None,
    regex: bool = False,
    normalize: bool = False,
    normalize_with_filters: bool = False,
    as_df=False,
) -> Union[List, pd.DataFrame]:
    """Get a list of word counts for each token in the doc.

    Args:
        docs (List[object]): A list of spaCy docs.
        limit (int): The maximum number of tokens to count.
        start (int): The index of the first token to count.
        end (int): The index of the last token to count after limit is applied.
        filters (List[Union[Dict[str, str], str]]): A list of Doc attributes to ignore.
        regex (bool): Whether to match the dictionary value using regex.
        normalize (bool): Whether to return raw counts or relative frequencies.
        normalize_with_filters (bool): Whether to normalize based on the number of tokens after filters are applied.
        as_df (bool): Whether to return a pandas dataframe.

    Returns:
        Union[List, pd.DataFrame]: A list of word count tuples for
        each token in the doc. Alternatively, a pandas dataframe.
    """
    tokens = []
    bool_filters = []
    dict_filters = {}
    if filters:
        for filter in filters:
            if isinstance(filter, dict):
                dict_filters[list(filter.keys())[0]] = list(filter.values())[0]
            else:
                bool_filters.append(filter)
    tokens = [
        token.text
        for doc in docs
        for token in doc
        if _bool_filter(token, bool_filters)
        and _dict_filter(token, dict_filters, regex=regex)
    ]
    term_counts = Counter(tokens).most_common(limit)[start:end]
    columns = ["term", "count"]
    if normalize_with_filters:
        normalize = True
        num_tokens = len(tokens)
    else:
        num_tokens = sum([len(doc) for doc in docs])
    if normalize:
        term_counts = [(x[0], x[1] / num_tokens) for x in term_counts]
        columns[1] = "frequency"
    if as_df:
        return pd.DataFrame(term_counts, columns=columns)
    else:
        return term_counts


def _bool_filter(token: object, filters: List[str]) -> bool:
    """Filter a token based on a list of boolean filters.

    Args:
        token (object): A spaCy token.
        filters (str): A list of boolean filters (the names of spaCy token attributes).

    Returns:
        bool: Whether the token passes the filters.
    """
    if filters and filters != []:
        for filter in filters:
            if getattr(token, filter):
                return False
            else:
                return True
    else:
        return True


def _dict_filter(token, filters: List[Dict[str, str]], regex: bool = False) -> bool:
    """Filter a token based on a list of dictionary filters.

    Args:
        token (object): A spaCy token.
        filters (List[Dict[str, str]]): A list of filter dictionaries with keys as spaCy token attributes.
        regex (bool): Whether to match the dictionary value using regex.

    Returns:
        bool: Whether the token passes the filters.
    """
    if filters and filters != {}:
        for filter, value in filters.items():
            if (
                regex
                and re.search(re.compile(value), getattr(token, filter)) is not None
            ):
                return False
            elif getattr(token, filter) == value:
                return False
            else:
                return True
    else:
        return True
