"""lexosdoc.py."""
import re
from collections import Counter
from typing import Any, Dict, List, Union

import pandas as pd
import spacy
from lexos.exceptions import LexosException

from lexos.tokenizer import extensions


class LexosDoc:
    """A wrapper class for a spaCy doc which allows for extra methods.

    A convenience that allows you to use Doc extensions without the
    underscore prefix.

    Note: There is probably no need for this class. We can just keep a
    library of functions in a file called `tokenizer.py` and import them.
    If certain functions get used commonly, they can be turned into Doc
    extensions.
    """

    def __init__(self, doc: object):
        """Initialize a LexosDoc object."""
        if isinstance(doc, spacy.tokens.doc.Doc):
            self.doc = doc
        else:
            raise LexosException("The input must be a spaCy doc.")

    def get_term_counts(
        self,
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
            self (object): A spaCy doc.
            limit (int): The maximum number of tokens to count.
            start (Any): The index of the first token to count.
            end (Any): The index of the last token to count after limit is applied.
            filters (List[Union[Dict[str, str], str]]): A list of Doc attributes to ignore.
            regex (bool): Whether to match the dictionary value using regex.
            normalize (bool): Whether to return raw counts or relative frequencies.
            normalize_with_filters (bool): Whether to normalize based on the number
                of tokens after filters are applied.
            as_df (bool): Whether to return a pandas dataframe.

        Returns:
            Union[List, pd.DataFrame]: A list of word count tuples for
            each token in the doc. Alternatively, a pandas dataframe.
        """
        tokens = []
        bool_filters = []
        dict_filters = {}
        if filters:
            self._validate_filters(filters)
            for filter in filters:
                if isinstance(filter, dict):
                    dict_filters[list(filter.keys())[0]] = list(filter.values())[0]
                else:
                    bool_filters.append(filter)
        tokens = [
            token.text
            for token in self.doc
            if self._bool_filter(token, bool_filters)
            and self._dict_filter(token, dict_filters, regex=regex)
        ]
        term_counts = Counter(tokens).most_common(limit)[start:end]
        columns = ["term", "count"]
        if normalize_with_filters:
            normalize = True
            num_tokens = len(tokens)
        else:
            num_tokens = len(self.doc)
        if normalize:
            term_counts = [(x[0], x[1] / num_tokens) for x in term_counts]
            columns[1] = "frequency"
        if as_df:
            return self._dataframe(term_counts, columns)
        else:
            return term_counts

    def get_tokens(self):
        """Return a list of tokens in the doc."""
        return [token.text for token in self.doc]

    def get_token_attrs(self):
        """Get a list of attributes for each token in the doc.

        Returns a dict with "spacy_attributes" and "extensions".

        Note: This function relies on sampling the first token in a doc
        to compile the list of attributes. It does not check for consistency.
        Currently, it is up to the user to reconcile inconsistencies between
        docs.
        """
        sample = self.doc[0]
        attrs = sorted([x for x in dir(sample) if not x.startswith("__") and x != "_"])
        exts = sorted(
            [f"_{x}" for x in dir(sample._) if x not in ["get", "has", "set"]]
        )
        return {"spacy_attributes": attrs, "extensions": exts}

    def to_dataframe(
        self, cols: List[str] = ["text"], show_ranges: bool = True
    ) -> pd.DataFrame:
        """Get a pandas dataframe of the doc attributes.

        Args:
            cols: A list of columns to include in the dataframe.
            show_ranges: Whether to include the token start and end positions in the dataframe.

        Returns a pandas dataframe of the doc attributes.

        Note: It is a good idea to call `LexosDoc.get_token_attrs()` first
        to check which attributes are available for the doc.
        """
        rows = []
        for i, token in enumerate(self.doc):
            t = []
            for col in cols:
                t.append(getattr(token, col))
            if show_ranges:
                ranges = self.doc.to_json()["tokens"][i]
                t.append(ranges["start"])
                t.append(ranges["end"])
            rows.append(t)
        if show_ranges:
            cols = cols + ["start", "end"]
        return self._dataframe(rows, cols)

    def _bool_filter(self, token: object, filters: List[str]) -> bool:
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

    def _dataframe(self, rows: List[dict], columns: List[str]) -> pd.DataFrame:
        """Return a pandas dataframe of the doc attributes.

        Args:
            rows (List[dict]): A list of dicts with the doc attributes.
            columns (List[str]): A list of column names.

        Returns:
            pd.DataFrame: A pandas dataframe of the doc attributes.

        Raises:
            LexosException: If a pandas exception occurs.
        """
        try:
            return pd.DataFrame(rows, columns=columns)
        except Exception as e:
            raise LexosException(e)

    def _dict_filter(
        self, token: object, filters: List[Dict[str, str]], regex: bool = False
    ) -> bool:
        """Filter a token based on a list of dictionary filters.

        Args:
            token (object): A spaCy token.
            filters (List[Dict[str, str]]): A list of filter dictionaries with keys
            as spaCy token attributes.
            regex (bool): Whether to match the dictionary value using regex.

        Returns:
            bool: Whether the token passes the filters.
        """
        if not isinstance(token, spacy.tokens.Token):
            raise LexosException("The input must be a spaCy token.")
        if not isinstance(regex, bool):
            raise LexosException("The regex flag must be a boolean.")
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

    def _validate_filters(self, filters: List[str]) -> None:
        """Ensure that filters are in the correct format.

        Args:
            filters (Union[List[Dict[str, str]], List[str])): A list of filter dictionaries with keys
            or a list of boolean filters (the names of spaCy token attributes).

        Returns:
            None

        Raises:
            LexosException: If the format for the filter is not correct.
        """
        message = "The filter must be a list of filter dictionaries with keys or a list of boolean filters (the names of spaCy token attributes)"
        if not isinstance(filters, list) or any(
            not isinstance(x, (dict, str)) for x in filters
        ):
            raise LexosException(message)
