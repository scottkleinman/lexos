"""dtm.py."""
from typing import Any, Dict, List, Union

from collections import Counter
import re

import pandas as pd


def get_doc_term_counts(docs,
                    limit: int = None,
                    start: Any = 0,
                    end: Any = None,
                    filters: List[Union[Dict[str, str], str]] = None,
                    regex: bool = False,
                    as_df = False) -> Union[List, pd.DataFrame]:
    """Get a list of word counts for each token in the doc.

    Args:
        self: A spaCy doc.
        limit: The maximum number of tokens to count.
        start: The index of the first token to count.
        end: The index of the last token to count after limit is applied.
        filters: A list of Doc attributes to ignore.
        regex (bool): Whether to match the dictionary value using regex.
        as_df: Whether to return a pandas dataframe.

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
    if as_df:
        return pd.DataFrame(term_counts, columns=["term", "count"])
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
        filters (List[Dict[str, str]]): A list of filter dictionaries with keys
        as spaCy token attributes.
        regex (bool): Whether to match the dictionary value using regex.

    Returns:
        bool: Whether the token passes the filters.
    """
    if filters and filters != {}:
        for filter, value in filters.items():
            if regex and re.search(re.compile(value), getattr(token, filter)) is not None:
                return False
            elif getattr(token, filter) == value:
                return False
            else:
                return True
    else:
        return True

