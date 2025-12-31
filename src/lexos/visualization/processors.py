"""processors.py.

This module contains functions to process data from various source types into term frequency dictionaries.

Last Update: August 11, 2025
Last Tested: December 5, 2025
"""

from collections import Counter
from itertools import chain
from typing import Any, Iterator, Optional

import pandas as pd
from pydantic import ConfigDict, Field, validate_call
from spacy.schemas import DocJSONSchema
from spacy.tokens import Doc, Span, Token

from lexos.dtm import DTM
from lexos.exceptions import LexosException
from lexos.util import ensure_list


@validate_call(config=ConfigDict(allow_arbitrary_types=True))
def process_data(
    data: Any,
    docs: Optional[int | str | list[int] | list[str]] = None,
    limit: Optional[int] = Field(
        None, gt=0, description="Limit on number of terms to return"
    ),
) -> dict[str, int]:
    """Process any supported data type into a consistent format of term counts.

    Args:
        data: The input data to process
        docs: Optional document selection for multi-document data
        limit: Optional limit on number of terms to return

    Returns:
        dict[str, int]: Dictionary with terms as keys and counts as values

    Raises:
        LexosException: If data type is unsupported
    """
    # Handle simple string input
    if isinstance(data, str):
        counts = Counter(data.split())  # TODO: Use better tokenizer

    # Handle spaCy objects
    elif isinstance(data, (Doc, Span)):
        counts = Counter([token.text for token in data])

    # Handle dictionary input (already in correct format)
    elif isinstance(data, dict):
        counts = Counter(data)

    # Handle list inputs
    elif isinstance(data, list):
        counts = _process_list_data(data, docs)

    # Handle DTM objects
    elif isinstance(data, DTM):
        counts = process_dtm(data, docs)

    # Handle DataFrame objects
    elif isinstance(data, pd.DataFrame):
        counts = process_dataframe(data, docs)

    # Unsupported data type
    else:
        raise LexosException(
            f"Unsupported data type: {type(data)}. "
            "Supported types: str, dict, list, DTM, DataFrame, spaCy Doc/Span objects."
        )

    # WARNING: This renders the code unusable if the data contains float counts
    # such as topic model distributions. It doesn't seem necessary for any of
    # our current use cases, so I'm commenting it out for now.
    # Ensure counts are integers
    # counts = Counter({k: int(v) for k, v in counts.items()})

    # Limit the number of terms if specified
    if limit is not None:
        counts = Counter(dict(counts.most_common(limit)))

    return dict(counts)


def _process_list_data(
    data: list, docs: Optional[int | str | list[int] | list[str]] = None
) -> Counter:
    """Process list-type data inputs.

    Args:
        data: List data to process
        docs: Optional document selection

    Returns:
        Counter: Counter object with term counts
    """
    if not data:
        return Counter()

    # Ensure all items in the list are of the same type
    first_item = data[0]
    first_type = type(first_item)
    if not all(isinstance(x, first_type) for x in data):
        raise LexosException(
            f"Mixed types found in list: {first_type} and {[type(x) for x in data]}"
        )

    # List of lists
    if isinstance(first_item, list):
        return process_list(data, docs)

    # List of spaCy objects
    if isinstance(first_item, (Doc, Span)):
        return process_docs(data, docs)

    # List of tokens
    if isinstance(first_item, Token):
        return Counter([token.text for token in data])

    # Simple list of strings
    return process_item(data)


def filter_docs(
    df: pd.DataFrame, docs: Optional[list[int] | list[str]] = None
) -> pd.DataFrame:
    """Filter the documents in a DTM.

    Args:
        df: A Document Term Matrix.
        docs: A list of document indices or labels to filter the DTM.

    Returns:
        A filtered DTM.
    """
    if docs:
        if isinstance(docs[0], str):
            return df[docs]
        elif isinstance(docs[0], int):
            return df.iloc[:, docs]
    return df


@validate_call(
    config=ConfigDict(
        arbitrary_types_allowed=True, json_schema_extra=DocJSONSchema.schema()
    )
)
def process_dataframe(
    df: pd.DataFrame, docs: Optional[int | str | list[int] | list[str]] = None
) -> Counter:
    """Generate a term frequency dictionary from a DTM.

    Args:
        df (pd.DataFrame): A Document Term Matrix object.
        docs (Optional[int | str | list[int] | list[str]]): A list of document indices or labels to filter the DTM.

    Returns:
        Counter: A Counter object with the terms as keys and the counts as values.
    """
    # Filter the documents
    df = filter_docs(df, ensure_list(docs))
    # Add the counts
    df = df.copy()
    df["counts"] = df.sum(axis=1)
    # Remove terms with zero counts
    df = df.query("counts > 0")
    # Return the counts as a Counter
    return Counter(df["counts"].to_dict())


@validate_call(
    config=ConfigDict(
        arbitrary_types_allowed=True, json_schema_extra=DocJSONSchema.schema()
    )
)
def process_dtm(
    dtm: DTM, docs: Optional[int | str | list[int] | list[str]] = None
) -> dict[str, int]:
    """Generate a term frequency dictionary from a DTM.

    Args:
        dtm (DTM): A Document Term Matrix object.
        docs (Optional[int | str | list[int] | list[str]]): A list of document indices or labels to filter the DTM.

    Returns:
        dict[str, int]: A dictionary with the terms as keys and the counts as values.
    """
    df = dtm.to_df()
    # Filter the documents
    df = filter_docs(df, ensure_list(docs))
    return process_dataframe(df)


@validate_call(
    config=ConfigDict(
        arbitrary_types_allowed=True, json_schema_extra=DocJSONSchema.schema()
    )
)
def process_list(
    data: list[list[Doc | Span] | list[str] | list[Token]],
    docs: Optional[int | list[int]],
) -> Counter:
    """Process a list of docs, spans, strings, or tokens.

    Args:
        data (list[list[Doc | Span] | list[str] | list[Token]]): The data.
        docs (Optional[int | list[int]]): A list of document indices to be selected from the DTM.

    Returns:
        Counter: A Counter object with the terms as keys and the counts as values.
    """
    if docs:
        # Filter the docs
        docs = ensure_list(docs)
        data = [item for i, item in enumerate(data) if i in docs]
        # Flatten the list
        data = list(chain(*data))
    # Get the terms
    if all(isinstance(item, str) for item in data):
        terms = [item for item in data]
    elif all(isinstance(item, Token) for item in data):
        terms = [item.text for item in data]
    elif all(isinstance(item, (Doc, Span)) for item in data):
        terms = [t.text for doc in data for t in doc]
    else:
        terms = list(chain(*data))
    return Counter(terms)


@validate_call(
    config=ConfigDict(
        arbitrary_types_allowed=True, json_schema_extra=DocJSONSchema.schema()
    )
)
def process_docs(
    data: list[Doc] | list[Span], docs: Optional[int | list[int]]
) -> Counter:
    """Process multiple docs or spans.

    Args:
        data (list[Doc] | list[Span]): The data.
        docs (Optional[int | list[int]]): A list of document indices to be selected from the DTM.

    Returns:
        Counter: A Counter object with the terms as keys and the counts as values.
    """
    if docs:
        # Filter the docs
        docs = ensure_list(docs)
        data = [item for i, item in enumerate(data) if i in docs]
    # Get the terms
    terms = [[token.text for token in doc] for doc in data]
    # Flatten the list
    terms = list(chain(*terms))
    return Counter(terms)


@validate_call(
    config=ConfigDict(
        arbitrary_types_allowed=True, json_schema_extra=DocJSONSchema.schema()
    )
)
def process_item(
    data: Doc | Span | list[str] | list[Token],
) -> Counter:
    """Process single docs, spans, and strings, or flat lists of strings or tokens.

    Args:
        data (Doc | Span | list[str] | list[Token]): The data.

    Returns:
        dict[str, int]: A dictionary with the terms as keys and the counts as values.
    """
    # Get the terms
    if isinstance(data, list) and isinstance(data[0], str):
        terms = [item for item in data]
    elif isinstance(data, list) and isinstance(data[0], Token):
        terms = [item.text for item in data]
    elif isinstance(data, (Doc, Span)):
        terms = [t.text for t in data]
    return Counter(terms)


@validate_call(
    config=ConfigDict(
        arbitrary_types_allowed=True, json_schema_extra=DocJSONSchema.schema()
    )
)
def multicloud_processor(
    data: DTM
    | pd.DataFrame
    | list[Doc]
    | list[Span]
    | list[list[str]]
    | list[list[Token]]
    | list[dict[str, int]],
    docs: Optional[int | str | list[int] | list[str]] = None,
) -> list[dict[str, int]]:
    """Process data into list of term-count dicts for multicloud visualization.

    Args:
        data (DTM | pd.DataFrame | list[Doc] | list[Span] | list[list[str]] | list[list[Token]] | list[dict[str, int]]]): The data.
        docs (Optional[int | str | list[int] | list[str]]): A list of document indices or labels to be selected from the DTM.

    Returns:
        list[dict[str, int]]: A list of dictionaries with the terms as keys and the counts as values.
    """
    # Convert DTM to DataFrame
    if isinstance(data, DTM):
        data = data.to_df()

    # Process DataFrame
    if isinstance(data, pd.DataFrame):
        df = filter_docs(data, ensure_list(docs))
        records = df.T.to_dict(orient="records")
        # Eliminate tokens with zero counts in each doc
        return [{k: v for k, v in record.items() if v != 0} for record in records]

    # Process other data types
    else:
        if docs:
            # Filter the docs
            docs = ensure_list(docs)
            if isinstance(docs[0], str):
                raise LexosException(
                    "Filtering by document labels is not yet supported for your data type. You may use list index numbers to select documents for processing."
                )
            else:
                data = [item for i, item in enumerate(data) if i in docs]
        try:
            # Docs and Spans
            if isinstance(data[0], (Doc, Span)):
                return [dict(Counter([token.text for token in doc])) for doc in data]

            # Lists of dicts
            elif isinstance(data, list) and isinstance(data[0], dict):
                return data

            # Lists of strings
            elif isinstance(data[0][0], str):
                return [dict(Counter(doc)) for doc in data]

            # Lists of Tokens
            elif isinstance(data[0][0], Token):
                return [dict(Counter([token.text for token in doc])) for doc in data]
        except IndexError:
            raise LexosException(
                "Data is empty or not in the expected format. "
                "Ensure you are passing a non-empty list of documents, spans, or strings."
            )


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
