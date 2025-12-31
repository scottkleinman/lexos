"""kwic.py.

Last Updated: December 6, 2025
Last Tested: July 28, 2025

A current limitation is that all spaCy docs must share the same model. This is due to the way spaCy loads models and the Matcher/PhraseMatcher, which are tied to the vocabulary of the loaded model. Without detecting the document models and loading each one, the only way to support lists of documents with different models is to create separate instances of the Kwic class for each set of documents created with a specific model.

Sample usage:

```python
    kwic = Kwic(nlp="en_core_web_sm")
    results = kwic(
        docs=["This is a test document.", "Another test document."],
        labels=["Doc 1", "Doc 2"],
        patterns=["test", "document"],
        window=5,
        matcher="tokens",
        case_sensitive=False,
        use_regex=False,
        as_df=True,
        sort_by="keyword",
        ascending=True,
    )
    print(results)
```
"""

import re
from typing import Optional

import pandas as pd
import spacy
from natsort import natsort_keygen, ns
from pydantic import BaseModel, ConfigDict, Field
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc

from lexos.exceptions import LexosException
from lexos.util import ensure_list


class Kwic(BaseModel):
    """Class for finding keywords in context (KWIC) in text or spaCy documents."""

    nlp: Optional[str] = Field(
        default="xx_sent_ud_sm", description="The spaCy model to use for tokenization."
    )
    alg: Optional[ns] = Field(
        default=ns.LOCALE, description="The sorting algorithm to use."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        """Initialize the Kwic class with a spaCy model."""
        super().__init__(**data)
        self.nlp = spacy.load(self.nlp)

        # Make sure the sorting algorithm is valid
        self._validate_sorting_algorithm()

    def __call__(
        self,
        docs: Optional[Doc | str | list[Doc | str]] = Field(
            default_factory=list,
            description="The spaCy Doc(s) or string(s) to search within.",
        ),
        labels: Optional[str | list[str]] = Field(
            None,
            description="A list of labels for the documents. Defaults to None.",
        ),
        patterns: list = Field(
            default_factory=list,
            description="A list of patterns to match. Can be regex strings or spaCy token patterns.",
        ),
        window: Optional[int] = Field(
            50,
            description="The number of tokens or characters to include before and after the match.",
        ),
        matcher: Optional[str] = Field(
            "characters",
            description="The type of matcher to use. Can be 'rule' for spaCy Matcher, 'phrase' for PhraseMatcher, 'tokens' for token patterns, or 'characters' for string matching.",
        ),
        case_sensitive: Optional[bool] = Field(
            False,
            description="If True, the matching will be case-sensitive. Defaults to False.",
        ),
        use_regex: Optional[bool] = Field(
            False,
            description="If True, use regex for matching with the 'tokens' setting. Defaults to False.",
        ),
        as_df: Optional[bool] = Field(
            True,
            description="If True, return results as a pandas DataFrame. Defaults to True.",
        ),
        sort_by: Optional[str] = Field(
            "keyword",  # Make sure this matches the column name exactly
            description="The column to sort the results by if as_df is True. Defaults to 'keyword'.",
        ),
        ascending: Optional[bool] = Field(
            True,
            description="If True, sort in ascending order. Defaults to True.",
        ),
    ) -> list[tuple[str, str, str]] | pd.DataFrame:
        """Call the Kwic instance to find keywords in context.

        Returns:
            list: A list of tuples, each containing the context before, the matched keyword,
                and the context after, or a DataFrame with the same content.
        """
        # Validate input types
        if matcher in ["rule", "phrase", "tokens"] and any(
            isinstance(doc, str) for doc in docs
        ):
            raise LexosException(
                "Docs must be spaCy Doc objects when using 'rule', 'phrase', or 'tokens' matcher. To search raw text strings, use the 'characters' matcher type, setting `use_regex` if you wish to use regex patterns."
            )

        # Ensure that docs and labels are lists of equal length
        docs = ensure_list(docs)
        if isinstance(labels, list):
            labels = ensure_list(labels)
            if len(docs) != len(labels) and labels:
                raise LexosException(
                    "The number of documents and labels must match. If you do not want to label the documents, set `labels` to None."
                )
        else:
            labels = [f"Doc {i + 1}" for i in range(len(docs))]

        # Assign search parameters and call match method
        match matcher:
            case "rule":
                matcher = Matcher(self.nlp.vocab)
                matcher.add("KWIC_PATTERNS", patterns)
                hits = self._match_tokens(docs, labels, window, matcher)
            case "phrase":
                if case_sensitive:
                    matcher = PhraseMatcher(self.nlp.vocab)
                else:
                    matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
                patterns = [self.nlp.make_doc(phrase) for phrase in patterns]
                matcher.add("KWIC_PATTERNS", patterns)
                hits = self._match_tokens(docs, labels, window, matcher)
            case "tokens":
                matcher = Matcher(self.nlp.vocab)
                patterns = ensure_list(patterns)
                patterns = self._convert_patterns_to_spacy(
                    patterns, case_sensitive, use_regex
                )
                matcher.add("KWIC_PATTERNS", patterns)
                hits = self._match_tokens(docs, labels, window, matcher)
            case _:
                docs = [doc.text if isinstance(doc, Doc) else doc for doc in docs]
                patterns = ensure_list(patterns)
                hits = list(
                    self._match_strings(
                        docs, labels, patterns, window, case_sensitive=case_sensitive
                    )
                )

        # Convert hits to DataFrame for sorting
        df = pd.DataFrame(
            hits, columns=["doc", "context_before", "keyword", "context_after"]
        )

        # Only sort if we have data and the sort_by column exists
        if not df.empty and sort_by in df.columns:
            df = df.sort_values(
                by=sort_by, ascending=ascending, key=natsort_keygen(alg=self.alg)
            )

        # If as_df is False, convert DataFrame to list of dictionaries
        if not as_df:
            result = list(df.to_records(index=False))
            return [tuple(item) for item in result]

        return df

    def _convert_patterns_to_spacy(
        self, patterns: list, case_sensitive: bool, use_regex: bool
    ) -> list:
        """Convert a list of string patterns to spaCy token patterns.

        Args:
            patterns (list): A list of string patterns to convert.
            case_sensitive (bool): If True, the patterns will be case-sensitive.
            use_regex (bool): If True, the patterns will be treated as regex patterns.

        Returns:
            list: A list of spaCy token patterns.
        """
        if use_regex:
            if case_sensitive:
                return [[{"TEXT": {"REGEX": pattern}}] for pattern in patterns]
            else:
                return [[{"LOWER": {"REGEX": pattern.lower()}}] for pattern in patterns]
        else:
            if case_sensitive:
                return [[{"TEXT": pattern}] for pattern in patterns]
            else:
                return [[{"LOWER": pattern}] for pattern in patterns]

    def _match_strings(
        self,
        docs: list[str],
        labels: list[str],
        patterns: list,
        window: int,
        case_sensitive: bool,
    ):
        """Match keywords in a string and return their context.

        Args:
            docs (list[str]): The text to search within.
            labels (str): A list of labels for the documents.
            patterns (list): A list of regex patterns to match.
            window (int): The number of characters to include before and after the match.
            case_sensitive (bool): If True, the matching will be case-sensitive.

        Yields:
            tuple (tuple): A tuple containing the context before, the matched keyword, and the context after.
        """
        flags = 0 if case_sensitive else re.IGNORECASE
        for i, doc in enumerate(docs):
            for pattern in patterns:
                for match in re.finditer(pattern, doc, flags=flags):
                    start = match.start()
                    end = match.end()
                    context_start = max(0, start - window)
                    context_end = min(len(doc), end + window)
                    context_before = doc[context_start:start]
                    context_after = doc[end:context_end]
                    yield (labels[i], context_before, match.group(), context_after)

    def _match_tokens(
        self, docs: list[Doc], labels: list[str], window: int, matcher: Matcher
    ) -> list[tuple[str, str, str, str]]:
        """Match keywords in a spaCy Doc and return their context.

        Args:
            docs (list[Doc]): The spaCy Doc(s) to search within.
            labels (list[str]): A list of labels for the documents.
            window (int): The number of tokens to include before and after the match.
            matcher (Matcher): The spaCy Matcher object with patterns added.

        Returns:
            list[tuple[str, str, str, str]]: A list of tuples, each containing the context before, the matched keyword, and the context after.
        """
        hits = []  # List to store the hits
        for i, doc in enumerate(docs):
            matches = matcher(doc)
            for _, start, end in matches:
                span = doc[start:end]  # The matched span (keyword)
                context_start = max(0, start - window)  #  Start of context window
                context_end = min(len(doc), end + window)  # End of context window
                context_before = doc[context_start : span.start]
                context_after = doc[span.end : context_end]  # Fixed indentation
                hits.append(
                    (labels[i], context_before.text, span.text, context_after.text)
                )  # Fixed indentation
        return hits

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
