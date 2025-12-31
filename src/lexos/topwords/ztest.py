"""ztest.py.

Module for performing z-tests using Pydantic models.

Last Update: November 8, 2025
Last Tested: November 8, 2025


NB. This implementation assumes that the documents are pre-tokenized
and that the tokens are accessible via the `Doc` object from spaCy.
If the input is strings, they will be converted to spaCy `Doc` objects,
which can slow down processing. Maybe there should be a fast mode which
disables features like stopword removal, punctuation removal, etc.?
But that might also disable ngrams.

Usage:

```python
ztest = ZTest(
    target_docs=[...],
    comparison_docs=[...],
    topn=10,
    case_sensitive=True,
    remove_stopwords=True,
    remove_punct=True,
    remove_digits=False,
    ngrams=1,
    model="xx_sent_ud_sm",
)
ztest.topwords # [('word1', z_score1), ('word2', z_score2), ...]
ztest.to_dict() # {"topwords": [{"term": "word1", "z_score": z_score1}, ...
ztest.to_df()
#      term    z_score
# 0   word1   z_score1
# 1   word2   z_score2
# ...
```
"""

from collections import Counter

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from spacy.tokens import Doc, Token

from lexos.tokenizer import Tokenizer
from lexos.topwords import TopWords
from lexos.util import ensure_list

if not Doc.has_extension("topwords"):
    Doc.set_extension("topwords", default=None, force=True)


class ZTest(TopWords):
    """Pydantic model for performing z-tests."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
    target_docs: str | Doc | list[str | Doc] = Field(
        ...,
        description="List of target documents.",
    )
    comparison_docs: str | Doc | list[str | Doc] = Field(
        ...,
        description="List of comparison documents.",
    )
    topn: int = Field(10, gt=0, description="Number of top words to return.")
    case_sensitive: bool | None = Field(
        True, description="Whether analysis is case sensitive."
    )
    remove_stopwords: bool | None = Field(
        True, description="Whether to remove stopwords."
    )
    remove_punct: bool | None = Field(
        True, description="Whether to remove punctuation."
    )
    remove_digits: bool | None = Field(False, description="Whether to remove digits.")
    ngrams: int | tuple[int, int] = Field(
        default=1,
        description="The ngram range for analysis, e.g., (1, 2) for unigrams and bigrams.",
    )
    model: str = Field(
        default="xx_sent_ud_sm",
        description="spaCy model name to use for tokenization.",
    )
    tokenizer: Tokenizer = Field(default_factory=Tokenizer, exclude=True)
    topwords: list[tuple[str, float]] = Field(
        default_factory=list,
        description="List of top words with their Z-scores.",
    )

    def __init__(self, **data) -> None:
        """Initialize the ZTest class, ensuring a tokenizer is set.

        If a tokenizer is not provided, creates one using the specified spaCy model.
        """
        # If tokenizer is not provided, create one with the specified model
        if "tokenizer" not in data or data["tokenizer"] is None:
            data["tokenizer"] = Tokenizer(model=data.get("model", "xx_sent_ud_sm"))
        super().__init__(**data)

        # Tokenize the docs if they are strings
        self.target_docs = [
            self.tokenizer.make_doc(doc) if isinstance(doc, str) else doc
            for doc in ensure_list(self.target_docs)
        ]
        self.comparison_docs = [
            self.tokenizer.make_doc(doc) if isinstance(doc, str) else doc
            for doc in ensure_list(self.comparison_docs)
        ]

        # Get the tokens for target and comparison documents
        target_tokens = self._get_doc_tokens(self.target_docs)
        comparison_tokens = self._get_doc_tokens(self.comparison_docs)

        # Get the term counts for target and comparison
        target_counts: Counter = Counter(target_tokens)
        comparison_counts: Counter = Counter(comparison_tokens)

        # Total number of tokens in target and comparison
        target_total: int = sum(target_counts.values())
        comparison_total: int = sum(comparison_counts.values())

        # Calculate Z-scores for each term
        results: list[tuple[str, float]] = []
        all_terms: set = set(target_counts) | set(comparison_counts)
        for term in all_terms:
            # Proportion of this term in target and comparison
            p1: float = target_counts[term] / target_total if target_total else 0
            p2: float = (
                comparison_counts[term] / comparison_total if comparison_total else 0
            )
            # Combined proportion of this term in both sets
            p: float = (
                (target_counts[term] + comparison_counts[term])
                / (target_total + comparison_total)
                if (target_total + comparison_total)
                else 0
            )
            n1, n2 = target_total, comparison_total

            # Only calculate Z if both sets have data and p is not 0 or 1
            if n1 > 0 and n2 > 0 and p > 0 and p < 1:
                # Standard error for the difference in proportions
                denominator = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
                if denominator != 0:
                    # Z-score: difference in proportions divided by standard error
                    z: float = (p1 - p2) / denominator
                else:
                    z = 0.0
            else:
                z = 0.0

            # Convert z to a plain Python float if it's a numpy type
            z_value = z.item() if hasattr(z, "item") else z
            results.append((term, z_value))

        # Filter out terms with a Z-score of 0.0 before sorting.
        filtered_results = [item for item in results if item[1] != 0.0]

        # Sort results by absolute Z-score in descending order
        sorted_results = sorted(
            filtered_results, key=lambda item: abs(item[1]), reverse=True
        )

        # Assign the top N results
        self.topwords = sorted_results[: self.topn]

        # Set the topwords attribute on each document
        for doc in self.target_docs:
            doc._.topwords = self.topwords
        for doc in self.comparison_docs:
            doc._.topwords = self.topwords

    def _filter_doc(self, doc: Doc) -> list[str]:
        """Filter tokens from a document based on preprocessing settings.

        Args:
            doc (Doc): A spaCy Doc object.

        Returns:
            list[str]: List of filtered token texts.
        """
        filtered_tokens = []
        for token in doc:
            if self.remove_stopwords and token.is_stop:
                continue
            if self.remove_punct and token.is_punct:
                continue
            if self.remove_digits and token.is_digit:
                continue
            if not self.case_sensitive:
                token_text = token.text.lower()
            else:
                token_text = token.text
            filtered_tokens.append(token_text)
        return filtered_tokens

    def _get_doc_tokens(self, docs: list[Doc]) -> list[str]:
        """Get tokens from a document after applying preprocessing.

        Args:
            docs (list[Doc]): List of spaCy Doc objects.

        Returns:
            list[str]: List of processed tokens from the documents.
        """
        # Get the ngram range
        if isinstance(self.ngrams, int):
            ngram_range = ensure_list(self.ngrams)
        else:
            ngram_range = self.ngrams

        tokens = []
        for doc in ensure_list(docs):
            # Handle single tokens
            if ngram_range == [1] or ngram_range == (1, 1):
                tokens.extend(self._filter_doc(doc))
            # Handle ngrams
            else:
                # Generate token list
                token_list = self._filter_doc(doc)
                # Generate ngrams
                for n in range(ngram_range[0], ngram_range[-1] + 1):
                    ngrams = zip(*[token_list[i:] for i in range(n)])
                    for ngram in ngrams:
                        tokens.append(" ".join(ngram))
        return tokens

    def to_dict(self):
        """Return the topwords as a dictionary with terms and Z-scores."""
        return {
            "topwords": [
                {"term": term, "z_score": z_score}
                for term, z_score in getattr(self, "topwords", [])
            ]
        }

    def to_df(self):
        """Return the topwords as a pandas DataFrame."""
        return pd.DataFrame(
            getattr(self, "topwords", []) or [], columns=["term", "z_score"]
        )

    def to_list_of_dicts(self):
        """Return the topwords as a list of dictionaries with 'term' and 'z_score'."""
        return [
            {"term": term, "z_score": z_score}
            for term, z_score in getattr(self, "topwords", [])
        ]
