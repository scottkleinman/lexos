"""keyterms.py.

Last Updated: November 10, 2025
Last Tested: November 10, 2025
"""

from typing import Any, Literal

import pandas as pd
from pydantic import ConfigDict, Field
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.schemas import DocJSONSchema
from spacy.tokens import Doc
from textacy import extract

from lexos.tokenizer import Tokenizer
from lexos.topwords import TopWords

validation_config = ConfigDict(
    arbitrary_types_allowed=True, json_schema_extra=DocJSONSchema.schema()
)

# Register a custom extension for keyterms if not already set
if not Doc.has_extension("keyterms"):
    Doc.set_extension("keyterms", default=None, force=True)


class KeyTerms(TopWords):
    """Extracts keyterms from text or a spaCy Doc using textacy algorithms."""

    document: str | Doc = Field(
        None, description="The raw text or spaCy doc to analyze."
    )
    method: Literal["textrank", "sgrank", "scake", "yake"] = Field(
        "textrank",
        description="Method for keyterm extraction (e.g., 'textrank', 'sgrank', 'scake', 'yake').",
    )
    topn: int = Field(10, gt=0, description="Number of top keyterms to return.")
    model: str = Field(
        "xx_sent_ud_sm",
        description="spaCy model name to use for tokenization.",
    )
    ngrams: int | tuple[int, int] = Field(
        1,
        description="The ngram range for keyterm extraction, e.g., 1 for unigrams, (1, 2) for unigrams and bigrams.",
    )
    tokenizer: Tokenizer = Field(default_factory=Tokenizer, exclude=True)
    normalize: str = Field(
        default="lemma",
        description="Normalization for keyterm extraction (e.g., 'lemma', 'lower', 'orth').",
    )

    model_config = validation_config
    keyterms: list[dict[str, Any]] | None = Field(
        default=None,
        description="Extracted keyterms.",
    )

    def __init__(self, **data) -> None:
        """Initialize the KeyTerms class, ensuring a tokenizer is set.

        If a tokenizer is not provided, creates one using the specified spaCy model.
        Extraction happens during initialization.
        """
        if "tokenizer" not in data or data["tokenizer"] is None:
            data["tokenizer"] = Tokenizer(model=data.get("model", "xx_sent_ud_sm"))
        super().__init__(**data)

        # Extract keyterms during initialization
        self._extract_keyterms()

    def _extract_keyterms(self):
        """Extract keyterms from the document.

        This method is called during initialization to populate self.keyterms
        and set the doc._.keyterms extension.
        """
        if isinstance(self.document, Doc):
            doc = self.document
        elif isinstance(self.document, str):
            doc = self.tokenizer(self.document)
        else:
            raise ValueError("The 'document' field must be a string or a spaCy Doc.")

        # Handle ngrams parameter - convert int to tuple
        if isinstance(self.ngrams, int):
            min_n = max_n = self.ngrams
        else:
            min_n, max_n = self.ngrams

        if self.method == "textrank":
            results = extract.keyterms.textrank(
                doc,
                normalize=self.normalize,
                topn=self.topn * 20,
            )
            results = [
                (term, score)
                for term, score in results
                if min_n <= len(term.split()) <= max_n
                and term.lower() not in STOP_WORDS
            ][: self.topn]
        elif self.method == "sgrank":
            # For sgrank, pass the ngrams parameter as a tuple
            ngrams_param = (
                (min_n, max_n) if isinstance(self.ngrams, int) else self.ngrams
            )
            results = extract.keyterms.sgrank(
                doc,
                normalize=self.normalize,
                ngrams=ngrams_param,
                topn=self.topn,
            )
        elif self.method == "scake":
            # scake doesn't accept ngrams parameter
            results = extract.keyterms.scake(
                doc,
                normalize=self.normalize,
                topn=self.topn,
            )
        elif self.method == "yake":
            # For yake, pass the ngrams parameter as a tuple
            ngrams_param = (
                (min_n, max_n) if isinstance(self.ngrams, int) else self.ngrams
            )
            results = extract.keyterms.yake(
                doc,
                normalize=self.normalize,
                ngrams=ngrams_param,
                topn=self.topn,
            )
        else:
            raise ValueError(
                "Invalid method. Choose 'textrank', 'sgrank', 'scake', or 'yake'."
            )

        self.keyterms = [{"term": term, "score": score} for term, score in results]
        doc._.keyterms = self.keyterms

    def to_dict(self):
        """Return the extracted keyterms as a dictionary."""
        return {
            "keyterms": [
                {"term": kw["term"], "score": kw["score"]}
                for kw in (self.keyterms or [])
            ]
        }

    def to_df(self):
        """Return the extracted keyterms as a pandas DataFrame."""
        return pd.DataFrame(getattr(self, "keyterms", []))

    def to_list(self):
        """Return the extracted keyterms as a list of (term, score) tuples."""
        return [(kw["term"], kw["score"]) for kw in getattr(self, "keyterms", [])]
