"""ngrams.py.

Last Update: December 4, 2025
Last Tested: May 26, 2025

Current usage:

ng = Ngrams()
# Set any instance attributes
ng.n = 3

# Generate ngrams from a Doc object
doc = nlp("This is a test.")

# Accepts any keyword arguments for textacy.extract.ngrams
# filter_digits turns off filter_nums
# output can be 'text', 'spans', or 'tuples'
ng.from_doc(doc, filter_digits=True, output="tuples")

# User multiple docs
ng.from_docs([doc, doc])

# Generate ngrams from a text
slicer = SliceTokenizer(2, False)
splitter = WhitespaceTokenizer()
# Accepts most keyword arguments for textacy.extract.ngrams but doesn't
# use Doc attributes for filtering
# You can submit a spaCy pipeline, and the text will be tokenised with it.
ng.from_text("This is a test.", tokenizer=slicer, drop_ws=True)
# Multiple texts
ng.from_texts(["This is a test.", "Another test."], tokenizer=splitter)

# Generate ngrams from a list of tokens
# Accepts most keyword arguments for textacy.extract.ngrams but doesn't
# use Doc attributes for filtering. Also accepts drop_ws.
ng.from_tokens(["This", "is", "a", "test."])

# Generate ngrams from a list of token lists
ng.from_token_lists([["This", "is", "a", "test."], ["Another", "test."]])
"""

import re
from typing import Any, Callable, Generator, Iterable, Optional

from cytoolz.itertoolz import frequencies
from pydantic import BaseModel, ConfigDict, Field, validate_call
from spacy.schemas import DocJSONSchema
from spacy.tokens import Doc, Token
from textacy.extract.basics import ngrams as textacy_ngrams

from lexos.exceptions import LexosException
from lexos.tokenizer import SliceTokenizer, WhitespaceTokenizer

validation_config = ConfigDict(
    arbitrary_types_allowed=True,
    json_schema_extra=DocJSONSchema.schema(),
    validate_assignment=True,
)


class Ngrams(BaseModel):
    """Generate ngrams from a text."""

    n: int = Field(
        default=2,
        description="The size of the ngrams.",
    )
    drop_ws: Optional[bool] = Field(
        default=True,
        description="Whether to drop whitespace from the ngrams.",
    )
    filter_digits: Optional[bool] = Field(
        default=False,
        description="If True, remove ngrams that contain any digits. Automatically sets filter_nums to False.",
    )
    filter_nums: Optional[bool] = Field(
        default=False,
        description="If True, remove ngrams that contain any numbers or number-like tokens.",
    )
    filter_punct: Optional[bool] = Field(
        default=True,
        description="Remove ngrams that contain any punctuation-only tokens.",
    )
    filter_stops: Optional[bool | list[str]] = Field(
        default=[],
        description="Remove ngrams that start or end with a stop word in the provided list.",
    )
    min_freq: Optional[int] = Field(
        default=1,
        description="Remove ngrams that occur in text, doc, or tokens fewer than min_freq times.",
    )
    output: Optional[str] = Field(
        default="text",
        description="The output format. Can be 'text', 'spans', or 'tuples'.",
    )
    tokenizer: Optional[Callable] = Field(
        default=WhitespaceTokenizer,
        description="The tokenizer to use.",
    )

    model_config = validation_config

    @property
    def stopwords(self) -> bool | list[str] | None:
        """Get the list of stopwords."""
        return self.filter_stops

    @validate_call(config=validation_config)
    def _filter_tokens(
        self,
        tokens: list[str],
        drop_ws: bool = True,
        filter_digits: bool = False,
        filter_punct: bool = True,
        filter_stops: list[str] = [],
    ) -> Generator:
        """Apply filters to a list of tokens.

        Args:
            tokens (list[str]): The list of tokens.
            drop_ws (bool): Whether to drop whitespace from the ngrams.
            filter_digits (bool): If True, remove ngrams that contain any digits.
            filter_punct (bool): Remove ngrams that contain any punctuation-only tokens.
            filter_stops (list[str]): Remove ngrams that start or end with a stop word in the provided list.

        Returns:
            Generator: A generator of filtered tokens.
        """
        if drop_ws:
            tokens = (t.strip() for t in tokens)
        if len(filter_stops) > 0:
            tokens = (t for t in tokens if t not in filter_stops)
        if filter_punct:
            tokens = (t for t in tokens if not re.match("\\W", t))
        if filter_digits:
            tokens = (t for t in tokens if not re.match("\\d", t))
        yield from tokens

    def _set_attributes(
        self, skip_set_attrs: bool = False, **kwargs: dict[str, Any]
    ) -> None:
        """Set the instance attributes based on keyword arguments.

        Args:
            skip_set_attrs (bool): Whether to skip setting the attributes.
            **kwargs (dict[str, Any]): The keyword arguments to set the attributes.
        """
        if not skip_set_attrs:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    @validate_call(config=validation_config)
    def from_doc(
        self,
        doc: Doc,
        n: int = 2,
        filter_digits: Optional[bool] = False,
        filter_nums: Optional[bool] = False,
        filter_punct: Optional[bool] = True,
        filter_stops: Optional[bool] = False,
        output: Optional[str] = "text",
        min_freq: Optional[int] = 1,
        skip_set_attrs: Optional[bool] = False,
        **kwargs: Any,
    ) -> Generator:
        """Generate a list of ngrams from a Doc.

        Args:
            doc (Doc): The source Doc.
            n (int): The size of the ngrams.
            filter_digits (bool): If True, remove ngrams that contain any digits.
            filter_nums (bool): If True, remove ngrams that contain any numbers or number-like tokens.
            filter_punct (bool): Remove ngrams that contain any punctuation-only tokens.
            filter_stops (bool): Remove ngrams that start or end with a stop word in the provided list.
            output (str): The output format. Can be 'text', 'spans', or 'tuples'.
            min_freq (int): Remove ngrams that occur in text, doc, or tokens fewer than min_freq times.
            skip_set_attrs (bool): Whether to skip setting the attributes.
            **kwargs (Any): Extra keyword arguments to pass to textacy.extract.basics.ngrams.

        Returns:
            Generator: A generator of ngrams.
        """
        attrs = {
            "n": n,
            "filter_digits": filter_digits,
            "filter_nums": filter_nums,
            "filter_punct": filter_punct,
            "filter_stops": filter_stops,
            "min_freq": min_freq,
            "output": output,
            "skip_set_attrs": skip_set_attrs,
        }
        attrs = {**attrs, **kwargs}
        self._set_attributes(**attrs)
        # Set filter_nums to false; we'll filter digits separately
        if filter_digits:
            self.filter_nums = False
        # Get the ngrams
        ngram_spans = textacy_ngrams(
            doc,
            n=self.n,
            filter_nums=self.filter_nums,
            filter_punct=self.filter_punct,
            filter_stops=self.filter_stops,
            **kwargs,
        )
        # Filter digits
        if filter_digits:
            ngram_spans = (
                ng for ng in ngram_spans if not any(token.is_digit for token in ng)
            )
        # Apply min_freq (for some reason, it doesn't work if passed to Textacy)
        if min_freq > 1:
            freqs = frequencies(ng.text.lower() for ng in ngram_spans)
            ngram_spans = (
                ng for ng in ngram_spans if freqs[ng.text.lower()] >= min_freq
            )
        # Yield the desired output
        if self.output == "text":
            for span in ngram_spans:
                yield span.text
        elif self.output == "spans":
            yield from ngram_spans
        elif self.output == "tuples":
            for span in ngram_spans:
                yield tuple([token.text for token in span])
        else:
            raise LexosException("Invalid output type.")

    @validate_call(config=validation_config)
    def from_docs(
        self,
        docs: Iterable[Doc],
        n: int = 2,
        filter_digits: Optional[bool] = False,
        filter_nums: Optional[bool] = False,
        filter_punct: Optional[bool] = True,
        filter_stops: Optional[bool] = False,
        min_freq: Optional[int] = 1,
        output: Optional[str] = "text",
        **kwargs: Any,
    ) -> list[Generator]:
        """Generate a list of ngrams from a Doc.

        Args:
            docs (Iterable[Doc]): An iterable of Docs.
            n (int): The size of the ngrams.
            filter_digits (bool): If True, remove ngrams that contain any digits.
            filter_nums (bool): If True, remove ngrams that contain any numbers or number-like tokens.
            filter_punct (bool): Remove ngrams that contain any punctuation-only tokens.
            filter_stops (list[str]): Remove ngrams that start or end with a stop word in the provided list.
            min_freq (int): Remove ngrams that occur in text, doc, or tokens fewer than min_freq times.
            output (str): The output format. Can be 'text', 'spans', or 'tuples'.
            **kwargs (Any): Extra keyword arguments to pass to textacy.extract.basics.ngrams.

        Returns:
            list[Generator]: A list of ngram generators.
        """
        attrs = {
            "n": n,
            "filter_digits": filter_digits,
            "filter_nums": filter_nums,
            "filter_punct": filter_punct,
            "filter_stops": filter_stops,
            "min_freq": min_freq,
            "output": output,
        }
        attrs = {**attrs, **kwargs}
        self._set_attributes(**attrs)
        ngram_list = []
        for doc in docs:
            ngram_list.append(self.from_doc(doc, skip_set_attrs=True))
        return ngram_list

    @validate_call(config=validation_config)
    def from_text(
        self,
        text: str,
        n: int = 2,
        drop_ws: Optional[bool] = True,
        filter_digits: Optional[bool] = False,
        filter_punct: Optional[bool] = True,
        filter_stops: Optional[Iterable[str]] = [],
        min_freq: Optional[int] = 1,
        output: Optional[str] = "text",
        skip_set_attrs: Optional[bool] = False,
        tokenizer: Optional[Callable] = WhitespaceTokenizer(),
    ) -> Generator:
        """Generate a list of ngrams from a list of tokens.

        Args:
            text (str): The text to generate ngrams from.
            n (int): The size of the ngrams.
            drop_ws (bool): Whether to drop whitespace from the ngrams.
            filter_digits (bool): If True, remove ngrams that contain any digits.
            filter_punct (bool): Remove ngrams that contain any punctuation-only tokens.
            filter_stops (Iterable[str]): Remove ngrams that start or end with a stop word in the provided list.
            min_freq (Optional[int]): Remove ngrams that occur in text fewer than min_freq times.
            output (str): The output format. Can be 'text' or 'tuples'.
            skip_set_attrs (bool): Whether to skip setting the attributes.
            tokenizer (Callable): The tokenizer to use.

        Returns:
            Generator: A generator of ngrams.
        """
        self._set_attributes(
            n=n,
            drop_ws=drop_ws,
            filter_digits=filter_digits,
            filter_punct=filter_punct,
            filter_stops=filter_stops,
            min_freq=min_freq,
            output=output,
            skip_set_attrs=skip_set_attrs,
        )
        tokens = tokenizer(text)
        # If the user tokenises with a spaCy pipeline, we need to extract the text
        if isinstance(tokens[0], Token):
            tokens = [token.text for token in tokens]
        tokens = list(
            self._filter_tokens(
                tokens,
                self.drop_ws,
                self.filter_digits,
                self.filter_punct,
                self.filter_stops,
            )
        )
        ngrams = zip(*[tokens[i:] for i in range(self.n)])
        if min_freq > 1:
            ngrams = list(ngrams)
            freqs = frequencies("".join(ng).lower() for ng in ngrams)
            ngrams = (ng for ng in ngrams if freqs["".join(ng).lower()] >= min_freq)
        if self.output == "text":
            for ngram in ngrams:
                yield " ".join(ngram)
        elif self.output == "tuples":
            yield from ngrams
        else:
            raise LexosException("Invalid output type.")

    @validate_call(config=validation_config)
    def from_texts(
        self,
        texts: Iterable[str],
        n: int = 2,
        drop_ws: Optional[bool] = True,
        filter_digits: Optional[bool] = False,
        filter_punct: Optional[bool] = True,
        filter_stops: Optional[Iterable[str]] = [],
        min_freq: Optional[int] = 1,
        output: Optional[str] = "text",
        tokenizer: Optional[Callable] = WhitespaceTokenizer,
    ) -> list[Generator]:
        """Generate a list of ngrams from a list of tokens.

        Args:
            texts (Iterable[str]): An iterable of texts.
            n (int): The size of the ngrams.
            drop_ws (bool): Whether to drop whitespace from the ngrams.
            filter_digits (bool): If True, remove ngrams that contain any digits.
            filter_punct (bool): Remove ngrams that contain any punctuation-only tokens.
            filter_stops (list[str]): Remove ngrams that start or end with a stop word in the provided list.
            min_freq (Optional[int]): Remove ngrams that occur in text fewer than min_freq times.
            output (str): The output format. Can be 'text' or 'tuples'.
            tokenizer (Callable): The tokenizer to use.

        Returns:
            list[Generator]: A list of ngram generators.
        """
        self._set_attributes(
            n=n,
            drop_ws=drop_ws,
            filter_punct=filter_punct,
            filter_stops=filter_stops,
            filter_digits=filter_digits,
            min_freq=min_freq,
            output=output,
            tokenizer=tokenizer,
        )
        ngram_list = []
        for text in texts:
            ngram_list.append(self.from_text(text, skip_set_attrs=True))
        return ngram_list

    @validate_call(config=validation_config)
    def from_tokens(
        self,
        tokens: Iterable[str],
        n: int = 2,
        drop_ws: Optional[bool] = True,
        filter_digits: Optional[bool] = False,
        filter_punct: Optional[bool] = True,
        filter_stops: Optional[Iterable[str]] = [],
        min_freq: Optional[int] = 1,
        output: Optional[str] = "text",
        skip_set_attrs: Optional[bool] = False,
    ) -> Generator:
        """Generate a ngrams from an iterable of tokens.

        Args:
            tokens (Iterable[str]): An iterable of tokens.
            n (int): The size of the ngrams.
            drop_ws (bool): Whether to drop whitespace from the ngrams.
            filter_digits (bool): If True, remove ngrams that contain any digits.
            filter_punct (bool): Remove ngrams that contain any punctuation-only tokens.
            filter_stops (Iterable[str]): Remove ngrams that start or end with a stop word in the provided list.
            min_freq (int): Remove ngrams that occur in tokens fewer than min_freq times.
            output (str): The output format. Can be 'text' or 'tuples'.
            skip_set_attrs (bool): Whether to skip setting the attributes.

        Returns:
            Generator: A generator of ngrams.
        """
        self._set_attributes(
            n=n,
            drop_ws=drop_ws,
            filter_digits=filter_digits,
            filter_punct=filter_punct,
            filter_stops=filter_stops,
            min_freq=min_freq,
            output=output,
            skip_set_attrs=skip_set_attrs,
        )

        tokens = list(
            self._filter_tokens(
                tokens,
                self.drop_ws,
                self.filter_digits,
                self.filter_punct,
                self.filter_stops,
            )
        )
        ngrams = zip(*[tokens[i:] for i in range(self.n)])
        if min_freq > 1:
            ngrams = list(ngrams)
            freqs = frequencies("".join(ng).lower() for ng in ngrams)
            ngrams = (ng for ng in ngrams if freqs["".join(ng).lower()] >= min_freq)
        if self.output == "text":
            ngrams = zip(*[tokens[i:] for i in range(self.n)])
            for ngram in ngrams:
                yield " ".join(ngram)
        elif self.output == "tuples":
            yield from ngrams
        else:
            raise LexosException("Invalid output type.")

    @validate_call(config=validation_config)
    def from_token_lists(
        self,
        token_lists: Iterable[Iterable[str]],
        n: int = 2,
        drop_ws: Optional[bool] = True,
        filter_digits: Optional[bool] = False,
        filter_punct: Optional[bool] = True,
        filter_stops: Optional[Iterable[str]] = [],
        min_freq: Optional[int] = 1,
        output: Optional[str] = "text",
    ) -> list[Generator]:
        """Generate a ngrams from an iterable of tokens.

        Args:
            token_lists (Iterable[Iterable[str]]): An iterable of token lists.
            n (int): The size of the ngrams.
            drop_ws (bool): Whether to drop whitespace from the ngrams.
            filter_digits (bool): If True, remove ngrams that contain any digits.
            filter_punct (bool): Remove ngrams that contain any punctuation-only tokens.
            filter_stops (list[str]): Remove ngrams that start or end with a stop word in the provided list.
            min_freq (int): Remove ngrams that occur in tokens fewer than min_freq times.
            output (str): The output format. Can be 'text' or 'tuples'.

        Returns:
            list[Generator]: A list of ngram generators.
        """
        self._set_attributes(
            n=n,
            drop_ws=drop_ws,
            filter_digits=filter_digits,
            filter_punct=filter_punct,
            filter_stops=filter_stops,
            min_freq=min_freq,
            output=output,
        )
        ngram_list = []
        for token_list in token_lists:
            ngram_list.append(self.from_tokens(token_list, skip_set_attrs=True))
        return ngram_list
