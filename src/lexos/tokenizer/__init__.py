"""__init__.py.

Last Update: December 17, 2025
Last Tested: December 17, 2025

Current usage:

tokenizer = Tokenizer()
# Modify the nlp pipeline any way you like
tokenizer.nlp.exclude = ["parser"]
# Helper methods
tokenizer.add_stopwords(["the", "and", "a"])
tokenizer.remove_stopwords(["the", "and", "a"])
tokenizer.add_extension("is_cat", default=False)
# Tokenize a single doc
doc = tokenizer.make_doc("This is a test.")
# Tokenize multiple docs
doc_stream = tokenizer.make_docs(["This is a test.", "This is another test."])
docs = list(doc_stream)
# make_doc and make_docs are public methods to which you can pass keyword arguments,
# but it can be easier to simply call the tokenizer directly (like spaCy's nlp):
# doc = tokenizer("This is a test.")
# docs = tokenizer(["This is a test.", "This is another test."])
"""

from itertools import batched
from typing import Any, Iterable, Optional

import spacy
from pydantic import BaseModel, ConfigDict, Field, validate_call
from spacy.language import Language
from spacy.schemas import DocJSONSchema
from spacy.tokens import Doc, Token

from lexos.exceptions import LexosException
from lexos.util import ensure_list

try:
    default_model = spacy.load("xx_sent_ud_sm")
except ImportError:
    raise LexosException(
        "The default model is not available. Please run `python -m spacy download xx_sent_ud_sm` from the command line."
    )


class Tokenizer(BaseModel):
    """A class for tokenizing text using spaCy."""

    model: Optional[str] = Field(
        default="xx_sent_ud_sm",
        description="The name of the spaCy model to be used for tokenization.",
    )
    max_length: Optional[int] = Field(
        default=2000000,
        description="The maximum length of the doc.",
    )
    disable: Optional[list[str]] = Field(
        default=[],
        description="A list of spaCy pipeline components to disable.",
    )
    stopwords: Optional[list[str] | str] = Field(
        default=[],
        description="A list of stop words to apply to docs.",
    )
    nlp: Optional[Language] = Field(
        default=default_model,
        description="The spaCy language object.",
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra=DocJSONSchema.schema(),
        validate_assignment=True,
    )

    def __init__(self, **data) -> None:
        """Initialise the Tokenizer class."""
        super().__init__(**data)
        try:
            self.nlp = spacy.load(self.model)
            self.nlp.max_length = self.max_length
        except OSError:
            raise LexosException(
                f"Error loading model {self.model}. Please check the name and try again. You may need to install the model on your system."
            )

    @validate_call
    def __call__(self, texts: str | Iterable[str]) -> Doc | Iterable[Doc]:
        """Tokenize a string or an iterable of strings.

        Args:
            texts (str | Iterable[str]): The text(s) to be tokenized.

        Returns:
            Doc | Iterable[Doc]: The tokenized doc(s).
        """
        if isinstance(texts, str):
            return self.make_doc(texts)
        elif isinstance(texts, Iterable):
            return self.make_docs(texts)

    @property
    def pipeline(self) -> list[str]:
        """Return the spaCy pipeline components."""
        return self.nlp.pipe_names

    @property
    def components(self) -> list[str]:
        """Return the spaCy pipeline components."""
        return self.nlp.components

    @property
    def disabled(self) -> list[str]:
        """Return the disabled spaCy pipeline components."""
        return self.nlp.disabled

    @validate_call
    def add_extension(self, name: str, default: str) -> None:
        """Add an extension to the spaCy Token class.

        Args:
            name (str): The name of the extension.
            default (str): The default value of the extension.
        """
        if not Token.has_extension(name):
            Token.set_extension(name, default=default, force=True)

    @validate_call
    def add_stopwords(self, stopwords: str | list[str]) -> None:
        """Add stopwords to the tokenizer.

        Args:
            stopwords (str | Iterable[str]): A list of stopwords to add to the model.
        """
        stopwords = ensure_list(stopwords)
        for term in stopwords:
            self.nlp.vocab[term].is_stop = True
        self.stopwords.extend(stopwords)

    @validate_call
    def make_doc(
        self, text: str, max_length: int = None, disable: list[str] = [], **kwargs: Any
    ) -> Doc:
        """Return a doc from a text.

        Args:
            text (str): The text to be parsed.
            max_length (int): The maximum length of the doc.
            disable (list[str]): A list of spaCy pipeline components to disable.
            kwargs (Any): Additional keyword arguments. Accepts any keyword arguments that
                can be passed to spaCy's `Language.pipe` method, such as `batch_size`.

        Returns:
            Doc: A spaCy doc object.
        """
        # Override instance settings with keyword arguments
        if max_length:
            self.max_length = max_length
            self.nlp.max_length = max_length
        if disable:
            # self.nlp.disabled.extend(disable)
            self.nlp.select_pipes(disable=disable)
        return next(self.nlp.pipe([text], disable=disable, **kwargs))

    @validate_call
    def make_docs(
        self,
        texts: Iterable[str],
        max_length: int = None,
        disable: Iterable[str] = [],
        **kwargs: Any,
    ) -> Iterable[Doc]:
        """Return a generator of docs from an iterable of texts.

        Args:
            texts (Iterable[str]): The texts to be parsed.
            max_length (int): The maximum length of the docs.
            kwargs (Any): Additional keyword arguments. Accepts any keyword arguments that
                can be passed to spaCy's `Language.pipe` method, such as `batch_size`.

        Yields:
            Iterable[Doc]: A generator of spaCy doc objects.
        """
        # Override instance settings with keyword arguments
        if max_length:
            self.max_length = max_length
            self.nlp.max_length = max_length
        if disable:
            # self.nlp.disabled.extend(disable)
            self.nlp.select_pipes(disable=disable)
        return self.nlp.pipe(texts, disable=disable, **kwargs)

    @validate_call
    def remove_extension(self, name: str) -> None:
        """Remove an extension from the spaCy Token class.

        Args:
            name (str): The name of the extension.
        """
        if Token.has_extension(name):
            Token.remove_extension(name)

    @validate_call
    def remove_stopwords(self, stopwords: str | list[str]) -> None:
        """Remove stopwords from the tokenizer.

        Args:
            stopwords (str | list[str]): A list of stopwords to remove from the model.
        """
        stopwords = ensure_list(stopwords)
        for term in stopwords:
            self.nlp.vocab[term].is_stop = False
        self.stopwords = [word for word in self.stopwords if word not in stopwords]


class SliceTokenizer(BaseModel, validate_assignment=True):
    """Simple slice tokenizer."""

    n: int = Field(description="The size of the tokens in characters.")
    drop_ws: Optional[bool] = Field(
        default=True, description="Whether to drop whitespace from the tokens."
    )

    @validate_call
    def __call__(self, text: str) -> list[str]:
        """Slice the text into tokens of n characters.

        Args:
            text (str): The text to tokenize.

        Returns:
            list[str]: A list of tokens.
        """
        if self.drop_ws:
            text = text.replace(" ", "")
        return ["".join(t) for t in batched(text, self.n)]


class WhitespaceTokenizer(BaseModel):
    """Simple whitespace tokenizer."""

    @validate_call
    def __call__(self, text: str) -> list[str]:
        """Split the text into tokens on whitespace.

        Args:
            text (str): The text to tokenize.

        Returns:
            list[str]: A list of tokens.
        """
        return text.split()
