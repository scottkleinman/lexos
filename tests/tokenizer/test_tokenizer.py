"""test_tokenizer.py.

Unit tests for the Tokenizer, SliceTokenizer, and WhitespaceTokenizer classes in lexos.tokenizer

Purpose:

These tests verify the correct behavior of the tokenizer class which generates
spaCy docs containing the tokens from input strings or lists of strings.

The tests ensure consistent and correct output across multiple conditions:
    - Correct tokenization of input strings and lists of strings
    - Adding and removing stop words
    - Adding and removing spaCy extensions
    - Tokenization using different language models
    - Handling of excpetions, including incorrect model names and invalid inputs

Functionality Tested:

 Tokenization using spaCy language models
 Correct output using make_doc(), make_docs(), and __call__ methods
 Correct modification of:
    - Stop words
    - Extensions
    - Pipeline components
Execption handling for:
    - Incorrect model names
    - Invalid inputs
    - Empty strings and lists
Using SliceTokenizer to slice text into tokens of n characters
Using WhitespaceTokenizer to split text on whitespace

Usage:
To run the tests for this module:
    uv run pytest tests/tokenizer/test_tokenizer.py

Coverage: 98%. Missing: 40-41
Last Updated: December 17, 2025
"""

from typing import Generator

import pytest
from spacy.tokens import Doc, Token

from lexos.exceptions import LexosException
from lexos.tokenizer import SliceTokenizer, Tokenizer, WhitespaceTokenizer


@pytest.fixture
def tokenizer() -> Tokenizer:
    """Fixture for the Tokenizer class."""
    return Tokenizer()


@pytest.fixture
def sliceTokenizer() -> SliceTokenizer:
    """Fixture for the SliceTokenizer class."""
    return SliceTokenizer(n=4)


@pytest.fixture
def whitespaceTokenizer() -> WhitespaceTokenizer:
    """Fixture for the WhitespaceTokenizer class."""
    return WhitespaceTokenizer()


def test_incorrect_model_exception() -> None:
    """Raises LexosException when an incorrect model is provided."""
    with pytest.raises(
        LexosException,
        match=f"Error loading model non_existent_model. Please check the name and try again. You may need to install the model on your system.",
    ):
        tokenizer = Tokenizer(model="non_existent_model")


def test_add_extension(tokenizer: Tokenizer) -> None:
    """Adds an extension to the tokenizer object."""
    tokenizer.add_extension("test_ext", default="default_value")
    assert Token.has_extension("test_ext")


def test_add_stopwords(tokenizer: Tokenizer) -> None:
    """Adds stopwords to the tokenizer object."""
    stopwords = ["the", "a", "an"]
    tokenizer.add_stopwords(stopwords)
    for term in stopwords:
        assert tokenizer.nlp.vocab[term].is_stop
        assert term in tokenizer.stopwords


def test_call(tokenizer: Tokenizer) -> None:
    """Creates a spaCy Doc object from a string."""
    doc = tokenizer("This is a test.")
    assert isinstance(doc, Doc)
    assert doc.text == "This is a test."


def test_call_multiple_texts(tokenizer: Tokenizer) -> None:
    """Creates a list of spaCy Doc objects from a list of strings."""
    docs = list(tokenizer(["This is a test.", "This is another test."]))
    assert isinstance(docs[0], Doc)
    assert docs[0].text == "This is a test."
    assert isinstance(docs[1], Doc)
    assert docs[1].text == "This is another test."


def test_make_doc(tokenizer: Tokenizer) -> None:
    """Creates a spaCy Doc object from a string that contains a max length and disabled pipes."""
    doc = tokenizer.make_doc("This is a test.")
    assert isinstance(doc, Doc)
    assert doc.text == "This is a test."
    doc = tokenizer.make_doc("This is another test.", max_length=40, disable=["senter"])
    assert isinstance(doc, Doc)
    assert tokenizer.max_length == 40
    assert "senter" in tokenizer.nlp.disabled


def test_make_docs(tokenizer: Tokenizer) -> None:
    """Creates a list of spaCy Doc objects from a list of strings that contains a max length and disabled pipes."""
    texts = ["This is a test.", "Another test."]
    docs = list(tokenizer.make_docs(texts))
    assert all(isinstance(doc, Doc) for doc in docs)
    assert [doc.text for doc in docs] == texts
    texts = ["This is a another test.", "Another another test."]
    docs = list(tokenizer.make_docs(texts, max_length=200, disable=["senter"]))
    assert all(isinstance(doc, Doc) for doc in docs)
    assert [doc.text for doc in docs] == texts
    assert tokenizer.max_length == 200
    assert "senter" in tokenizer.nlp.disabled


def test_remove_extension(tokenizer: Tokenizer) -> None:
    """Removes an extension from the tokenizer object."""
    tokenizer.add_extension("test_ext", default="default_value")
    tokenizer.remove_extension("test_ext")
    assert not Token.has_extension("test_ext")


def test_remove_stopwords(tokenizer: Tokenizer) -> None:
    """Removes stopwords from the tokenizer object."""
    stopwords = ["the", "a", "an"]
    tokenizer.add_stopwords(stopwords)
    assert tokenizer.nlp.vocab["the"].is_stop
    tokenizer.remove_stopwords("the")
    assert not tokenizer.nlp.vocab["the"].is_stop
    assert "the" not in tokenizer.stopwords


def test_pipeline(tokenizer: Tokenizer) -> None:
    """Returns the spaCy pipeline."""
    pipeline = tokenizer.pipeline
    assert pipeline == ["senter"]


def test_components(tokenizer: Tokenizer) -> None:
    """Returns the spaCy pipeline components."""
    components = tokenizer.components
    assert isinstance(components, list)
    assert len(components) == 1
    assert components[0][0] == "senter"


def test_disabled(tokenizer: Tokenizer) -> None:
    """Returns the disabled spaCy pipeline components."""
    disabled = tokenizer.disabled
    assert disabled == []


def test_slice_tokenizer(sliceTokenizer: SliceTokenizer) -> None:
    """Slice the text into tokens of n characters."""
    text = "This is a test."
    slices = sliceTokenizer(text)
    # Default n=4, drop_ws=True, so spaces are removed: 'Thisisatest.'
    # Slices: ['This', 'isat', 'est.']
    assert slices == ["This", "isat", "est."]


def test_white_space_tokenizer(whitespaceTokenizer: WhitespaceTokenizer) -> None:
    """Split the text into tokens on whitespace."""
    text = "This is a test."
    tokens = whitespaceTokenizer(text)
    assert list(tokens) == ["This", "is", "a", "test."]
