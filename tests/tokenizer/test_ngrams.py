"""test_ngrams.py.

Unit tests for the Ngrams class in lexos.tokenizer.ngrams

Purpose:

These tests verify the correct behavior of the Ngrams class which generates
sequences of n-grams (typically bigrams) from various forms of input:
    - spaCy Doc objects
    - plain text strings
    - lists of tokens
    - lists of Docs, texts, or token lists

The tests ensure consistent and correct output across multiple output formats:
    - 'tuples' : pairs of string tokens
    - 'text'   : space-joined strings of tokens
    - 'spans'  : spaCy Span objects (if using a spaCy tokenizer)

Functionality Tested:

 Output format correctness
 Filtering options:
    - Stop words (`filter_stops`)
    - Digits (`filter_digits`)
    - Punctuation (`filter_punct`)
 Minimum frequency filtering (`min_freq`)
 Compatibility with different tokenizers:
    - spaCy NLP pipeline
    - SliceTokenizer (custom character-level tokenizer)
 Robustness on edge cases:
    - Empty strings
    - Text shorter than required n-gram size
    - Very long strings

Usage:
To run the tests for this module:
    uv run pytest tests/tokenizer/test_ngrams.py

Coverage: 100%
Last updated: May 27, 2025
"""

import pytest
import spacy

from lexos.exceptions import LexosException
from lexos.tokenizer import SliceTokenizer, WhitespaceTokenizer
from lexos.tokenizer.ngrams import Ngrams

nlp = spacy.load("en_core_web_sm")


@pytest.fixture
def ng() -> Ngrams:
    """Fixture for the Ngrams class."""
    return Ngrams()


def test_ngrams_stopwords(ng: Ngrams) -> None:
    """Adds stopwords to the ngrams object."""
    ng.filter_stops = ["Stop", "Words"]
    assert ng.stopwords == ["Stop", "Words"]


def test_ngrams_from_doc_output(ng: Ngrams) -> None:
    """Test output formats (tuples, text, spans) from spaCy Doc input."""
    doc = nlp("This is a test.")
    ngrams = ng.from_doc(doc, output="tuples")
    assert list(ngrams) == [("This", "is"), ("is", "a"), ("a", "test")]
    ngrams = ng.from_doc(doc, output="text")
    assert list(ngrams) == ["This is", "is a", "a test"]
    ngrams = ng.from_doc(doc, output="spans")
    assert [span.text for span in ngrams] == ["This is", "is a", "a test"]


def test_ngrams_from_text_output(ng: Ngrams) -> None:
    """Test output formats from plain text input."""
    text = "This is a test."
    ngrams = ng.from_text(text, output="tuples")
    assert list(ngrams) == [("This", "is"), ("is", "a"), ("a", "test.")]
    ngrams = ng.from_text(text, output="text")
    assert list(ngrams) == ["This is", "is a", "a test."]


def test_ngrams_from_tokens_output(ng: Ngrams) -> None:
    """Test output formats from list of tokens."""
    tokens = ["This", "is", "a", "test", "."]
    ngrams = ng.from_tokens(tokens, output="tuples")
    assert list(ngrams) == [("This", "is"), ("is", "a"), ("a", "test")]
    ngrams = ng.from_tokens(tokens, output="text")
    assert list(ngrams) == ["This is", "is a", "a test"]


def test_ngrams_from_docs(ng: Ngrams) -> None:
    """Test n-grams from multiple spaCy Doc objects."""
    doc = nlp("This is a test.")
    ngrams = ng.from_docs([doc, doc], output="text")
    for doc_ng in ngrams:
        assert list(doc_ng) == ["This is", "is a", "a test"]


def test_ngrams_from_texts(ng: Ngrams) -> None:
    """Test n-grams from multiple plain text strings."""
    text = "This is a test."
    ngrams = ng.from_texts([text, text], output="text")
    for doc_ng in ngrams:
        assert list(doc_ng) == ["This is", "is a", "a test."]


def test_ngrams_from_token_lists(ng: Ngrams) -> None:
    """Test n-grams from multiple lists of tokens."""
    tokens = ["This", "is", "a", "test", "."]
    ngrams = ng.from_token_lists([tokens, tokens], output="text")
    for doc_ng in ngrams:
        assert list(doc_ng) == ["This is", "is a", "a test"]


def test_ngrams_from_doc_filter_nums(ng: Ngrams) -> None:
    """Test filtering numeric tokens from spaCy Doc input."""
    doc = nlp("This is test ten of 10.")
    ngrams = ng.from_doc(doc, filter_nums=True, output="tuples")
    assert list(ngrams) == [("This", "is"), ("is", "test")]


def test_ngrams_from_doc_filter_digits(ng: Ngrams) -> None:
    """Test filtering digits from spaCy Doc input."""
    doc = nlp("This is test ten of 10.")
    ngrams = ng.from_doc(doc, filter_digits=True, output="tuples")
    assert list(ngrams) == [
        ("This", "is"),
        ("is", "test"),
        ("test", "ten"),
        ("ten", "of"),
    ]


def test_ngrams_from_doc_filter_punct(ng: Ngrams) -> None:
    """Test filtering punctuation from spaCy Doc input."""
    doc = nlp("This is test.")
    ngrams = ng.from_doc(doc, filter_punct=False, output="tuples")
    assert list(ngrams) == [("This", "is"), ("is", "test"), ("test", ".")]


def test_ngrams_from_doc_min_freq(ng: Ngrams) -> None:
    """Test minimum frequency filtering from spaCy Doc input."""
    doc = nlp("This is test.")
    ngrams = ng.from_doc(doc, min_freq=2, output="tuples")
    assert list(ngrams) == []


def test_ngrams_from_doc_filter_stops(ng: Ngrams) -> None:
    """Test filtering stop words from spaCy Doc input."""
    doc = nlp("This is really big test.")
    ngrams = ng.from_doc(doc, filter_stops=True, output="tuples")
    assert list(ngrams) == [("big", "test")]
    ngrams = ng.from_doc(doc, filter_stops=False, output="tuples")
    assert list(ngrams) == [
        ("This", "is"),
        ("is", "really"),
        ("really", "big"),
        ("big", "test"),
    ]


def test_ngrams_from_doc_exception(ng: Ngrams) -> None:
    """Generate an exception when an invalid output type is provided using from_doc."""
    doc = nlp("This test should raise an exception.")
    with pytest.raises(LexosException, match="Invalid output type."):
        list(ng.from_doc(doc, output="invalid_format"))


def test_ngrams_from_text_filter_digits(ng: Ngrams) -> None:
    """Test filtering digits from plain text input."""
    text = "This is test ten of 10."
    ngrams = ng.from_text(text, filter_digits=True, output="tuples")
    assert list(ngrams) == [
        ("This", "is"),
        ("is", "test"),
        ("test", "ten"),
        ("ten", "of"),
    ]


def test_ngrams_from_text_filter_punct(ng: Ngrams) -> None:
    """Test filtering punctuation from plain text input."""
    text = "This is test."
    ngrams = ng.from_text(text, filter_punct=False, output="tuples")
    assert list(ngrams) == [("This", "is"), ("is", "test.")]


def test_ngrams_from_text_filter_stops(ng: Ngrams) -> None:
    """Test filtering stop words from plain text input."""
    text = "This is test."
    ngrams = ng.from_text(text, filter_stops=["is"], output="tuples")
    assert list(ngrams) == [("This", "test.")]


def test_ngrams_from_text_min_freq(ng: Ngrams) -> None:
    """Test minimum frequency filtering from plain text input."""
    text = "This is test."
    ngrams = ng.from_text(text, min_freq=2, output="tuples")
    assert list(ngrams) == []


def test_ngrams_from_text_exception(ng: Ngrams) -> None:
    """Generate an exception when an invalid output type is provided using from_text."""
    text = "This test should raise an exception."
    with pytest.raises(LexosException, match="Invalid output type."):
        list(ng.from_text(text, output="invalid_format"))


def test_ngrams_from_tokens_filter_digits(ng: Ngrams) -> None:
    """Test filtering digits from list of tokens."""
    tokens = ["This", "is", "test", "ten", "of", "10", "."]
    ngrams = ng.from_tokens(tokens, filter_digits=True, output="tuples")
    assert list(ngrams) == [
        ("This", "is"),
        ("is", "test"),
        ("test", "ten"),
        ("ten", "of"),
    ]


def test_ngrams_from_tokens_filter_punct(ng: Ngrams) -> None:
    """Test filtering punctuation from list of tokens."""
    tokens = ["This", "is", "a", "test", "."]
    ngrams = ng.from_tokens(tokens, filter_punct=False, output="tuples")
    assert list(ngrams) == [("This", "is"), ("is", "a"), ("a", "test"), ("test", ".")]


def test_ngrams_from_tokens_filter_stops(ng: Ngrams) -> None:
    """Test filtering stop words from list of tokens."""
    tokens = ["This", "is", "a", "test", "."]
    ngrams = ng.from_tokens(tokens, filter_stops=["is"], output="tuples")
    assert list(ngrams) == [("This", "a"), ("a", "test")]


def test_ngrams_from_tokens_min_freq(ng: Ngrams) -> None:
    """Test minimum frequency filtering from list of tokens."""
    tokens = ["This ", "is", "a", "test", "."]
    ngrams = ng.from_tokens(tokens, min_freq=2, output="tuples")
    assert list(ngrams) == []


def test_ngrams_from_text_slice_tokenizer(ng: Ngrams) -> None:
    """Test n-grams from text using SliceTokenizer."""
    text = "This is test."
    tokenizer = SliceTokenizer(n=2, drop_ws=True)
    ngrams = ng.from_text(text, tokenizer=tokenizer, output="tuples")
    assert list(ngrams) == [("Th", "is"), ("is", "is"), ("is", "te"), ("te", "st")]
    tokenizer = SliceTokenizer(n=2, drop_ws=False)
    ngrams = ng.from_text(text, tokenizer=tokenizer, output="tuples")
    assert list(ngrams) == [
        ("Th", "is"),
        ("is", "i"),
        ("i", "s"),
        ("s", "te"),
        ("te", "st"),
    ]


def test_ngrams_from_text_nlp(ng: Ngrams) -> None:
    """Test n-grams from text using spaCy NLP pipeline."""
    text = "This is test."
    ngrams = ng.from_text(text, tokenizer=nlp, output="tuples")
    assert list(ngrams) == [("This", "is"), ("is", "test")]


def test_stopwords_property() -> None:
    """Test the stopwords property of Ngrams."""
    ng = Ngrams(filter_stops=["and", "the"])
    assert ng.stopwords == ["and", "the"]


def test_from_text_invalid_output() -> None:
    """Test exception raised on invalid output from text input."""
    ng = Ngrams()
    with pytest.raises(LexosException, match="Invalid output type."):
        list(ng.from_text("This is a test.", output="invalid_output"))


def test_from_doc_invalid_output() -> None:
    """Test exception raised on invalid output from Doc input."""
    ng = Ngrams()
    doc = nlp("This is a test.")
    with pytest.raises(LexosException, match="Invalid output type."):
        list(ng.from_doc(doc, output="invalid_output"))


def test_from_tokens_invalid_output() -> None:
    """Test exception raised on invalid output from token list input."""
    ng = Ngrams()
    tokens = ["This", "is", "a", "test"]
    with pytest.raises(LexosException, match="Invalid output type."):
        list(ng.from_tokens(tokens, output="invalid_output"))
