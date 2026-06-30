"""Tests for keyterms_util shared utility module.

Coverage target: Major uses of the utilities file functions used for
textrank.py, scake.py, sgrank and yake.py so far.
"""

import math

import pytest
import spacy
from spacy.tokens import Doc

from lexos.topwords.keyterms.keyterms_util import (
    _resolve_topn,
    _to_term_sequence,
    is_unicode_punctuation,
    terms_to_strings,
)

@pytest.fixture(scope="module")
def nlp():
    """Create a lightweight spaCy pipeline for token-based tests."""
    return spacy.blank("en")


class TestIsUnicodePunctuation:
    """Test is_unicode_punctuation()."""

    def test_unicode_punctuation_detection(self):
        """Unicode punctuation should be detected correctly."""
        assert is_unicode_punctuation("!") is True
        assert is_unicode_punctuation("…") is True
        assert is_unicode_punctuation("A") is False
        assert is_unicode_punctuation(" ") is False
        assert is_unicode_punctuation("a!") is False


class TestToTermSequence:
    """Test _to_term_sequence() for both Doc and str inputs."""

    def test_doc_input_returns_list_of_tokens(self, nlp):
        """A spaCy Doc should be converted to a list of Token objects."""
        doc = nlp("hello world")

        result = _to_term_sequence(doc)

        assert isinstance(result, list)
        assert len(result) == 2

    def test_doc_preserves_token_count(self, nlp):
        """Token count in the result should match the Doc length."""
        doc = nlp("one two three four five")

        result = _to_term_sequence(doc)

        assert len(result) == len(doc)

    def test_string_splits_words_correctly(self):
        """Word tokens should be present in the result for a plain string."""
        result = _to_term_sequence("does this work")

        assert "does" in result
        assert "this" in result
        assert "work" in result
   
    def test_string_keeps_punctuation_as_separate_token(self):
        """Punctuation characters should appear as their own tokens."""
        result = _to_term_sequence("hi, jax")

        assert "hi" in result
        assert "," in result
        assert "jax" in result


class TestResolveTopn:
    """Test _resolve_topn() for int and float inputs."""

    def test_resolve_topn_float_and_int(self):
        """_resolve_topn should preserve ints and convert float ratios."""
        assert _resolve_topn(3, candidate_count=10) == 3
        assert _resolve_topn(0.4, candidate_count=10) == 4

    def test_int_topn_of_one_is_returned_unchanged(self):
        """An integer topn of 1 should be returned as is."""
        assert _resolve_topn(1, candidate_count=100) == 1

    def test_float_topn_with_zero_candidates_returns_zero(self):
        """Float topn with zero candidates should return 0."""
        assert _resolve_topn(0.5, candidate_count=0) == 0


class TestTermsToStrings:
    """Test terms_to_strings()."""

    def test_terms_to_strings_invalid_mode_raises(self):
        """Invalid normalization mode should fail fast."""
        with pytest.raises(ValueError, match="by="):
            list(terms_to_strings(["Token"], by="invalid"))

    def test_terms_to_strings_lower_for_doc_tokens(self, nlp):
        """Lower normalization should lowercase spaCy token text."""
        doc = nlp("Hello World Test")
        out = list(terms_to_strings(doc[:3], by="lower"))

        assert len(out) == 3
        assert all(term == term.lower() for term in out)
        assert all(isinstance(term, str) for term in out)

    def test_orth_preserves_token_casing(self, nlp):
        """'orth' mode should return token text exactly as written."""
        doc = nlp("Its Cold")

        result = list(terms_to_strings(doc[:2], by="orth"))

        assert result == ["Its", "Cold"]

    def test_none_preserves_token_casing(self, nlp):
        """None mode should be identically to 'orth'."""
        doc = nlp("Its Cold")

        result = list(terms_to_strings(doc[:2], by=None))

        assert result == ["Its", "Cold"]

    def test_lemma_returns_strings_for_tokens(self, nlp):
        """'lemma' mode should return a string for each Token."""
        doc = nlp("music rocks")

        result = list(terms_to_strings(doc[:2], by="lemma"))

        assert len(result) == 2
        assert all(isinstance(t, str) for t in result)

    def test_invalid_mode_raises_value_error(self):
        """An unrecognised `by` value should raise a ValueError."""
        with pytest.raises(ValueError, match="by="):
            list(terms_to_strings(["hello"], by="invalid"))

    def test_output_length_matches_input_length(self, nlp):
        """Output should have exactly one string per input term."""
        doc = nlp("one two three")

        result = list(terms_to_strings(doc, by="lower"))

        assert len(result) == len(doc)