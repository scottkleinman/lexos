"""Tests for TextRank keyterm extraction module.

Coverage target: high-value behavior for public API and helper utilities.
"""

import math

import pandas as pd
import pytest
import spacy
from pydantic_core._pydantic_core import ValidationError as PydanticValidationError
from spacy.tokens import Doc

from lexos.topwords.keyterms.textrank import (
    TextRank,
    _build_position_bias,
    _resolve_topn,
    _validate_textrank_args,
    get_longest_subsequence_candidates,
    is_unicode_punctuation,
    terms_to_strings,
    textrank,
)


@pytest.fixture(scope="module")
def nlp():
    """Create a lightweight spaCy pipeline for doc-based tests."""
    return spacy.blank("en")


@pytest.fixture
def sample_text():
    """Provide a representative text with repeated key terms."""
    return (
        "Machine learning systems learn from data. "
        "Machine learning models improve with data and feedback. "
        "Neural networks are a kind of machine learning model."
    )


@pytest.fixture
def sample_doc(nlp, sample_text):
    """Convert sample text to a spaCy Doc."""
    return nlp(sample_text)


class TestTextRankFunction:
    """Test the standalone textrank() function."""

    def test_textrank_accepts_string_input(self, sample_text):
        """String input should produce term-score tuples."""
        results = textrank(sample_text, topn=5)

        assert isinstance(results, list)
        assert len(results) <= 5
        assert all(isinstance(item, tuple) and len(item) == 2 for item in results)
        assert all(
            isinstance(term, str) and isinstance(score, float)
            for term, score in results
        )

    def test_textrank_accepts_doc_input(self, sample_doc):
        """Doc input should also produce non-empty results for valid text."""
        results = textrank(sample_doc, topn=5)

        assert isinstance(results, list)
        assert len(results) > 0

    def test_candidate_weighting_frequency_boosts_repeats(self):
        """Frequency weighting should not underperform unique weighting on repeated terms."""
        text = "apple apple apple banana"

        unique_scores = dict(
            textrank(
                text,
                include_pos=None,
                normalize="lower",
                ngrams=1,
                candidate_weighting="unique",
                topn=10,
            )
        )
        frequency_scores = dict(
            textrank(
                text,
                include_pos=None,
                normalize="lower",
                ngrams=1,
                candidate_weighting="frequency",
                topn=10,
            )
        )

        assert "apple" in unique_scores and "apple" in frequency_scores
        assert frequency_scores["apple"] >= unique_scores["apple"]

    def test_textrank_respects_float_topn_ratio(self):
        """Float topn should be interpreted as a ratio of candidate terms."""
        text = "alpha beta gamma delta"

        results = textrank(
            text,
            include_pos=None,
            normalize="lower",
            ngrams=1,
            topn=0.5,
        )

        # 4 candidates * 0.5 -> round(2.0) == 2
        assert len(results) <= 2

    def test_textrank_invalid_candidate_weighting_raises(self, sample_text):
        """Invalid candidate_weighting should raise a ValueError."""
        with pytest.raises(ValueError, match="candidate_weighting"):
            textrank(sample_text, candidate_weighting="invalid")  # type: ignore[arg-type]


class TestTextRankClass:
    """Test the TextRank model wrapper API."""

    def test_init_without_doc_raises_validation_error(self):
        """TextRank requires `doc` at initialization."""
        with pytest.raises(PydanticValidationError, match="doc"):
            TextRank()

    def test_init_sets_keyterms_and_to_dict(self, sample_doc):
        """Initialization should populate keyterms and serialize to dict shape."""
        model = TextRank(doc=sample_doc, topn=5)

        payload = model.to_dict()

        assert model.keyterms is not None
        assert "keyterms" in payload
        assert isinstance(payload["keyterms"], list)
        assert all("term" in item and "score" in item for item in payload["keyterms"])

    def test_to_df_returns_dataframe(self, sample_doc):
        """to_df() should convert extracted terms to a DataFrame."""
        model = TextRank(doc=sample_doc, topn=5)

        df = model.to_df()

        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert "term" in df.columns
            assert "score" in df.columns


class TestTextRankHelpers:
    """Test helper functions with deterministic expectations."""

    def test_validate_args_normalizes_sets_and_ngrams(self):
        """Validation should coerce values to expected internal forms."""
        include_pos, stopwords, ngrams, topn = _validate_textrank_args(
            include_pos=("NOUN", "ADJ"),
            stopwords=("The", "And"),
            ngrams=(2, 1, 2),
            candidate_weighting="unique",
            topn=3,
        )

        assert include_pos == {"NOUN", "ADJ"}
        assert stopwords == {"the", "and"}
        assert ngrams == (1, 2)
        assert topn == 3

    def test_validate_args_rejects_invalid_ngrams(self):
        """Invalid ngrams values should raise a ValueError."""
        with pytest.raises(ValueError, match="ngrams"):
            _validate_textrank_args(
                include_pos=None,
                stopwords=None,
                ngrams=0,
                candidate_weighting="unique",
                topn=5,
            )

    def test_position_bias_is_normalized_distribution(self):
        """Position bias values should sum to 1 for non-empty term lists."""
        word_pos = _build_position_bias(["a", "b", "a"])

        assert set(word_pos.keys()) == {"a", "b"}
        assert math.isclose(sum(word_pos.values()), 1.0, rel_tol=1e-9)
        assert word_pos["a"] > word_pos["b"]

    def test_resolve_topn_float_and_int(self):
        """_resolve_topn should preserve ints and convert float ratios."""
        assert _resolve_topn(3, candidate_count=10) == 3
        assert _resolve_topn(0.4, candidate_count=10) == 4

    def test_terms_to_strings_invalid_mode_raises(self):
        """Invalid normalization mode should fail fast."""
        with pytest.raises(ValueError, match="by="):
            list(terms_to_strings(["Token"], by="invalid"))

    def test_unicode_punctuation_detection(self):
        """Unicode punctuation should be detected correctly."""
        assert is_unicode_punctuation("!") is True
        assert is_unicode_punctuation("…") is True
        assert is_unicode_punctuation("A") is False

    def test_longest_subsequence_candidates_for_string(self):
        """Longest matching runs should be returned as tuple candidates."""
        text = "aa bb cc dd"

        # Only tokens starting with a or b should match.
        candidates = list(
            get_longest_subsequence_candidates(
                text,
                match_func=lambda tok: isinstance(tok, str)
                and tok[0].lower() in {"a", "b"},
            )
        )

        assert candidates == [("aa", "bb")]

    def test_terms_to_strings_lower_for_doc_tokens(self, sample_doc):
        """Lower normalization should lowercase token text for spaCy terms."""
        out = list(terms_to_strings(sample_doc[:3], by="lower"))

        assert len(out) == 3
        assert all(term == term.lower() for term in out)
        assert all(isinstance(term, str) for term in out)
