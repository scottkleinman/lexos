"""Tests for YAKE keyterm extraction module."""

import pandas as pd
import pytest
import spacy

from lexos.topwords.keyterms.yake import Yake, yake


@pytest.fixture(scope="module")
def nlp():
    """Create a small spaCy pipeline for doc-based tests."""
    return spacy.blank("en")


@pytest.fixture
def sample_text():
    """Provide representative text for keyterm extraction."""
    return (
        "Machine learning models improve with data. "
        "Neural networks are a kind of machine learning model."
    )


class BareToken:
    """Minimal token-like object without POS attributes."""

    def __init__(self, text: str):
        self.text = text
        self.is_stop = False
        self.is_punct = False
        self.is_space = False
        self.is_upper = text.isupper()
        self.is_title = text.istitle()


class TestYakeFunction:
    """Test the standalone yake() function."""

    def test_yake_accepts_string_input(self, sample_text):
        """Raw string input should return scored keyterms."""
        results = yake(sample_text, topn=5)

        assert isinstance(results, list)
        assert len(results) <= 5
        assert all(
            isinstance(term, str) and isinstance(score, float)
            for term, score in results
        )

    def test_yake_accepts_list_of_strings(self):
        """List[str] input should be accepted and scored."""
        terms = ["machine", "learning", "improves", "analysis", ".", "models"]

        results = yake(terms, include_pos=None, topn=5)

        assert isinstance(results, list)
        assert len(results) > 0

    def test_yake_accepts_tokens_without_pos(self):
        """Token-like objects without POS attributes should not fail include_pos filtering."""
        terms = [
            BareToken("Machine"),
            BareToken("learning"),
            BareToken("analysis"),
            BareToken("pipeline"),
        ]

        results = yake(terms, include_pos=("NOUN",), ngrams=(1, 2), topn=5)

        assert isinstance(results, list)
        assert len(results) > 0

    def test_yake_rejects_invalid_topn_float(self, sample_text):
        """Topn float outside (0, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="topn"):
            yake(sample_text, topn=1.5)


class TestYakeClass:
    """Test the Yake class wrapper."""

    def test_init_without_doc_raises_validation_error(self):
        """Yake requires `doc` at initialization."""
        with pytest.raises(Exception, match="doc"):
            Yake()

    def test_init_extracts_to_dict_and_df(self, sample_text):
        """Initialization should populate keyterms and serialize correctly."""
        model = Yake(doc=sample_text, topn=5)

        data = model.to_dict()
        df = model.to_df()

        assert model.keyterms is not None
        assert "keyterms" in data
        assert isinstance(data["keyterms"], list)
        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert "term" in df.columns
            assert "score" in df.columns
