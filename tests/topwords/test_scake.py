"""Tests for sCAKE keyterm extraction module.

Coverage target: high-value behavior for public API and helper utilities.
"""

import collections

import networkx as nx
import pandas as pd
import pytest
import spacy
from pydantic_core._pydantic_core import ValidationError as PydanticValidationError

from lexos.topwords.keyterms.scake import (
    SCake,
    _build_cooc_matrix,
    _compute_node_truss_levels,
    _compute_word_scores,
    _get_candidates,
    _is_valid_tok_doc,
    _is_valid_tok_str,
    _validate_scake_args,
    scake,
)
from lexos.topwords.keyterms.keyterms_util import (
    _to_term_sequence,
    terms_to_strings,
)


@pytest.fixture(scope="module")
def nlp():
    """Create a lightweight spaCy pipeline for doc-based tests."""
    return spacy.blank("en")


@pytest.fixture
def sample_text():
    """Provide a representative text with repeated key terms across sentences."""
    return (
        "Machine learning systems learn from data. "
        "Machine learning models improve with data and feedback. "
        "Neural networks are a kind of machine learning model."
    )


@pytest.fixture
def sample_doc(nlp, sample_text):
    """Convert sample text to a spaCy Doc (no POS tagger — models the xx_sent_ud_sm case)."""
    return nlp(sample_text)


class TestSCakeFunction:
    """Test the standalone scake() function."""

    def test_scake_accepts_string_input(self, sample_text):
        """String input should produce term-score tuples without raising an error."""
        results = scake(sample_text, topn=5)

        assert isinstance(results, list)
        assert len(results) <= 5
        assert all(isinstance(item, tuple) and len(item) == 2 for item in results)
        assert all(isinstance(term, str) and isinstance(score, float)for term, score in results)

    
    # ---FAILING--- To fix later
    def test_scake_accepts_doc_input(self, sample_doc):
        """Doc input should produce non-empty results for valid text."""
        results = scake(sample_doc, topn=5)

        assert isinstance(results, list)
        assert len(results) > 0

    def test_scake_returns_empty_for_empty_string(self):
        """An empty string should return an empty list without raising."""
        results = scake("")

        assert results == []

    def test_scake_returns_empty_for_empty_doc(self, nlp):
        """An empty Doc should return an empty list without raising."""
        results = scake(nlp(""))

        assert results == []

    def test_scake_respects_float_topn_ratio(self):
        """Float topn should be interpreted as a ratio of candidate terms."""
        text = "a b c d"

        results = scake(
            text,
            include_pos=None,
            normalize="lower",
            topn=0.5,
        )

        assert len(results) <= 2

    def test_scake_include_pos_none_works_without_tagger(self, sample_doc):
        """When include_pos=None , it shouldnt raise on a doc from blank pipeline."""
        results = scake(sample_doc, include_pos=None, topn=5)

        assert isinstance(results, list)

    def test_scake_invalid_float_topn_raises(self, sample_text):
        """Float topn outside (0, 1] should raise a ValueError."""
        with pytest.raises(ValueError, match="topn"):
            scake(sample_text, topn=1.5)

    def test_scake_scores_are_non_negative(self, sample_text):
        """All returned scores should be positive floats."""
        results = scake(sample_text, include_pos=None, topn=10)

        assert all(score >= 0.0 for _, score in results)

    def test_scake_results_sorted_descending(self, sample_text):
        """Results should be sorted in descending order of score."""
        results = scake(sample_text, include_pos=None, topn=10)

        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

class TestSCakeClass:
    """Test the SCake Pydantic model wrapper."""

    def test_init_without_doc_raises_validation_error(self):
        """SCake requires `doc` at initialization."""
        with pytest.raises(PydanticValidationError, match="doc"):
            SCake()

    def test_init_sets_keyterms(self, sample_doc):
        """Initialization should fill the keyterms field."""
        model = SCake(doc=sample_doc, topn=5)

        assert model.keyterms is not None
        assert isinstance(model.keyterms, list)

    def test_to_dict_shape(self, sample_doc):
        """The to_dict() should return a dict with 'keyterms' of dicts."""
        model = SCake(doc=sample_doc, topn=5)

        payload = model.to_dict()

        assert "keyterms" in payload
        assert isinstance(payload["keyterms"], list)
        assert all("term" in item and "score" in item for item in payload["keyterms"])

    def test_to_df_returns_dataframe(self, sample_doc):
        """Should return a DataFrame with 'term' and 'score' columns."""
        model = SCake(doc=sample_doc, topn=5)

        df = model.to_df()

        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert "term" in df.columns
            assert "score" in df.columns

    def test_to_dict_and_to_df_are_consistent(self, sample_doc):
        """Should reflect the same keyterms in the same order."""
        model = SCake(doc=sample_doc, topn=5)

        dict_terms = [item["term"] for item in model.to_dict()["keyterms"]]
        df_terms = model.to_df()["term"].tolist()

        assert dict_terms == df_terms

class TestSCakeHelpers:
    """Test sCAKE-specific private helper functions."""

    def test_validate_args_accepts_valid_inputs(self):
        """Valid args should be returned without raising."""
        include_pos_set, topn = _validate_scake_args(
            include_pos=("NOUN", "ADJ"),
            topn=5,
        )

        assert include_pos_set == {"NOUN", "ADJ"}
        assert topn == 5

    def test_validate_args_none_include_pos(self):
        """Return None when include_pos=None, not an empty set."""
        include_pos_set, _ = _validate_scake_args(include_pos=None, topn=5)

        assert include_pos_set is None

    def test_validate_args_float_topn_valid(self):
        """Float topn in (0, 1] should be accepted without raising."""
        _, topn = _validate_scake_args(include_pos=None, topn=0.5)

        assert topn == 0.5

    def test_validate_args_float_topn_invalid_raises(self):
        """Float topn outside (0, 1] should raise a ValueError."""
        with pytest.raises(ValueError, match="topn"):
            _validate_scake_args(include_pos=None, topn=2.0)


    def test_is_valid_tok_doc_allows_token_when_no_pos(self, nlp):
        """Tokens with no POS tag should pass POS filtering."""
        doc = nlp("learning")
        tok = doc[0]

        assert tok.pos_ == ""
        assert _is_valid_tok_doc(tok, include_pos={"NOUN", "ADJ"}) is True

    def test_is_valid_tok_str_accepts_word(self):
        """Plain word strings should be accepted."""
        assert _is_valid_tok_str("haiiiii") is True

    def test_is_valid_tok_str_rejects_punctuation_and_whitespace(self):
        """Punctuation strings should be rejected."""
        assert _is_valid_tok_str("!") is False
        assert _is_valid_tok_str("…") is False 
        assert _is_valid_tok_str(" ") is False
        assert _is_valid_tok_str("\t") is False

    def test_build_cooc_matrix_string_input_produces_pairs(self):
        """Co-occurrence matrix for plain-string input should count word pairs."""
        text = "apple banana apple cherry"
        terms = _to_term_sequence(text)
        normalized = list(terms_to_strings(terms, by="lower"))

        cooc = _build_cooc_matrix(text, terms, normalized, include_pos=None)

        assert isinstance(cooc, collections.Counter)
        assert all(isinstance(k, tuple) and len(k) == 2 for k in cooc)

    def test_build_cooc_matrix_single_word_is_empty(self):
        """A single-word input produces no word pairs."""
        text = "apple"
        terms = _to_term_sequence(text)
        normalized = list(terms_to_strings(terms, by="lower"))

        cooc = _build_cooc_matrix(text, terms, normalized, include_pos=None)

        assert len(cooc) == 0

    def test_get_candidates_returns_set_of_tuples(self, nlp):
        """ The _get_candidates should return a set of string tuples."""
        doc = nlp("machine learning is great")
        terms = list(doc)
        normalized = list(terms_to_strings(terms, by="lower"))

        candidates = _get_candidates(terms, normalized, include_pos=None)

        assert isinstance(candidates, set)
        assert all(isinstance(c, tuple) for c in candidates)
        assert all(isinstance(w, str) for c in candidates for w in c)

    def test_get_candidates_plain_strings(self):
        """Should work with plain string token sequences."""
        text = "i am tired"
        terms = _to_term_sequence(text)
        normalized = list(terms_to_strings(terms, by="lower"))

        candidates = _get_candidates(terms, normalized, include_pos=None)

        assert len(candidates) > 0

    def test_compute_node_truss_levels_triangle(self):
        """A triangle graph should give all nodes a positive truss level."""
        graph = nx.Graph()
        graph.add_edges_from([("a", "b"), ("b", "c"), ("a", "c")])

        levels = _compute_node_truss_levels(graph)

        assert set(levels.keys()) == {"a", "b", "c"}
        assert all(v >= 0 for v in levels.values())

    def test_compute_node_truss_levels_path_gives_zero(self):
        """A path graph with no triangles should give all nodes 0."""
        graph = nx.Graph()
        graph.add_edges_from([("a", "b"), ("b", "c")])

        levels = _compute_node_truss_levels(graph)

        assert all(v == 0 for v in levels.values())
        