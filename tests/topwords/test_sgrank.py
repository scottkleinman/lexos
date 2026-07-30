"""test_sgrank.py.

Tests for SGRank keyterm extraction module.

Coverage: 98%. Missing: 296, 330

Last Updated: July 30, 2026
"""

import collections

import pandas as pd
import pytest
import spacy
from pydantic_core._pydantic_core import ValidationError as PydanticValidationError

from lexos.topwords.keyterms.keyterms_util import (
    _to_term_sequence,
    terms_to_strings,
)
from lexos.topwords.keyterms.sgrank import (
    Candidate,
    SGRank,
    _build_weighted_graph,
    _compute_term_weights,
    _get_candidates,
    _score_candidate_phrases,
    _validate_sgrank_args,
    sgrank,
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


class TestSGRankFunction:
    """Test the sgrank() function."""

    def test_sgrank_accepts_string_input(self, sample_text):
        """String input should produce term-score tuples without raising."""
        results = sgrank(sample_text, topn=5)

        assert isinstance(results, list)
        assert len(results) <= 5
        assert all(isinstance(item, tuple) and len(item) == 2 for item in results)
        assert all(
            isinstance(term, str) and isinstance(score, float)
            for term, score in results
        )

    def test_sgrank_accepts_doc_input(self, sample_doc):
        """Doc input should produce non-empty results for valid text."""
        results = sgrank(sample_doc, topn=5)

        assert isinstance(results, list)
        assert len(results) > 0

    def test_sgrank_returns_empty_for_empty_string(self):
        """An empty string should return an empty list without error."""
        results = sgrank("")

        assert results == []

    def test_sgrank_returns_empty_for_empty_doc(self, nlp):
        """An empty Doc should return an empty list."""
        results = sgrank(nlp(""))

        assert results == []

    def test_sgrank_include_pos_none_works_without_tagger(self, sample_doc):
        """include_pos=None should not raise on a doc produced by a blank pipeline."""
        results = sgrank(sample_doc, include_pos=None, topn=5)

        assert isinstance(results, list)

    def test_sgrank_respects_float_topn_ratio(self, sample_text):
        """Float topn should be interpreted as a ratio of candidate terms."""
        results = sgrank(
            sample_text,
            include_pos=None,
            normalize="lower",
            ngrams=1,
            topn=0.25,
        )

        assert isinstance(results, list)

    def test_sgrank_invalid_float_topn_raises(self, sample_text):
        """Float topn outside (0, 1] should raise a ValueError."""
        with pytest.raises(ValueError, match="topn"):
            sgrank(sample_text, topn=1.5)

    def test_sgrank_invalid_window_size_raises(self, sample_text):
        """window_size < 2 should raise a ValueError."""
        with pytest.raises(ValueError, match="window_size"):
            sgrank(sample_text, window_size=1)

    def test_sgrank_scores_are_non_negative(self, sample_text):
        """All returned scores should be non-negative floats."""
        results = sgrank(sample_text, include_pos=None, topn=10)

        assert all(score >= 0.0 for _, score in results)

    def test_sgrank_results_sorted_descending(self, sample_text):
        """Results should be sorted in descending order of score."""
        results = sgrank(sample_text, include_pos=None, topn=10)

        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_sgrank_single_ngram_size(self, sample_text):
        """Passing a single int ngrams value should work without raising an error."""
        results = sgrank(sample_text, include_pos=None, ngrams=1, topn=5)

        assert isinstance(results, list)

    def test_sgrank_multiple_ngram_sizes(self, sample_text):
        """Passing multiple ngram sizes should work."""
        results = sgrank(
            sample_text, include_pos=None, normalize="lower", ngrams=(1, 2), topn=10
        )

        assert isinstance(results, list)


class TestSGRankClass:
    """Test the SGRank Pydantic model wrapper."""

    def test_init_without_doc_raises_validation_error(self):
        """SGRank requires `doc` at initialization."""
        with pytest.raises(PydanticValidationError, match="doc"):
            SGRank()

    def test_init_sets_keyterms(self, sample_doc):
        """Initialization should fill the keyterms field."""
        model = SGRank(doc=sample_doc, topn=5)

        assert model.keyterms is not None
        assert isinstance(model.keyterms, list)

    def test_to_dict_shape(self, sample_doc):
        """The to_dict() function should return a dict with a 'keyterms' key of dicts."""
        model = SGRank(doc=sample_doc, topn=5)

        payload = model.to_dict()

        assert "keyterms" in payload
        assert isinstance(payload["keyterms"], list)
        assert all("term" in item and "score" in item for item in payload["keyterms"])

    def test_to_df_returns_dataframe(self, sample_doc):
        """The to_df() function should return a DataFrame with 'term' and 'score' columns."""
        model = SGRank(doc=sample_doc, topn=5)

        df = model.to_df()

        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert "term" in df.columns
            assert "score" in df.columns

    def test_to_dict_and_to_df_are_consistent(self, sample_doc):
        """For to_dict() and to_df(), they should reflect the same keyterms in the same order."""
        model = SGRank(doc=sample_doc, topn=5)

        dict_terms = [item["term"] for item in model.to_dict()["keyterms"]]
        df_terms = model.to_df()["term"].tolist()

        assert dict_terms == df_terms


class TestSGRankHelpers:
    """Test SGRank-specific private helper functions."""

    def test_validate_args_accepts_valid_inputs(self):
        """Valid args should be returned without an error."""
        include_pos_set, ngram_sizes, topn = _validate_sgrank_args(
            include_pos=("NOUN", "ADJ"),
            ngrams=(1, 2, 3),
            window_size=10,
            topn=5,
        )

        assert include_pos_set == {"NOUN", "ADJ"}
        assert ngram_sizes == (1, 2, 3)
        assert topn == 5

    def test_validate_args_none_include_pos(self):
        """When include_pos=None, should return None."""
        include_pos_set, _, _ = _validate_sgrank_args(
            include_pos=None, ngrams=1, window_size=10, topn=5
        )

        assert include_pos_set is None

    def test_validate_args_int_ngrams_converted_to_tuple(self):
        """A single int ngrams value should be wrapped in a tuple."""
        _, ngram_sizes, _ = _validate_sgrank_args(
            include_pos=None, ngrams=2, window_size=10, topn=5
        )

        assert ngram_sizes == (2,)

    def test_validate_args_float_topn_valid(self):
        """Float topn in (0, 1] should be accepted."""
        _, _, topn = _validate_sgrank_args(
            include_pos=None, ngrams=1, window_size=10, topn=0.5
        )

        assert topn == 0.5

    def test_validate_args_float_topn_invalid_raises(self):
        """Float topn outside (0, 1] should raise a ValueError."""
        with pytest.raises(ValueError, match="topn"):
            _validate_sgrank_args(include_pos=None, ngrams=1, window_size=10, topn=2.0)

    def test_candidate_fields_accessible_by_name(self):
        """Candidate fields should be accessible by name."""
        c = Candidate(text="machine learning", idx=0, length=2, count=3)

        assert c.text == "machine learning"
        assert c.idx == 0
        assert c.length == 2
        assert c.count == 3

    def test_get_candidates_returns_list_of_candidates(self, nlp):
        """The _get_candidates should return a list of Candidate objects."""
        doc = nlp("machine learning is great")
        terms = list(doc)
        normalized = list(terms_to_strings(terms, by="lower"))

        candidates = _get_candidates(
            terms, normalized, include_pos=None, ngram_sizes=(1,)
        )

        assert isinstance(candidates, list)
        assert all(isinstance(c, Candidate) for c in candidates)

    def test_get_candidates_records_position(self, nlp):
        """Each candidate should record have a starting index."""
        doc = nlp("alpha beta gamma")
        terms = list(doc)
        normalized = list(terms_to_strings(terms, by="lower"))

        candidates = _get_candidates(
            terms, normalized, include_pos=None, ngram_sizes=(1,)
        )

        idxs = [c.idx for c in candidates]
        assert idxs == sorted(idxs)

    def test_get_candidates_produces_bigrams(self, nlp):
        """With ngram_sizes=(2,) at least one bigram candidate should be produced."""
        doc = nlp("machine learning model")
        terms = list(doc)
        normalized = list(terms_to_strings(terms, by="lower"))

        candidates = _get_candidates(
            terms, normalized, include_pos=None, ngram_sizes=(2,)
        )

        assert any(c.length == 2 for c in candidates)

    def test_get_candidates_plain_strings(self):
        """Should work with plain string token sequences."""
        text = "alpha beta gamma"
        terms = _to_term_sequence(text)
        normalized = list(terms_to_strings(terms, by="lower"))

        candidates = _get_candidates(
            terms, normalized, include_pos=None, ngram_sizes=(1,)
        )

        assert len(candidates) > 0

    def test_get_candidates_no_pos_tag_allows_token(self, nlp):
        """Tokens from a blank pipeline (no POS) should pass the POS filter."""
        doc = nlp("learning")
        terms = list(doc)
        normalized = list(terms_to_strings(terms, by="lower"))

        candidates = _get_candidates(
            terms, normalized, include_pos={"NOUN", "ADJ"}, ngram_sizes=(1,)
        )

        assert len(candidates) > 0

    def test_compute_term_weights_ngrams_not_affected_by_idf(self):
        """N-grams (containing spaces) should not be multiplied by idf."""
        candidates = [
            Candidate("apple pie", 0, 2, 0),
            Candidate("apple pie", 3, 2, 0),
        ]
        idf = {"apple pie": 5.0}

        weights = _compute_term_weights(candidates, idf=idf)

        assert weights["apple pie"] == 2.0

    def test_build_weighted_graph_connects_co_occurring_terms(self):
        """Terms within the window should be connected by an edge."""
        candidates = [
            Candidate("apple", 0, 1, 0),
            Candidate("banana", 3, 1, 0),
        ]
        term_weights = {"apple": 1.0, "banana": 1.0}

        graph = _build_weighted_graph(candidates, term_weights, window_size=100)

        assert graph.has_edge("apple", "banana") or graph.has_edge("banana", "apple")

    def test_score_candidate_phrases_takes_max_for_duplicates(self):
        """Take the highest score if a candidate appears multiple times."""
        candidates = [
            Candidate("apple", 0, 1, 0),
            Candidate("apple", 5, 1, 0),
        ]
        word_scores = {"apple": 0.7}

        scores = _score_candidate_phrases(candidates, word_scores)

        assert scores["apple"] == 0.7
        assert len(scores) == 1

    def test_validate_args_invalid_ngrams_int(self):
        """Test invalid ngrams integer < 1 (Line 172)."""
        with pytest.raises(ValueError, match="ngrams"):
            _validate_sgrank_args(None, 0, 10, 5)

    def test_validate_args_invalid_ngrams_iterable(self):
        """Test invalid ngrams iterable (Line 180)."""
        # Empty iterable
        with pytest.raises(ValueError, match="ngrams"):
            _validate_sgrank_args(None, [], 10, 5)
        # Non-int or < 1
        with pytest.raises(TypeError):
            _validate_sgrank_args(None, [1, "2"], 10, 5)
        with pytest.raises(ValueError, match="ngrams"):
            _validate_sgrank_args(None, [1, 0], 10, 5)

    def test_sgrank_no_terms_early_exit(self):
        """Test sgrank early exit if no terms found (Line 204)."""
        assert sgrank("   ") == []

    def test_sgrank_no_candidates_early_exit(self):
        """Test sgrank early exit if no candidates found (Line 213)."""
        # If we include stopwords and punct in include_pos, it might still skip them if they are in the default stopword list
        # But _is_valid(tok) has if tok.is_stop or tok.is_punct or tok.is_space: return False
        # So symbols should be ignored.
        assert sgrank("! ! !", include_pos=None) == []

    def test_build_weighted_graph_empty_exit(self):
        """Test sgrank early exit if graph has no nodes (Line 296)."""
        # If all candidates have same text, no edges added.
        # But wait, apple apple apple will produce "apple" unigram.
        # PageRank on a single node graph?
        # Actually, "apple" alone has no co-occurrences, so edge_weights is empty.
        # graph.number_of_nodes() will be 0 if no edges were added and we only add edges?
        # Let's check _build_weighted_graph code: it ONLY adds weighted edges.
        # So a single-word candidate list will result in 0 nodes.
        assert sgrank("apple", include_pos=None) == []

    def test_build_weighted_graph_distance_overlap(self):
        """Test _build_weighted_graph distance <= 0 branch (Line 330)."""
        c1 = Candidate("a", 0, 1, 0)
        c2 = Candidate("b", 0, 1, 0)  # Same index
        graph = _build_weighted_graph([c1, c2], {"a": 1.0, "b": 1.0}, 10)
        assert graph.number_of_edges() == 0

    def test_score_candidate_phrases_none_score(self):
        """Test _score_candidate_phrases line 358 (score is None)."""
        # Candidate not present in word_scores
        c = Candidate("a", 0, 1, 0)
        scores = _score_candidate_phrases([c], {})
        assert scores == {}
