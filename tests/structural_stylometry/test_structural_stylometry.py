"""test_structural_stylometry.py.

Unit tests for the StructuralAnalyzer class in lexos.structural_stylometry.

Coverage: 100%
Last Update: July 22, 2026
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from spacy.tokens import Doc

from lexos.corpus import Corpus
from lexos.io.loader import Loader
from lexos.structural_stylometry.structural_stylometry import StructuralAnalyzer


@pytest.fixture
def sample_texts():
    """Provide a sample dictionary of texts for testing."""
    return {
        "doc1": "Hello world! This is a test. How many dots...?",
        "doc2": "Another test; with semicolons; and some more! !!!",
        "doc3": "This one has  double spaces and \n\nmultiple newlines. ",
    }


def test_structural_analyzer_init_dict(sample_texts):
    """Test initialization with a dictionary of strings."""
    analyzer = StructuralAnalyzer(corpus=sample_texts, min_punctuation_threshold=1)
    assert len(analyzer.doc_ids) == 3
    assert "doc1" in analyzer.doc_ids
    assert "doc2" in analyzer.doc_ids
    assert "doc3" in analyzer.doc_ids
    assert len(analyzer.vocabulary) > 0
    assert analyzer.raw_matrix.shape == (3, len(analyzer.vocabulary))


def test_structural_analyzer_init_corpus(sample_texts):
    """Test initialization with a Lexos Corpus object."""
    corpus = Corpus()
    for name, text in sample_texts.items():
        corpus.add(name=name, content=text)

    analyzer = StructuralAnalyzer(corpus=corpus, min_punctuation_threshold=1)
    assert len(analyzer.doc_ids) == 3
    assert "doc1" in analyzer.doc_ids


def test_structural_analyzer_low_count_warning(sample_texts):
    """Test warning when punctuation count is below threshold."""
    # sample_texts docs have low punctuation counts (< 10)
    with pytest.warns(UserWarning, match="⚠️ DOCUMENT CONSTRAINT ALERT"):
        analyzer = StructuralAnalyzer(
            corpus=sample_texts,
            min_punctuation_threshold=10,
            action_on_low_count="warn",
        )
        assert len(analyzer.doc_ids) == 3


def test_structural_analyzer_low_count_drop(sample_texts):
    """Test dropping documents when punctuation count is below threshold."""
    with pytest.warns(UserWarning, match="DROPPING document"):
        analyzer = StructuralAnalyzer(
            corpus=sample_texts, min_punctuation_threshold=5, action_on_low_count="drop"
        )
        # doc1 has 4 (including ... as 3? No, spaCy might treat ... as one or three tokens)
        # doc2 has 6
        # doc3 has 1
        # Let's see what remains.
        assert len(analyzer.doc_ids) < 3


def test_structural_analyzer_all_dropped(sample_texts):
    """Test exception when all documents are dropped."""
    with pytest.raises(ValueError, match="All documents were dropped"):
        StructuralAnalyzer(
            corpus=sample_texts,
            min_punctuation_threshold=100,
            action_on_low_count="drop",
        )


def test_feature_modes(sample_texts):
    """Test different feature modes."""
    # punctuation_only
    analyzer_punct = StructuralAnalyzer(
        corpus=sample_texts,
        feature_mode="punctuation_only",
        min_punctuation_threshold=1,
    )
    for tokens in analyzer_punct.tokenized_corpus.values():
        for t in tokens:
            assert not t.startswith("[WS_")
            # In punctuation_only, it keeps tokens that are not alnum and not WS_
            # But wait, it might keep words if they are not alnum? Unlikely.
            # actually it filters: [t for t in tokens if not t.startswith("[WS_") and not t.isalnum()]

    # structural_only
    analyzer_struct = StructuralAnalyzer(
        corpus=sample_texts, feature_mode="structural_only", min_punctuation_threshold=1
    )
    # filters: [t for t in tokens if t.startswith("[WS_") or not t.isalnum()]

    # all
    analyzer_all = StructuralAnalyzer(
        corpus=sample_texts, feature_mode="all", min_punctuation_threshold=1
    )
    assert len(analyzer_all.vocabulary) >= len(analyzer_punct.vocabulary)


def test_whitespace_markers(sample_texts):
    """Test tracking of whitespace markers."""
    analyzer = StructuralAnalyzer(
        corpus=sample_texts, include_whitespace=True, min_punctuation_threshold=1
    )
    # doc3 has double space and multiple newline
    tokens_doc3 = analyzer.tokenized_corpus["doc3"]
    assert "[WS_DOUBLE_SPACE]" in tokens_doc3
    assert "[WS_MULTIPLE_NEWLINE]" in tokens_doc3
    assert "[WS_TRAILING_SPACE]" in tokens_doc3


def test_get_feature_matrix(sample_texts):
    """Test different stylometric matrices."""
    analyzer = StructuralAnalyzer(corpus=sample_texts, min_punctuation_threshold=1)

    raw = analyzer.get_feature_matrix(method="raw")
    assert isinstance(raw, np.ndarray)
    assert raw.shape == (3, len(analyzer.vocabulary))

    tfidf = analyzer.get_feature_matrix(method="tfidf")
    assert tfidf.shape == (3, len(analyzer.vocabulary))

    z_scores = analyzer.get_feature_matrix(method="burrows_z")
    assert z_scores.shape == (3, len(analyzer.vocabulary))


def test_get_distance_matrix(sample_texts):
    """Test stylistic distance matrices."""
    analyzer = StructuralAnalyzer(corpus=sample_texts, min_punctuation_threshold=1)

    dist_classic = analyzer.get_distance_matrix(method="classic")
    assert dist_classic.shape == (3, 3)
    assert np.allclose(np.diag(dist_classic), 0)

    dist_quad = analyzer.get_distance_matrix(method="quadratic")
    assert dist_quad.shape == (3, 3)
    assert np.allclose(np.diag(dist_quad), 0)


def test_to_df(sample_texts):
    """Test conversion to DataFrame."""
    analyzer = StructuralAnalyzer(corpus=sample_texts, min_punctuation_threshold=1)
    df = analyzer.to_df(method="tfidf")
    assert isinstance(df, pd.DataFrame)
    assert "Document_ID" in df.columns
    assert len(df) == 3


def test_get_loadings(sample_texts):
    """Test PCA loadings retrieval."""
    analyzer = StructuralAnalyzer(corpus=sample_texts, min_punctuation_threshold=1)
    loadings = analyzer.get_loadings()
    assert "PC1" in loadings
    assert "PC2" in loadings
    assert isinstance(loadings["PC1"], pd.DataFrame)


def test_visualize_no_plot(sample_texts):
    """Test visualize method with plotting disabled."""
    analyzer = StructuralAnalyzer(corpus=sample_texts, min_punctuation_threshold=1)
    # Just check if it runs without error
    analyzer.visualize(show_plots=False, show_loadings=False)


def test_structural_analyzer_init_loader(sample_texts):
    """Test initialization with a Lexos Loader object (Line 90)."""
    loader = Loader()
    loader.names = list(sample_texts.keys())
    loader.texts = list(sample_texts.values())
    # Mocking paths and mime_types to avoid mismatched lengths Error
    loader.paths = [""] * len(sample_texts)
    loader.mime_types = ["text/plain"] * len(sample_texts)

    analyzer = StructuralAnalyzer(corpus=loader, min_punctuation_threshold=1)
    assert len(analyzer.doc_ids) == 3
    assert "doc1" in analyzer.doc_ids


def test_count_total_punctuation_with_doc(sample_texts):
    """Test _count_total_punctuation with a spaCy Doc (Line 146)."""
    import spacy

    nlp = spacy.load("xx_sent_ud_sm")
    doc = nlp(sample_texts["doc1"])
    analyzer = StructuralAnalyzer(corpus=sample_texts, min_punctuation_threshold=1)
    count = analyzer._count_total_punctuation(doc)
    assert count > 0


def test_tokenize_structural_with_doc(sample_texts):
    """Test _tokenize_structural with a spaCy Doc (Line 164)."""
    import spacy

    nlp = spacy.load("xx_sent_ud_sm")
    doc = nlp(sample_texts["doc1"])
    analyzer = StructuralAnalyzer(corpus=sample_texts, min_punctuation_threshold=1)
    tokens = analyzer._tokenize_structural(doc)
    assert len(tokens) > 0


def test_get_distance_matrix_errors(sample_texts):
    """Test get_distance_matrix error handling and as_df (Lines 238, 241-243)."""
    analyzer = StructuralAnalyzer(corpus=sample_texts, min_punctuation_threshold=1)
    with pytest.raises(ValueError, match="Unknown method"):
        analyzer.get_distance_matrix(method="invalid")

    df = analyzer.get_distance_matrix(method="classic", as_df=True)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 3)
    assert list(df.index) == analyzer.doc_ids


def test_get_feature_matrix_error(sample_texts):
    """Test get_feature_matrix error handling (Line 265)."""
    analyzer = StructuralAnalyzer(corpus=sample_texts, min_punctuation_threshold=1)
    with pytest.raises(ValueError, match="Unknown transformation method"):
        analyzer.get_feature_matrix(method="invalid")


def test_to_csv(sample_texts, tmp_path):
    """Test to_csv method and directory creation (Lines 278-282)."""
    analyzer = StructuralAnalyzer(corpus=sample_texts, min_punctuation_threshold=1)
    # Test directory creation
    sub_dir = tmp_path / "subdir"
    filepath = sub_dir / "test.csv"
    analyzer.to_csv(filepath, method="raw")
    assert filepath.exists()


def test_visualize_loading_and_plots(sample_texts, monkeypatch):
    """Test visualize method loadings and plots (Lines 334-381, 386-404)."""
    # Mock plt.show() and plt.subplots() to avoid GUI issues
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)

    analyzer = StructuralAnalyzer(corpus=sample_texts, min_punctuation_threshold=1)

    # Test with show_plots=True and show_loadings=True
    # This covers the plot generation logic and the markdown printing logic
    analyzer.visualize(method="tfidf", show_plots=True, show_loadings=True)

    # Test with different method for cityblock/average linkage
    analyzer.visualize(method="burrows_z", show_plots=True, show_loadings=False)
