"""test_structural_stylometry.py.

Coverage: 100%

Last Updated: September 4, 2026
"""

import importlib

import numpy as np
import pandas as pd
import pytest
import spacy


@pytest.fixture
def structural_module(monkeypatch):
    """Import the structural stylometry module with a lightweight fake spaCy model."""

    class FakeToken:
        def __init__(self, text, pos_="X", is_punct=False, is_space=False):
            self.text = text
            self.pos_ = pos_
            self.is_punct = is_punct
            self.is_space = is_space
            self.lower_ = text.lower()

    class FakeDoc:
        def __init__(self, text):
            self.text = text
            tokens = []
            for token in text.split():
                if token in {",", ".", "!", "?"}:
                    tokens.append(FakeToken(token, pos_="PUNCT", is_punct=True))
                elif token in {"\n", "\t"}:
                    tokens.append(FakeToken(token, is_space=True))
                else:
                    tokens.append(FakeToken(token))
            self._tokens = tokens

        def __iter__(self):
            return iter(self._tokens)

    class FakeNLP:
        def __call__(self, text):
            return FakeDoc(text)

    monkeypatch.setattr(spacy, "load", lambda model: FakeNLP())

    module = importlib.import_module(
        "lexos.structural_stylometry.structural_stylometry"
    )
    return importlib.reload(module)


def test_structural_analyzer_initializes_and_builds_matrix(structural_module):
    """The analyzer should initialize from a simple corpus and build a feature matrix."""
    analyzer = structural_module.StructuralAnalyzer(
        corpus={"doc1": "Hello, world!", "doc2": "Hello world?"},
        model="xx_sent_ud_sm",
    )

    assert analyzer.doc_ids == ["doc1", "doc2"]
    assert analyzer.raw_matrix.shape[0] == 2
    assert analyzer.raw_matrix.shape[1] == len(analyzer.vocabulary)
    assert analyzer.raw_matrix.shape[1] > 0


def test_structural_analyzer_returns_feature_transformations(structural_module):
    """The analyzer should expose raw, tf-idf, and Burrows-Z matrices."""
    analyzer = structural_module.StructuralAnalyzer(
        corpus={"doc1": "Hello, world!", "doc2": "Hello world?"},
        model="xx_sent_ud_sm",
    )

    raw = analyzer.get_feature_matrix("raw")
    tfidf = analyzer.get_feature_matrix("tfidf")
    burrows_z = analyzer.get_feature_matrix("burrows_z")

    assert raw.shape == tfidf.shape == burrows_z.shape
    assert np.isfinite(burrows_z).all()
    assert np.any(tfidf > 0)


def test_structural_analyzer_distance_matrix_and_export(structural_module, tmp_path):
    """The analyzer should compute distances and export tabular data."""
    analyzer = structural_module.StructuralAnalyzer(
        corpus={"doc1": "Hello, world!", "doc2": "Hello world?"},
        model="xx_sent_ud_sm",
    )

    classic = analyzer.get_distance_matrix(method="classic")
    quadratic = analyzer.get_distance_matrix(method="quadratic")
    df = analyzer.to_df(method="raw")

    assert classic.shape == (2, 2)
    assert quadratic.shape == (2, 2)
    assert np.allclose(classic.diagonal(), 0)
    assert np.allclose(quadratic.diagonal(), 0)
    assert "Document_ID" in df.columns
    assert df.shape[0] == 2

    csv_path = tmp_path / "stylometry.csv"
    analyzer.to_csv(csv_path, method="raw")

    assert csv_path.exists()
    saved_df = pd.read_csv(csv_path)
    assert saved_df.shape[0] == 2


def test_structural_analyzer_visualize(monkeypatch, structural_module):
    """Visualize should run without error when plots are enabled."""
    analyzer = structural_module.StructuralAnalyzer(
        corpus={"doc1": "Hello, world!", "doc2": "Hello world?"},
        model="xx_sent_ud_sm",
    )

    monkeypatch.setattr(structural_module.plt, "show", lambda: None)

    analyzer.visualize(method="raw", show_plots=True, show_loadings=False)


def test_structural_analyzer_handles_drop_and_value_error(structural_module):
    """Low punctuation documents should be dropped or trigger a ValueError when all documents are dropped."""
    with pytest.warns(UserWarning, match="DROPPING document from corpus"):
        with pytest.raises(ValueError, match="All documents were dropped"):
            structural_module.StructuralAnalyzer(
                corpus={"doc1": "Hello world"},
                model="xx_sent_ud_sm",
                min_punctuation_threshold=10,
                action_on_low_count="drop",
            )


def test_structural_analyzer_accepts_loader_and_corpus_inputs(
    monkeypatch, structural_module, tmp_path
):
    """The analyzer should normalize Loader and Corpus inputs into plain dictionaries."""
    from lexos.corpus import Corpus
    from lexos.io.loader import Loader

    loader = Loader()
    loader.names = ["doc1"]
    loader.texts = ["Hello, world!"]

    loader_analyzer = structural_module.StructuralAnalyzer(
        corpus=loader,
        model="xx_sent_ud_sm",
        min_punctuation_threshold=0,
    )
    assert loader_analyzer.doc_ids == ["doc1"]

    corpus = Corpus(corpus_dir=str(tmp_path))
    corpus.records = {
        "doc1": type("Record", (), {"name": "doc1", "content": "Hello, world!"})()
    }

    corpus_analyzer = structural_module.StructuralAnalyzer(
        corpus=corpus,
        model="xx_sent_ud_sm",
        min_punctuation_threshold=0,
    )
    assert corpus_analyzer.doc_ids == ["doc1"]


def test_structural_analyzer_handles_doc_inputs_and_feature_modes(
    monkeypatch, structural_module
):
    """The analyzer should support direct spaCy-style document objects and both feature modes."""

    class DummyToken:
        def __init__(self, text, pos_="X", is_punct=False, is_space=False):
            self.text = text
            self.pos_ = pos_
            self.is_punct = is_punct
            self.is_space = is_space

    class DummyDocType:
        def __init__(self):
            self._tokens = [
                DummyToken("Hello"),
                DummyToken(",", pos_="PUNCT", is_punct=True),
                DummyToken(" ", is_space=True),
                DummyToken("world"),
                DummyToken("!", pos_="PUNCT", is_punct=True),
            ]

        def __iter__(self):
            return iter(self._tokens)

    monkeypatch.setattr(structural_module, "Doc", DummyDocType)
    doc = DummyDocType()

    analyzer = structural_module.StructuralAnalyzer.model_construct(
        model="xx_sent_ud_sm",
        include_whitespace=False,
        feature_mode="all",
        min_punctuation_threshold=0,
    )

    assert analyzer._count_total_punctuation(doc) == 2

    analyzer.feature_mode = "punctuation_only"
    assert analyzer._tokenize_structural(doc) == [",", "!"]

    analyzer.feature_mode = "structural_only"
    assert analyzer._tokenize_structural(doc) == [",", "!"]


def test_structural_analyzer_handles_error_branches_and_visualization(
    monkeypatch, structural_module, tmp_path
):
    """The analyzer should raise for unsupported methods and support CSV/visualization output paths."""
    analyzer = structural_module.StructuralAnalyzer(
        corpus={"doc1": "Hello, world!", "doc2": "Hello world?"},
        model="xx_sent_ud_sm",
        min_punctuation_threshold=0,
    )

    with pytest.raises(ValueError, match="Unknown method"):
        analyzer.get_distance_matrix(method="unknown")

    with pytest.raises(ValueError, match="Unknown transformation method"):
        analyzer.get_feature_matrix(method="unknown")

    distance_df = analyzer.get_distance_matrix(method="classic", as_df=True)
    assert isinstance(distance_df, pd.DataFrame)
    assert distance_df.shape == (2, 2)

    csv_path = tmp_path / "nested" / "stylometry.csv"
    analyzer.to_csv(csv_path, method="raw")
    assert csv_path.exists()

    monkeypatch.setattr(structural_module.plt, "show", lambda: None)
    monkeypatch.setattr(
        pd.DataFrame, "to_markdown", lambda self, *args, **kwargs: "table"
    )

    analyzer.visualize(method="burrows_z", top_n=1, show_plots=True, show_loadings=True)

    loadings = analyzer.get_loadings(method="raw")
    assert set(loadings) == {"PC1", "PC2"}
