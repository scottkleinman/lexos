"""Tests for the PyRMallet backend.

Coverage: 99%. Missing: 226, 303, 494

Last Updated: September 8, 2026
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lexos.exceptions import LexosException
from lexos.topic_modeling.mallet import Mallet, PyRMallet


class FakeLDA:
    """Minimal stand-in for the pyrmallet LatentDirichletAllocation API."""

    def __init__(self, *args, **kwargs):
        """Initialize the fake LDA model with predefined components and distributions."""
        self.components_ = np.array(
            [
                [0.9, 0.1, 0.0],
                [0.2, 0.7, 0.1],
            ],
            dtype=float,
        )
        self.doc_topic_distributions_ = np.array(
            [
                [0.8, 0.2],
                [0.3, 0.7],
            ],
            dtype=float,
        )
        self.feature_names_in_ = np.array(["alpha", "beta", "gamma"], dtype=object)
        self.n_features_in_ = 3

    def fit(self, docs):
        self.docs_ = docs
        return self

    def transform(self, docs):
        """Transform the given documents into topic distributions."""
        return np.array(
            [
                [0.9, 0.1],
                [0.4, 0.6],
            ],
            dtype=float,
        )


@pytest.fixture
def tmp_model_dir(tmp_path):
    """Create and return a temporary model directory for tests."""
    d = tmp_path / "pyrmallet_model"
    d.mkdir()
    return d


def test_pyrmallet_train_sets_canonical_metadata(tmp_model_dir, monkeypatch):
    """PyRMallet should persist the same canonical metadata keys as Mallet."""
    import lexos.topic_modeling.mallet.pyrmallet as pyrmallet_mod

    monkeypatch.setattr(
        pyrmallet_mod,
        "LatentDirichletAllocation",
        FakeLDA,
    )

    model = PyRMallet(model_dir=str(tmp_model_dir))
    model.import_data(["alpha beta gamma", "beta gamma"])
    model.train(num_topics=2, num_iterations=5, verbose=False)

    assert model.CANONICAL_DOC_TOPIC_KEY in model.metadata
    assert model.CANONICAL_TOPIC_KEYS_KEY in model.metadata
    assert model.CANONICAL_TERM_WEIGHTS_KEY in model.metadata
    assert model.CANONICAL_INFERENCER_KEY in model.metadata
    assert Path(model.metadata[model.CANONICAL_DOC_TOPIC_KEY]).exists()
    assert Path(model.metadata[model.CANONICAL_TOPIC_KEYS_KEY]).exists()
    assert Path(model.metadata[model.CANONICAL_TERM_WEIGHTS_KEY]).exists()


def test_pyrmallet_infer_returns_distributions(tmp_model_dir, monkeypatch):
    """Inference should return the same list-of-lists distribution contract as Mallet."""
    import lexos.topic_modeling.mallet.pyrmallet as pyrmallet_mod

    monkeypatch.setattr(
        pyrmallet_mod,
        "LatentDirichletAllocation",
        FakeLDA,
    )

    model = PyRMallet(model_dir=str(tmp_model_dir))
    model.import_data(["alpha beta gamma", "beta gamma"])
    model.train(num_topics=2, num_iterations=5, verbose=False)

    distributions = model.infer(docs=["alpha beta", "gamma"], show=False)

    assert isinstance(distributions, list)
    assert len(distributions) == 2
    assert all(len(doc) == 2 for doc in distributions)
    assert all(isinstance(value, float) for doc in distributions for value in doc)


def test_pyrmallet_maps_compatibility_defaults(tmp_model_dir, monkeypatch):
    """The adapter should preserve the meaningful PyRMallet defaults used for compatibility."""
    import lexos.topic_modeling.mallet.pyrmallet as pyrmallet_mod

    captured = {}

    class RecordingLDA(FakeLDA):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(
        pyrmallet_mod,
        "LatentDirichletAllocation",
        RecordingLDA,
    )

    model = PyRMallet(model_dir=str(tmp_model_dir))
    model.import_data(["alpha beta gamma", "beta gamma"], remove_stopwords=True)
    model.train(num_topics=2, num_iterations=5, optimize_interval=10, verbose=False)

    assert "stopwords" in captured
    assert captured["stopwords"] is not None
    assert captured["min_doc_freq"] == 1
    assert captured["max_doc_fraction"] == 1.0
    assert captured["optimize_interval"] == 10

    model.infer(docs=["alpha beta", "gamma"], show=False)
    assert captured["n_inference_iter"] == 50


def test_pyrmallet_doc_topic_file_matches_model_topic_count(tmp_model_dir, monkeypatch):
    """Doc-topic output must remain one dense vector per document to satisfy MALLET consumers."""
    import lexos.topic_modeling.mallet.pyrmallet as pyrmallet_mod

    monkeypatch.setattr(
        pyrmallet_mod,
        "LatentDirichletAllocation",
        FakeLDA,
    )

    model = PyRMallet(model_dir=str(tmp_model_dir))
    model.import_data(["alpha beta gamma", "beta gamma", "gamma alpha", "beta alpha"])
    model.train(num_topics=2, num_iterations=5, verbose=False)

    assert len(model.distributions) == 2
    assert all(len(dist) == 2 for dist in model.distributions)
    assert model.get_top_docs(topic=0, n=2).shape[0] == 2


def test_mallet_backend_factory_selects_pyrmallet(tmp_model_dir, monkeypatch):
    """The factory should route to the PyRMallet backend when requested."""
    import lexos.topic_modeling.mallet.pyrmallet as pyrmallet_mod

    monkeypatch.setattr(
        pyrmallet_mod,
        "LatentDirichletAllocation",
        FakeLDA,
    )

    model = Mallet(model_dir=str(tmp_model_dir), backend="pyrmallet")

    assert isinstance(model, PyRMallet)
    assert model.backend == "pyrmallet"

    default_model = Mallet(model_dir=str(tmp_model_dir), backend="java")
    assert isinstance(default_model, Mallet)
    assert default_model.backend == "java"


def test_pyrmallet_validation_and_empty_file_output(tmp_model_dir, monkeypatch):
    """Exercise backend validation and the empty doc-topic output path."""
    import lexos.topic_modeling.mallet.pyrmallet as pyrmallet_mod

    monkeypatch.setattr(pyrmallet_mod, "LatentDirichletAllocation", None)
    with pytest.raises(Exception):
        PyRMallet(model_dir=str(tmp_model_dir))

    monkeypatch.setattr(pyrmallet_mod, "LatentDirichletAllocation", FakeLDA)
    model = PyRMallet(model_dir=str(tmp_model_dir))
    empty_path = tmp_model_dir / "empty_doc_topic.txt"
    model._write_doc_topic_file(str(empty_path), [])
    assert empty_path.read_text() == "#doc\tlabel\n"


def test_pyrmallet_stopword_and_fit_fallback_branches(tmp_model_dir, monkeypatch):
    """Exercise the stopword fallback and verbose training helper branches."""
    import lexos.topic_modeling.mallet.pyrmallet as pyrmallet_mod

    monkeypatch.setattr(pyrmallet_mod, "LatentDirichletAllocation", FakeLDA)
    model = PyRMallet(model_dir=str(tmp_model_dir))

    assert model._compatibility_stopwords(False) is None
    with patch.dict(
        sys.modules,
        {
            "sklearn": None,
            "sklearn.feature_extraction": None,
            "sklearn.feature_extraction.text": None,
        },
    ):
        assert model._compatibility_stopwords(True) == []

    class FakeTqdm:
        def __init__(self, *args, **kwargs):
            self.n = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def set_description(self, value):
            self.description = value

        def refresh(self):
            pass

    monkeypatch.setattr(pyrmallet_mod, "tqdm", FakeTqdm)
    fitted = model._fit_lda(FakeLDA(), ["alpha beta", "gamma delta"], 5, True)
    assert fitted is not None


def test_pyrmallet_model_array_normalization_and_padding(tmp_model_dir):
    """Exercise the feature-name fallback and array resizing/padding branches."""
    model = PyRMallet(model_dir=str(tmp_model_dir))

    class SparseArrayLDA:
        def __init__(self):
            self.components_ = np.ones((1, 2), dtype=float)

    doc_topic, topic_word, feature_names = model._normalize_model_arrays(
        SparseArrayLDA(),
        ["alpha beta", "gamma delta"],
        2,
    )

    assert len(doc_topic) == 2
    assert topic_word.shape == (2, 4)
    assert feature_names == ["alpha", "beta", "delta", "gamma"]


def test_pyrmallet_training_inference_and_metadata_validation_branches(
    tmp_model_dir, monkeypatch
):
    """Cover train/infer validation, invalid input handling, and metadata persistence."""
    import lexos.topic_modeling.mallet.pyrmallet as pyrmallet_mod

    monkeypatch.setattr(pyrmallet_mod, "LatentDirichletAllocation", FakeLDA)
    monkeypatch.setattr(pyrmallet_mod.msg, "good", MagicMock())

    model = PyRMallet(model_dir=str(tmp_model_dir))
    model.import_data(["alpha beta gamma", "beta gamma"])
    model.train(num_topics=2, num_iterations=5, verbose=True)
    assert pyrmallet_mod.msg.good.called

    assert model.infer(docs=["alpha beta"], show=True) is None

    training_file = tmp_model_dir / "docs.txt"
    training_file.write_text("alpha beta\n\ngamma\n", encoding="utf-8")
    assert model._convert_docs_to_training_list(str(training_file)) == [
        "alpha beta",
        "gamma",
    ]

    with pytest.raises(LexosException):
        model._convert_docs_to_training_list(True)
    with pytest.raises(LexosException):
        model.import_data(True)
    with pytest.raises(LexosException):
        model.import_data(["alpha", 1])

    bare = PyRMallet()
    with pytest.raises(LexosException):
        bare.train()

    untrained = PyRMallet(model_dir=str(tmp_model_dir))
    untrained.metadata.pop("path_to_training_data", None)
    with pytest.raises(LexosException):
        untrained.train()

    fresh_inference_model = PyRMallet(model_dir=str(tmp_model_dir))
    fresh_inference_model.metadata.clear()
    with pytest.raises(LexosException):
        fresh_inference_model.infer(["alpha beta"])

    model.set_metadata("flag", "value")
    meta_path = tmp_model_dir / "meta.json"
    assert meta_path.exists()
    assert json.loads(meta_path.read_text())["flag"] == "value"
