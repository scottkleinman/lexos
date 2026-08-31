"""test_classifier.py.

Coverage: 100%

Last Updated: August 30, 2026
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import spacy
from pydantic import ValidationError

from lexos.classification import (
    BaseClassificationPipeline,
    Classifier,
    ClassifierData,
    SklearnClassifierPipeline,
)
from lexos.classification.classifier import _is_dtm_like, _is_spacy_doc


class _DummyPipeline(BaseClassificationPipeline):
    """Minimal concrete pipeline used only to hit abstract API wrappers."""

    def fit(self, data, labels):
        return self

    def predict(self, data):
        return ["a", "b"]

    def predict_scores(self, data):
        return [{"a": 0.9, "b": 0.1}]

    def evaluate(self, data, labels):
        return {"accuracy": 1.0}

    @property
    def model(self):
        return object()


def test_classifier_data_alignment_and_document_helpers() -> None:
    """Exercise row-alignment helpers and normalized document conversion."""
    nlp = spacy.blank("en")
    doc = nlp.make_doc("alpha beta gamma")

    assert _is_dtm_like(None) is False
    assert _is_spacy_doc(doc) is True
    assert _is_spacy_doc(object()) is False

    class FakeDTM:
        def __init__(self):
            self.doc_term_matrix = np.array([[1, 0], [0, 1]])
            self.vectorizer = object()
            self.labels = ["A", "B"]
            self.docs = [doc, doc]
            self.titles = ["t-1", "t-2"]

    prepared = ClassifierData.from_input(FakeDTM())
    assert prepared.row_count() == 2
    assert prepared.as_texts() == ["alpha beta gamma", "alpha beta gamma"]

    frame = pd.DataFrame({"text": ["u", "v"], "label": ["A", "B"], "title": ["x", "y"]})
    frame_data = ClassifierData.from_input(frame)
    assert frame_data.labels == ["A", "B"]
    assert frame_data.titles == ["x", "y"]

    matrix_data = ClassifierData.from_input(
        np.array([[1, 2], [3, 4]]), labels=["A", "B"]
    )
    assert matrix_data.matrix is not None
    assert matrix_data.row_count() == 2

    with pytest.raises(ValueError):
        ClassifierData._validate_label_count(2, 3, "bad label count")

    with pytest.raises(ValueError):
        ClassifierData.from_input(None)

    subset = ClassifierData.from_input(
        ["one", "two", "three"], labels=["A", "B", "C"]
    ).subset([0, 2])
    assert subset.labels == ["A", "C"]
    assert subset.values == ["one", "three"]

    assert ClassifierData._test_count_for_group(2, 0.9) == 1
    assert ClassifierData._dev_count_for_group(2, 0.8) == 1
    assert ClassifierData._split_random_indices(1, 0.5, 0) == {"train": [0], "test": []}


def test_classifier_frontend_methods_and_split_metadata() -> None:
    """Cover the public Classifier facade wrappers and split metadata preservation."""
    with pytest.raises(ValidationError):
        Classifier(pipeline=object())

    classifier = Classifier(
        data=["text-1", "text-2", "text-3", "text-4"],
        labels=["A", "B", "A", "B"],
        titles=["t-1", "t-2", "t-3", "t-4"],
        pipeline=_DummyPipeline(),
    )

    resolved_data, resolved_labels = classifier._resolve_data_and_labels(
        data=["x", "y"], labels=["C", "D"], titles=["u", "v"]
    )
    assert resolved_labels == ["C", "D"]
    assert resolved_data.titles == ["u", "v"]

    with pytest.raises(ValueError):
        Classifier(pipeline=None).fit()

    assert classifier.predict(["x"]) == ["a", "b"]
    assert classifier.predict_scores(["x"]) == [{"a": 0.9, "b": 0.1}]
    assert classifier.__call__(["x"]) == ["a", "b"]

    split = classifier.split_data(
        test_size=0.5, random_state=0, titles=["u", "v", "w", "z"]
    )
    assert set(split["train"]["titles"]) | set(split["test"]["titles"]) == {
        "u",
        "v",
        "w",
        "z",
    }
    assert classifier.train_test_split(test_size=0.5, random_state=0)["train"]["labels"]
    assert classifier.train_dev_split(dev_size=0.5, random_state=0)["train"]["labels"]


def test_base_pipeline_contract_and_abstract_wrapper() -> None:
    """Exercise the abstract pipeline contract and __call__ wrapper."""
    pipeline = _DummyPipeline()
    assert pipeline.model is not None
    assert pipeline.__call__(["x"]) == ["a", "b"]


def test_classifier_split_data_keeps_labels_and_titles_aligned() -> None:
    """Split output should preserve both metadata and label alignment."""
    texts = ["history text", "finance text", "arts text", "politics text"]
    labels = ["history", "finance", "arts", "politics"]
    titles = ["title-1", "title-2", "title-3", "title-4"]

    classifier = Classifier(
        data=texts,
        labels=labels,
        titles=titles,
        pipeline=SklearnClassifierPipeline(),
    )

    split = classifier.split_data(test_size=0.5, random_state=7)

    assert set(split["train"]["labels"]) | set(split["test"]["labels"]) == set(labels)
    assert set(split["train"]["titles"]) | set(split["test"]["titles"]) == set(titles)
    assert len(split["train"]["titles"]) + len(split["test"]["titles"]) == len(titles)


def test_classifier_predict_scores_are_sorted_by_score_desc() -> None:
    """Scores should be ordered from strongest to weakest for each prediction row."""
    texts = [
        "history of literature",
        "market report and finance",
        "poetry and essays",
        "government policy and elections",
    ]
    labels = ["history", "finance", "arts", "politics"]

    classifier = Classifier(
        data=texts, labels=labels, pipeline=SklearnClassifierPipeline()
    )
    classifier.fit()
    scores = classifier.predict_scores(texts)

    assert len(scores) == len(texts)
    assert all(isinstance(row, dict) for row in scores)
    assert all(
        list(row.keys()) == sorted(row, key=row.get, reverse=True)
        for row in scores
        if row
    )


def test_classifier_data_from_dtm_and_split_preserves_labels() -> None:
    """ClassifierData should normalize DTM input and split it cleanly."""
    from lexos.dtm import DTM

    docs = [
        ["alpha", "beta"],
        ["alpha"],
        ["beta"],
        ["gamma"],
        ["alpha", "gamma"],
        ["beta", "gamma"],
        ["alpha", "beta"],
        ["gamma"],
        ["beta"],
        ["alpha"],
    ]
    labels = ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"]

    dtm = DTM(docs=docs, labels=labels)
    dtm.__call__(docs=docs, labels=labels)

    prepared = ClassifierData.from_input(dtm)
    split = prepared.split(test_size=0.5, stratify=True, random_state=42)

    assert prepared.row_count() == len(docs)
    assert len(split["train"].labels) + len(split["test"].labels) == len(docs)
    assert set(split["train"].labels) | set(split["test"].labels) == set(labels)


def test_classifier_uses_dtm_labels_when_classifier_labels_are_not_supplied() -> None:
    """A DTM should provide its labels when the classifier instance has no explicit labels."""
    docs = [
        ["novel", "history", "literature"],
        ["literature", "poetry", "novel"],
        ["market", "stocks", "earnings"],
        ["market", "investors", "report"],
        ["cathedral", "medieval", "history"],
        ["architecture", "medieval", "cathedral"],
    ]
    labels = ["literature", "literature", "finance", "finance", "history", "history"]

    from lexos.dtm import DTM

    dtm = DTM(docs=docs, labels=labels)
    dtm.__call__(docs=docs, labels=labels)

    classifier = Classifier(data=dtm, pipeline=SklearnClassifierPipeline())
    split = classifier.train_test_split(test_size=0.33, random_state=42)

    assert len(split["train"]["labels"]) + len(split["test"]["labels"]) == len(labels)
    assert set(split["train"]["labels"]) | set(split["test"]["labels"]) == set(labels)


def test_classifier_data_split_preserves_titles_and_returns_them_in_split_data() -> (
    None
):
    """Title metadata should survive row selection and appear in split output."""
    texts = ["alpha", "beta", "gamma", "delta"]
    labels = ["A", "B", "A", "B"]
    titles = ["title-1", "title-2", "title-3", "title-4"]

    prepared = ClassifierData.from_input(texts, labels=labels, titles=titles)
    split = prepared.split(test_size=0.5, stratify=True, random_state=42)

    assert split["train"].titles is not None
    assert split["test"].titles is not None
    assert len(split["train"].titles) + len(split["test"].titles) == len(titles)
    assert set(split["train"].titles) | set(split["test"].titles) == set(titles)

    classifier = Classifier(
        data=texts, labels=labels, pipeline=SklearnClassifierPipeline()
    )
    split_result = classifier.split_data(
        data=texts,
        labels=labels,
        titles=titles,
        test_size=0.5,
        random_state=42,
    )

    assert split_result["train"]["titles"] is not None
    assert split_result["test"]["titles"] is not None
    assert set(split_result["train"]["titles"]) | set(
        split_result["test"]["titles"]
    ) == set(titles)


def test_classifier_fit_ignores_stale_titles_when_training_on_a_split_subset() -> None:
    """A split subset should not inherit a stale full-dataset title list from the classifier."""
    texts = ["alpha", "beta", "gamma", "delta"]
    labels = ["A", "B", "A", "B"]
    titles = ["title-1", "title-2", "title-3", "title-4"]

    classifier = Classifier(
        data=texts,
        labels=labels,
        titles=titles,
        pipeline=SklearnClassifierPipeline(),
    )
    split = classifier.train_test_split(test_size=0.5, random_state=42)

    classifier.fit(
        data=split["train"]["data"],
        labels=split["train"]["labels"],
    )

    assert classifier.labels == split["train"]["labels"]
    assert classifier.titles == [] or len(classifier.titles) == len(
        split["train"]["data"]
    )


def test_stratified_split_reserves_test_rows_for_small_classes() -> None:
    """Small classes should still contribute at least one item to the test split."""
    texts = [f"text-{i}" for i in range(6)]
    labels = ["A", "A", "B", "B", "C", "C"]

    prepared = ClassifierData.from_input(texts, labels)
    split = prepared.split(test_size=0.2, stratify=True, random_state=42)

    assert len(split["train"].labels) + len(split["test"].labels) == len(labels)
    assert len(split["test"].labels) > 0


def test_classifier_fit_accepts_matrix_like_split_output() -> None:
    """Classifier.fit should accept the sparse matrix returned by split_data for DTM input."""
    from lexos.dtm import DTM

    docs = [
        ["novel", "history", "literature"],
        ["literature", "poetry", "novel"],
        ["market", "stocks", "earnings"],
        ["market", "investors", "report"],
        ["cathedral", "medieval", "history"],
        ["architecture", "medieval", "cathedral"],
    ]
    labels = ["literature", "literature", "finance", "finance", "history", "history"]

    dtm = DTM(docs=docs, labels=labels)
    dtm.__call__(docs=docs, labels=labels)

    classifier = Classifier(data=dtm, pipeline=SklearnClassifierPipeline())
    split = classifier.train_test_split(test_size=0.33, random_state=42)

    fitted = classifier.fit(
        data=split["train"]["data"],
        labels=split["train"]["labels"],
    )
    predictions = fitted.predict(split["test"]["data"])

    assert len(predictions) == len(split["test"]["labels"])


def test_classifier_data_and_split_edge_cases() -> None:
    """Exercise the remaining row-alignment and split-validation branches."""
    nlp = spacy.blank("en")
    doc = nlp.make_doc("alpha beta gamma")

    fake_dtm = type(
        "FakeDTM",
        (),
        {
            "doc_term_matrix": np.array([[1, 0], [0, 1]]),
            "vectorizer": object(),
            "labels": ["A", "B"],
            "docs": [doc, doc],
            "titles": ["t-1", "t-2"],
        },
    )()
    prepared = ClassifierData.from_input(fake_dtm)
    assert prepared.as_texts() == ["alpha beta gamma", "alpha beta gamma"]
    assert prepared.subset([1]).titles == ["t-2"]

    frame = pd.DataFrame({"text": ["u", "v"], "label": ["A", "B"]})
    frame_data = ClassifierData.from_input(frame, titles=["x", "y"])
    assert frame_data.titles == ["x", "y"]

    with pytest.raises(ValueError):
        ClassifierData._resolve_row_alignment(
            2,
            ["A"],
            ["x", "y"],
            "bad labels",
            "bad titles",
        )

    with pytest.raises(ValueError):
        ClassifierData.from_input([], labels=["A", "B"])

    assert ClassifierData._as_text_for_doc(None) == ""
    assert (
        ClassifierData._as_text_for_doc(type("TextDoc", (), {"text": "hello"})())
        == "hello"
    )
    assert ClassifierData._as_text_for_doc(["one", "two"]) == "one two"

    split = prepared.split(test_size=0.5, dev_size=0.25, random_state=0)
    assert set(split["train"].labels) | set(split["test"].labels) | set(
        split["dev"].labels
    ) == {"A", "B"}

    with pytest.raises(ValueError):
        ClassifierData._validate_split_parameters(0, 0.5, None)
    with pytest.raises(ValueError):
        ClassifierData._validate_split_parameters(2, 0.0, None)
    with pytest.raises(ValueError):
        ClassifierData._validate_split_parameters(2, 0.5, 0.0)


def test_classifier_predict_and_evaluate_error_paths() -> None:
    """Cover the remaining prediction, scoring, and evaluation branches."""

    class NoScorePipeline(_DummyPipeline):
        def predict_scores(self, data):
            raise NotImplementedError

    classifier = Classifier(
        data=["one", "two"], labels=["A", "B"], pipeline=_DummyPipeline()
    )
    assert classifier.predict() == ["a", "b"]
    assert classifier.predict_scores() == [{"a": 0.9, "b": 0.1}]
    assert classifier.evaluate() == {"accuracy": 1.0}

    with pytest.raises(ValueError):
        Classifier(data=None, pipeline=_DummyPipeline()).predict()
    with pytest.raises(ValueError):
        Classifier(data=None, pipeline=_DummyPipeline()).predict_scores()
    with pytest.raises(ValueError):
        Classifier(data=None, pipeline=_DummyPipeline()).evaluate()
    with pytest.raises(NotImplementedError):
        Classifier(
            data=["one"], labels=["A"], pipeline=NoScorePipeline()
        ).predict_scores()

    classifier = Classifier(
        data=["one", "two", "three", "four"],
        labels=["A", "B", "A", "B"],
        titles=["t-1", "t-2", "t-3", "t-4"],
        pipeline=_DummyPipeline(),
    )
    assert classifier._resolve_data_and_labels(
        data=None,
        labels=None,
        titles=None,
    )[0].titles == ["t-1", "t-2", "t-3", "t-4"]

    with pytest.raises(ValueError):
        Classifier(data=None, pipeline=None).fit()


def test_classifier_remaining_helper_branches() -> None:
    """Exercise the final row-alignment, split, and wrapper branches in ClassifierData and Classifier."""
    assert ClassifierData._matrix_row_count(np.array([[1, 2], [3, 4]])) == 2
    assert ClassifierData._matrix_row_count(["a", "b"]) == 2
    assert ClassifierData._validate_label_count(None, 2, "msg") is None

    dtm_like = type(
        "DTMLike",
        (),
        {
            "doc_term_matrix": np.array([[1, 0], [0, 1]]),
            "vectorizer": object(),
            "labels": ["A", "B"],
            "docs": ["alpha", "beta"],
            "titles": ["t-1", "t-2"],
        },
    )()
    dtm_data = ClassifierData._from_dtm_input(
        dtm_like, labels=["A", "B"], titles=["x", "y"]
    )
    assert dtm_data.titles == ["x", "y"]
    assert dtm_data.row_count() == 2

    dataframe = pd.DataFrame({"text": ["u", "v"], "label": ["A", "B"]})
    frame_data = ClassifierData._from_dataframe_input(
        dataframe, labels=["X", "Y"], titles=["T1", "T2"]
    )
    assert frame_data.labels == ["X", "Y"]
    assert frame_data.titles == ["T1", "T2"]

    matrix_data = ClassifierData._from_matrix_input(
        np.array([[1, 0], [0, 1]]), labels=["L1", "L2"], titles=["n1", "n2"]
    )
    assert matrix_data.matrix is not None
    assert matrix_data.titles == ["n1", "n2"]

    seq_data = ClassifierData._from_sequence_input(
        ["one", "two"], labels=["A", "B"], titles=["t1", "t2"]
    )
    assert seq_data.values == ["one", "two"]
    assert seq_data.labels == ["A", "B"]

    scalar_data = ClassifierData.from_input("plain text", labels=["A"], titles=["book"])
    assert scalar_data.values == "plain text"
    assert scalar_data.titles == ["book"]

    docs_data = ClassifierData(
        values=["alpha", "beta"],
        labels=["A", "B"],
        docs=[
            type("Doc", (), {"text": "alpha"})(),
            type("Doc", (), {"text": "beta"})(),
        ],
    )
    assert docs_data.as_texts() == ["alpha", "beta"]
    assert ClassifierData.from_input(
        ["alpha", "beta"], labels=["A", "B"]
    ).as_texts() == [
        "alpha",
        "beta",
    ]

    subset_df = ClassifierData.from_input(
        pd.DataFrame({"text": ["one", "two", "three"]}),
        labels=["A", "B", "C"],
        titles=["t1", "t2", "t3"],
    ).subset([0, 2])
    assert subset_df.labels == ["A", "C"]
    assert subset_df.titles == ["t1", "t3"]

    grouped = ClassifierData.from_input(
        ["a", "b", "c"], labels=["A", "A", "B"]
    )._group_label_indices()
    assert grouped == {"A": [0, 1], "B": [2]}

    split_indices = ClassifierData.from_input(
        ["a", "b", "c"], labels=["A", "A", "B"]
    )._split_indices_by_labels(
        test_size=0.5,
        dev_size=0.5,
        random_state=9,
    )
    assert set(split_indices["train"]) | set(split_indices["test"]) == {0, 2}
    assert "dev" in split_indices

    classifier = Classifier(
        data=["alpha", "beta", "gamma", "delta"],
        labels=["A", "A", "B", "B"],
        titles=["t1", "t2", "t3", "t4"],
        pipeline=_DummyPipeline(),
    )
    classifier.data = ["alpha", "beta"]
    classifier.labels = ["A", "B"]
    classifier.titles = ["u1", "u2"]
    resolved_data, resolved_labels = classifier._resolve_data_and_labels(
        data=["x", "y"], labels=None, titles=None
    )
    assert resolved_labels == ["A", "B"]
    assert resolved_data.titles == ["u1", "u2"]

    split = classifier.split_data(test_size=0.5, dev_size=0.25, random_state=11)
    assert set(split["train"]["labels"]) | set(split["test"]["labels"]) == {"B"}
    assert split["train"]["titles"] is not None
    assert split["dev"]["labels"]

    classifier.fit(data=["x", "y"], labels=["A", "B"])
    assert classifier.predict(["x"]) == ["a", "b"]
    assert classifier.predict_scores(["x"]) == [{"a": 0.9, "b": 0.1}]
    assert classifier.evaluate(["x"], ["A"]) == {"accuracy": 1.0}
    assert classifier(["x"]) == ["a", "b"]

    with pytest.raises(ValueError):
        ClassifierData._validate_split_parameters(0, 0.5, None)
    with pytest.raises(ValueError):
        ClassifierData._validate_split_parameters(3, 0.0, None)
    with pytest.raises(ValueError):
        ClassifierData._validate_split_parameters(3, 0.5, 0.0)
    with pytest.raises(ValueError):
        ClassifierData._validate_split_parameters(3, 0.5, 1.0)

    class NoScorePipeline(_DummyPipeline):
        def predict_scores(self, data):
            raise NotImplementedError

    with pytest.raises(NotImplementedError):
        Classifier(
            data=["one"], labels=["A"], pipeline=NoScorePipeline()
        ).predict_scores()


def test_classifier_final_uncovered_line_targets() -> None:
    """Exercise remaining abstract and edge branches in classifier helpers."""
    base = BaseClassificationPipeline()
    with pytest.raises(NotImplementedError):
        _ = base.model
    with pytest.raises(NotImplementedError):
        base.fit(["x"], ["A"])
    with pytest.raises(NotImplementedError):
        base.predict(["x"])
    with pytest.raises(NotImplementedError):
        base.predict_scores(["x"])
    with pytest.raises(NotImplementedError):
        base.evaluate(["x"], ["A"])
    with pytest.raises(NotImplementedError):
        base.save("unused")
    with pytest.raises(NotImplementedError):
        BaseClassificationPipeline.load("unused")

    # DTM normalization branch where both matrix and docs are missing.
    dtm_without_matrix_or_docs = type(
        "DTMNoMatrixDocs",
        (),
        {
            "doc_term_matrix": None,
            "vectorizer": object(),
            "labels": ["A", "B"],
            "docs": None,
        },
    )()
    dtm_data = ClassifierData.from_input(dtm_without_matrix_or_docs)
    assert dtm_data.row_count() == 2

    # from_input branch for list of spaCy docs.
    nlp = spacy.blank("en")
    spacy_docs = [nlp.make_doc("alpha beta"), nlp.make_doc("gamma")]
    docs_data = ClassifierData.from_input(spacy_docs, labels=["A", "B"])
    assert docs_data.docs is not None
    assert docs_data.row_count() == 2

    # row_count branches.
    assert ClassifierData(values=["a"], labels=["L"], docs=["doc"]).row_count() == 1
    assert ClassifierData(values=["a", "b"], labels=["L1", "L2"]).row_count() == 2
    assert ClassifierData(values="scalar", labels=["L"]).row_count() == 1

    # as_text helper branches.
    assert ClassifierData._as_text_for_doc(123) == "123"
    frame = ClassifierData(values=pd.DataFrame({"text": ["x"]}), labels=["A"])
    assert frame.as_texts() == ["{'text': 'x'}"]
    values_with_spacy_doc = ClassifierData(
        values=[nlp.make_doc("one two")], labels=["A"]
    )
    assert values_with_spacy_doc.as_texts() == ["one two"]
    assert ClassifierData(values="plain", labels=["A"]).as_texts() == ["plain"]

    # split helper branches.
    assert ClassifierData._test_count_for_group(1, 0.9) == 0
    tiny_group = ClassifierData.from_input(["x", "y"], labels=["A", "A"])
    split_with_dev = tiny_group._split_indices_by_labels(
        test_size=0.5,
        dev_size=0.5,
        random_state=0,
    )
    assert split_with_dev["dev"] == []
    assert ClassifierData._split_random_indices(2, 0.9, 1)["test"]
    assert ClassifierData._split_random_indices(2, 0.1, 1)["test"]

    # split() path that explicitly avoids stratification.
    random_split = ClassifierData.from_input(
        ["a", "b", "c"], labels=["A", "B", "C"]
    ).split(test_size=0.34, stratify=False, random_state=3)
    assert len(random_split["train"].labels) + len(random_split["test"].labels) == 3

    classifier = Classifier(data=["x"], labels=["A"], pipeline=_DummyPipeline())

    # Hit validator TypeError branch directly by mutating and calling validator.
    classifier.pipeline = object()  # type: ignore[assignment]
    with pytest.raises(TypeError):
        classifier._validate_pipeline()

    with pytest.raises(ValueError):
        Classifier(data=["x"], labels=["A"], pipeline=None).fit()
    with pytest.raises(ValueError):
        Classifier(data=["x"], labels=["A"], pipeline=None).predict()
    with pytest.raises(ValueError):
        Classifier(data=["x"], labels=["A"], pipeline=None).evaluate()

    # Construct without validation to hit the no-predict_scores implementation guard.
    raw = Classifier.model_construct(
        data=["x"],
        labels=["A"],
        titles=[],
        pipeline=object(),
    )
    with pytest.raises(NotImplementedError):
        raw.predict_scores(["x"])

    with pytest.raises(ValueError):
        Classifier(data=["x"], labels=[], pipeline=_DummyPipeline()).evaluate(["x"])


def test_classifier_last_four_coverage_lines() -> None:
    """Cover final row-count, split-count, and pipeline-guard branches."""
    # Cover ClassifierData._from_dtm_input branch where row_count comes from docs length.
    dtm_docs_only = type(
        "DTMDocsOnly",
        (),
        {
            "doc_term_matrix": None,
            "vectorizer": object(),
            "labels": ["A", "B"],
            "docs": ["doc-a", "doc-b"],
        },
    )()
    dtm_docs_data = ClassifierData.from_input(dtm_docs_only)
    assert dtm_docs_data.row_count() == 2

    # Cover ClassifierData.row_count DataFrame branch.
    frame_data = ClassifierData.from_input(
        pd.DataFrame({"text": ["x", "y"], "label": ["A", "B"]})
    )
    assert frame_data.row_count() == 2

    # Cover _dev_count_for_group early return.
    assert ClassifierData._dev_count_for_group(1, 0.5) == 0

    # Cover Classifier.predict_scores pipeline guard.
    with pytest.raises(ValueError):
        Classifier(data=["x"], labels=["A"], pipeline=None).predict_scores(["x"])
