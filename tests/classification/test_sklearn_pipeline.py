"""test_sklearn_pipeline.py.

Coverage: 100%

Last Updated: August 30, 2026
"""

import numpy as np
import pytest
import spacy
from sklearn.multiclass import OneVsRestClassifier

import lexos.classification.sklearn_pipeline as sklearn_module
from lexos.classification import ClassifierData, SklearnClassifierPipeline


def test_sklearn_pipeline_direct_branches() -> None:
    """Exercise the remaining scikit-learn helper branches and matrix logic."""
    pipeline = SklearnClassifierPipeline(max_iter=2000)

    assert pipeline._coerce_label_list("A") == ["A"]
    assert pipeline._coerce_label_list(["A", ["B", "C"]]) == ["A", "B", "C"]
    assert pipeline._is_multi_label(["A", ["B", "C"]]) is True
    assert pipeline._prepare_targets(["A", "B"]) == (["A", "B"], None)
    multi_key, multi_matrix = pipeline._prepare_targets([["A", "B"], ["C"]])
    assert set(multi_key).issubset({"A", "B", "C"})
    assert multi_matrix.shape == (2, 3)

    assert pipeline._get_estimator_for(False) is pipeline.estimator
    assert pipeline._get_estimator_for(True) is not None

    matrix = np.array([[1, 0], [0, 1]])
    pipeline._fit_label_matrix(matrix, [["A"], ["B"]], False)
    assert pipeline._target_label_count == 1
    pipeline._fit_label_matrix(matrix, [["A", "B"], ["B"]], True)
    assert pipeline._target_label_count == 2

    pipeline._fit_text_matrix(["alpha", "beta"], [["A"], ["B"]], False)
    assert pipeline.estimator is not None

    with pytest.raises(ValueError):
        pipeline._fit_text_matrix(["alpha"], [["A"], ["B"]], False)

    transformed = pipeline._prepare_prediction_matrix(np.array([[1, 0]]))
    assert transformed.shape == (1, 2)
    assert pipeline._build_global_label_rank(
        [{"A": 0.7, "B": 0.2}, {"A": 0.6, "B": 0.8}]
    )
    assert pipeline._select_top_labels({"A": 0.9, "B": 0.2}) == ["A"]

    fit_data = ["history literature", "market report", "poetry review"]
    fit_labels = [["history", "literature"], ["market"], ["poetry"]]
    pipeline.fit(fit_data, fit_labels)
    preds = pipeline.predict(fit_data)
    assert len(preds) == 3
    assert all(isinstance(item, list) for item in preds)
    scores = pipeline.predict_scores(fit_data)
    assert len(scores) == 3
    assert pipeline.evaluate(fit_data, fit_labels)["accuracy"] >= 0.0


def test_sklearn_pipeline_multi_label_fit_and_predict() -> None:
    """Multi-label documents should be fit and predict as list-valued outputs."""
    texts = [
        "history literature review",
        "history politics essay",
        "finance market report",
        "finance policy analysis",
        "art history review",
    ]
    labels = [
        ["history", "literature"],
        ["history", "politics"],
        ["finance", "market"],
        ["finance", "policy"],
        ["art", "history"],
    ]

    pipeline = SklearnClassifierPipeline(max_iter=2000)
    pipeline.fit(texts, labels)
    predictions = pipeline.predict(texts)

    assert len(predictions) == len(texts)
    assert all(isinstance(prediction, list) for prediction in predictions)


def test_sklearn_pipeline_predict_scores_respect_document_ranking() -> None:
    """Prediction score dictionaries should be ordered by descending confidence."""
    texts = [
        "history literature review",
        "finance market report",
        "art and poetry",
        "government policies and elections",
    ]
    labels = [
        ["history", "literature"],
        ["finance", "market"],
        ["art", "poetry"],
        ["government", "politics"],
    ]

    pipeline = SklearnClassifierPipeline(max_iter=2000)
    pipeline.fit(texts, labels)
    scores = pipeline.predict_scores(texts)

    assert len(scores) == len(texts)
    assert all(isinstance(row, dict) for row in scores)
    assert all(
        list(row.keys()) == sorted(row, key=row.get, reverse=True)
        for row in scores
        if row
    )


def test_sklearn_pipeline_helper_branches_and_serialization(tmp_path) -> None:
    """Cover the helper branches for matrix fitting, score ranking, and save/load."""
    nlp = spacy.blank("en")
    doc = nlp.make_doc("history literature review")

    pipeline = SklearnClassifierPipeline(
        max_iter=2000,
        multi_label_wrapper=OneVsRestClassifier,
    )
    assert pipeline._coerce_label_list(None) == []
    assert pipeline._prepare_label_lists(["A", ["B", "C"]]) == [["A"], ["B", "C"]]
    assert pipeline._get_estimator_for(False) is pipeline.estimator
    assert pipeline._get_estimator_for(True) is not None
    assert pipeline._coerce_text_for_doc(doc) == "history literature review"

    matrix = np.array([[1, 0], [0, 1]])
    pipeline.fit(matrix, ["A", "B"])
    assert pipeline.predict(np.array([[1, 0]])) == ["A"]

    score_row = {"history": 0.9, "literature": 0.7, "politics": 0.2}
    assert list(pipeline._rank_score_row(score_row).keys()) == [
        "history",
        "literature",
        "politics",
    ]
    pipeline.score_ranking = "global"
    pipeline._global_label_rank = {"history": 0, "literature": 1, "politics": 2}
    assert list(pipeline._rank_score_row(score_row).keys()) == [
        "history",
        "literature",
        "politics",
    ]

    data = ClassifierData.from_input(
        ["history literature", "politics essay"], labels=["A", "B"]
    )
    pipeline.fit(data, ["A", "B"])
    assert pipeline._coerce_texts(data) == ["history literature", "politics essay"]

    saved = tmp_path / "sklearn_pipeline.joblib"
    pipeline.save(saved)
    restored = SklearnClassifierPipeline.load(saved)
    assert restored.score_ranking == pipeline.score_ranking
    assert restored.vectorizer is not None

    pipeline.score_ranking = "document"
    rows = [{"history": 0.9, "politics": 0.1}, {"politics": 0.8, "history": 0.5}]
    assert pipeline._build_global_label_rank(rows) == {"history": 0, "politics": 1}

    with pytest.raises(ValueError):
        pipeline._fit_text_matrix(["a"], [["A"], ["B"]], False)

    matrix_data = ClassifierData.from_input(
        np.array([[1, 0], [0, 1]]), labels=["A", "B"]
    )
    pipeline.fit(matrix_data, ["A", "B"])
    assert pipeline.predict_scores(np.array([[1, 0]]))


def test_sklearn_pipeline_remaining_helper_branches() -> None:
    """Exercise the remaining multi-label ranking and classifier-data prediction branches."""
    pipeline = SklearnClassifierPipeline(
        max_iter=2000,
        multi_label_wrapper=OneVsRestClassifier,
    )

    assert pipeline._prepare_targets([]) == ([], None)
    assert pipeline._prepare_targets(["A", "B"]) == (["A", "B"], None)
    multi_key, multi_matrix = pipeline._prepare_targets([["A", "B"], ["C"]])
    assert set(multi_key) == {"A", "B", "C"}
    assert multi_matrix.shape == (2, 3)

    matrix = np.array([[1, 0], [0, 1]])
    pipeline._fit_label_matrix(matrix, [["A"], ["B"]], False)
    assert pipeline._target_label_count == 1
    pipeline._fit_label_matrix(matrix, [["A", "B"], ["B"]], True)
    assert pipeline._target_label_count == 2
    assert hasattr(pipeline, "_label_binarizer")

    score_rows = pipeline._score_rows_for_matrix(matrix)
    assert len(score_rows) == 2
    assert all(isinstance(row, dict) for row in score_rows)

    pipeline.score_ranking = "global"
    pipeline._global_label_rank = {"A": 0, "B": 1}
    ranked = pipeline._rank_score_row({"A": 0.7, "B": 0.4})
    assert list(ranked.keys()) == ["A", "B"]

    data = ClassifierData.from_input(
        ["history literature", "politics essay"], labels=["A", "B"]
    )
    pipeline.fit(data, ["A", "B"])
    assert pipeline._coerce_texts(data) == ["history literature", "politics essay"]
    prediction = pipeline.predict(data)
    assert len(prediction) == 2
    assert all(isinstance(row, list) for row in prediction)
    assert pipeline.predict_scores(data)

    score_rows = [{"A": 0.90, "B": 0.40}, {"A": 0.30, "B": 0.80}]
    assert pipeline._build_global_label_rank(score_rows) == {"B": 0, "A": 1}

    matrix_data = ClassifierData.from_input(
        np.array([[1, 0], [0, 1]]), labels=["A", "B"]
    )
    pipeline.fit(matrix_data, ["A", "B"])
    assert pipeline._prepare_prediction_matrix(matrix_data).shape == (2, 2)
    assert pipeline.evaluate(matrix_data, ["A", "B"]) == {"accuracy": 1.0}


def test_sklearn_pipeline_remaining_edge_branches(monkeypatch) -> None:
    """Cover the remaining import guards, empty returns, and fallback prediction branches."""
    pipeline = SklearnClassifierPipeline(max_iter=2000)

    with monkeypatch.context() as m:
        m.setattr(sklearn_module, "TfidfVectorizer", None)
        m.setattr(sklearn_module, "LogisticRegression", None)
        with pytest.raises(ImportError):
            sklearn_module.SklearnClassifierPipeline(max_iter=2000)

    assert pipeline._coerce_label_list(7) == ["7"]
    assert pipeline.model is pipeline.estimator
    assert pipeline._rank_score_row({}) == {}
    assert pipeline._build_global_label_rank([]) == {}

    data = ClassifierData(values=["x", "y"], labels=["A", "B"])
    assert pipeline._coerce_texts(data) == ["x", "y"]
    assert pipeline._coerce_texts(["one", "two"]) == ["one", "two"]

    class FakeDTM:
        doc_term_matrix = None
        vectorizer = object()
        labels = ["A", "B"]
        docs = ["x", "y"]

    with pytest.raises(ValueError):
        pipeline.fit(FakeDTM(), ["A", "B"])

    pipeline.estimator = None
    with pytest.raises(ValueError):
        pipeline._prepare_prediction_matrix(["hello"])
    with pytest.raises(ValueError):
        pipeline.evaluate(["hello"], ["A"])

    pipeline.estimator = pipeline.__class__(max_iter=2000).estimator
    pipeline.vectorizer = None
    with pytest.raises(ValueError):
        pipeline._prepare_prediction_matrix(["hello"])

    class DTMWithoutMatrix:
        doc_term_matrix = None
        vectorizer = object()

    with pytest.raises(ValueError):
        pipeline._prepare_prediction_matrix(DTMWithoutMatrix())

    class DecisionOnlyEstimator:
        def decision_function(self, matrix):
            return np.array([[0.9, 0.1], [0.2, 0.8]])

    pipeline.estimator = DecisionOnlyEstimator()
    pipeline.vectorizer = object()
    pipeline._label_binarizer = sklearn_module.MultiLabelBinarizer().fit([["A"], ["B"]])
    assert pipeline._score_rows_for_matrix(np.array([[1, 0], [0, 1]])) == [
        {"A": 0.9, "B": 0.1},
        {"A": 0.2, "B": 0.8},
    ]

    class NoScoreEstimator:
        pass

    pipeline.estimator = NoScoreEstimator()
    pipeline._label_binarizer = None
    with pytest.raises(NotImplementedError):
        pipeline._score_rows_for_matrix(np.array([[1, 0]]))

    class EmptyProbaEstimator:
        def predict(self, matrix):
            return np.array([[1, 0, 0]])

        def predict_proba(self, matrix):
            return []

    pipeline.estimator = EmptyProbaEstimator()
    pipeline.vectorizer = object()
    pipeline._label_binarizer = sklearn_module.MultiLabelBinarizer().fit(
        [["A", "B"], ["C"]]
    )
    prediction = pipeline.predict(np.array([[1, 0, 0]]))
    assert isinstance(prediction, list)
    assert len(prediction) == 1
    assert isinstance(prediction[0], list)

    class SingleLabelEstimator:
        def predict(self, matrix):
            return np.array(["A"])

    pipeline.estimator = SingleLabelEstimator()
    pipeline._label_binarizer = None
    with monkeypatch.context() as m:
        m.setattr(sklearn_module, "accuracy_score", None)
        with pytest.raises(ImportError):
            pipeline.evaluate(["hello"], ["A"])

    pipeline.estimator = SingleLabelEstimator()
    pipeline.vectorizer = sklearn_module.TfidfVectorizer()
    pipeline.vectorizer.fit(["hello world", "goodbye world"])
    pipeline._label_binarizer = None
    assert pipeline.evaluate(["hello world"], ["A"]) == {"accuracy": 1.0}


def test_sklearn_pipeline_final_uncovered_lines(monkeypatch) -> None:
    """Exercise the final uncovered sklearn branches with minimal targeted inputs."""
    with monkeypatch.context() as m:
        m.setattr(sklearn_module, "LogisticRegression", None)
        with pytest.raises(ImportError):
            sklearn_module.SklearnClassifierPipeline(vectorizer=object())

    pipeline = SklearnClassifierPipeline(max_iter=2000)

    with monkeypatch.context() as m:
        m.setattr(sklearn_module, "TfidfVectorizer", None)
        m.setattr(sklearn_module, "LogisticRegression", None)
        with pytest.raises(ImportError):
            pipeline.fit(["text"], ["A"])

    nlp = spacy.blank("en")
    classifier_data = ClassifierData(
        values=["alpha", "beta"],
        labels=["A", "B"],
        docs=[nlp.make_doc("alpha"), nlp.make_doc("beta")],
    )
    assert pipeline._coerce_texts(classifier_data) == ["alpha", "beta"]

    class HasDocs:
        docs = ["gamma", "delta"]

    assert pipeline._coerce_texts(HasDocs()) == ["gamma", "delta"]

    class IterOnly:
        def __iter__(self):
            return iter([1, 2])

    assert pipeline._coerce_texts(IterOnly()) == ["1", "2"]

    class DTMForFit:
        doc_term_matrix = np.array([[1, 0], [0, 1]])
        vectorizer = object()

    assert pipeline.fit(DTMForFit(), ["A", "B"]) is pipeline

    pipeline.estimator = object()
    pipeline.vectorizer = None
    with pytest.raises(ValueError):
        pipeline._prepare_prediction_matrix(
            ClassifierData(values=["hello"], labels=["A"])
        )

    class DTMForPredict:
        doc_term_matrix = np.array([[1, 0]])
        vectorizer = object()

    assert np.array_equal(
        pipeline._prepare_prediction_matrix(DTMForPredict()),
        np.array([[1, 0]]),
    )

    class NoScoreEstimator:
        pass

    pipeline.estimator = NoScoreEstimator()
    pipeline._label_binarizer = sklearn_module.MultiLabelBinarizer().fit([["A"], ["B"]])
    assert pipeline._score_rows_for_matrix(np.array([[1, 0]])) == []

    pipeline.estimator = None
    with pytest.raises(ValueError):
        pipeline.predict_scores(["hello"])
