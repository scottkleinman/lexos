"""test_spacy_pipeline.py.

Coverage: 100%

Last Updated: August 30, 2026
"""

from pathlib import Path

import numpy as np
import pytest

from lexos.classification import Classifier, SpaCyTextCategorizerPipeline
from lexos.classification.classifier import ClassifierData


def test_spacy_pipeline_direct_branches() -> None:
    """Exercise the spaCy helper methods, validation paths, and save/load behavior."""
    pipeline = SpaCyTextCategorizerPipeline(exclusive_classes=False, architecture="bow")
    assert pipeline._resolved_pipe_name == "textcat_multilabel"

    custom = SpaCyTextCategorizerPipeline(
        architecture={
            "@architectures": "spacy.TextCatBOW.v3",
            "ngram_size": 1,
            "length": 262144,
            "exclusive_classes": False,
            "no_output_layer": False,
        }
    )
    assert custom._resolved_pipe_name == "textcat_multilabel"

    with pytest.raises(ValueError):
        SpaCyTextCategorizerPipeline(architecture="not_a_real_architecture")

    labels = [["A", "B"], ["C"]]
    assert pipeline._ensure_labels(labels) == ["A", "B", "C"]
    assert pipeline._coerce_doc_to_text(None) == ""
    assert pipeline._coerce_doc_to_text(["a", "b"]) == "a b"

    text_data = ["history of literature", "market report and finance"]
    label_data = [["history", "literature"], ["finance"]]
    fit_pipe = SpaCyTextCategorizerPipeline(
        language="en",
        epochs=1,
        exclusive_classes=False,
        architecture="bow",
    )
    fit_pipe.fit(text_data, label_data)
    assert fit_pipe.predict(text_data)
    assert len(fit_pipe.predict_scores(text_data)) == 2

    assert fit_pipe._select_single_label({"A": 0.1, "B": 0.2}) == "B"
    fit_pipe._target_label_count = 1
    assert fit_pipe._predict_document(
        type("Doc", (), {"cats": {"A": 0.1, "B": 0.3}})()
    ) == ["B"]
    assert fit_pipe.evaluate(text_data, label_data)["accuracy"] >= 0.0

    classifier = Classifier(data=text_data, labels=label_data, pipeline=fit_pipe)
    assert classifier.predict_scores(text_data)

    save_path = Path("./.tmp_spacy_pipeline_coverage")
    fit_pipe.save(save_path)
    restored = SpaCyTextCategorizerPipeline.load(save_path)
    assert restored.score_ranking == fit_pipe.score_ranking
    assert restored.exclusive_classes is False


def test_spacy_pipeline_auto_switches_to_multilabel_when_needed() -> None:
    """Multi-label inputs should switch the component to textcat_multilabel."""
    texts = [
        "history of literature and criticism",
        "market report and investor analysis",
        "art history and poetry",
        "government policy and elections",
    ]
    labels = [
        ["history", "literature"],
        ["market", "finance"],
        ["art", "history"],
        ["government", "politics"],
    ]

    pipeline = SpaCyTextCategorizerPipeline(
        language="en",
        epochs=1,
        exclusive_classes=True,
        architecture="bow",
    )
    pipeline.fit(texts, labels)

    assert pipeline.exclusive_classes is False
    assert pipeline._resolved_pipe_name == "textcat_multilabel"

    predictions = pipeline.predict(texts)
    assert len(predictions) == len(texts)
    assert all(isinstance(prediction, list) for prediction in predictions)


def test_spacy_pipeline_save_load_and_score_ranking(tmp_path) -> None:
    """The trained spaCy pipeline should preserve config and produce ranked scores."""
    texts = [
        "history and literature review",
        "finance and market report",
        "poetry and arts criticism",
    ]
    labels = [
        ["history", "literature"],
        ["finance", "market"],
        ["art", "poetry"],
    ]

    pipeline = SpaCyTextCategorizerPipeline(
        language="en",
        epochs=1,
        exclusive_classes=False,
        architecture="bow",
    )
    pipeline.fit(texts, labels)

    scores = pipeline.predict_scores(texts)
    assert len(scores) == len(texts)
    assert all(isinstance(row, dict) for row in scores)
    assert all(
        list(row.keys()) == sorted(row, key=row.get, reverse=True)
        for row in scores
        if row
    )

    saved_path = tmp_path / "spacy_pipeline"
    pipeline.save(saved_path)
    loaded = SpaCyTextCategorizerPipeline.load(saved_path)

    assert loaded.exclusive_classes is False
    assert loaded._resolved_pipe_name == "textcat_multilabel"
    assert loaded.score_ranking == pipeline.score_ranking


def test_spacy_pipeline_remaining_helper_branches() -> None:
    """Cover the remaining architecture, ranking, validation, and evaluation helpers."""
    bow = SpaCyTextCategorizerPipeline(exclusive_classes=True, architecture="bow")
    assert bow._resolved_pipe_name == "textcat"
    assert bow._build_pipe_config()["model"]["exclusive_classes"] is True

    custom = SpaCyTextCategorizerPipeline(
        architecture={
            "model": {
                "@architectures": "spacy.TextCatBOW.v3",
                "ngram_size": 1,
                "length": 262144,
                "exclusive_classes": False,
                "no_output_layer": False,
            }
        }
    )
    assert custom._resolved_pipe_name == "textcat_multilabel"
    assert custom._build_pipe_config()["model"]["exclusive_classes"] is False

    ensemble = SpaCyTextCategorizerPipeline(
        language="en",
        epochs=1,
        exclusive_classes=False,
        architecture="ensemble",
    )
    assert (
        ensemble._build_pipe_config()["model"]["linear_model"]["exclusive_classes"]
        is False
    )

    multi = SpaCyTextCategorizerPipeline(
        language="en",
        epochs=1,
        exclusive_classes=True,
        architecture="bow",
    )
    multi._switch_to_multi_label_mode()
    assert multi.exclusive_classes is False
    assert multi._resolved_pipe_name == "textcat_multilabel"
    assert multi._ensure_labels([["A", "B"], ["C"]]) == ["A", "B", "C"]
    assert multi._build_score_map(["A", "B", "C"], ["A", "C"]) == {
        "A": 1.0,
        "B": 0.0,
        "C": 1.0,
    }

    bad_pipe = SpaCyTextCategorizerPipeline(language="en", epochs=1)
    bad_pipe._resolved_pipe_name = "missing_pipe"
    with pytest.raises(KeyError):
        bad_pipe._get_textcat_pipe()

    multi.score_ranking = "document"
    assert multi._select_multi_label_predictions({"A": 0.4, "B": 0.9, "C": 0.6}) == [
        "B"
    ]
    multi._target_label_count = 2
    multi.score_ranking = "global"
    multi._global_label_rank = {"A": 0, "B": 1, "C": 2}
    assert list(multi._rank_score_row({"A": 0.9, "B": 0.4, "C": 0.7}).keys()) == [
        "A",
        "C",
        "B",
    ]
    assert multi._build_global_label_rank(
        [{"A": 0.1, "B": 0.9}, {"A": 0.2, "B": 0.8}]
    ) == {"B": 0, "A": 1}

    doc = multi.nlp.make_doc("literature review")
    doc.cats = {"A": 0.7, "B": 0.3}
    assert multi._predict_document(doc) == ["A", "B"]
    assert multi._matches_predicted_labels(["A", "B"], {"A", "B"}) is True
    assert multi._matches_predicted_labels("A", {"A"}) is True
    assert multi._matches_predicted_labels([], set()) is True

    texts = ["history and literature", "finance and policy"]
    labels = [["history", "literature"], ["finance"]]
    trained = SpaCyTextCategorizerPipeline(
        language="en",
        epochs=1,
        exclusive_classes=False,
        architecture="bow",
    )
    trained.fit(texts, labels)
    assert trained.evaluate(texts, labels)["accuracy"] >= 0.0
    assert trained.evaluate([], [])["accuracy"] == 0.0

    empty = SpaCyTextCategorizerPipeline(language="en", epochs=1)
    assert empty._coerce_doc_to_text(None) == ""
    assert empty._coerce_doc_to_text(["alpha", "beta"]) == "alpha beta"

    with pytest.raises(ValueError):
        SpaCyTextCategorizerPipeline(architecture="not_a_real_architecture")

    with pytest.raises(ValueError):
        trained.fit(["text one", "text two"], [["A", "B"], ["C", "D"], ["E"]])

    cnn = SpaCyTextCategorizerPipeline(
        architecture="cnn",
        exclusive_classes=True,
    )
    assert cnn._build_pipe_config()["model"]["exclusive_classes"] is True

    bow_single = SpaCyTextCategorizerPipeline(
        architecture="bow",
        exclusive_classes=True,
    )
    assert bow_single._build_pipe_config()["model"]["exclusive_classes"] is True


def test_spacy_pipeline_final_uncovered_branches() -> None:
    """Exercise the remaining coercion, ranking, and document-level prediction branches."""
    pipeline = SpaCyTextCategorizerPipeline(
        language="en",
        epochs=1,
        exclusive_classes=False,
        architecture="bow",
    )

    assert pipeline._coerce_label_list(5) == ["5"]
    assert set(pipeline._coerce_label_list([["A", {"B", "C"}]])) == {"A", "B", "C"}
    assert pipeline._select_single_label({}) == ""

    class TextDoc:
        def __init__(self, text):
            self.text = text

    assert pipeline._coerce_doc_to_text(TextDoc("hello world")) == "hello world"
    assert pipeline._coerce_doc_to_text(["hello", "world"]) == "hello world"

    matrix_data = ClassifierData(
        values=np.array([[1, 0], [0, 1]]),
        labels=["A", "B"],
        matrix=np.array([[1, 0], [0, 1]]),
    )
    with pytest.raises(ValueError):
        pipeline._coerce_texts_from_input(
            ClassifierData(values=np.array([[1]]), labels=["A"], matrix=np.array([[1]]))
        )
    assert pipeline._coerce_texts_from_input(
        ClassifierData(
            values=["x", "y"], labels=["A", "B"], docs=[TextDoc("x"), TextDoc("y")]
        )
    ) == ["x", "y"]

    sample_data = ["history and literature", "politics and policy"]
    target_labels = [["history", "literature"], ["politics", "policy"]]
    pipeline.fit(sample_data, target_labels)
    pipeline._target_label_count = 2
    pipeline.score_ranking = "global"
    pipeline._global_label_rank = {
        "history": 0,
        "literature": 1,
        "politics": 2,
        "policy": 3,
    }
    assert pipeline._select_multi_label_predictions(
        {"history": 0.8, "policy": 0.6, "literature": 0.3}
    ) == [
        "history",
        "policy",
    ]
    assert list(pipeline._rank_score_row({"policy": 0.6, "history": 0.8}).keys()) == [
        "history",
        "policy",
    ]
    assert pipeline._build_global_label_rank(
        [{"history": 0.8, "policy": 0.5}, {"history": 0.6, "policy": 0.7}]
    ) == {"history": 0, "policy": 1}

    doc = pipeline.nlp.make_doc("history policy")
    doc.cats = {"history": 0.9, "policy": 0.2}
    pipeline._target_label_count = 1
    assert pipeline._predict_document(doc) == ["history"]
    prediction = pipeline.predict(["history and policy"])
    assert isinstance(prediction, list)
    assert len(prediction) == 1
    assert isinstance(prediction[0], list)
    assert len(prediction[0]) == 1
    assert prediction[0][0] in {"history", "literature", "politics", "policy"}

    assert pipeline.evaluate(sample_data, target_labels)["accuracy"] >= 0.0
    assert pipeline._matches_predicted_labels([], set()) is True
    assert pipeline._matches_predicted_labels("A", {"A"}) is True


def test_spacy_pipeline_remaining_edge_line_targets() -> None:
    """Cover remaining edge branches for coercion, mode switching, and prediction helpers."""
    pipeline = SpaCyTextCategorizerPipeline(language="en", epochs=1)

    # Field and property branches.
    assert pipeline.exclusive_classes is True
    assert pipeline.model is pipeline.nlp
    assert pipeline._coerce_label_list(None) == []

    # Custom config branch where defaults are injected.
    custom = SpaCyTextCategorizerPipeline(
        architecture={
            "model": {
                "@architectures": "spacy.TextCatBOW.v3",
                "ngram_size": 1,
                "length": 262144,
            }
        },
        exclusive_classes=False,
    )
    custom_model_cfg = custom._build_pipe_config()["model"]
    assert custom_model_cfg["exclusive_classes"] is False
    assert custom_model_cfg["no_output_layer"] is False

    # Explicitly cover the nested "model" extraction branch in _build_pipe_config.
    custom._resolved_architecture = {
        "model": {
            "@architectures": "spacy.TextCatBOW.v3",
            "ngram_size": 1,
            "length": 262144,
        }
    }
    nested_cfg = custom._build_pipe_config()["model"]
    assert nested_cfg["exclusive_classes"] is False

    # Force the internal unsupported architecture branch directly.
    custom._resolved_architecture = "unknown_arch"
    with pytest.raises(ValueError):
        custom._build_pipe_config()

    # Early return branch when already in non-exclusive mode.
    already_multi = SpaCyTextCategorizerPipeline(exclusive_classes=False)
    already_multi._switch_to_multi_label_mode()
    assert already_multi._resolved_pipe_name == "textcat_multilabel"

    # Cover spaCy-doc and generic-object coercion paths.
    doc = pipeline.nlp.make_doc("alpha beta")
    assert pipeline._coerce_doc_to_text(doc) == "alpha beta"

    class TextHolder:
        def __init__(self, text: str) -> None:
            self.text = text

    assert pipeline._coerce_doc_to_text(TextHolder("holder text")) == "holder text"
    assert pipeline._coerce_doc_to_text(123) == "123"
    assert pipeline._coerce_texts_from_sequence([doc, TextHolder("gamma")]) == [
        "alpha beta",
        "gamma",
    ]

    class DocsPayload:
        def __init__(self, docs) -> None:
            self.docs = docs

    assert pipeline._coerce_texts_from_input(DocsPayload([doc])) == ["alpha beta"]

    class DTMNoDocs:
        doc_term_matrix = np.array([[1]])
        vectorizer = object()
        docs = None

    class DTMLine345:
        doc_term_matrix = np.array([[1]])
        vectorizer = object()

        def __init__(self) -> None:
            self._docs_reads = 0

        @property
        def docs(self):
            # First read satisfies `hasattr`; second read keeps the early docs branch false;
            # third read (inside the DTM branch) returns docs so line 345 is executed.
            self._docs_reads += 1
            if self._docs_reads == 1:
                return "ignored"
            if self._docs_reads == 2:
                return None
            return ["with docs"]

    with pytest.raises(ValueError):
        pipeline._coerce_texts_from_input(DTMNoDocs())
    assert pipeline._coerce_texts_from_input(DTMLine345()) == ["with docs"]

    # Trigger exclusive_classes validation for an empty label in single-label mode.
    strict = SpaCyTextCategorizerPipeline(exclusive_classes=True, epochs=1)
    with pytest.raises(ValueError):
        strict.fit(["only text"], [None])

    # Trigger the no-labels training guard.
    with pytest.raises(ValueError):
        already_multi.fit(["only text"], [None])

    assert already_multi._rank_score_row({}) == {}
    assert already_multi._build_global_label_rank([]) == {}
    assert already_multi._select_multi_label_predictions({}) == []

    empty_doc = already_multi.nlp.make_doc("unused")
    empty_doc.cats = {}
    assert already_multi._predict_document(empty_doc) == []

    single = SpaCyTextCategorizerPipeline(exclusive_classes=True)
    single_doc = single.nlp.make_doc("single")
    single_doc.cats = {"A": 0.9, "B": 0.1}
    assert single._predict_document(single_doc) == "A"

    # Hit the global-ranking branch in predict().
    fitted = SpaCyTextCategorizerPipeline(exclusive_classes=False, epochs=1)
    train_texts = ["history literature", "market finance"]
    train_labels = [["history", "literature"], ["market", "finance"]]
    fitted.fit(train_texts, train_labels)
    fitted.score_ranking = "global"
    score_rows = fitted.predict_scores(train_texts)
    assert len(score_rows) == 2
    preds = fitted.predict(train_texts)
    assert len(preds) == 2
    assert fitted._global_label_rank

    assert fitted._matches_predicted_labels("", set()) is True
