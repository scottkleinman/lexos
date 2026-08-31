"""sklearn_pipeline.py.

Last Updated: August 30, 2026
Last Tested: August 30, 2026
"""

from pathlib import Path
from typing import Any, Literal, Sequence

import joblib
from pydantic import Field, PrivateAttr

from lexos.classification.classifier import BaseClassificationPipeline, ClassifierData

MultiLabelWrapper = Any

try:  # pragma: no cover - optional dependency
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import MultiLabelBinarizer
except ImportError:  # pragma: no cover - handled at runtime
    TfidfVectorizer = None
    LogisticRegression = None
    accuracy_score = None
    OneVsRestClassifier = None
    MultiLabelBinarizer = None


class SklearnClassifierPipeline(BaseClassificationPipeline):
    """A scikit-learn-based classification backend."""

    vectorizer: Any = Field(
        default=None,
        description="Text vectorizer to transform raw text before fitting.",
    )
    estimator: Any = Field(
        default=None, description="Underlying scikit-learn estimator."
    )
    max_iter: int = Field(
        default=1000, description="Maximum number of iterations for iterative solvers."
    )
    multi_label_wrapper: Any = Field(
        default=None,
        description=(
            "Optional factory that wraps a base estimator for multi-label tasks. "
            "Receives a base estimator and returns a multi-label-compatible estimator."
        ),
    )
    score_ranking: Literal["document", "global"] = Field(
        default="document",
        description=(
            "How to rank labels when selecting multi-label outputs: 'document' "
            "sorts each document's scores separately, while 'global' ranks labels "
            "according to their score across the whole prediction set."
        ),
    )
    _target_label_count: int = PrivateAttr(default=1)
    _global_label_rank: dict[str, int] = PrivateAttr(default_factory=dict)

    def __init__(self, **data):
        """Initialize the SklearnClassifierPipeline with the specified settings."""
        super().__init__(**data)
        if self.vectorizer is None:
            if TfidfVectorizer is None:
                raise ImportError(
                    "scikit-learn is required for SklearnClassifierPipeline."
                )
            self.vectorizer = TfidfVectorizer()
        if self.estimator is None:
            if LogisticRegression is None:
                raise ImportError(
                    "scikit-learn is required for SklearnClassifierPipeline."
                )
            self.estimator = LogisticRegression(max_iter=self.max_iter)

    def save(self, path: str | Path) -> None:
        """Save the fitted sklearn pipeline together with its configuration."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.model_dump(mode="python"),
            "vectorizer": self.vectorizer,
            "estimator": self.estimator,
            "label_binarizer": getattr(self, "_label_binarizer", None),
            "target_label_count": getattr(self, "_target_label_count", 1),
            "global_label_rank": getattr(self, "_global_label_rank", {}),
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "SklearnClassifierPipeline":
        """Load a saved sklearn pipeline and restore its fitted state and config."""
        payload = joblib.load(Path(path))
        pipeline = cls(**payload["config"])
        pipeline.vectorizer = payload["vectorizer"]
        pipeline.estimator = payload["estimator"]
        pipeline._label_binarizer = payload.get("label_binarizer")
        pipeline._target_label_count = payload.get("target_label_count", 1)
        pipeline._global_label_rank = payload.get("global_label_rank", {})
        return pipeline

    def _coerce_label_list(self, value: Any) -> list[str]:
        """Normalize a single label or list-valued document target to a flat label list.

        Args:
            value: The raw label or list of labels for a single document.

        Returns:
            A flat list of string labels.
        """
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple, set)):
            labels: list[str] = []
            for item in value:
                labels.extend(self._coerce_label_list(item))
            return labels
        return [str(value)]

    def _is_multi_label(self, labels: Sequence[Any]) -> bool:
        """Return True when any document target contains more than one label.

        Args:
            labels: A sequence of raw label targets for each document.

        Returns:
            True if any document has more than one label, False otherwise.
        """
        return any(len(self._coerce_label_list(label)) > 1 for label in labels)

    def _prepare_targets(self, labels: Sequence[Any]) -> tuple[list[str], Any | None]:
        """Convert single-label or multi-label targets into a training-ready representation.

        Args:
            labels: A sequence of raw label targets for each document.

        Returns:
            A tuple containing the list of unique labels and the encoded target matrix (or None for single-label).
        """
        if not labels:
            return [], None
        if not self._is_multi_label(labels):
            single_labels: list[str] = []
            for document_label in labels:
                single_labels.extend(self._coerce_label_list(document_label))
            return [str(label) for label in single_labels], None

        flattened = [self._coerce_label_list(label) for label in labels]
        mlb = MultiLabelBinarizer()
        encoded = mlb.fit_transform(flattened)
        return [str(value) for value in mlb.classes_], encoded

    @property
    def model(self) -> Any:
        """Return the underlying scikit-learn estimator.

        Returns:
            The underlying scikit-learn estimator object.
        """
        return self.estimator

    def _prepare_label_lists(self, labels: Sequence[Any]) -> list[list[str]]:
        """Normalize raw label targets to a list of per-document label sets.

        Args:
            labels: A sequence of raw label targets for each document.

        Returns:
            A list of per-document label lists.
        """
        return [self._coerce_label_list(label) for label in labels]

    def _get_estimator_for(self, is_multi: bool) -> Any:
        """Create the underlying estimator for the given label scheme.

        Args:
            is_multi: A boolean indicating whether the label scheme is multi-label.

        Returns:
            The underlying scikit-learn estimator object configured for the label scheme.
        """
        if not is_multi:
            return self.estimator

        base_estimator = LogisticRegression(max_iter=self.max_iter)
        if self.multi_label_wrapper is not None:
            return self.multi_label_wrapper(base_estimator)
        return OneVsRestClassifier(base_estimator)

    def _fit_label_matrix(
        self, matrix: Any, label_list: list[list[str]], is_multi: bool
    ) -> None:
        """Fit a matrix-like feature representation against single- or multi-label targets.

        Args:
            matrix: A matrix-like feature representation of the input data.
            label_list: A list of per-document label lists.
            is_multi: A boolean indicating whether the label scheme is multi-label.

        Returns:
            None
        """
        if is_multi:
            self._target_label_count = max(
                (len(item) for item in label_list), default=1
            )
            self.estimator = self._get_estimator_for(is_multi=True)
            mlb = MultiLabelBinarizer()
            transformed_targets = mlb.fit_transform(label_list)
            self.estimator.fit(matrix, transformed_targets)
            self._label_binarizer = mlb
            return

        self._target_label_count = 1
        self.estimator.fit(
            matrix, [str(item[0]) if item else "" for item in label_list]
        )

    def _fit_text_matrix(
        self, texts: Sequence[str], label_list: list[list[str]], is_multi: bool
    ) -> None:
        """Vectorize texts and fit the model against the prepared labels.

        Args:
            texts: A sequence of raw text documents.
            label_list: A list of per-document label lists.
            is_multi: A boolean indicating whether the label scheme is multi-label.

        Returns:
            None
        """
        if len(texts) != len(label_list):
            raise ValueError("Number of texts must match the number of labels.")

        transformed = self.vectorizer.fit_transform(texts)
        self._fit_label_matrix(transformed, label_list, is_multi)

    @staticmethod
    def _is_spacy_doc(doc: Any) -> bool:
        """Return True when `doc` is a spaCy Doc-like object."""
        return (
            doc is not None
            and type(doc).__module__.startswith("spacy.")
            and hasattr(doc, "vocab")
            and hasattr(doc, "__iter__")
        )

    def _coerce_text_for_doc(self, doc: Any) -> str:
        """Convert a document-like item to text while preserving spaCy tokenization."""
        if self._is_spacy_doc(doc):
            return " ".join(token.text for token in doc)
        return str(doc)

    def _coerce_texts(self, data: Any) -> list[str]:
        """Normalize raw data into strings while preserving spaCy Doc tokenization."""
        if isinstance(data, ClassifierData):
            if data.docs is not None:
                return [self._coerce_text_for_doc(doc) for doc in data.docs]
            return data.as_texts()

        if hasattr(data, "docs") and getattr(data, "docs", None) is not None:
            return [self._coerce_text_for_doc(doc) for doc in data.docs]

        if isinstance(data, (list, tuple)):
            return [self._coerce_text_for_doc(item) for item in data]

        return [str(item) for item in data]

    def fit(self, data: Any, labels: Sequence[Any]) -> "SklearnClassifierPipeline":
        """Fit a scikit-learn classifier on the provided data.

        Args:
            data: Raw text, a ClassifierData object, a Lexos DTM, or another matrix-like object.
            labels: A sequence of label strings or per-document label collections.

        Returns:
            The fitted `SklearnClassifierPipeline` instance.
        """
        if TfidfVectorizer is None or LogisticRegression is None:
            raise ImportError("scikit-learn is required for SklearnClassifierPipeline.")

        label_list = self._prepare_label_lists(labels)
        is_multi = any(len(item) > 1 for item in label_list)

        if isinstance(data, ClassifierData):
            matrix = data.matrix
            if matrix is not None:
                self._fit_label_matrix(matrix, label_list, is_multi)
                return self

            texts = self._coerce_texts(data)
            self._fit_text_matrix(texts, label_list, is_multi)
            return self

        if hasattr(data, "doc_term_matrix") and hasattr(data, "vectorizer"):
            matrix = getattr(data, "doc_term_matrix")
            if matrix is None:
                raise ValueError("DTM input is missing a document-term matrix.")
            self._fit_label_matrix(matrix, label_list, is_multi)
            return self

        if hasattr(data, "shape") and getattr(data, "ndim", None) == 2:
            self._fit_label_matrix(data, label_list, is_multi)
            return self

        texts = self._coerce_texts(data)
        self._fit_text_matrix(texts, label_list, is_multi)
        return self

    def _rank_score_row(self, score_row: dict[str, float]) -> dict[str, float]:
        """Rank a single score row according to the configured scoring policy."""
        if not score_row:
            return {}

        if self.score_ranking == "document":
            return {
                label: score
                for label, score in sorted(
                    score_row.items(), key=lambda item: item[1], reverse=True
                )
            }

        global_rank = getattr(self, "_global_label_rank", {})
        return {
            label: score
            for label, score in sorted(
                score_row.items(),
                key=lambda item: (
                    item[1],
                    -global_rank.get(item[0], 0),
                ),
                reverse=True,
            )
        }

    def _prepare_prediction_matrix(self, data: Any) -> Any:
        """Normalize an input payload into the matrix structure expected by the estimator."""
        if self.estimator is None:
            raise ValueError("The pipeline must be fitted before calling predict().")

        if isinstance(data, ClassifierData):
            if data.matrix is not None:
                return data.matrix
            if self.vectorizer is None:
                raise ValueError(
                    "The pipeline must be fitted before calling predict()."
                )
            return self.vectorizer.transform(data.as_texts())

        if hasattr(data, "doc_term_matrix") and hasattr(data, "vectorizer"):
            matrix = getattr(data, "doc_term_matrix")
            if matrix is None:
                raise ValueError("DTM input is missing a document-term matrix.")
            return matrix

        if hasattr(data, "shape") and getattr(data, "ndim", None) == 2:
            return data

        if self.vectorizer is None:
            raise ValueError("The pipeline must be fitted before calling predict().")
        return self.vectorizer.transform(self._coerce_texts(data))

    def _build_global_label_rank(
        self, score_rows: Sequence[dict[str, float]]
    ) -> dict[str, int]:
        """Compute a corpus-level rank for labels from a sequence of score rows."""
        if not score_rows:
            return {}

        label_scores: dict[str, list[float]] = {}
        for row in score_rows:
            for label, score in row.items():
                label_scores.setdefault(label, []).append(float(score))

        return {
            label: rank
            for rank, label in enumerate(
                sorted(
                    label_scores,
                    key=lambda label: (
                        sum(label_scores[label]) / len(label_scores[label]),
                        label,
                    ),
                    reverse=True,
                )
            )
        }

    def _score_rows_for_matrix(self, matrix: Any) -> list[dict[str, float]]:
        """Return raw label scores for each row in the given matrix."""
        label_binarizer = getattr(self, "_label_binarizer", None)
        if label_binarizer is not None:
            classes = list(label_binarizer.classes_)
            if hasattr(self.estimator, "predict_proba"):
                scores = self.estimator.predict_proba(matrix)
            elif hasattr(self.estimator, "decision_function"):
                scores = self.estimator.decision_function(matrix)
            else:
                return []
            return [
                {str(classes[idx]): float(score) for idx, score in enumerate(row)}
                for row in scores
            ]

        if hasattr(self.estimator, "predict_proba"):
            scores = self.estimator.predict_proba(matrix)
            return [
                {str(index): float(score) for index, score in enumerate(row)}
                for row in scores
            ]

        raise NotImplementedError(
            f"{type(self.estimator).__name__} does not implement predict_scores()."
        )

    def _select_top_labels(self, score_row: dict[str, float]) -> list[str]:
        """Return the top labels for a single document according to the configured ranking."""
        ranked = self._rank_score_row(score_row)
        top_count = max(1, self._target_label_count)
        return list(ranked.keys())[:top_count]

    def predict(self, data: Any) -> list[str]:
        """Predict labels for a sequence of texts, a ClassifierData object, or a Lexos DTM.

        Args:
            data: Raw text, a ClassifierData object, a Lexos DTM, or another matrix-like object.

        Returns:
            A list of predicted label strings.
        """
        if self.estimator is None:
            raise ValueError("The pipeline must be fitted before calling predict().")

        matrix_to_predict = self._prepare_prediction_matrix(data)
        predictions = self.estimator.predict(matrix_to_predict)

        if getattr(self, "_label_binarizer", None) is None:
            return [str(value) for value in predictions]

        score_rows = self._score_rows_for_matrix(matrix_to_predict)
        if score_rows:
            if self.score_ranking == "global":
                self._global_label_rank = self._build_global_label_rank(score_rows)
            else:
                self._global_label_rank = {}
            return [self._select_top_labels(row) for row in score_rows]

        decoded = self._label_binarizer.inverse_transform(predictions)
        return [[str(label) for label in labels] for labels in decoded]

    def predict_scores(self, data: Any) -> list[dict[str, float]]:
        """Return the per-document confidence scores for each label.

        Args:
            data: Raw text, a ClassifierData object, a Lexos DTM, or another matrix-like object.

        Returns:
            A list of dictionaries containing each label's score for each document.
        """
        if self.estimator is None:
            raise ValueError(
                "The pipeline must be fitted before calling predict_scores()."
            )

        matrix_to_predict = self._prepare_prediction_matrix(data)
        raw_rows = self._score_rows_for_matrix(matrix_to_predict)

        if self.score_ranking == "global" and raw_rows:
            self._global_label_rank = self._build_global_label_rank(raw_rows)
        else:
            self._global_label_rank = {}

        return [self._rank_score_row(row) for row in raw_rows]

    def evaluate(self, data: Any, labels: Sequence[Any]) -> dict[str, float]:
        """Evaluate using standard accuracy.

        Args:
            data: The input text data to evaluate.
            labels: A sequence of true label strings or per-document label lists.

        Returns:
            A dictionary containing the accuracy of the predictions.
        """
        if accuracy_score is None:
            raise ImportError("scikit-learn is required to compute evaluation metrics.")
        predictions = self.predict(data)
        if getattr(self, "_label_binarizer", None) is not None:
            gold = [set(self._coerce_label_list(label)) for label in labels]
            pred = [
                set(item) if isinstance(item, list) else set() for item in predictions
            ]
            matches = sum(1 for a, b in zip(gold, pred) if a == b)
            return {"accuracy": float(matches / len(labels))}
        return {"accuracy": float(accuracy_score(list(labels), predictions))}


__all__ = ["SklearnClassifierPipeline"]
