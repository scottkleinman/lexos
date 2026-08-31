"""spacy_pipeline.py.

Last Updated: August 30, 2026
Last Tested: August 30, 2026
"""

import json
from pathlib import Path
from typing import Any, Literal, Sequence

import spacy
from pydantic import Field, PrivateAttr, model_validator
from spacy.language import Language
from spacy.training import Example

from lexos.classification.classifier import BaseClassificationPipeline, ClassifierData


class SpaCyTextCategorizerPipeline(BaseClassificationPipeline):
    """A spaCy `TextCategorizer` wrapper with a Lexos-friendly API."""

    language: str = Field(default="en", description="SpaCy language shortcut to use.")
    nlp: Language | None = Field(
        default=None, description="Underlying spaCy language object."
    )
    exclusive_classes: bool = Field(
        default=True, description="Whether classes are mutually exclusive."
    )
    architecture: str | dict[str, Any] = Field(
        default="bow", description="Text categorizer architecture or custom config."
    )
    epochs: int = Field(default=10, description="Number of training epochs.")
    score_ranking: Literal["document", "global"] = Field(
        default="document",
        description=(
            "How to rank labels when selecting a multi-label prediction: "
            "'document' ranks within each document, 'global' ranks using the "
            "global score ordering across all predicted documents."
        ),
    )
    _labels: list[str] = PrivateAttr(default_factory=list)
    _resolved_pipe_name: str = PrivateAttr(default="textcat")
    _resolved_architecture: str = PrivateAttr(default="bow")
    _target_label_count: int = PrivateAttr(default=1)

    @model_validator(mode="after")
    def _validate_and_resolve_settings(self):
        """Resolve the configured textcat behavior into a valid spaCy component config.

        Args:
            mode (str): The validation mode, typically "after".
        """
        if isinstance(self.architecture, dict):
            custom_cfg = dict(self.architecture)
            if "model" in custom_cfg:
                self._resolved_architecture = custom_cfg["model"]
            else:
                self._resolved_architecture = custom_cfg
            if "exclusive_classes" in self._resolved_architecture:
                self.exclusive_classes = bool(
                    self._resolved_architecture["exclusive_classes"]
                )
            self._resolved_pipe_name = (
                "textcat_multilabel" if not self.exclusive_classes else "textcat"
            )
            return self

        if self.architecture not in {"bow", "cnn", "ensemble"}:
            raise ValueError(
                "Unsupported spaCy architecture. Expected one of: bow, cnn, ensemble."
            )

        self._resolved_architecture = self.architecture
        self._resolved_pipe_name = (
            "textcat_multilabel" if not self.exclusive_classes else "textcat"
        )
        return self

    def __init__(self, **data):
        """Initialize the SpaCyTextCategorizerPipeline with the specified settings."""
        super().__init__(**data)
        if self.nlp is None:
            self.nlp = spacy.blank(self.language)

        pipe_name = self._resolved_pipe_name
        if pipe_name not in self.nlp.pipe_names:
            cfg = self._build_pipe_config()
            self.nlp.add_pipe(pipe_name, config=cfg)

    def save(self, path: str | Path) -> None:
        """Save the trained spaCy pipeline and its configuration to disk."""
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        config = self.model_dump(mode="python", exclude={"nlp"})
        config["_labels"] = list(self._labels)
        config["_target_label_count"] = int(getattr(self, "_target_label_count", 1))
        config["_global_label_rank"] = dict(getattr(self, "_global_label_rank", {}))
        config_path = target / "pipeline_config.json"
        config_path.write_text(json.dumps(config, indent=2))
        self.nlp.to_disk(str(target / "model"))

    @classmethod
    def load(cls, path: str | Path) -> "SpaCyTextCategorizerPipeline":
        """Load a saved spaCy pipeline instance and restore its configuration."""
        target = Path(path)
        config = json.loads((target / "pipeline_config.json").read_text())
        pipeline = cls(**{k: v for k, v in config.items() if not k.startswith("_")})
        pipeline.nlp = spacy.load(str(target / "model"))
        pipeline._labels = list(config.get("_labels", []))
        pipeline._target_label_count = int(config.get("_target_label_count", 1))
        pipeline._global_label_rank = dict(config.get("_global_label_rank", {}))
        pipeline._resolved_pipe_name = (
            "textcat_multilabel" if not pipeline.exclusive_classes else "textcat"
        )
        return pipeline

    def _build_pipe_config(self) -> dict[str, Any]:
        """Resolve the configured architecture into the model config spaCy expects.

        Returns:
            dict[str, Any]: The resolved spaCy pipe configuration.
        """
        if isinstance(self._resolved_architecture, dict):
            custom_cfg = dict(self._resolved_architecture)
            if "@architectures" not in custom_cfg and "model" in custom_cfg:
                custom_cfg = dict(custom_cfg["model"])
            if "exclusive_classes" not in custom_cfg:
                custom_cfg["exclusive_classes"] = self.exclusive_classes
            if "no_output_layer" not in custom_cfg:
                custom_cfg["no_output_layer"] = False
            return {"model": custom_cfg}

        architecture = self._resolved_architecture
        base = {
            "@architectures": "spacy.TextCatBOW.v3",
            "ngram_size": 1,
            "length": 262144,
            "no_output_layer": False,
        }
        if architecture == "bow":
            base["exclusive_classes"] = self.exclusive_classes
            return {"model": base}

        if architecture == "cnn":
            base = {
                "@architectures": "spacy.TextCatReduce.v1",
                "exclusive_classes": self.exclusive_classes,
                "use_reduce_first": False,
                "use_reduce_last": False,
                "use_reduce_max": False,
                "use_reduce_mean": True,
                "tok2vec": {
                    "@architectures": "spacy.HashEmbedCNN.v2",
                    "pretrained_vectors": None,
                    "width": 96,
                    "depth": 4,
                    "embed_size": 2000,
                    "window_size": 1,
                    "maxout_pieces": 3,
                    "subword_features": True,
                },
            }
            return {"model": base}

        if architecture == "ensemble":
            return {
                "model": {
                    "@architectures": "spacy.TextCatEnsemble.v2",
                    "tok2vec": {
                        "@architectures": "spacy.Tok2Vec.v2",
                        "embed": {
                            "@architectures": "spacy.MultiHashEmbed.v2",
                            "width": 64,
                            "rows": [2000, 2000, 500, 1000, 500],
                            "attrs": ["NORM", "LOWER", "PREFIX", "SUFFIX", "SHAPE"],
                            "include_static_vectors": False,
                        },
                        "encode": {
                            "@architectures": "spacy.MaxoutWindowEncoder.v2",
                            "width": 64,
                            "window_size": 1,
                            "maxout_pieces": 3,
                            "depth": 2,
                        },
                    },
                    "linear_model": {
                        "@architectures": "spacy.TextCatBOW.v3",
                        "exclusive_classes": self.exclusive_classes,
                        "length": 262144,
                        "ngram_size": 1,
                        "no_output_layer": False,
                    },
                }
            }

        raise ValueError(f"Unsupported spaCy architecture: {architecture}")

    @property
    def model(self) -> Language:
        """Return the underlying spaCy language model.

        Returns:
            The underlying spaCy `Language` object representing the text categorizer model.
        """
        return self.nlp

    def _coerce_label_list(self, value: Any) -> list[str]:
        """Normalize a single label or a per-document label collection to a list of strings.

        Args:
            value: A single label, a collection of labels, or None.

        Returns:
            A list of strings representing the normalized labels.
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

    def _build_score_map(
        self, categories: Sequence[str], labels: Any
    ) -> dict[str, float]:
        """Build a spaCy `cats` score map from a document's active label or label set.

        Args:
            categories: A sequence of all possible category labels.
            labels: The active label or collection of labels for the current document.

        Returns:
            A dictionary mapping each category to a float score (1.0 for active labels, 0.0 otherwise).
        """
        score_map = {category: 0.0 for category in categories}
        for label in self._coerce_label_list(labels):
            score_map[str(label)] = 1.0
        return score_map

    def _get_textcat_pipe(self):
        """Return the active spaCy text categorizer component for this pipeline."""
        pipe_name = self._resolved_pipe_name
        if pipe_name not in self.nlp.pipe_names:
            raise KeyError(
                f"[E001] No component '{pipe_name}' found in pipeline. "
                f"Available names: {list(self.nlp.pipe_names)}"
            )
        return self.nlp.get_pipe(pipe_name)

    def _switch_to_multi_label_mode(self) -> None:
        """Switch the underlying spaCy text categorizer to multi-label mode."""
        if self.exclusive_classes is False:
            self._resolved_pipe_name = "textcat_multilabel"
            return

        self.exclusive_classes = False
        current_pipe_name = self._resolved_pipe_name
        target_pipe_name = "textcat_multilabel"

        if (
            current_pipe_name != target_pipe_name
            and current_pipe_name in self.nlp.pipe_names
        ):
            self.nlp.remove_pipe(current_pipe_name)
        self._resolved_pipe_name = target_pipe_name
        if target_pipe_name not in self.nlp.pipe_names:
            self.nlp.add_pipe(target_pipe_name, config=self._build_pipe_config())

    def _ensure_labels(self, labels: Sequence[Any]) -> list[str]:
        """Ensure that the specified labels are added to the text categorizer.

        Args:
            labels: A sequence of label strings or per-document label collections.

        Returns:
            A sorted list of unique labels that are now present in the text categorizer.
        """
        flattened = []
        for item in labels:
            flattened.extend(self._coerce_label_list(item))
        unique_labels = sorted({str(label) for label in flattened})
        textcat_pipe = self._get_textcat_pipe()
        for label in unique_labels:
            if label not in textcat_pipe.labels:
                textcat_pipe.add_label(label)
        self._labels = unique_labels
        return unique_labels

    @staticmethod
    def _is_spacy_doc(doc: Any) -> bool:
        """Return True when `doc` is a spaCy Doc-like object."""
        return (
            doc is not None
            and type(doc).__module__.startswith("spacy.")
            and hasattr(doc, "vocab")
            and hasattr(doc, "__iter__")
        )

    @staticmethod
    def _coerce_doc_to_text(doc: Any) -> str:
        """Convert a document-like object to a text string while preserving tokenization."""
        if doc is None:
            return ""
        if SpaCyTextCategorizerPipeline._is_spacy_doc(doc):
            return " ".join(token.text for token in doc)
        if hasattr(doc, "text") and not SpaCyTextCategorizerPipeline._is_spacy_doc(doc):
            return str(doc.text)
        if isinstance(doc, (list, tuple, set)):
            return " ".join(str(item) for item in doc)
        return str(doc)

    def _coerce_texts_from_sequence(self, docs: Sequence[Any]) -> list[str]:
        """Normalize a sequence of doc-like items into plain text strings."""
        return [self._coerce_doc_to_text(doc) for doc in docs]

    def _coerce_texts_from_input(self, data: Any) -> list[str]:
        """Convert raw text, a standardized dataset, or Lexos docs into spaCy text strings.

        Args:
            data: The input data, which can be raw text, a standardized dataset, or Lexos docs.

        Returns:
            A list of spaCy-compatible text strings extracted from the input data.
        """
        if isinstance(data, ClassifierData):
            if data.matrix is not None and data.docs is None:
                raise ValueError(
                    "SpaCy text categorization requires tokenized document text, not just a sparse DTM matrix."
                )
            return data.as_texts()

        if hasattr(data, "docs") and getattr(data, "docs", None) is not None:
            return self._coerce_texts_from_sequence(data.docs)

        if hasattr(data, "doc_term_matrix") and hasattr(data, "vectorizer"):
            docs = getattr(data, "docs", None)
            if docs is None:
                raise ValueError(
                    "SpaCy text categorization requires tokenized document text, not just a sparse DTM matrix."
                )
            return self._coerce_texts_from_sequence(docs)

        return [str(item) for item in data]

    def fit(self, data: Any, labels: Sequence[Any]) -> "SpaCyTextCategorizerPipeline":
        """Fit a spaCy text categorizer on raw text or tokenized Lexos docs.

        Args:
            data: The input data, which can be raw text, a standardized dataset, or Lexos docs.
            labels: The corresponding labels for each document in the input data.

        Returns:
            The fitted SpaCyTextCategorizerPipeline instance.
        """
        texts = self._coerce_texts_from_input(data)
        label_values = list(labels)

        if len(texts) != len(label_values):
            raise ValueError("Number of texts must match the number of labels.")

        normalized_labels = [self._coerce_label_list(label) for label in label_values]
        if self.exclusive_classes and any(len(item) > 1 for item in normalized_labels):
            self._switch_to_multi_label_mode()

        if self.exclusive_classes:
            for i, item in enumerate(normalized_labels):
                if len(item) != 1:
                    raise ValueError(
                        "exclusive_classes=True requires exactly one label per document; "
                        f"found {item!r} for row {i}."
                    )

        categories = self._ensure_labels(normalized_labels)
        if not categories:
            raise ValueError("At least one label is required before training.")

        self._target_label_count = max(
            (len(item) for item in normalized_labels), default=1
        )
        self._global_label_rank = {}

        training_examples = []
        for text, item_labels in zip(texts, normalized_labels):
            doc = self.nlp.make_doc(text)
            score_map = self._build_score_map(categories, item_labels)
            training_examples.append(Example.from_dict(doc, {"cats": score_map}))

        self.nlp.initialize()
        optimizer = self.nlp.create_optimizer()
        for _ in range(self.epochs):
            losses = {}
            self.nlp.update(training_examples, sgd=optimizer, losses=losses)

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

    def _build_global_label_rank(
        self, score_rows: Sequence[dict[str, float]]
    ) -> dict[str, int]:
        """Compute corpus-level label ordering from a collection of score rows."""
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

    def _score_rows_for_texts(self, texts: Sequence[str]) -> list[dict[str, float]]:
        """Return the score map for each document produced by the spaCy model."""
        return [
            {str(label): float(score) for label, score in (doc.cats or {}).items()}
            for doc in self.nlp.pipe(texts)
        ]

    def predict_scores(self, data: Any) -> list[dict[str, float]]:
        """Return the raw spaCy confidence scores attached to each predicted document.

        Args:
            data: The input data, which can be raw text, a standardized dataset, or Lexos docs.

        Returns:
            A list of dictionaries containing the raw confidence scores for each label.
        """
        texts = self._coerce_texts_from_input(data)
        score_rows = self._score_rows_for_texts(texts)
        if self.score_ranking == "global":
            self._global_label_rank = self._build_global_label_rank(score_rows)
        else:
            self._global_label_rank = {}

        return [self._rank_score_row(row) for row in score_rows]

    @staticmethod
    def _select_single_label(scores: dict[str, float]) -> str:
        """Return the strongest single-label prediction."""
        if not scores:
            return ""
        return max(scores, key=scores.get)

    def _select_multi_label_predictions(self, scores: dict[str, float]) -> list[str]:
        """Return the highest-scoring labels according to the configured ranking mode."""
        if not scores:
            return []

        target_count = max(1, getattr(self, "_target_label_count", 1))
        if self.score_ranking == "document":
            ranked_labels = [
                label
                for label, _ in sorted(
                    scores.items(), key=lambda item: item[1], reverse=True
                )
            ]
            return ranked_labels[:target_count]

        global_rank = getattr(self, "_global_label_rank", {})
        ranked_labels = [
            label
            for label in sorted(
                scores,
                key=lambda label: (scores[label], -global_rank.get(label, 0)),
                reverse=True,
            )
        ]
        return ranked_labels[:target_count]

    def _predict_document(self, doc: Any) -> list[str] | str:
        """Predict a label or a set of labels for one spaCy document result."""
        scores = doc.cats or {}
        if not scores:
            return [] if not self.exclusive_classes else ""
        if self.exclusive_classes:
            return self._select_single_label(scores)
        return self._select_multi_label_predictions(scores)

    def predict(self, data: Any) -> list[list[str] | str]:
        """Predict labels for raw text or tokenized Lexos documents.

        Args:
            data: The input data, which can be raw text, a standardized dataset, or Lexos docs.

        Returns:
            A list of predicted labels for each document.
        """
        texts = self._coerce_texts_from_input(data)
        doc_iter = list(self.nlp.pipe(texts))
        if self.score_ranking == "global":
            score_rows = [
                {str(label): float(score) for label, score in (doc.cats or {}).items()}
                for doc in doc_iter
            ]
            self._global_label_rank = self._build_global_label_rank(score_rows)
        else:
            self._global_label_rank = {}
        return [self._predict_document(doc) for doc in doc_iter]

    @staticmethod
    def _matches_predicted_labels(
        predicted: list[str] | str, gold_labels: set[str]
    ) -> bool:
        """Return whether a prediction matches the gold label set."""
        if isinstance(predicted, list):
            return set(predicted) == gold_labels
        if not predicted:
            return not gold_labels
        return predicted == next(iter(gold_labels), "")

    def evaluate(self, data: Any, labels: Sequence[Any]) -> dict[str, float]:
        """Evaluate the trained pipeline using accuracy.

        Args:
            data: The input text data to evaluate.
            labels: A sequence of true label strings or per-document label lists.

        Returns:
            A dictionary containing the accuracy of the predictions.
        """
        if not labels:
            return {"accuracy": 0.0}

        predictions = self.predict(data)
        matches = 0
        for predicted, label in zip(predictions, labels):
            gold_labels = set(self._coerce_label_list(label))
            if self._matches_predicted_labels(predicted, gold_labels):
                matches += 1
        return {"accuracy": matches / len(labels)}


__all__ = ["SpaCyTextCategorizerPipeline"]
