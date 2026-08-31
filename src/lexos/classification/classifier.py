"""classifier.py.

This module provides a simple, user-facing classifier facade backed by
pluggable pipeline strategies.

Last Updated: August 30, 2026
Last Tested: August 30, 2026
"""

import random
from collections import defaultdict
from typing import Any, Sequence

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator


def _is_dtm_like(value: Any) -> bool:
    """Return True when `value` looks like a Lexos DTM object.

    Args:
        value: The object to check for DTM-like attributes.

    Returns:
        True if `value` has the attributes typical of a Lexos DTM object, False otherwise.
    """
    return (
        value is not None
        and hasattr(value, "doc_term_matrix")
        and hasattr(value, "vectorizer")
        and hasattr(value, "labels")
    )


def _is_spacy_doc(value: Any) -> bool:
    """Return True when `value` is a spaCy Doc-like object."""
    return (
        value is not None
        and type(value).__module__.startswith("spacy.")
        and hasattr(value, "vocab")
        and hasattr(value, "__iter__")
    )


class BaseClassificationPipeline(BaseModel):
    """Abstract strategy interface for classification backends.

    Subclasses implement the concrete logic for each method, such as spaCy
    `TextCategorizer` or a scikit-learn estimator.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = Field(default="classifier", description="Human-readable pipeline name.")

    @property
    def model(self) -> Any:
        """Return the underlying backend model object.

        Returns:
            The underlying backend model object.
        """
        raise NotImplementedError

    def fit(self, data: Any, labels: Sequence[str]) -> Any:
        """Train the pipeline on the supplied data and labels.

        Args:
            data: The input data to train on.
            labels: The corresponding labels for the input data.

        Returns:
            The trained pipeline instance.
        """
        raise NotImplementedError

    def predict(self, data: Any) -> Sequence[str]:
        """Predict labels for the supplied data.

        Args:
            data: The input data to make predictions on.

        Returns:
            A list of predicted labels for the input data.
        """
        raise NotImplementedError

    def predict_scores(self, data: Any) -> Sequence[dict[str, float]]:
        """Return the prediction probabilities or confidences for each input item.

        Args:
            data: The input data to make predictions on.

        Returns:
            A list of dictionaries containing prediction probabilities or confidences for each input item.
        """
        raise NotImplementedError

    def evaluate(self, data: Any, labels: Sequence[str]) -> dict[str, float]:
        """Evaluate the fitted pipeline on a dataset.

        Args:
            data: The input data to evaluate on.
            labels: The corresponding labels for the input data.

        Returns:
            A dictionary containing evaluation metrics for the input data and labels.
        """
        raise NotImplementedError

    def save(self, path: str | Any) -> None:
        """Persist the fitted pipeline and its configuration to disk."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Any) -> "BaseClassificationPipeline":
        """Load a saved pipeline instance from disk."""
        raise NotImplementedError

    def __call__(self, data: Any) -> Sequence[str]:
        """Convenience wrapper for predicting on a single data payload."""
        return self.predict(data)


class ClassifierData(BaseModel):
    """Standardized input wrapper for training and prediction data.

    This object centralizes the data-shape concerns ensuring that data is handled consistently before it is passed to `Classifier`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    values: Any
    labels: list[Any] = Field(default_factory=list)
    docs: Any = None
    titles: list[Any] | None = None
    matrix: Any = None
    source: str = "raw"

    def __init__(
        self,
        values: Any,
        labels: Sequence[Any] | None = None,
        *,
        docs: Any = None,
        titles: Sequence[Any] | None = None,
        matrix: Any = None,
        source: str = "raw",
    ) -> None:
        """Initialize the ClassifierData object.

        Args:
            values: The main data values.
            labels: Optional sequence of labels corresponding to the data.
            docs: Optional sequence of document objects.
            titles: Optional sequence of titles for the data items.
            matrix: Optional matrix representation of the data.
            source: A string indicating the source of the data.
        """
        super().__init__(
            values=values,
            labels=list(labels) if labels is not None else [],
            docs=docs,
            titles=list(titles) if titles is not None else None,
            matrix=matrix,
            source=source,
        )

    @staticmethod
    def _matrix_row_count(matrix: Any) -> int:
        """Return the number of rows in a matrix-like object, including list-backed inputs.

        Args:
            matrix: The matrix-like object to count rows for.

        Returns:
            The number of rows in the matrix-like object.
        """
        shape = getattr(matrix, "shape", None)
        if shape is not None:
            return int(shape[0])
        return len(matrix)

    @staticmethod
    def _validate_label_count(
        expected_count: int | None, actual_count: int, context: str
    ) -> None:
        """Raise a ValueError when label length diverges from the data shape.

        Args:
            expected_count: The expected number of labels.
            actual_count: The actual number of labels.
            context: A message to include in the ValueError if the counts do not match.

        Raises:
            ValueError: If the expected count is not None and does not match the actual count.
        """
        if expected_count is not None and expected_count != actual_count:
            raise ValueError(f"{context}")

    @classmethod
    def _from_dtm_input(
        cls,
        data: Any,
        labels: Sequence[Any] | None = None,
        titles: Sequence[Any] | None = None,
    ) -> "ClassifierData":
        """Normalize a Lexos DTM object into a standardized Dataset wrapper.

        Args:
            data: The Lexos DTM object to normalize.
            labels: Optional sequence of labels to associate with the data. If not provided, labels will be inferred from the DTM object.
            titles: Optional sequence of titles aligned with the records.

        Returns:
            An instance of the ClassifierData class wrapping the normalized data.
        """
        matrix = getattr(data, "doc_term_matrix", None)
        docs = getattr(data, "docs", None)
        resolved_labels = list(getattr(data, "labels", []) or [])
        resolved_titles = (
            list(getattr(data, "titles", []) or []) if hasattr(data, "titles") else []
        )
        if labels is not None and len(labels) > 0:
            resolved_labels = list(labels)
        if titles is not None and len(titles) > 0:
            resolved_titles = list(titles)

        if matrix is not None:
            row_count = cls._matrix_row_count(matrix)
        elif docs is not None:
            row_count = len(docs)
        else:
            row_count = len(resolved_labels)

        resolved_labels, resolved_titles = cls._resolve_row_alignment(
            row_count,
            resolved_labels,
            resolved_titles,
            "Label count must match the row count in the DTM document-term matrix."
            if matrix is not None
            else "Label count must match the number of stored DTM docs.",
            "Title count must match the number of rows in the dataset.",
        )

        return cls(
            values=matrix if matrix is not None else docs,
            labels=resolved_labels,
            docs=docs,
            titles=resolved_titles,
            matrix=matrix,
            source="dtm",
        )

    @classmethod
    def _from_dataframe_input(
        cls,
        data: pd.DataFrame,
        labels: Sequence[Any] | None = None,
        titles: Sequence[Any] | None = None,
    ) -> "ClassifierData":
        """Normalize a pandas DataFrame input into a standardized Dataset wrapper.

        Args:
            data: The pandas DataFrame to normalize.
            labels: Optional sequence of labels to associate with the data. If not provided, labels will be inferred from the DataFrame.

        Returns:
            An instance of the ClassifierData class wrapping the normalized data.
        """
        frame_labels = (
            list(data["label"].astype(str)) if "label" in data.columns else []
        )
        resolved_labels = list(frame_labels if labels is None else labels)
        resolved_titles = list(data["title"]) if "title" in data.columns else []
        if titles is not None and len(titles) > 0:
            resolved_titles = list(titles)

        resolved_labels, resolved_titles = cls._resolve_row_alignment(
            len(data),
            resolved_labels,
            resolved_titles,
            "Label count must match the row count in the DataFrame.",
            "Title count must match the row count in the DataFrame.",
        )

        return cls(
            values=data,
            labels=resolved_labels,
            titles=resolved_titles,
            source="dataframe",
        )

    @classmethod
    def _from_matrix_input(
        cls,
        data: Any,
        labels: Sequence[Any] | None = None,
        titles: Sequence[Any] | None = None,
    ) -> "ClassifierData":
        """Normalize matrix-like inputs into a standardized Dataset wrapper.

        Args:
            data: The matrix-like data to normalize.
            labels: Optional sequence of labels to associate with the data. If not provided, labels will be inferred from the matrix-like data.

        Returns:
            An instance of the ClassifierData class wrapping the normalized data.
        """
        resolved_labels = list(labels or [])
        resolved_titles = list(titles) if titles is not None else []
        row_count = cls._matrix_row_count(data)

        resolved_labels, resolved_titles = cls._resolve_row_alignment(
            row_count,
            resolved_labels,
            resolved_titles,
            "Label count must match the row count in the matrix-like data.",
            "Title count must match the row count in the matrix-like data.",
        )

        return cls(
            values=data,
            labels=resolved_labels,
            titles=resolved_titles,
            matrix=data,
            source="matrix",
        )

    @classmethod
    def _from_sequence_input(
        cls,
        data: Sequence[Any],
        labels: Sequence[Any] | None = None,
        titles: Sequence[Any] | None = None,
    ) -> "ClassifierData":
        """Normalize native Python sequences into a standardized Dataset wrapper.

        Args:
            data: The sequence of data items to normalize.
            labels: Optional sequence of labels to associate with the data. If not provided, labels will be inferred from the sequence.

        Returns:
            An instance of the ClassifierData class wrapping the normalized data.
        """
        resolved_labels = list(labels or [])
        resolved_titles = list(titles) if titles is not None else []

        resolved_labels, resolved_titles = cls._resolve_row_alignment(
            len(data),
            resolved_labels,
            resolved_titles,
            "Number of labels must match the number of data items.",
            "Title count must match the number of data items.",
        )

        return cls(
            values=list(data),
            labels=resolved_labels,
            titles=resolved_titles,
            source="raw",
        )

    @staticmethod
    def _resolve_row_alignment(
        row_count: int,
        labels: Sequence[Any] | None,
        titles: Sequence[Any] | None,
        labels_message: str,
        titles_message: str,
    ) -> tuple[list[Any], list[Any] | None]:
        """Validate label and title counts against a row count and return normalized values."""
        resolved_labels = list(labels or [])
        resolved_titles = list(titles) if titles is not None else []

        ClassifierData._validate_label_count(
            len(resolved_labels),
            row_count,
            labels_message,
        )
        if resolved_titles:
            ClassifierData._validate_label_count(
                len(resolved_titles),
                row_count,
                titles_message,
            )

        return resolved_labels, resolved_titles or None

    @classmethod
    def from_input(
        cls,
        data: Any,
        labels: Sequence[Any] | None = None,
        titles: Sequence[Any] | None = None,
    ) -> "ClassifierData":
        """Normalize raw text, DataFrames, and Lexos DTM objects to a standard form.

        Args:
            data: The input data to normalize. Can be raw text, a pandas DataFrame, a Lexos DTM object, or a matrix-like structure.
            labels: Optional sequence of labels to associate with the data. If not provided, labels will be inferred from the input data.
            titles: Optional sequence of titles aligned with the rows.

        Returns:
            An instance of the ClassifierData class wrapping the normalized data.
        """
        if _is_dtm_like(data):
            return cls._from_dtm_input(data, labels, titles)
        if isinstance(data, pd.DataFrame):
            return cls._from_dataframe_input(data, labels, titles)
        if hasattr(data, "shape") and getattr(data, "ndim", None) == 2:
            return cls._from_matrix_input(data, labels, titles)
        if isinstance(data, (list, tuple)):
            if data and _is_spacy_doc(data[0]):
                resolved_labels, resolved_titles = cls._resolve_row_alignment(
                    len(data),
                    labels,
                    titles,
                    "Number of labels must match the number of data items.",
                    "Title count must match the number of data items.",
                )
                return cls(
                    values=list(data),
                    labels=resolved_labels,
                    titles=resolved_titles,
                    docs=list(data),
                    source="raw",
                )
            return cls._from_sequence_input(data, labels, titles)
        if data is None:
            raise ValueError("No data was supplied to the classifier.")

        return cls(
            values=data,
            labels=list(labels or []),
            titles=list(titles) if titles is not None else None,
            source="raw",
        )

    def row_count(self) -> int:
        """Return the number of rows represented by the standardized input.

        Returns:
            The number of rows represented by the standardized input.
        """
        if self.matrix is not None:
            return self.matrix.shape[0]
        if self.docs is not None:
            return len(self.docs)
        if isinstance(self.values, pd.DataFrame):
            return len(self.values)
        if isinstance(self.values, (list, tuple)):
            return len(self.values)
        return len(self.labels)

    @staticmethod
    def _as_text_for_doc(doc: Any) -> str:
        """Convert a document-like object to text while preserving original tokenization.

        Args:
            doc: The document-like object to convert to text. Can be a spaCy Doc, an object with a `text` attribute, or a sequence of items.

        Returns:
            The text representation of the document-like object.
        """
        if doc is None:
            return ""

        module_name = type(doc).__module__ if doc is not None else ""
        if (
            module_name.startswith("spacy.")
            and hasattr(doc, "vocab")
            and hasattr(doc, "__iter__")
        ):
            return " ".join(token.text for token in doc)
        if hasattr(doc, "text") and not (
            hasattr(doc, "vocab") and hasattr(doc, "__iter__")
        ):
            return str(doc.text)
        if isinstance(doc, (list, tuple, set)):
            return " ".join(str(item) for item in doc)
        return str(doc)

    def as_texts(self) -> list[str]:
        """Return the data as plain text strings when possible.

        Returns:
            A list of plain text strings representing the data.
        """
        if self.docs is not None:
            texts: list[str] = []
            for doc in self.docs:
                texts.append(self._as_text_for_doc(doc))
            return texts

        if isinstance(self.values, pd.DataFrame):
            return [str(item) for item in self.values.to_dict(orient="records")]

        if isinstance(self.values, (list, tuple)):
            texts: list[str] = []
            for item in self.values:
                if _is_spacy_doc(item):
                    texts.append(" ".join(token.text for token in item))
                else:
                    texts.append(str(item))
            return texts

        return [str(self.values)]

    def subset(self, indices: Sequence[int]) -> "ClassifierData":
        """Return a new data object containing only the selected row indices.

        Args:
            indices: A sequence of row indices to include in the subset.

        Returns:
            A new `ClassifierData` object containing only the selected rows.
        """
        idx_list = list(indices)
        selected_titles = (
            [self.titles[i] for i in idx_list] if self.titles is not None else None
        )
        if self.matrix is not None:
            return ClassifierData(
                values=self.matrix[idx_list],
                labels=[self.labels[i] for i in idx_list],
                docs=[self.docs[i] for i in idx_list]
                if self.docs is not None
                else None,
                titles=selected_titles,
                matrix=self.matrix[idx_list],
                source=self.source,
            )

        if isinstance(self.values, pd.DataFrame):
            data = self.values.iloc[idx_list]
            return ClassifierData(
                values=data,
                labels=[self.labels[i] for i in idx_list],
                titles=selected_titles,
                source=self.source,
            )

        sliced = [self.values[i] for i in idx_list]
        return ClassifierData(
            values=sliced,
            labels=[self.labels[i] for i in idx_list],
            titles=selected_titles,
            source=self.source,
        )

    @staticmethod
    def _test_count_for_group(group_size: int, test_size: float) -> int:
        """Compute the number of rows to reserve for testing within one label group.

        Args:
            group_size: The number of rows in the label group.
            test_size: The proportion of the group to reserve for testing.

        Returns:
            The number of rows to reserve for testing within the group.
        """
        if group_size <= 1:
            return 0

        count = int(round(group_size * test_size))
        if count >= group_size:
            count = group_size - 1
        if count <= 0:
            count = 1
        return count

    @staticmethod
    def _dev_count_for_group(group_size: int, dev_size: float) -> int:
        """Compute the number of rows to reserve for development within a train split.

        Args:
            group_size: The number of rows in the train split.
            dev_size: The proportion of the train split to reserve for development.

        Returns:
            The number of rows to reserve for development within the train split.
        """
        if group_size <= 1:
            return 0

        count = int(round(group_size * dev_size))
        if count >= group_size:
            count = max(0, group_size - 1)
        if count <= 0:
            count = 1 if group_size > 1 else 0
        return count

    def _group_label_indices(self) -> dict[str, list[int]]:
        """Group row indices by label value for stratified splitting.

        Returns:
            A dictionary mapping each label value to a list of row indices that have that label.
        """
        grouped: dict[str, list[int]] = defaultdict(list)
        for idx, label in enumerate(self.labels):
            grouped[str(label)].append(idx)
        return grouped

    def _split_indices_by_labels(
        self,
        test_size: float,
        dev_size: float | None,
        random_state: int,
    ) -> dict[str, list[int]]:
        """Split row indices while preserving each class distribution.

        Args:
            test_size: The proportion of the dataset to reserve for testing.
            dev_size: The proportion of the training set to reserve for development, or None if no development set is needed.
            random_state: The seed for the random number generator to ensure reproducibility.

        Returns:
            A dictionary containing the split row indices with keys "train", "test", and optionally "dev".
        """
        grouped = self._group_label_indices()
        train_indices: list[int] = []
        test_indices: list[int] = []
        rng = random.Random(random_state)

        for label_indices in grouped.values():
            rng.shuffle(label_indices)
            label_test_count = self._test_count_for_group(len(label_indices), test_size)
            test_indices.extend(label_indices[:label_test_count])
            train_indices.extend(label_indices[label_test_count:])

        rng.shuffle(train_indices)
        rng.shuffle(test_indices)

        result = {"train": train_indices, "test": test_indices}
        if dev_size is not None:
            if len(train_indices) <= 1:
                result["dev"] = []
            else:
                dev_count = self._dev_count_for_group(len(train_indices), dev_size)
                dev_indices = train_indices[:dev_count]
                result["dev"] = dev_indices
                result["train"] = [
                    idx for idx in train_indices if idx not in set(dev_indices)
                ]
        return result

    @staticmethod
    def _validate_split_parameters(
        n_rows: int, test_size: float, dev_size: float | None
    ) -> None:
        """Validate dataset split parameters before partitioning.

        Args:
            n_rows: The total number of rows in the dataset.
            test_size: The proportion of the dataset to reserve for testing.
            dev_size: The proportion of the training set to reserve for development, or None if no development set is needed.

        Raises:
            ValueError: If any of the split parameters are invalid.
        """
        if n_rows == 0:
            raise ValueError("Cannot split an empty dataset.")
        if test_size <= 0 or test_size >= 1:
            raise ValueError("test_size must be between 0 and 1.")
        if dev_size is not None and (dev_size <= 0 or dev_size >= 1):
            raise ValueError("dev_size must be between 0 and 1.")

    @staticmethod
    def _split_random_indices(
        n_rows: int, test_size: float, random_state: int
    ) -> dict[str, list[int]]:
        """Create a simple non-stratified random split.

        Args:
            n_rows: The total number of rows in the dataset.
            test_size: The proportion of the dataset to reserve for testing.
            random_state: The seed for the random number generator to ensure reproducibility.

        Returns:
            A dictionary containing the split row indices with keys "train" and "test".
        """
        rng = random.Random(random_state)
        indices = list(range(n_rows))
        rng.shuffle(indices)

        test_count = int(round(n_rows * test_size))
        if n_rows > 1 and test_count >= n_rows:
            test_count = n_rows - 1
        if n_rows > 1 and test_count <= 0:
            test_count = 1
        if n_rows <= 1:
            test_count = 0

        test_indices = set(indices[:test_count])
        train_indices = [idx for idx in indices if idx not in test_indices]
        return {"train": train_indices, "test": list(test_indices)}

    def split(
        self,
        test_size: float = 0.2,
        dev_size: float | None = None,
        random_state: int = 42,
        stratify: bool = True,
    ) -> dict[str, "ClassifierData"]:
        """Split a standardized dataset into train/test/dev partitions.

        Args:
            test_size: The proportion of the dataset to reserve for testing.
            dev_size: The proportion of the training set to reserve for development, or None if no development set is needed.
            random_state: The seed for the random number generator to ensure reproducibility.
            stratify: Whether to perform a stratified split based on the labels.

        Returns:
            A dictionary containing the split datasets with keys "train", "test", and optionally "dev".
        """
        n_rows = self.row_count()
        self._validate_split_parameters(n_rows, test_size, dev_size)

        if stratify and self.labels:
            split = self._split_indices_by_labels(
                test_size=test_size,
                dev_size=dev_size,
                random_state=random_state,
            )
        else:
            split = self._split_random_indices(n_rows, test_size, random_state)

        train_result = self.subset(split["train"])
        test_result = self.subset(split["test"])

        result: dict[str, ClassifierData] = {
            "train": train_result,
            "test": test_result,
        }

        if dev_size is not None and "dev" in split:
            result["dev"] = self.subset(split["dev"])

        return result


class Classifier(BaseModel):
    """High-level classification orchestration for non-technical users.

    The `Classifier` class is intentionally backend-agnostic. Users supply data,
    labels, and a pipeline object that implements the backend-specific training and
    inference logic. This keeps the public API stable across SpaCy, scikit-learn,
    and other classification methods.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: Any | None = Field(default=None, description="Training or prediction data.")
    labels: Sequence[Any] = Field(
        default_factory=list, description="Classification labels."
    )
    titles: Sequence[Any] = Field(
        default_factory=list, description="Document titles aligned with the data rows."
    )
    pipeline: BaseClassificationPipeline | None = Field(
        default=None,
        description="Classification backend strategy; e.g. a spaCy or scikit-learn pipeline.",
    )

    _fitted: bool = PrivateAttr(default=False)
    _last_fit_data: Any = PrivateAttr(default=None)
    _last_fit_labels: list[Any] = PrivateAttr(default_factory=list)

    @model_validator(mode="after")
    def _validate_pipeline(self) -> "Classifier":
        """Validate that the pipeline is correctly set and an instance of BaseClassificationPipeline.

        Returns:
            The validated Classifier instance.

        Raises:
            TypeError: If the pipeline is not an instance of BaseClassificationPipeline.
        """
        if self.pipeline is not None and not isinstance(
            self.pipeline, BaseClassificationPipeline
        ):
            raise TypeError(
                "pipeline must be an instance of BaseClassificationPipeline."
            )
        return self

    def _resolve_data_and_labels(
        self,
        data: Any | None = None,
        labels: Sequence[Any] | None = None,
        titles: Sequence[Any] | None = None,
    ) -> tuple[ClassifierData, list[Any]]:
        """Resolve the input data and labels, normalizing them into a ClassifierData instance.

        Args:
            data: The input data to resolve. If None, the stored data is used.
            labels: The input labels to resolve. If None, the stored labels are used.
            titles: Optional titles aligned with the input rows.

        Returns:
            A tuple containing the normalized ClassifierData instance and the corresponding list of labels.
        """
        resolved_data = self.data if data is None else data
        explicit_labels = list(labels) if labels is not None else None
        if explicit_labels is None:
            explicit_labels = list(self.labels) if self.labels else None

        explicit_titles = list(titles) if titles is not None else None
        normalized = ClassifierData.from_input(
            resolved_data,
            explicit_labels,
            titles=explicit_titles,
        )

        if explicit_titles is None and self.titles:
            if len(self.titles) == normalized.row_count():
                normalized.titles = list(self.titles)

        return normalized, normalized.labels

    def fit(
        self, data: Any | None = None, labels: Sequence[Any] | None = None
    ) -> "Classifier":
        """Fit the classifier using the supplied pipeline backend.

        Args:
            data: The input data to fit the classifier on. If None, the stored data is used.
            labels: The corresponding labels for the input data. If None, the stored labels are used.

        Returns:
            The fitted Classifier instance.
        """
        if self.pipeline is None:
            raise ValueError(
                "A classification pipeline must be configured before calling fit()."
            )

        resolved_data, resolved_labels = self._resolve_data_and_labels(data, labels)
        self.pipeline.fit(resolved_data, resolved_labels)
        self.data = resolved_data.values
        self.labels = resolved_labels
        self.titles = (
            list(resolved_data.titles) if resolved_data.titles is not None else []
        )
        self._fitted = True
        self._last_fit_data = resolved_data
        self._last_fit_labels = resolved_labels
        return self

    def predict(self, data: Any | None = None) -> list[str]:
        """Predict labels for the supplied data or for the stored training data.

        Args:
            data: The input data to predict labels for. If None, the stored data is used.

        Returns:
            A list of predicted labels for each document.
        """
        if self.pipeline is None:
            raise ValueError(
                "A classification pipeline must be configured before calling predict()."
            )

        if data is None:
            if self.data is None:
                raise ValueError(
                    "No prediction data was supplied and no fitted data is available."
                )
            data = self.data

        predictions = self.pipeline.predict(data)
        return [
            value if isinstance(value, (list, tuple, set)) else str(value)
            for value in predictions
        ]

    def predict_scores(self, data: Any | None = None) -> list[dict[str, float]]:
        """Return the underlying confidence scores for each prediction when available.

        Args:
            data: The input data to predict scores for. If None, the stored data is used.

        Returns:
            A list of dictionaries containing confidence scores for each prediction.
        """
        if self.pipeline is None:
            raise ValueError(
                "A classification pipeline must be configured before calling predict_scores()."
            )

        if data is None:
            if self.data is None:
                raise ValueError(
                    "No prediction data was supplied and no fitted data is available."
                )
            data = self.data

        if not hasattr(self.pipeline, "predict_scores"):
            raise NotImplementedError(
                f"{type(self.pipeline).__name__} does not implement predict_scores()."
            )

        scores = self.pipeline.predict_scores(data)
        # Sort the scores
        for row in scores:
            row.update(
                dict(sorted(row.items(), key=lambda item: item[1], reverse=True))
            )
        return scores

    def evaluate(
        self, data: Any | None = None, labels: Sequence[Any] | None = None
    ) -> dict[str, float]:
        """Evaluate the fitted pipeline on the supplied data and labels.

        Args:
            data: The input data to evaluate the pipeline on. If None, the stored data is used.
            labels: The corresponding labels for the input data. If None, the stored labels are used.

        Returns:
            A dictionary containing evaluation metrics for the predictions.
        """
        if self.pipeline is None:
            raise ValueError(
                "A classification pipeline must be configured before calling evaluate()."
            )

        if data is None:
            if self.data is None:
                raise ValueError(
                    "No evaluation data was supplied and no fitted data is available."
                )
            data = self.data
        if labels is None:
            if not self.labels:
                raise ValueError("No labels were provided for evaluation.")
            labels = self.labels

        return self.pipeline.evaluate(data, list(labels))

    def split_data(
        self,
        data: Any | None = None,
        labels: Sequence[str] | None = None,
        titles: Sequence[Any] | None = None,
        test_size: float = 0.2,
        dev_size: float | None = None,
        random_state: int = 42,
        stratify: bool = True,
    ) -> dict[str, Any]:
        """Split data into train/test/dev partitions.

        Args:
            data: data to split; defaults to the classifier's stored data.
            labels: labels aligned to the data; defaults to the classifier's labels.
            titles: optional titles aligned to the rows; preserved in the output.
            test_size: fraction of the data reserved for testing.
            dev_size: optional fraction reserved for development / validation.
            random_state: deterministic random seed.
            stratify: whether to preserve label distributions across splits.

        Returns:
            Dictionary containing the partitions keyed by `train`, `test`, and optional
            `dev` data plus the corresponding labels and titles.
        """
        resolved_data, resolved_labels = self._resolve_data_and_labels(
            data,
            labels,
            titles=titles,
        )

        split = resolved_data.split(
            test_size=test_size,
            dev_size=dev_size,
            random_state=random_state,
            stratify=stratify,
        )

        result = {
            "train": {
                "data": split["train"].values,
                "labels": split["train"].labels,
                "titles": split["train"].titles,
            },
            "test": {
                "data": split["test"].values,
                "labels": split["test"].labels,
                "titles": split["test"].titles,
            },
        }
        if "dev" in split:
            result["dev"] = {
                "data": split["dev"].values,
                "labels": split["dev"].labels,
                "titles": split["dev"].titles,
            }
        return result

    def train_test_split(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
        titles: Sequence[Any] | None = None,
    ) -> dict[str, Any]:
        """Convenience wrapper for train/test splitting.

        Args:
            test_size: fraction of the data reserved for testing.
            random_state: deterministic random seed.
            stratify: whether to preserve label distributions across splits.
            titles: optional titles aligned with the rows; preserved in the output.

        Returns:
            Dictionary containing the train and test partitions keyed by `train` and `test` data plus the corresponding labels and titles.
        """
        return self.split_data(
            titles=titles,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

    def train_dev_split(
        self,
        dev_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
        titles: Sequence[Any] | None = None,
    ) -> dict[str, Any]:
        """Convenience wrapper for train/dev splitting.

        Args:
            dev_size: fraction of the data reserved for development / validation.
            random_state: deterministic random seed.
            stratify: whether to preserve label distributions across splits.
            titles: optional titles aligned with the rows; preserved in the output.

        Returns:
            Dictionary containing the train and dev partitions keyed by `train` and `dev` data plus the corresponding labels and titles.
        """
        return self.split_data(
            titles=titles,
            dev_size=dev_size,
            random_state=random_state,
            stratify=stratify,
        )

    def __call__(self, data: Any) -> list[str]:
        """Convenience wrapper for prediction calls.

        Args:
            data: The input data to make predictions on.

        Returns:
            A list of predicted labels for the input data.
        """
        return self.predict(data)


__all__ = ["BaseClassificationPipeline", "Classifier"]
