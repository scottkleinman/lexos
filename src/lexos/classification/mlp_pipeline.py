"""mlp_pipeline.py.

Compatibility MLP classification pipeline built on the Lexos sklearn backend.

Last Updated: November 23, 2026
"""

from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from pydantic import Field

from lexos.classification.sklearn_pipeline import SklearnClassifierPipeline
from lexos.dtm import DTM
from lexos.tokenizer import WhitespaceTokenizer

try:  # pragma: no cover - optional dependency
    from imblearn.over_sampling import SMOTE
except ImportError:  # pragma: no cover - handled at runtime
    SMOTE = None

try:  # pragma: no cover - optional dependency
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
except ImportError:  # pragma: no cover - handled at runtime
    accuracy_score = None
    balanced_accuracy_score = None
    classification_report = None
    confusion_matrix = None
    f1_score = None
    StratifiedKFold = None
    train_test_split = None
    MLPClassifier = None
    StandardScaler = None


def _to_dense(matrix: Any) -> np.ndarray:
    """Return a dense array from a sparse or dense matrix-like object."""
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


class MLPClassifierPipeline(SklearnClassifierPipeline):
    """A scikit-learn MLP-backed classification pipeline."""

    hidden_layer_sizes: tuple[int, ...] = Field(
        default=(64,), description="Hidden layer sizes for the MLP estimator."
    )
    activation: Literal["identity", "logistic", "tanh", "relu"] = Field(
        default="relu", description="Activation function for hidden layers."
    )
    solver: Literal["lbfgs", "sgd", "adam"] = Field(
        default="adam", description="Weight optimization solver."
    )
    alpha: float = Field(
        default=1e-4, description="L2 regularization term for the MLP."
    )
    learning_rate_init: float = Field(
        default=1e-3, description="Initial learning rate for MLP optimization."
    )
    random_state: int = Field(
        default=42, description="Random seed used by the default MLP estimator."
    )
    min_df: int | float = Field(
        default=1, description="Minimum document frequency used when building DTMs."
    )
    include_bigrams: bool = Field(
        default=True, description="Whether to include adjacent token bigrams."
    )
    use_smote: bool = Field(
        default=False,
        description="Whether to apply SMOTE oversampling during training/evaluation.",
    )
    test_size: float = Field(
        default=0.2, description="Holdout split ratio used by execute_training()."
    )
    cv_splits: int = Field(
        default=5, description="Number of stratified folds used by execute_training()."
    )
    run_holdout: bool = Field(
        default=False,
        description="Whether execute_training() should compute holdout metrics.",
    )
    run_cv: bool = Field(
        default=False,
        description="Whether execute_training() should compute cross-validation metrics.",
    )

    def __init__(self, **data: Any) -> None:
        """Initialize an MLP pipeline, using MLPClassifier unless overridden."""
        super().__init__(**data)
        if data.get("estimator") is None:
            if MLPClassifier is None:
                raise ImportError(
                    "scikit-learn is required for MLPClassifierPipeline."
                )
            self.estimator = MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                solver=self.solver,
                alpha=self.alpha,
                learning_rate_init=self.learning_rate_init,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )

    def _tokenize_items(self, items: Sequence[Any]) -> list[list[str]]:
        """Convert incoming items to token lists suitable for DTM construction."""
        tokenizer = WhitespaceTokenizer()
        token_lists: list[list[str]] = []
        for item in items:
            if isinstance(item, str):
                unigrams = [tok for tok in tokenizer(item.lower()) if tok.isalpha()]
            elif hasattr(item, "__iter__") and not hasattr(item, "shape"):
                unigrams = [str(tok).lower() for tok in item if str(tok).isalpha()]
            else:
                unigrams = [str(item).lower()] if str(item).isalpha() else []

            if self.include_bigrams:
                bigrams = [f"{a}_{b}" for a, b in zip(unigrams, unigrams[1:])]
                token_lists.append(unigrams + bigrams)
            else:
                token_lists.append(unigrams)

        if any(len(tokens) == 0 for tokens in token_lists):
            raise ValueError(
                "At least one document produced zero tokens after preprocessing."
            )
        return token_lists

    def _normalize_active_features(self, active_features: Any | None) -> list[str] | None:
        """Normalize feature selectors to a list of feature names."""
        if active_features is None:
            return None
        if isinstance(active_features, pd.DataFrame):
            return active_features.columns.tolist()
        if isinstance(active_features, pd.Index):
            return active_features.tolist()
        if isinstance(active_features, str):
            return [active_features]
        return list(active_features)

    def _filter_active_features(
        self, baseline_features: list[str], matrix: Any, active_features: Any | None
    ) -> Any:
        """Filter matrix columns down to selected active features."""
        selected = self._normalize_active_features(active_features)
        if selected is None:
            return matrix

        if isinstance(matrix, pd.DataFrame):
            missing = [feature for feature in selected if feature not in matrix.columns]
            if missing:
                raise ValueError(
                    "Requested active features are not present in the baseline feature set: "
                    f"{missing}"
                )
            return matrix.loc[:, selected]

        index_map = {feature: idx for idx, feature in enumerate(baseline_features)}
        valid_indices = [index_map[feature] for feature in selected if feature in index_map]
        if isinstance(matrix, np.ndarray):
            return matrix[:, valid_indices]
        return matrix.tocsr()[:, valid_indices]

    def _apply_smote(self, feature_matrix: Any, labels: np.ndarray) -> tuple[Any, np.ndarray]:
        """Apply SMOTE oversampling when enabled."""
        if not self.use_smote:
            return feature_matrix, labels
        if SMOTE is None:
            raise ImportError("SMOTE requested, but imbalanced-learn is not installed.")
        smote = SMOTE(random_state=self.random_state)
        return smote.fit_resample(_to_dense(feature_matrix), labels)

    def _validate_execute_training_dependencies(self) -> None:
        """Ensure sklearn evaluation components are importable."""
        if any(
            dep is None
            for dep in (
                accuracy_score,
                balanced_accuracy_score,
                classification_report,
                confusion_matrix,
                f1_score,
                StratifiedKFold,
                train_test_split,
                StandardScaler,
            )
        ):
            raise ImportError(
                "scikit-learn is required for execute_training() in MLPClassifierPipeline."
            )

    def execute_training(
        self,
        train_data: Sequence[Any] | pd.DataFrame,
        labels: Sequence[str],
        active_features: Any | None = None,
    ) -> dict[str, Any]:
        """Fit final model and optional holdout/CV metrics, mirroring legacy MLP output."""
        self._validate_execute_training_dependencies()
        y = np.asarray(labels)
        active_features = self._normalize_active_features(active_features)
        is_dataframe = isinstance(train_data, pd.DataFrame)
        positions = np.arange(len(train_data))

        holdout_metrics: dict[str, float] | None = None
        holdout_report = pd.DataFrame()
        holdout_confusion_matrix = pd.DataFrame()
        cv_fold_metrics = pd.DataFrame()
        cv_mean_metrics: dict[str, float] | None = None

        if self.run_holdout:
            train_pos, test_pos = train_test_split(
                positions,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y,
                shuffle=True,
            )
            y_train = y[train_pos]
            y_test = y[test_pos]

            if is_dataframe:
                x_train_raw = train_data.iloc[train_pos]
                x_test_raw = train_data.iloc[test_pos]
                holdout_baseline = train_data.columns.tolist()
            else:
                tokens_train = self._tokenize_items([train_data[i] for i in train_pos])
                tokens_test = self._tokenize_items([train_data[i] for i in test_pos])
                dtm_holdout = DTM()
                dtm_holdout(
                    docs=tokens_train,
                    labels=[f"train_doc_{i}" for i in train_pos],
                    min_df=self.min_df,
                )
                x_train_raw = dtm_holdout.doc_term_matrix
                x_test_raw = dtm_holdout.vectorizer.transform(tokens_test)
                holdout_baseline = dtm_holdout.sorted_terms_list

            x_train_raw = self._filter_active_features(
                holdout_baseline, x_train_raw, active_features
            )
            x_test_raw = self._filter_active_features(
                holdout_baseline, x_test_raw, active_features
            )

            scaler_holdout = StandardScaler(with_mean=False)
            x_train_scaled = scaler_holdout.fit_transform(x_train_raw)
            x_test_scaled = scaler_holdout.transform(x_test_raw)
            x_train_model, y_train_model = self._apply_smote(x_train_scaled, y_train)

            holdout_model = MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                solver=self.solver,
                alpha=self.alpha,
                learning_rate_init=self.learning_rate_init,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
            holdout_model.fit(_to_dense(x_train_model), y_train_model)
            y_pred = holdout_model.predict(_to_dense(x_test_scaled))

            holdout_metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
            }
            holdout_report = pd.DataFrame(
                classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            ).T

            class_labels = sorted(np.unique(y).tolist())
            cm = confusion_matrix(y_test, y_pred, labels=class_labels)
            holdout_confusion_matrix = pd.DataFrame(
                cm,
                index=[f"true_{label}" for label in class_labels],
                columns=[f"pred_{label}" for label in class_labels],
            )

        if self.run_cv:
            rows: list[dict[str, float]] = []
            cv = StratifiedKFold(
                n_splits=self.cv_splits,
                shuffle=True,
                random_state=self.random_state,
            )

            for fold, (tr_idx, va_idx) in enumerate(cv.split(positions, y), start=1):
                y_tr = y[tr_idx]
                y_va = y[va_idx]
                if is_dataframe:
                    x_tr_raw = train_data.iloc[tr_idx]
                    x_va_raw = train_data.iloc[va_idx]
                    cv_baseline = train_data.columns.tolist()
                else:
                    tokens_train = self._tokenize_items([train_data[i] for i in tr_idx])
                    tokens_valid = self._tokenize_items([train_data[i] for i in va_idx])
                    dtm_fold = DTM()
                    dtm_fold(
                        docs=tokens_train,
                        labels=[f"fold_doc_{i}" for i in tr_idx],
                        min_df=self.min_df,
                    )
                    x_tr_raw = dtm_fold.doc_term_matrix
                    x_va_raw = dtm_fold.vectorizer.transform(tokens_valid)
                    cv_baseline = dtm_fold.sorted_terms_list

                x_tr_raw = self._filter_active_features(
                    cv_baseline, x_tr_raw, active_features
                )
                x_va_raw = self._filter_active_features(
                    cv_baseline, x_va_raw, active_features
                )

                scaler = StandardScaler(with_mean=False)
                x_tr_scaled = scaler.fit_transform(x_tr_raw)
                x_va_scaled = scaler.transform(x_va_raw)
                x_tr_model, y_tr_model = self._apply_smote(x_tr_scaled, y_tr)

                fold_model = MLPClassifier(
                    hidden_layer_sizes=self.hidden_layer_sizes,
                    activation=self.activation,
                    solver=self.solver,
                    alpha=self.alpha,
                    learning_rate_init=self.learning_rate_init,
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                )
                fold_model.fit(_to_dense(x_tr_model), y_tr_model)
                fold_pred = fold_model.predict(_to_dense(x_va_scaled))

                rows.append(
                    {
                        "fold": float(fold),
                        "accuracy": float(accuracy_score(y_va, fold_pred)),
                        "balanced_accuracy": float(
                            balanced_accuracy_score(y_va, fold_pred)
                        ),
                        "macro_f1": float(f1_score(y_va, fold_pred, average="macro")),
                    }
                )

            cv_fold_metrics = pd.DataFrame(rows)
            cv_mean_metrics = {
                "accuracy": float(cv_fold_metrics["accuracy"].mean()),
                "balanced_accuracy": float(cv_fold_metrics["balanced_accuracy"].mean()),
                "macro_f1": float(cv_fold_metrics["macro_f1"].mean()),
            }

        if is_dataframe:
            x_full_raw = train_data
            final_dtm = None
            baseline_features = train_data.columns.tolist()
        else:
            tokens = self._tokenize_items(train_data)
            final_dtm = DTM()
            final_dtm(
                docs=tokens,
                labels=[f"train_doc_{i}" for i in range(len(tokens))],
                min_df=self.min_df,
            )
            x_full_raw = final_dtm.doc_term_matrix
            baseline_features = final_dtm.sorted_terms_list

        x_full_raw = self._filter_active_features(
            baseline_features, x_full_raw, active_features
        )

        scaler_final = StandardScaler(with_mean=False)
        x_full_scaled = scaler_final.fit_transform(x_full_raw)
        x_full_model, y_full_model = self._apply_smote(x_full_scaled, y)

        final_model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        final_model.fit(_to_dense(x_full_model), y_full_model)

        final_pred = final_model.predict(_to_dense(x_full_scaled))
        final_metrics = {
            "accuracy": float(accuracy_score(y, final_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, final_pred)),
            "macro_f1": float(f1_score(y, final_pred, average="macro")),
        }
        final_report = pd.DataFrame(
            classification_report(y, final_pred, output_dict=True, zero_division=0)
        ).T

        return {
            "final_model": final_model,
            "final_dtm": final_dtm,
            "final_scaler": scaler_final,
            "final_metrics": final_metrics,
            "final_report": final_report,
            "holdout_metrics": holdout_metrics,
            "holdout_report": holdout_report,
            "holdout_confusion_matrix": holdout_confusion_matrix,
            "cv_fold_metrics": cv_fold_metrics,
            "cv_mean_metrics": cv_mean_metrics,
        }


# Backward-compatible alias for older code that imported `MLPPipeline`.
MLPPipeline = MLPClassifierPipeline


__all__ = ["MLPClassifierPipeline", "MLPPipeline"]
