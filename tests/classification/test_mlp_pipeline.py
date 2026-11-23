"""test_mlp_pipeline.py.

Coverage target: core MLP pipeline compatibility paths.
"""

import pandas as pd
from sklearn.neural_network import MLPClassifier

from lexos.classification import MLPClassifierPipeline, MLPPipeline


def test_mlp_pipeline_uses_mlp_estimator_by_default() -> None:
    """Default estimator should be sklearn MLPClassifier."""
    pipeline = MLPClassifierPipeline(max_iter=200)
    assert isinstance(pipeline.estimator, MLPClassifier)
    assert pipeline.model is pipeline.estimator


def test_mlp_pipeline_alias_and_fit_predict_round_trip() -> None:
    """Legacy MLPPipeline alias should fit and predict through current backend API."""
    pipeline = MLPPipeline(max_iter=200)
    texts = [
        "history literature",
        "history archive",
        "finance markets",
        "finance report",
    ]
    labels = ["history", "history", "finance", "finance"]

    pipeline.fit(texts, labels)
    predictions = pipeline.predict(texts)
    scores = pipeline.predict_scores(texts)

    assert len(predictions) == len(texts)
    assert all(isinstance(value, str) for value in predictions)
    assert len(scores) == len(texts)
    assert all(isinstance(row, dict) for row in scores)


def test_mlp_pipeline_execute_training_with_holdout_and_cv() -> None:
    """execute_training should return final, holdout, and CV metrics when enabled."""
    pipeline = MLPClassifierPipeline(
        max_iter=200,
        run_holdout=True,
        run_cv=True,
        cv_splits=2,
        test_size=0.5,
    )
    texts = [
        "history literature",
        "history archive",
        "finance market",
        "finance report",
    ]
    labels = ["history", "history", "finance", "finance"]

    result = pipeline.execute_training(texts, labels)

    assert "final_model" in result
    assert result["holdout_metrics"] is not None
    assert not result["cv_fold_metrics"].empty
    assert result["cv_mean_metrics"] is not None
    assert "accuracy" in result["final_metrics"]


def test_mlp_pipeline_execute_training_with_dataframe_and_active_features() -> None:
    """execute_training should support DataFrame inputs and active feature filtering."""
    pipeline = MLPClassifierPipeline(max_iter=200, run_holdout=False, run_cv=False)
    data = pd.DataFrame(
        {
            "feature_1": [1.0, 0.9, 0.1, 0.2],
            "feature_2": [0.0, 0.1, 1.0, 0.8],
            "feature_3": [0.2, 0.3, 0.7, 0.6],
        }
    )
    labels = ["A", "A", "B", "B"]

    result = pipeline.execute_training(data, labels, active_features=["feature_1"])

    assert result["holdout_metrics"] is None
    assert result["cv_mean_metrics"] is None
    assert "macro_f1" in result["final_metrics"]
