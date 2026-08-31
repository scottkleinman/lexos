"""Public classification API for Lexos.

This package exposes the new backend-agnostic `Classifier` interface while
preserving compatibility with older Lexos imports such as `Pipeline` and
`MLPPipeline`.
"""

from lexos.classification.classifier import (
    BaseClassificationPipeline,
    Classifier,
    ClassifierData,
)
from lexos.classification.mlp_pipeline import MLPPipeline
from lexos.classification.sklearn_pipeline import SklearnClassifierPipeline
from lexos.classification.spacy_pipeline import SpaCyTextCategorizerPipeline
from lexos.classification.util import PredictionSaver, save_predictions

Pipeline = BaseClassificationPipeline

__all__ = [
    "BaseClassificationPipeline",
    "Classifier",
    "ClassifierData",
    "Pipeline",
    "SpaCyTextCategorizerPipeline",
    "SklearnClassifierPipeline",
    "MLPPipeline",
    "PredictionSaver",
    "save_predictions",
]
