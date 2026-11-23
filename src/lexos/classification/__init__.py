"""__init__.py.

Last Updated: August 30, 2026
Last Tested: August 30, 2026
"""

from lexos.classification.classifier import (
    BaseClassificationPipeline,
    Classifier,
    ClassifierData,
)
from lexos.classification.mlp_pipeline import MLPClassifierPipeline, MLPPipeline
from lexos.classification.sklearn_pipeline import SklearnClassifierPipeline
from lexos.classification.spacy_pipeline import SpaCyTextCategorizerPipeline

Pipeline = BaseClassificationPipeline

__all__ = [
    "BaseClassificationPipeline",
    "Classifier",
    "ClassifierData",
    "MLPClassifierPipeline",
    "MLPPipeline",
    "Pipeline",
    "SpaCyTextCategorizerPipeline",
    "SklearnClassifierPipeline",
]
