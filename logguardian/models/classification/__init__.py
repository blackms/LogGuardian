"""
Classification modules for anomaly detection.

This package contains modules for classifying log sequences as normal or anomalous.
"""

from logguardian.models.classification.base_classifier import BaseLogClassifier
from logguardian.models.classification.llm_classifier import LlamaLogClassifier

__all__ = ["BaseLogClassifier", "LlamaLogClassifier"]