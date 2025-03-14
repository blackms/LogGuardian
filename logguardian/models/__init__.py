"""
Model implementations for LogGuardian.

This package contains modules for:
1. Semantic feature extraction using BERT
2. Embedding alignment between different model spaces
3. Classification using LLM models
"""

from logguardian.models.feature_extraction import BertFeatureExtractor
from logguardian.models.embedding_alignment import EmbeddingProjector

__all__ = ["BertFeatureExtractor", "EmbeddingProjector"]