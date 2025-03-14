"""
LogGuardian: A Python-based log anomaly detection system leveraging large language models.

This system combines semantic extraction via BERT with LLM-based classification to detect
anomalies in system logs.
"""

__version__ = "0.1.0"

from logguardian.pipeline import LogGuardian

__all__ = ["LogGuardian"]