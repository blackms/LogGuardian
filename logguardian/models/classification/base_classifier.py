"""
Base abstract class for log sequence classifiers.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple

import numpy as np
import torch


class BaseLogClassifier(ABC):
    """
    Abstract base class for all log sequence classifiers.
    
    This class defines the interface that all log sequence classifiers must implement.
    Classifiers are responsible for determining whether a sequence of log messages
    contains anomalies.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the classifier.
        
        Args:
            config: Optional configuration dictionary for the classifier
        """
        self.config = config or {}
    
    @abstractmethod
    def classify(
        self, 
        log_sequence: Union[List[str], List[torch.Tensor], torch.Tensor],
        raw_output: bool = False
    ) -> Union[int, float, Dict[str, Any]]:
        """
        Classify a single log sequence.
        
        Args:
            log_sequence: Sequence of log messages or embeddings
            raw_output: Whether to return raw model outputs
            
        Returns:
            Classification result (1 for anomaly, 0 for normal)
            If raw_output is True, returns a dictionary with detailed outputs
        """
        pass
    
    def classify_batch(
        self, 
        log_sequences: Union[List[List[str]], List[List[torch.Tensor]], List[torch.Tensor]],
        raw_output: bool = False
    ) -> Union[List[int], List[float], List[Dict[str, Any]]]:
        """
        Classify a batch of log sequences.
        
        Args:
            log_sequences: Batch of log sequences
            raw_output: Whether to return raw model outputs
            
        Returns:
            List of classification results
        """
        return [self.classify(seq, raw_output=raw_output) for seq in log_sequences]
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the classifier to a file.
        
        Args:
            path: Path to save the classifier to
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: str, **kwargs) -> 'BaseLogClassifier':
        """
        Load a classifier from a file.
        
        Args:
            path: Path to load the classifier from
            
        Returns:
            Loaded classifier
        """
        pass