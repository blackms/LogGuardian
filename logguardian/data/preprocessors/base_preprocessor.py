"""
Base abstract class for log preprocessors.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any


class BaseLogPreprocessor(ABC):
    """
    Abstract base class for all log preprocessors.
    
    This class defines the interface that all log preprocessors must implement.
    Log preprocessors are responsible for cleaning and standardizing logs before
    they are passed to the feature extraction stage.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the log preprocessor.
        
        Args:
            config: Optional configuration dictionary for the preprocessor
        """
        self.config = config or {}
    
    @abstractmethod
    def preprocess(self, log_message: str) -> str:
        """
        Preprocess a single log message.
        
        Args:
            log_message: The raw log message to preprocess
            
        Returns:
            The preprocessed log message
        """
        pass
    
    def preprocess_batch(self, log_messages: List[str]) -> List[str]:
        """
        Preprocess a batch of log messages.
        
        Args:
            log_messages: List of raw log messages to preprocess
            
        Returns:
            List of preprocessed log messages
        """
        return [self.preprocess(log) for log in log_messages]
    
    def fit(self, log_messages: List[str]) -> 'BaseLogPreprocessor':
        """
        Fit the preprocessor on a dataset of log messages.
        
        This method can be overridden by subclasses to learn preprocessing parameters
        from a dataset of log messages.
        
        Args:
            log_messages: List of log messages to fit on
            
        Returns:
            self
        """
        return self
    
    def save(self, path: str) -> None:
        """
        Save the preprocessor to a file.
        
        Args:
            path: Path to save the preprocessor to
        """
        pass
    
    @classmethod
    def load(cls, path: str) -> 'BaseLogPreprocessor':
        """
        Load a preprocessor from a file.
        
        Args:
            path: Path to load the preprocessor from
            
        Returns:
            Loaded preprocessor
        """
        pass