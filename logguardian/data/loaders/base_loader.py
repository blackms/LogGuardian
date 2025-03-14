"""
Base abstract class for dataset loaders.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union, Iterator

from loguru import logger

from logguardian.data.preprocessors.base_preprocessor import BaseLogPreprocessor


class BaseDataLoader(ABC):
    """
    Abstract base class for all dataset loaders.
    
    This class defines the interface that all dataset loaders must implement.
    DataLoaders are responsible for loading and parsing log datasets into a format
    that can be used for training and evaluation.
    """
    
    def __init__(
        self,
        preprocessor: Optional[BaseLogPreprocessor] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the data loader.
        
        Args:
            preprocessor: Optional preprocessor to use for preprocessing logs
            config: Optional configuration dictionary for the loader
        """
        self.preprocessor = preprocessor
        self.config = config or {}
        self.data_path = self.config.get("data_path", "")
        
    @abstractmethod
    def load(self, path: Optional[str] = None) -> Tuple[List[str], List[int]]:
        """
        Load and parse a log dataset.
        
        Args:
            path: Path to the dataset. If None, use the path from config
            
        Returns:
            Tuple of (log_messages, labels) where:
                - log_messages is a list of log messages
                - labels is a list of labels (1 for anomaly, 0 for normal)
        """
        pass
    
    def load_batch(
        self, 
        path: Optional[str] = None, 
        batch_size: int = 1000
    ) -> Iterator[Tuple[List[str], List[int]]]:
        """
        Load and parse a log dataset in batches.
        
        Args:
            path: Path to the dataset. If None, use the path from config
            batch_size: Size of each batch
            
        Returns:
            Iterator yielding tuples of (log_messages, labels) for each batch
        """
        # Default implementation loads the whole dataset and yields batches
        # Subclasses should override this if they can load in a more memory-efficient way
        all_logs, all_labels = self.load(path)
        
        for i in range(0, len(all_logs), batch_size):
            yield all_logs[i:i+batch_size], all_labels[i:i+batch_size]
    
    def preprocess(self, logs: List[str]) -> List[str]:
        """
        Preprocess a list of log messages.
        
        Args:
            logs: List of log messages to preprocess
            
        Returns:
            List of preprocessed log messages
        """
        if self.preprocessor is None:
            logger.warning("No preprocessor set, returning raw logs")
            return logs
        
        return self.preprocessor.preprocess_batch(logs)
    
    def get_train_test_split(
        self, 
        path: Optional[str] = None, 
        test_size: float = 0.2, 
        shuffle: bool = True,
        random_state: Optional[int] = None
    ) -> Tuple[List[str], List[int], List[str], List[int]]:
        """
        Load dataset and split into train and test sets.
        
        Args:
            path: Path to the dataset. If None, use the path from config
            test_size: Proportion of the dataset to use for testing
            shuffle: Whether to shuffle the data before splitting
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_logs, train_labels, test_logs, test_labels)
        """
        from sklearn.model_selection import train_test_split
        
        logs, labels = self.load(path)
        
        train_logs, test_logs, train_labels, test_labels = train_test_split(
            logs, labels, test_size=test_size, shuffle=shuffle, random_state=random_state
        )
        
        return train_logs, train_labels, test_logs, test_labels