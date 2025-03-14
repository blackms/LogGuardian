"""
PyTorch Dataset class for log anomaly detection.
"""
from typing import List, Dict, Any, Optional, Tuple, Union, Callable

import numpy as np
import torch
from torch.utils.data import Dataset
from loguru import logger


class LogDataset(Dataset):
    """
    PyTorch Dataset for log sequences.
    
    This dataset wraps log data and provides an interface compatible with
    PyTorch's DataLoader for efficient batching and iteration.
    """
    
    def __init__(
        self,
        logs: List[str],
        labels: Optional[List[int]] = None,
        tokenizer=None,
        max_length: int = 128,
        window_size: int = 10,
        stride: int = 5,
        transform: Optional[Callable] = None
    ):
        """
        Initialize the log dataset.
        
        Args:
            logs: List of preprocessed log messages
            labels: List of labels (1 for anomaly, 0 for normal)
            tokenizer: Tokenizer to convert logs to token IDs
            max_length: Maximum length of tokenized sequences
            window_size: Size of sliding windows for sequence creation
            stride: Stride of sliding windows
            transform: Optional transform to apply to the data
        """
        self.logs = logs
        self.labels = labels if labels is not None else [0] * len(logs)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        
        # Create sequences by sliding a window over the logs
        self.sequences, self.sequence_labels = self._create_sequences()
        
        logger.info(f"Created LogDataset with {len(self.sequences)} sequences")
        
    def _create_sequences(self) -> Tuple[List[List[str]], List[int]]:
        """
        Create sequences by sliding a window over the logs.
        
        Returns:
            Tuple of (sequences, sequence_labels) where:
                - sequences is a list of log sequences
                - sequence_labels is a list of sequence labels
        """
        sequences = []
        sequence_labels = []
        
        for i in range(0, len(self.logs) - self.window_size + 1, self.stride):
            sequence = self.logs[i:i+self.window_size]
            # A sequence is anomalous if any log in it is anomalous
            label = 1 if any(self.labels[i:i+self.window_size]) else 0
            
            sequences.append(sequence)
            sequence_labels.append(label)
        
        return sequences, sequence_labels
    
    def __len__(self) -> int:
        """
        Get the number of sequences in the dataset.
        
        Returns:
            Number of sequences
        """
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sequence by index.
        
        Args:
            idx: Index of the sequence
            
        Returns:
            Dictionary with sequence data and label
        """
        sequence = self.sequences[idx]
        label = self.sequence_labels[idx]
        
        # If tokenizer is provided, tokenize the sequence
        if self.tokenizer is not None:
            # Join logs with a separator for tokenization
            sequence_text = " [SEP] ".join(sequence)
            
            # Tokenize the sequence
            tokens = self.tokenizer(
                sequence_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Convert to dict and remove batch dimension
            item = {k: v.squeeze(0) for k, v in tokens.items()}
            item["label"] = torch.tensor(label, dtype=torch.long)
            
        else:
            # If no tokenizer, just return the raw sequence
            item = {
                "sequence": sequence,
                "label": label
            }
        
        # Apply transform if provided
        if self.transform is not None:
            item = self.transform(item)
            
        return item
    
    @classmethod
    def from_loader(
        cls,
        loader,
        path: Optional[str] = None,
        tokenizer=None,
        max_length: int = 128,
        window_size: int = 10,
        stride: int = 5,
        preprocess: bool = True,
        transform: Optional[Callable] = None
    ) -> 'LogDataset':
        """
        Create a dataset from a data loader.
        
        Args:
            loader: DataLoader instance
            path: Path to the dataset
            tokenizer: Tokenizer to convert logs to token IDs
            max_length: Maximum length of tokenized sequences
            window_size: Size of sliding windows for sequence creation
            stride: Stride of sliding windows
            preprocess: Whether to preprocess the logs
            transform: Optional transform to apply to the data
            
        Returns:
            LogDataset instance
        """
        logs, labels = loader.load(path)
        
        if preprocess and loader.preprocessor is not None:
            logs = loader.preprocessor.preprocess_batch(logs)
        
        return cls(
            logs=logs,
            labels=labels,
            tokenizer=tokenizer,
            max_length=max_length,
            window_size=window_size,
            stride=stride,
            transform=transform
        )