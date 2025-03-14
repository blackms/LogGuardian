"""
Loader for Liberty supercomputer logs dataset.
"""
import os
import re
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple, List
from loguru import logger

from logguardian.data.loaders.base_loader import BaseDataLoader


class LibertyLoader(BaseDataLoader):
    """
    Loader for the Liberty supercomputer logs dataset.
    
    The Liberty dataset contains logs collected from the Liberty supercomputer
    at Sandia National Labs (SNL) in Albuquerque. Each log message is labeled
    as either normal or anomalous.
    
    Liberty log format example:
    1151018935 2006.06.23 R32-M1-N2-C:J09-U11 2006-06-23-15.42.15.497585 R32-M1-N2-C:J09-U11 RAS KERNEL INFO generating core 14627
    """
    
    def __init__(
        self,
        preprocessor=None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Liberty loader.
        
        Args:
            preprocessor: Log preprocessor to apply to each log message
            config: Configuration parameters
        """
        super().__init__(preprocessor, config)
        
        # Default configuration
        self.data_path = self.config.get("data_path", "")
        self.window_size = self.config.get("window_size", 100)
        self.step_size = self.config.get("step_size", 100)
        self.label_type = self.config.get("label_type", "sliding_window")
        
        # Store dataset once loaded
        self.logs = None
        self.labels = None
        self.timestamps = None
    
    def _parse_liberty_logs(self, filepath: str) -> Tuple[List[str], List[int], List[float]]:
        """
        Parse Liberty logs from file.
        
        Args:
            filepath: Path to Liberty log file
            
        Returns:
            Tuple of (log_messages, labels, timestamps)
        """
        # Regular expression to extract Liberty log parts
        # Liberty logs have the same format as BGL logs
        liberty_regex = r'(\d+)\s+([\d\.]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+(\w+)\s+(\w+)\s+(\w+)\s+(.*)'
        
        logs = []
        labels = []
        timestamps = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    match = re.match(liberty_regex, line.strip())
                    if match:
                        timestamp = float(match.group(1))
                        date = match.group(2)
                        node = match.group(3)
                        time = match.group(4)
                        location = match.group(5)
                        severity = match.group(6)
                        component = match.group(7)
                        level = match.group(8)
                        content = match.group(9)
                        
                        # Construct log message (focusing on the content part)
                        log_message = content
                        
                        # Determine if log is anomalous based on severity
                        # In Liberty, "FATAL", "ERROR", "SEVERE", "WARNING" indicate anomalies
                        is_anomaly = 1 if severity in ["FATAL", "ERROR", "SEVERE", "WARNING"] else 0
                        
                        logs.append(log_message)
                        labels.append(is_anomaly)
                        timestamps.append(timestamp)
                except Exception as e:
                    logger.warning(f"Error parsing line: {line.strip()}, Error: {e}")
        
        logger.info(f"Parsed {len(logs)} log messages with {sum(labels)} anomalies")
        return logs, labels, timestamps
    
    def _create_sliding_windows(
        self,
        logs: List[str],
        labels: List[int],
        timestamps: List[float]
    ) -> Tuple[List[List[str]], List[int], List[float]]:
        """
        Create sliding windows from logs.
        
        A window is labeled as anomalous if it contains at least one anomalous log.
        
        Args:
            logs: List of log messages
            labels: List of labels (0 for normal, 1 for anomalous)
            timestamps: List of timestamps
            
        Returns:
            Tuple of (log_sequences, sequence_labels, sequence_timestamps)
        """
        log_sequences = []
        sequence_labels = []
        sequence_timestamps = []
        
        # Create sliding windows
        for i in range(0, len(logs) - self.window_size + 1, self.step_size):
            # Get window of logs and labels
            window_logs = logs[i:i+self.window_size]
            window_labels = labels[i:i+self.window_size]
            window_timestamps = timestamps[i:i+self.window_size]
            
            # A window is anomalous if any log in it is anomalous
            is_anomalous = 1 if sum(window_labels) > 0 else 0
            
            # Use the timestamp of the first log in the window
            timestamp = window_timestamps[0]
            
            log_sequences.append(window_logs)
            sequence_labels.append(is_anomalous)
            sequence_timestamps.append(timestamp)
        
        # If there are remaining logs, add the last window
        if len(logs) > self.window_size and (len(logs) - self.window_size) % self.step_size != 0:
            window_logs = logs[-self.window_size:]
            window_labels = labels[-self.window_size:]
            window_timestamps = timestamps[-self.window_size:]
            
            is_anomalous = 1 if sum(window_labels) > 0 else 0
            timestamp = window_timestamps[0]
            
            log_sequences.append(window_logs)
            sequence_labels.append(is_anomalous)
            sequence_timestamps.append(timestamp)
        
        logger.info(f"Created {len(log_sequences)} log sequences with {sum(sequence_labels)} anomalies")
        return log_sequences, sequence_labels, sequence_timestamps
    
    def load(self, path: Optional[str] = None) -> Tuple[List[List[str]], List[int]]:
        """
        Load and process Liberty logs.
        
        Args:
            path: Path to Liberty log file (overrides config path)
            
        Returns:
            Tuple of (log_sequences, sequence_labels)
        """
        # Use provided path or config path
        filepath = path or self.data_path
        
        if not filepath:
            raise ValueError("No data path provided.")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Liberty log file not found at {filepath}")
        
        logger.info(f"Loading Liberty logs from {filepath}")
        
        # Parse logs
        logs, labels, timestamps = self._parse_liberty_logs(filepath)
        
        # Apply preprocessing if available
        if self.preprocessor:
            logger.info("Applying preprocessing to log messages")
            logs = [self.preprocessor.preprocess(log) for log in logs]
        
        # Create sequences using sliding windows
        if self.label_type == "sliding_window":
            log_sequences, sequence_labels, sequence_timestamps = self._create_sliding_windows(
                logs, labels, timestamps
            )
        else:
            # If no windowing is needed, each log is its own sequence
            log_sequences = [[log] for log in logs]
            sequence_labels = labels
            sequence_timestamps = timestamps
        
        # Store dataset
        self.logs = log_sequences
        self.labels = sequence_labels
        self.timestamps = sequence_timestamps
        
        return log_sequences, sequence_labels
    
    def get_train_test_split(
        self,
        test_size: float = 0.2,
        shuffle: bool = False,
        random_state: Optional[int] = None
    ) -> Tuple[List[List[str]], List[int], List[List[str]], List[int]]:
        """
        Split the dataset into train and test sets.
        
        For Liberty, the split is chronological by default, where the first
        (1-test_size) of the data is used for training, and the remaining
        test_size is used for testing. This reflects real-world scenarios
        where we train on past data and test on future data.
        
        Args:
            test_size: Proportion of data to use for testing
            shuffle: Whether to shuffle the data before splitting
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_logs, train_labels, test_logs, test_labels)
        """
        if self.logs is None or self.labels is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        if shuffle:
            logger.warning("Shuffling data for Liberty might not be realistic for time-series data.")
            
            # Shuffle data
            if random_state is not None:
                np.random.seed(random_state)
            
            # Generate shuffled indices
            indices = np.random.permutation(len(self.logs))
            
            # Split indices
            split_idx = int(len(indices) * (1 - test_size))
            train_indices = indices[:split_idx]
            test_indices = indices[split_idx:]
            
            # Split data
            train_logs = [self.logs[i] for i in train_indices]
            train_labels = [self.labels[i] for i in train_indices]
            test_logs = [self.logs[i] for i in test_indices]
            test_labels = [self.labels[i] for i in test_indices]
        else:
            # Chronological split
            split_idx = int(len(self.logs) * (1 - test_size))
            
            train_logs = self.logs[:split_idx]
            train_labels = self.labels[:split_idx]
            test_logs = self.logs[split_idx:]
            test_labels = self.labels[split_idx:]
        
        logger.info(f"Train set: {len(train_logs)} sequences, {sum(train_labels)} anomalies")
        logger.info(f"Test set: {len(test_logs)} sequences, {sum(test_labels)} anomalies")
        
        return train_logs, train_labels, test_logs, test_labels
    
    def get_test_data(self) -> Tuple[List[List[str]], List[int]]:
        """
        Get the test data (useful after train/test split is performed).
        
        Returns:
            Tuple of (test_logs, test_labels)
        """
        if self.logs is None or self.labels is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        # By default, use the last 20% of the data as test data
        split_idx = int(len(self.logs) * 0.8)
        
        test_logs = self.logs[split_idx:]
        test_labels = self.labels[split_idx:]
        
        return test_logs, test_labels