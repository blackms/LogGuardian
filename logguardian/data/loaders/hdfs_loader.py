"""
Data loader for the HDFS benchmark dataset.
"""
import os
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
from loguru import logger

from logguardian.data.loaders.base_loader import BaseDataLoader
from logguardian.data.preprocessors.base_preprocessor import BaseLogPreprocessor


class HDFSLoader(BaseDataLoader):
    """
    Loader for the HDFS (Hadoop Distributed File System) log dataset.
    
    The HDFS dataset is a widely used benchmark for log anomaly detection
    containing logs from Hadoop Distributed File System operations.
    
    Dataset format:
    - Log file: contains raw log messages
    - Label file: contains block IDs with anomaly labels
    """
    
    def __init__(
        self,
        preprocessor: Optional[BaseLogPreprocessor] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the HDFS data loader.
        
        Args:
            preprocessor: Optional preprocessor to use for preprocessing logs
            config: Optional configuration dictionary with keys:
                - data_path: Path to the dataset directory
                - log_file: Name of the log file (default: HDFS.log)
                - label_file: Name of the label file (default: anomaly_label.csv)
        """
        super().__init__(preprocessor, config)
        
        # Set default file names if not provided in config
        self.log_file = self.config.get("log_file", "HDFS.log")
        self.label_file = self.config.get("label_file", "anomaly_label.csv")
        
    def load(self, path: Optional[str] = None) -> Tuple[List[str], List[int]]:
        """
        Load and parse the HDFS dataset.
        
        Args:
            path: Path to the dataset directory. If None, use the path from config
            
        Returns:
            Tuple of (log_messages, labels) where:
                - log_messages is a list of log messages
                - labels is a list of labels (1 for anomaly, 0 for normal)
        """
        data_path = path or self.data_path
        
        if not data_path:
            raise ValueError("No data path provided. Set path in config or pass it as an argument.")
        
        log_path = os.path.join(data_path, self.log_file)
        label_path = os.path.join(data_path, self.label_file)
        
        logger.info(f"Loading HDFS dataset from {data_path}")
        
        # Load raw logs
        logs = self._load_logs(log_path)
        
        # Load and parse labels
        block_labels = self._load_labels(label_path)
        
        # Extract block IDs from logs
        processed_logs, labels = self._process_logs_with_labels(logs, block_labels)
        
        # Preprocess logs if a preprocessor is provided
        if self.preprocessor is not None:
            processed_logs = self.preprocessor.preprocess_batch(processed_logs)
        
        logger.info(f"Loaded {len(processed_logs)} logs with {sum(labels)} anomalies")
        
        return processed_logs, labels
    
    def _load_logs(self, log_path: str) -> List[str]:
        """
        Load raw logs from the log file.
        
        Args:
            log_path: Path to the log file
            
        Returns:
            List of raw log messages
        """
        logger.info(f"Loading logs from {log_path}")
        
        with open(log_path, 'r', encoding='utf-8') as f:
            logs = f.readlines()
        
        logger.info(f"Loaded {len(logs)} raw log messages")
        
        return logs
    
    def _load_labels(self, label_path: str) -> Dict[str, int]:
        """
        Load anomaly labels from the label file.
        
        Args:
            label_path: Path to the label file
            
        Returns:
            Dictionary mapping block IDs to labels (1 for anomaly, 0 for normal)
        """
        logger.info(f"Loading labels from {label_path}")
        
        # Read label file (format: block_id,label)
        df = pd.read_csv(label_path)
        
        # Convert to dictionary
        block_labels = {
            str(row['BlockId']): int(row['Label'])
            for _, row in df.iterrows()
        }
        
        logger.info(f"Loaded labels for {len(block_labels)} blocks")
        
        return block_labels
    
    def _process_logs_with_labels(
        self, 
        logs: List[str], 
        block_labels: Dict[str, int]
    ) -> Tuple[List[str], List[int]]:
        """
        Process logs and match them with their labels.
        
        Args:
            logs: List of raw log messages
            block_labels: Dictionary mapping block IDs to labels
            
        Returns:
            Tuple of (processed_logs, labels)
        """
        processed_logs = []
        labels = []
        
        # Regular expression to extract block ID from log message
        import re
        block_id_pattern = re.compile(r'blk_[-\d]+')
        
        for log in logs:
            # Extract block ID from log
            match = block_id_pattern.search(log)
            
            if match:
                block_id = match.group(0)
                
                # Get label for this block_id (default to 0 if not found)
                label = block_labels.get(block_id, 0)
                
                processed_logs.append(log.strip())
                labels.append(label)
            else:
                # No block ID in this log line, assume normal
                processed_logs.append(log.strip())
                labels.append(0)
        
        return processed_logs, labels