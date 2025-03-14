"""
Main pipeline for log anomaly detection.
"""
import os
from typing import List, Dict, Any, Optional, Union, Tuple

import torch
import numpy as np
from loguru import logger

from logguardian.data.preprocessors import BaseLogPreprocessor, SystemLogPreprocessor
from logguardian.models.feature_extraction import BertFeatureExtractor
from logguardian.models.embedding_alignment import EmbeddingProjector
from logguardian.models.classification import BaseLogClassifier, LlamaLogClassifier


class LogGuardian:
    """
    End-to-end pipeline for log anomaly detection.
    
    This class integrates all components of the system:
    1. Log preprocessing
    2. Semantic feature extraction
    3. Embedding alignment
    4. Sequence classification
    
    It provides a unified interface for detecting anomalies in log data.
    """
    
    def __init__(
        self,
        preprocessor: Optional[BaseLogPreprocessor] = None,
        feature_extractor: Optional[BertFeatureExtractor] = None,
        embedding_projector: Optional[EmbeddingProjector] = None,
        classifier: Optional[BaseLogClassifier] = None,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the LogGuardian pipeline.
        
        Args:
            preprocessor: Log preprocessor component
            feature_extractor: Feature extraction component
            embedding_projector: Embedding alignment component
            classifier: Sequence classifier component
            device: Device to run on (cuda/cpu)
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Initializing LogGuardian pipeline on device: {self.device}")
        
        # Initialize or load components
        self.preprocessor = preprocessor or self._initialize_preprocessor()
        self.feature_extractor = feature_extractor or self._initialize_feature_extractor()
        self.embedding_projector = embedding_projector or self._initialize_embedding_projector()
        self.classifier = classifier or self._initialize_classifier()
        
        logger.info("LogGuardian pipeline initialized successfully")
    
    def _initialize_preprocessor(self) -> BaseLogPreprocessor:
        """
        Initialize the log preprocessor component.
        
        Returns:
            Initialized preprocessor
        """
        preprocessor_config = self.config.get("preprocessor", {})
        
        logger.info("Initializing default SystemLogPreprocessor")
        return SystemLogPreprocessor(preprocessor_config)
    
    def _initialize_feature_extractor(self) -> BertFeatureExtractor:
        """
        Initialize the feature extraction component.
        
        Returns:
            Initialized feature extractor
        """
        extractor_config = self.config.get("feature_extractor", {})
        model_name = extractor_config.get("model_name", "bert-base-uncased")
        
        logger.info(f"Initializing BertFeatureExtractor with model: {model_name}")
        return BertFeatureExtractor(
            model_name=model_name,
            device=str(self.device),
            config=extractor_config
        )
    
    def _initialize_embedding_projector(self) -> EmbeddingProjector:
        """
        Initialize the embedding alignment component.
        
        Returns:
            Initialized embedding projector
        """
        projector_config = self.config.get("embedding_projector", {})
        
        # Default dimensions
        input_dim = projector_config.get("input_dim", 768)  # BERT base hidden size
        output_dim = projector_config.get("output_dim", 4096)  # Llama hidden size
        
        logger.info(f"Initializing EmbeddingProjector: {input_dim} → {output_dim}")
        return EmbeddingProjector(
            input_dim=input_dim,
            output_dim=output_dim,
            device=str(self.device),
            config=projector_config
        )
    
    def _initialize_classifier(self) -> BaseLogClassifier:
        """
        Initialize the classifier component.
        
        Returns:
            Initialized classifier
        """
        classifier_config = self.config.get("classifier", {})
        model_name = classifier_config.get("model_name", "meta-llama/Llama-3-8b")
        
        logger.info(f"Initializing LlamaLogClassifier with model: {model_name}")
        return LlamaLogClassifier(
            model_name_or_path=model_name,
            device=str(self.device),
            config=classifier_config
        )
    
    def detect(
        self, 
        logs: Union[str, List[str]],
        batch_size: int = 16,
        window_size: int = 10,
        stride: int = 5,
        raw_output: bool = False
    ) -> Union[List[int], List[Dict[str, Any]]]:
        """
        Detect anomalies in log data.
        
        Args:
            logs: Single log message or list of log messages
            batch_size: Size of batches for processing
            window_size: Size of sliding window for sequences
            stride: Stride of sliding window
            raw_output: Whether to return raw classification outputs
            
        Returns:
            List of anomaly labels (1 for anomaly, 0 for normal)
            If raw_output is True, returns a list of dictionaries with detailed outputs
        """
        # Ensure logs is a list
        if isinstance(logs, str):
            logs = [logs]
        
        # Step 1: Preprocess logs
        preprocessed_logs = self._preprocess_logs(logs)
        
        # Step 2: Create sequences with sliding window
        sequences = self._create_sequences(preprocessed_logs, window_size, stride)
        
        # Step 3: Extract features
        feature_batch_size = min(batch_size, len(sequences))
        features = self._extract_features(sequences, feature_batch_size)
        
        # Step 4: Project features
        projected_features = self._project_features(features)
        
        # Step 5: Classify sequences
        classifier_batch_size = min(batch_size, len(projected_features))
        results = self._classify_sequences(projected_features, classifier_batch_size, raw_output)
        
        return results
    
    def _preprocess_logs(self, logs: List[str]) -> List[str]:
        """
        Preprocess log messages.
        
        Args:
            logs: List of raw log messages
            
        Returns:
            List of preprocessed log messages
        """
        logger.info(f"Preprocessing {len(logs)} log messages")
        return self.preprocessor.preprocess_batch(logs)
    
    def _create_sequences(
        self, 
        logs: List[str], 
        window_size: int, 
        stride: int
    ) -> List[List[str]]:
        """
        Create sequences by sliding a window over the logs.
        
        Args:
            logs: List of preprocessed log messages
            window_size: Size of sliding window
            stride: Stride of sliding window
            
        Returns:
            List of log sequences
        """
        sequences = []
        
        for i in range(0, len(logs) - window_size + 1, stride):
            sequences.append(logs[i:i+window_size])
        
        # If there are remaining logs, add the last window
        if len(logs) > window_size and (len(logs) - window_size) % stride != 0:
            sequences.append(logs[-window_size:])
        
        # Handle case where there are fewer logs than window_size
        if len(sequences) == 0 and len(logs) > 0:
            sequences.append(logs)
        
        logger.info(f"Created {len(sequences)} sequences with window_size={window_size}, stride={stride}")
        
        return sequences
    
    def _extract_features(
        self, 
        sequences: List[List[str]], 
        batch_size: int
    ) -> List[torch.Tensor]:
        """
        Extract features from log sequences.
        
        Args:
            sequences: List of log sequences
            batch_size: Size of batches for processing
            
        Returns:
            List of feature vectors for each sequence
        """
        logger.info(f"Extracting features from {len(sequences)} sequences")
        
        # Flatten sequences for feature extraction
        flattened_logs = []
        sequence_lengths = []
        
        for sequence in sequences:
            flattened_logs.extend(sequence)
            sequence_lengths.append(len(sequence))
        
        # Extract features in batches
        all_features = []
        
        for i in range(0, len(flattened_logs), batch_size):
            batch_logs = flattened_logs[i:i+batch_size]
            batch_features = self.feature_extractor.encode(batch_logs)
            all_features.append(batch_features)
        
        # Concatenate all features
        if len(all_features) > 0:
            if isinstance(all_features[0], torch.Tensor):
                features = torch.cat(all_features, dim=0)
            else:
                features = np.concatenate(all_features, axis=0)
        else:
            # Create empty tensor/array of appropriate shape
            if isinstance(self.feature_extractor, BertFeatureExtractor):
                hidden_size = self.feature_extractor.model.config.hidden_size
                features = torch.zeros((0, hidden_size), device=self.device)
            else:
                features = np.array([])
        
        # Reconstruct sequences
        sequence_features = []
        start_idx = 0
        
        for length in sequence_lengths:
            end_idx = start_idx + length
            sequence_features.append(features[start_idx:end_idx])
            start_idx = end_idx
        
        return sequence_features
    
    def _project_features(
        self, 
        features: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Project features to target embedding space.
        
        Args:
            features: List of feature vectors for each sequence
            
        Returns:
            List of projected feature vectors
        """
        logger.info(f"Projecting features for {len(features)} sequences")
        
        projected_features = []
        
        for sequence_features in features:
            projected = self.embedding_projector.project(sequence_features)
            projected_features.append(projected)
        
        return projected_features
    
    def _classify_sequences(
        self, 
        projected_features: List[torch.Tensor], 
        batch_size: int,
        raw_output: bool
    ) -> Union[List[int], List[Dict[str, Any]]]:
        """
        Classify sequences as normal or anomalous.
        
        Args:
            projected_features: List of projected feature vectors
            batch_size: Size of batches for processing
            raw_output: Whether to return raw classification outputs
            
        Returns:
            List of anomaly labels or detailed outputs
        """
        logger.info(f"Classifying {len(projected_features)} sequences")
        
        # Process in batches
        all_results = []
        
        for i in range(0, len(projected_features), batch_size):
            batch_features = projected_features[i:i+batch_size]
            batch_results = self.classifier.classify_batch(
                batch_features, 
                raw_output=raw_output
            )
            all_results.extend(batch_results)
        
        return all_results
    
    def save(self, path: str) -> None:
        """
        Save the pipeline to a directory.
        
        Args:
            path: Path to save the pipeline to
        """
        os.makedirs(path, exist_ok=True)
        
        # Save each component
        self.preprocessor.save(os.path.join(path, "preprocessor"))
        self.feature_extractor.save(os.path.join(path, "feature_extractor"))
        self.embedding_projector.save(os.path.join(path, "embedding_projector"))
        self.classifier.save(os.path.join(path, "classifier"))
        
        # Save configuration
        import json
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Saved LogGuardian pipeline to {path}")
    
    @classmethod
    def load(
        cls, 
        path: str, 
        device: Optional[str] = None,
        **kwargs
    ) -> 'LogGuardian':
        """
        Load a pipeline from a directory.
        
        Args:
            path: Path to load the pipeline from
            device: Device to load the pipeline onto
            
        Returns:
            Loaded pipeline
        """
        # Load configuration
        import json
        with open(os.path.join(path, "config.json"), "r") as f:
            config = json.load(f)
        
        # Load components
        preprocessor = SystemLogPreprocessor.load(os.path.join(path, "preprocessor"))
        
        feature_extractor = BertFeatureExtractor.load(
            os.path.join(path, "feature_extractor"),
            device=device
        )
        
        embedding_projector = EmbeddingProjector.load(
            os.path.join(path, "embedding_projector"),
            device=device
        )
        
        classifier = LlamaLogClassifier.load(
            os.path.join(path, "classifier"),
            device=device,
            **kwargs
        )
        
        # Create pipeline
        pipeline = cls(
            preprocessor=preprocessor,
            feature_extractor=feature_extractor,
            embedding_projector=embedding_projector,
            classifier=classifier,
            device=device,
            config=config
        )
        
        logger.info(f"Loaded LogGuardian pipeline from {path}")
        
        return pipeline