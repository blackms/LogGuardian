"""
BERT-based feature extractor for log messages.
"""
import os
from typing import List, Dict, Any, Optional, Union, Tuple

import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer
from loguru import logger


class BertFeatureExtractor(nn.Module):
    """
    Feature extractor based on BERT for encoding log messages.
    
    This module uses a pre-trained BERT model to encode log messages into
    semantic vector representations. These vectors capture the semantic
    meaning of the log messages and can be used for anomaly detection.
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 128,
        pooling_strategy: str = "cls",
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the BERT feature extractor.
        
        Args:
            model_name: Name or path of the pre-trained BERT model
            max_length: Maximum sequence length for tokenization
            pooling_strategy: Strategy for pooling token embeddings:
                - "cls": Use the [CLS] token embedding (default)
                - "mean": Use the mean of all token embeddings
                - "max": Use the max of all token embeddings
            device: Device to run the model on. If None, use CUDA if available
            config: Optional configuration dictionary
        """
        super().__init__()
        
        self.config = config or {}
        self.model_name = model_name
        self.max_length = max_length
        self.pooling_strategy = pooling_strategy
        
        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Initializing BertFeatureExtractor with model {model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self._load_model_and_tokenizer()
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        logger.info("BertFeatureExtractor initialized successfully")
    
    def _load_model_and_tokenizer(self):
        """
        Load the pre-trained BERT model and tokenizer.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.debug(f"Loaded tokenizer: {self.model_name}")
            
            self.model = AutoModel.from_pretrained(self.model_name)
            logger.debug(f"Loaded model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading model or tokenizer: {e}")
            raise
    
    def tokenize(self, texts: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """
        Tokenize input texts using the BERT tokenizer.
        
        Args:
            texts: Single text or list of texts to tokenize
            
        Returns:
            Dictionary with input_ids, attention_mask, and token_type_ids
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move to device
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        
        return tokens
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        return_all_hidden_states: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Extract features from tokenized inputs.
        
        Args:
            input_ids: Token IDs from tokenizer (batch_size, seq_length)
            attention_mask: Attention mask from tokenizer (batch_size, seq_length)
            token_type_ids: Optional token type IDs (batch_size, seq_length)
            return_all_hidden_states: Whether to return all hidden states
            
        Returns:
            If return_all_hidden_states is False:
                Tensor of shape (batch_size, hidden_size) with extracted features
            If return_all_hidden_states is True:
                Tuple of (features, all_hidden_states)
        """
        # Prepare inputs
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": return_all_hidden_states,
        }
        
        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids
        
        # Get BERT outputs
        with torch.no_grad():
            outputs = self.model(**model_inputs)
        
        # Get hidden states
        last_hidden_state = outputs.last_hidden_state
        
        # Apply pooling strategy
        if self.pooling_strategy == "cls":
            # Use [CLS] token embedding (first token)
            features = last_hidden_state[:, 0, :]
        elif self.pooling_strategy == "mean":
            # Use mean of all token embeddings (accounting for padding)
            features = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1, keepdim=True)
        elif self.pooling_strategy == "max":
            # Use max of all token embeddings (accounting for padding)
            features = (last_hidden_state * attention_mask.unsqueeze(-1)).max(1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        if return_all_hidden_states:
            all_hidden_states = outputs.hidden_states
            return features, all_hidden_states
        
        return features
    
    def encode(
        self, 
        texts: Union[str, List[str]], 
        batch_size: int = 32,
        return_numpy: bool = False,
        return_all_hidden_states: bool = False
    ) -> Union[torch.Tensor, np.ndarray, Tuple]:
        """
        Encode texts into feature vectors.
        
        Args:
            texts: Single text or list of texts to encode
            batch_size: Batch size for processing
            return_numpy: Whether to return numpy arrays instead of torch tensors
            return_all_hidden_states: Whether to return all hidden states
            
        Returns:
            Feature vectors for the input texts
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        # Process in batches
        all_features = []
        all_hidden_states = [] if return_all_hidden_states else None
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            tokens = self.tokenize(batch_texts)
            
            # Forward pass
            if return_all_hidden_states:
                batch_features, batch_hidden_states = self.forward(
                    tokens["input_ids"],
                    tokens["attention_mask"],
                    tokens.get("token_type_ids"),
                    return_all_hidden_states=True
                )
                all_hidden_states.append(batch_hidden_states)
            else:
                batch_features = self.forward(
                    tokens["input_ids"],
                    tokens["attention_mask"],
                    tokens.get("token_type_ids"),
                    return_all_hidden_states=False
                )
            
            all_features.append(batch_features)
        
        # Concatenate batches
        features = torch.cat(all_features, dim=0)
        
        # Convert to numpy if requested
        if return_numpy:
            features = features.cpu().numpy()
            if return_all_hidden_states:
                all_hidden_states = [
                    [h.cpu().numpy() for h in hidden_states] 
                    for hidden_states in all_hidden_states
                ]
        
        if return_all_hidden_states:
            return features, all_hidden_states
        
        return features
    
    def save(self, path: str):
        """
        Save the model and tokenizer to a directory.
        
        Args:
            path: Path to directory where model should be saved
        """
        os.makedirs(path, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(path)
        
        # Save config
        config_path = os.path.join(path, "extractor_config.json")
        import json
        with open(config_path, "w") as f:
            json.dump({
                "model_name": self.model_name,
                "max_length": self.max_length,
                "pooling_strategy": self.pooling_strategy
            }, f)
        
        logger.info(f"Saved BertFeatureExtractor to {path}")
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> 'BertFeatureExtractor':
        """
        Load a saved model from a directory.
        
        Args:
            path: Path to directory containing the saved model
            device: Device to load the model onto
            
        Returns:
            Loaded BertFeatureExtractor
        """
        # Load config
        import json
        config_path = os.path.join(path, "extractor_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Create extractor with path as model_name to load from local files
        extractor = cls(
            model_name=path,
            max_length=config["max_length"],
            pooling_strategy=config["pooling_strategy"],
            device=device
        )
        
        logger.info(f"Loaded BertFeatureExtractor from {path}")
        
        return extractor