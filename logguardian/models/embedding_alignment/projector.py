"""
Embedding projector for aligning BERT and LLM embedding spaces.
"""
import os
from typing import Dict, Any, Optional, Union

import torch
import torch.nn as nn
import numpy as np
from loguru import logger


class EmbeddingProjector(nn.Module):
    """
    Linear projection layer for aligning embedding spaces between models.
    
    This module maps embedding vectors from the source model space (e.g., BERT)
    to the target model space (e.g., Llama) to ensure compatibility between
    different language models in the pipeline.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the embedding projector.
        
        Args:
            input_dim: Dimension of input embeddings (e.g., BERT hidden size)
            output_dim: Dimension of output embeddings (e.g., Llama hidden size)
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
            device: Device to run the model on. If None, use CUDA if available
            config: Optional configuration dictionary
        """
        super().__init__()
        
        self.config = config or {}
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout
        self.use_batch_norm = use_batch_norm
        
        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Initializing EmbeddingProjector: {input_dim} → {output_dim}")
        logger.info(f"Using device: {self.device}")
        
        # Build the projection layers
        self._build_projection_layers()
        
        # Move model to device
        self.to(self.device)
        
        logger.info("EmbeddingProjector initialized successfully")
    
    def _build_projection_layers(self):
        """
        Build the projection layers.
        """
        layers = []
        
        # Linear projection
        layers.append(nn.Linear(self.input_dim, self.output_dim))
        
        # Batch normalization (optional)
        if self.use_batch_norm:
            layers.append(nn.BatchNorm1d(self.output_dim))
        
        # Activation
        layers.append(nn.ReLU())
        
        # Dropout
        layers.append(nn.Dropout(self.dropout_rate))
        
        # Create sequential model
        self.projection = nn.Sequential(*layers)
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Project embeddings from source to target space.
        
        Args:
            embeddings: Input embeddings tensor of shape (batch_size, input_dim)
            
        Returns:
            Projected embeddings tensor of shape (batch_size, output_dim)
        """
        return self.projection(embeddings)
    
    def project(
        self, 
        embeddings: Union[torch.Tensor, np.ndarray],
        return_numpy: bool = False
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Project embeddings and handle numpy conversion if needed.
        
        Args:
            embeddings: Input embeddings as torch tensor or numpy array
            return_numpy: Whether to return numpy arrays instead of torch tensors
            
        Returns:
            Projected embeddings in the target space
        """
        # Convert numpy to torch if needed
        is_numpy = isinstance(embeddings, np.ndarray)
        if is_numpy:
            embeddings = torch.from_numpy(embeddings).to(self.device)
        
        # Ensure the model is in eval mode
        self.eval()
        
        # Forward pass
        with torch.no_grad():
            projected = self.forward(embeddings)
        
        # Convert back to numpy if requested or if input was numpy
        if return_numpy or is_numpy:
            projected = projected.cpu().numpy()
        
        return projected
    
    def save(self, path: str):
        """
        Save the projection model to a file.
        
        Args:
            path: Path to directory where model should be saved
        """
        os.makedirs(path, exist_ok=True)
        
        # Save model parameters
        model_path = os.path.join(path, "projector.pt")
        torch.save(self.state_dict(), model_path)
        
        # Save config
        config_path = os.path.join(path, "projector_config.json")
        import json
        with open(config_path, "w") as f:
            json.dump({
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "dropout": self.dropout_rate,
                "use_batch_norm": self.use_batch_norm
            }, f)
        
        logger.info(f"Saved EmbeddingProjector to {path}")
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> 'EmbeddingProjector':
        """
        Load a saved model from a directory.
        
        Args:
            path: Path to directory containing the saved model
            device: Device to load the model onto
            
        Returns:
            Loaded EmbeddingProjector
        """
        # Load config
        import json
        config_path = os.path.join(path, "projector_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Create projector
        projector = cls(
            input_dim=config["input_dim"],
            output_dim=config["output_dim"],
            dropout=config["dropout"],
            use_batch_norm=config["use_batch_norm"],
            device=device
        )
        
        # Load model parameters
        model_path = os.path.join(path, "projector.pt")
        projector.load_state_dict(torch.load(model_path, map_location=projector.device))
        projector.eval()
        
        logger.info(f"Loaded EmbeddingProjector from {path}")
        
        return projector