"""
Three-stage trainer for LogGuardian as described in the LogLLM paper.
"""
import os
import json
import copy
from typing import Dict, Any, Optional, Union, Tuple, List, Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from loguru import logger
from tqdm.auto import tqdm

from logguardian.training.trainer import Trainer
from logguardian.pipeline import LogGuardian


class ThreeStageTrainer:
    """
    Three-stage trainer for LogGuardian model.
    
    This trainer implements the three-stage training procedure described in the
    LogLLM paper:
    
    1. Stage 1: Fine-tune Llama to capture the answer template
    2. Stage 2: Train the embedder of log messages (BERT + projector)
    3. Stage 3: Fine-tune the entire model end-to-end
    """
    
    def __init__(
        self,
        model: LogGuardian,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the three-stage trainer.
        
        Args:
            model: LogGuardian model to train
            device: Device to use for training
            config: Configuration parameters
        """
        self.model = model
        self.config = config or {}
        
        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Create separate trainers for each stage
        self.stage1_trainer = None
        self.stage2_trainer = None
        self.stage3_trainer = None
        
        # Training metrics for each stage
        self.metrics = {
            "stage1": {},
            "stage2": {},
            "stage3": {}
        }
        
        # Overall best model
        self.best_model_state = None
        self.best_model_metrics = None
        
        logger.info(f"Initialized three-stage trainer on device: {self.device}")
    
    def setup_stage1(
        self,
        learning_rate: float = 5e-4,
        warmup_steps: int = 100,
        weight_decay: float = 0.01,
        **kwargs
    ) -> None:
        """
        Set up Stage 1: Fine-tune Llama to capture the answer template.
        
        Args:
            learning_rate: Learning rate for fine-tuning
            warmup_steps: Warmup steps for learning rate scheduler
            weight_decay: Weight decay for optimizer
            **kwargs: Additional parameters
        """
        logger.info("Setting up Stage 1: Fine-tune Llama to capture the answer template")
        
        # Create a copy of the model for Stage 1
        stage1_model = copy.deepcopy(self.model)
        
        # Freeze BERT and projector
        for param in stage1_model.feature_extractor.parameters():
            param.requires_grad = False
        
        for param in stage1_model.embedding_projector.parameters():
            param.requires_grad = False
        
        # Only train the classifier (Llama)
        for param in stage1_model.classifier.parameters():
            param.requires_grad = True
        
        # Create trainer for Stage 1
        self.stage1_trainer = Trainer(stage1_model, device=self.device, config=kwargs.get("stage1_config"))
        
        # Configure optimizer with higher learning rate for Stage 1
        self.stage1_trainer.configure_optimizers(
            optimizer_type=kwargs.get("optimizer_type", "adamw"),
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            scheduler_type=kwargs.get("scheduler_type", "linear_warmup"),
            warmup_steps=warmup_steps,
            total_steps=kwargs.get("total_steps", 1000)
        )
    
    def setup_stage2(
        self,
        learning_rate: float = 5e-5,
        warmup_steps: int = 0,
        weight_decay: float = 0.01,
        **kwargs
    ) -> None:
        """
        Set up Stage 2: Train the embedder of log messages (BERT + projector).
        
        Args:
            learning_rate: Learning rate for fine-tuning
            warmup_steps: Warmup steps for learning rate scheduler
            weight_decay: Weight decay for optimizer
            **kwargs: Additional parameters
        """
        logger.info("Setting up Stage 2: Train the embedder of log messages")
        
        # Create a copy of the model for Stage 2
        stage2_model = copy.deepcopy(self.model)
        
        # Use the fine-tuned classifier from Stage 1 if available
        if self.stage1_trainer is not None and hasattr(self.stage1_trainer, "model"):
            stage2_model.classifier = copy.deepcopy(self.stage1_trainer.model.classifier)
        
        # Freeze the classifier (Llama)
        for param in stage2_model.classifier.parameters():
            param.requires_grad = False
        
        # Only train BERT and projector
        for param in stage2_model.feature_extractor.parameters():
            param.requires_grad = True
        
        for param in stage2_model.embedding_projector.parameters():
            param.requires_grad = True
        
        # Create trainer for Stage 2
        self.stage2_trainer = Trainer(stage2_model, device=self.device, config=kwargs.get("stage2_config"))
        
        # Configure optimizer for Stage 2
        self.stage2_trainer.configure_optimizers(
            optimizer_type=kwargs.get("optimizer_type", "adamw"),
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            scheduler_type=kwargs.get("scheduler_type", "linear_warmup"),
            warmup_steps=warmup_steps,
            total_steps=kwargs.get("total_steps", 2000)
        )
    
    def setup_stage3(
        self,
        learning_rate: float = 5e-5,
        warmup_steps: int = 0,
        weight_decay: float = 0.01,
        **kwargs
    ) -> None:
        """
        Set up Stage 3: Fine-tune the entire model end-to-end.
        
        Args:
            learning_rate: Learning rate for fine-tuning
            warmup_steps: Warmup steps for learning rate scheduler
            weight_decay: Weight decay for optimizer
            **kwargs: Additional parameters
        """
        logger.info("Setting up Stage 3: Fine-tune the entire model end-to-end")
        
        # Create a copy of the model for Stage 3
        stage3_model = copy.deepcopy(self.model)
        
        # Use the trained components from previous stages if available
        if self.stage2_trainer is not None and hasattr(self.stage2_trainer, "model"):
            stage3_model.feature_extractor = copy.deepcopy(self.stage2_trainer.model.feature_extractor)
            stage3_model.embedding_projector = copy.deepcopy(self.stage2_trainer.model.embedding_projector)
        
        if self.stage1_trainer is not None and hasattr(self.stage1_trainer, "model"):
            stage3_model.classifier = copy.deepcopy(self.stage1_trainer.model.classifier)
        
        # Train all components together
        for param in stage3_model.parameters():
            param.requires_grad = True
        
        # Create trainer for Stage 3
        self.stage3_trainer = Trainer(stage3_model, device=self.device, config=kwargs.get("stage3_config"))
        
        # Configure optimizer for Stage 3 with lower learning rate
        self.stage3_trainer.configure_optimizers(
            optimizer_type=kwargs.get("optimizer_type", "adamw"),
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            scheduler_type=kwargs.get("scheduler_type", "linear_warmup"),
            warmup_steps=warmup_steps,
            total_steps=kwargs.get("total_steps", 2000)
        )
    
    def run_stage1(
        self,
        train_loader: DataLoader,
        criterion: Callable,
        eval_loader: Optional[DataLoader] = None,
        metrics: Optional[Dict[str, Callable]] = None,
        num_epochs: int = 1,
        num_samples: Optional[int] = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run Stage 1: Fine-tune Llama to capture the answer template.
        
        Args:
            train_loader: DataLoader for training data
            criterion: Loss function
            eval_loader: DataLoader for evaluation data
            metrics: Dictionary of metric functions
            num_epochs: Number of epochs to train
            num_samples: Number of samples to use for training (if None, use all)
            **kwargs: Additional parameters for training
            
        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Running Stage 1: Fine-tune Llama with {num_samples if num_samples else 'all'} samples")
        
        # If num_samples is specified, create a subset of the training data
        if num_samples is not None and num_samples < len(train_loader.dataset):
            from torch.utils.data import Subset
            import random
            
            # Randomly select indices
            all_indices = list(range(len(train_loader.dataset)))
            random.shuffle(all_indices)
            selected_indices = all_indices[:num_samples]
            
            # Create subset
            subset = Subset(train_loader.dataset, selected_indices)
            
            # Create new dataloader
            subset_loader = DataLoader(
                subset,
                batch_size=train_loader.batch_size,
                shuffle=True,
                num_workers=train_loader.num_workers if hasattr(train_loader, "num_workers") else 0
            )
            
            # Use subset for training
            train_loader = subset_loader
        
        # Ensure the trainer is set up
        if self.stage1_trainer is None:
            self.setup_stage1(**kwargs)
        
        # Run training
        stage1_metrics = self.stage1_trainer.train(
            train_loader=train_loader,
            criterion=criterion,
            eval_loader=eval_loader,
            metrics=metrics,
            num_epochs=num_epochs,
            gradient_accumulation_steps=kwargs.get("gradient_accumulation_steps", 1),
            max_grad_norm=kwargs.get("max_grad_norm", 1.0),
            log_interval=kwargs.get("log_interval", 10),
            checkpoint_dir=kwargs.get("checkpoint_dir_stage1"),
            save_best_only=kwargs.get("save_best_only", True),
            early_stopping_patience=kwargs.get("early_stopping_patience")
        )
        
        # Store metrics
        self.metrics["stage1"] = stage1_metrics
        
        return stage1_metrics
    
    def run_stage2(
        self,
        train_loader: DataLoader,
        criterion: Callable,
        eval_loader: Optional[DataLoader] = None,
        metrics: Optional[Dict[str, Callable]] = None,
        num_epochs: int = 2,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run Stage 2: Train the embedder of log messages (BERT + projector).
        
        Args:
            train_loader: DataLoader for training data
            criterion: Loss function
            eval_loader: DataLoader for evaluation data
            metrics: Dictionary of metric functions
            num_epochs: Number of epochs to train
            **kwargs: Additional parameters for training
            
        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Running Stage 2: Train embedder (BERT + projector) for {num_epochs} epochs")
        
        # Ensure the trainer is set up
        if self.stage2_trainer is None:
            self.setup_stage2(**kwargs)
        
        # Run training
        stage2_metrics = self.stage2_trainer.train(
            train_loader=train_loader,
            criterion=criterion,
            eval_loader=eval_loader,
            metrics=metrics,
            num_epochs=num_epochs,
            gradient_accumulation_steps=kwargs.get("gradient_accumulation_steps", 1),
            max_grad_norm=kwargs.get("max_grad_norm", 1.0),
            log_interval=kwargs.get("log_interval", 10),
            checkpoint_dir=kwargs.get("checkpoint_dir_stage2"),
            save_best_only=kwargs.get("save_best_only", True),
            early_stopping_patience=kwargs.get("early_stopping_patience")
        )
        
        # Store metrics
        self.metrics["stage2"] = stage2_metrics
        
        return stage2_metrics
    
    def run_stage3(
        self,
        train_loader: DataLoader,
        criterion: Callable,
        eval_loader: Optional[DataLoader] = None,
        metrics: Optional[Dict[str, Callable]] = None,
        num_epochs: int = 2,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run Stage 3: Fine-tune the entire model end-to-end.
        
        Args:
            train_loader: DataLoader for training data
            criterion: Loss function
            eval_loader: DataLoader for evaluation data
            metrics: Dictionary of metric functions
            num_epochs: Number of epochs to train
            **kwargs: Additional parameters for training
            
        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Running Stage 3: Fine-tune entire model for {num_epochs} epochs")
        
        # Ensure the trainer is set up
        if self.stage3_trainer is None:
            self.setup_stage3(**kwargs)
        
        # Run training
        stage3_metrics = self.stage3_trainer.train(
            train_loader=train_loader,
            criterion=criterion,
            eval_loader=eval_loader,
            metrics=metrics,
            num_epochs=num_epochs,
            gradient_accumulation_steps=kwargs.get("gradient_accumulation_steps", 1),
            max_grad_norm=kwargs.get("max_grad_norm", 1.0),
            log_interval=kwargs.get("log_interval", 10),
            checkpoint_dir=kwargs.get("checkpoint_dir_stage3"),
            save_best_only=kwargs.get("save_best_only", True),
            early_stopping_patience=kwargs.get("early_stopping_patience")
        )
        
        # Store metrics
        self.metrics["stage3"] = stage3_metrics
        
        # Save the best model state
        if hasattr(self.stage3_trainer, "model"):
            self.best_model_state = copy.deepcopy(self.stage3_trainer.model.state_dict())
            self.best_model_metrics = stage3_metrics
        
        return stage3_metrics
    
    def train(
        self,
        train_loader: DataLoader,
        criterion: Callable,
        eval_loader: Optional[DataLoader] = None,
        metrics: Optional[Dict[str, Callable]] = None,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run the complete three-stage training procedure.
        
        Args:
            train_loader: DataLoader for training data
            criterion: Loss function
            eval_loader: DataLoader for evaluation data
            metrics: Dictionary of metric functions
            **kwargs: Additional parameters for training
            
        Returns:
            Dictionary of training metrics for all stages
        """
        logger.info("Starting three-stage training procedure")
        
        # Run Stage 1: Fine-tune Llama to capture the answer template
        stage1_metrics = self.run_stage1(
            train_loader=train_loader,
            criterion=criterion,
            eval_loader=eval_loader,
            metrics=metrics,
            num_epochs=kwargs.get("num_epochs_stage1", 1),
            num_samples=kwargs.get("num_samples_stage1", 1000),
            **kwargs
        )
        
        # Run Stage 2: Train the embedder of log messages (BERT + projector)
        stage2_metrics = self.run_stage2(
            train_loader=train_loader,
            criterion=criterion,
            eval_loader=eval_loader,
            metrics=metrics,
            num_epochs=kwargs.get("num_epochs_stage2", 2),
            **kwargs
        )
        
        # Run Stage 3: Fine-tune the entire model end-to-end
        stage3_metrics = self.run_stage3(
            train_loader=train_loader,
            criterion=criterion,
            eval_loader=eval_loader,
            metrics=metrics,
            num_epochs=kwargs.get("num_epochs_stage3", 2),
            **kwargs
        )
        
        # Update the original model with the best model state
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        logger.info("Completed three-stage training procedure")
        
        # Save final metrics
        if kwargs.get("checkpoint_dir"):
            metrics_path = os.path.join(kwargs.get("checkpoint_dir"), "three_stage_metrics.json")
            serializable_metrics = {}
            
            # Convert tensors to lists for JSON serialization
            for stage, stage_metrics in self.metrics.items():
                serializable_metrics[stage] = {}
                for key, value in stage_metrics.items():
                    if isinstance(value, list):
                        serializable_metrics[stage][key] = [float(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v for v in value]
                    elif isinstance(value, dict):
                        serializable_metrics[stage][key] = {}
                        for k, v in value.items():
                            if isinstance(v, list):
                                serializable_metrics[stage][key][k] = [float(item) if isinstance(item, (torch.Tensor, np.ndarray)) else item for item in v]
                            else:
                                serializable_metrics[stage][key][k] = float(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v
                    else:
                        serializable_metrics[stage][key] = float(value) if isinstance(value, (torch.Tensor, np.ndarray)) else value
                        
            with open(metrics_path, "w") as f:
                json.dump(serializable_metrics, f, indent=2)
            
            logger.info(f"Saved three-stage training metrics to {metrics_path}")
        
        return self.metrics
    
    def save(self, path: str) -> None:
        """
        Save the three-stage trainer state.
        
        Args:
            path: Directory to save the trainer state
        """
        os.makedirs(path, exist_ok=True)
        
        # Save model
        model_path = os.path.join(path, "model.pt")
        torch.save(self.model.state_dict(), model_path)
        
        # Save metrics
        metrics_path = os.path.join(path, "metrics.json")
        
        # Convert metrics to serializable format
        serializable_metrics = {}
        for stage, stage_metrics in self.metrics.items():
            serializable_metrics[stage] = {}
            for key, value in stage_metrics.items():
                if isinstance(value, list):
                    serializable_metrics[stage][key] = [float(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v for v in value]
                elif isinstance(value, dict):
                    serializable_metrics[stage][key] = {}
                    for k, v in value.items():
                        if isinstance(v, list):
                            serializable_metrics[stage][key][k] = [float(item) if isinstance(item, (torch.Tensor, np.ndarray)) else item for item in v]
                        else:
                            serializable_metrics[stage][key][k] = float(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v
                else:
                    serializable_metrics[stage][key] = float(value) if isinstance(value, (torch.Tensor, np.ndarray)) else value
        
        with open(metrics_path, "w") as f:
            json.dump(serializable_metrics, f, indent=2)
        
        # Save config
        config_path = os.path.join(path, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Saved three-stage trainer to {path}")
    
    def load(self, path: str) -> None:
        """
        Load the three-stage trainer state.
        
        Args:
            path: Directory to load the trainer state from
        """
        # Load model
        model_path = os.path.join(path, "model.pt")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # Load metrics if available
        metrics_path = os.path.join(path, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                self.metrics = json.load(f)
        
        # Load config if available
        config_path = os.path.join(path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = json.load(f)
        
        logger.info(f"Loaded three-stage trainer from {path}")