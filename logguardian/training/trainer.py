"""
Base trainer for LogGuardian models.
"""
import os
import json
import time
from typing import Dict, Any, Optional, Union, Tuple, List, Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from loguru import logger
from tqdm.auto import tqdm


class Trainer:
    """
    Base trainer class for LogGuardian models.
    
    This class provides common functionality for training models,
    such as setting up optimizers, tracking metrics, and saving
    checkpoints.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
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
        
        # Initialize optimizer and scheduler
        self.optimizer = None
        self.scheduler = None
        
        # Training metrics
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": {},
            "val_metrics": {},
            "best_val_loss": float("inf"),
            "best_val_metric": 0.0,
            "best_epoch": 0
        }
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        
        logger.info(f"Initialized trainer on device: {self.device}")
    
    def configure_optimizers(
        self,
        optimizer_type: str = "adamw",
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        scheduler_type: Optional[str] = "linear_warmup",
        warmup_steps: int = 0,
        **kwargs
    ):
        """
        Configure optimizers and learning rate schedulers.
        
        Args:
            optimizer_type: Type of optimizer to use
            learning_rate: Learning rate
            weight_decay: Weight decay factor
            scheduler_type: Type of learning rate scheduler
            warmup_steps: Number of warmup steps for scheduler
            **kwargs: Additional parameters for optimizer and scheduler
        """
        # Configure optimizer
        if optimizer_type.lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                **kwargs.get("optimizer_kwargs", {})
            )
        elif optimizer_type.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                **kwargs.get("optimizer_kwargs", {})
            )
        elif optimizer_type.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                **kwargs.get("optimizer_kwargs", {})
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        # Configure learning rate scheduler
        if scheduler_type is not None:
            if scheduler_type == "linear_warmup":
                from transformers import get_linear_schedule_with_warmup
                
                total_steps = kwargs.get("total_steps", 1000)
                self.scheduler = get_linear_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps
                )
            elif scheduler_type == "cosine":
                from transformers import get_cosine_schedule_with_warmup
                
                total_steps = kwargs.get("total_steps", 1000)
                self.scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps
                )
            elif scheduler_type == "plateau":
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode="min",
                    factor=0.1,
                    patience=5,
                    verbose=True
                )
            else:
                raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
        logger.info(f"Configured optimizer: {optimizer_type}, scheduler: {scheduler_type}")
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        criterion: Callable,
        metrics: Optional[Dict[str, Callable]] = None,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        log_interval: int = 10
    ) -> Dict[str, float]:
        """
        Train the model for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            criterion: Loss function
            metrics: Dictionary of metric functions
            gradient_accumulation_steps: Number of steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping
            log_interval: Interval for logging training metrics
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0
        total_samples = 0
        epoch_metrics = {name: 0.0 for name in (metrics or {}).keys()}
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {self.current_epoch}")):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            
            # Calculate loss
            if isinstance(outputs, tuple):
                loss = criterion(outputs[0], batch.get("labels"))
            else:
                loss = criterion(outputs, batch.get("labels"))
            
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Track total loss
            total_loss += loss.item() * gradient_accumulation_steps
            total_samples += batch.get("labels").size(0) if "labels" in batch else 1
            
            # Update parameters after accumulating gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                
                # Optimizer step
                self.optimizer.step()
                
                # Scheduler step (if using step-based scheduler)
                if self.scheduler is not None and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Update global step
                self.global_step += 1
            
            # Compute metrics
            if metrics:
                with torch.no_grad():
                    # Get predictions
                    if isinstance(outputs, tuple):
                        preds = outputs[0].argmax(dim=-1) if outputs[0].dim() > 1 else outputs[0]
                    else:
                        preds = outputs.argmax(dim=-1) if outputs.dim() > 1 else outputs
                    
                    # Calculate metrics
                    for name, metric_fn in metrics.items():
                        epoch_metrics[name] += metric_fn(preds.cpu(), batch.get("labels").cpu()).item()
            
            # Log progress
            if (batch_idx + 1) % log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                logger.info(f"Epoch: {self.current_epoch}, Batch: {batch_idx+1}/{len(train_loader)}, "
                          f"Loss: {avg_loss:.4f}, LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        
        # Compute average metrics
        avg_loss = total_loss / len(train_loader)
        avg_metrics = {name: value / len(train_loader) for name, value in epoch_metrics.items()}
        
        # Update training metrics
        self.metrics["train_loss"].append(avg_loss)
        for name, value in avg_metrics.items():
            if name not in self.metrics["train_metrics"]:
                self.metrics["train_metrics"][name] = []
            self.metrics["train_metrics"][name].append(value)
        
        # Log epoch results
        duration = time.time() - start_time
        metrics_str = ", ".join([f"{name}: {value:.4f}" for name, value in avg_metrics.items()])
        logger.info(f"Training Epoch {self.current_epoch} completed in {duration:.2f}s, "
                    f"Loss: {avg_loss:.4f}, {metrics_str}")
        
        return {"loss": avg_loss, **avg_metrics}
    
    def evaluate(
        self,
        eval_loader: DataLoader,
        criterion: Callable,
        metrics: Optional[Dict[str, Callable]] = None
    ) -> Dict[str, float]:
        """
        Evaluate the model on the validation set.
        
        Args:
            eval_loader: DataLoader for evaluation data
            criterion: Loss function
            metrics: Dictionary of metric functions
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0
        total_samples = 0
        all_preds = []
        all_labels = []
        epoch_metrics = {name: 0.0 for name in (metrics or {}).keys()}
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluation"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                
                # Calculate loss
                if isinstance(outputs, tuple):
                    loss = criterion(outputs[0], batch.get("labels"))
                    preds = outputs[0].argmax(dim=-1) if outputs[0].dim() > 1 else outputs[0]
                else:
                    loss = criterion(outputs, batch.get("labels"))
                    preds = outputs.argmax(dim=-1) if outputs.dim() > 1 else outputs
                
                # Track total loss
                total_loss += loss.item()
                total_samples += batch.get("labels").size(0) if "labels" in batch else 1
                
                # Store predictions and labels for metrics
                all_preds.append(preds.cpu())
                all_labels.append(batch.get("labels").cpu())
        
        # Concatenate predictions and labels
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Compute average loss
        avg_loss = total_loss / len(eval_loader)
        
        # Compute metrics
        avg_metrics = {}
        if metrics:
            for name, metric_fn in metrics.items():
                avg_metrics[name] = metric_fn(all_preds, all_labels).item()
        
        # Update validation metrics
        self.metrics["val_loss"].append(avg_loss)
        for name, value in avg_metrics.items():
            if name not in self.metrics["val_metrics"]:
                self.metrics["val_metrics"][name] = []
            self.metrics["val_metrics"][name].append(value)
        
        # Update best validation metrics
        if avg_loss < self.metrics["best_val_loss"]:
            self.metrics["best_val_loss"] = avg_loss
            self.metrics["best_epoch"] = self.current_epoch
        
        # If using F1 score or accuracy as primary metric
        primary_metric = avg_metrics.get("f1", avg_metrics.get("accuracy", None))
        if primary_metric is not None and primary_metric > self.metrics["best_val_metric"]:
            self.metrics["best_val_metric"] = primary_metric
            self.metrics["best_epoch"] = self.current_epoch
        
        # Log evaluation results
        metrics_str = ", ".join([f"{name}: {value:.4f}" for name, value in avg_metrics.items()])
        logger.info(f"Validation Epoch {self.current_epoch}, "
                    f"Loss: {avg_loss:.4f}, {metrics_str}")
        
        # Update scheduler if using ReduceLROnPlateau
        if self.scheduler is not None and isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(avg_loss)
        
        return {"loss": avg_loss, **avg_metrics}
    
    def train(
        self,
        train_loader: DataLoader,
        criterion: Callable,
        eval_loader: Optional[DataLoader] = None,
        metrics: Optional[Dict[str, Callable]] = None,
        num_epochs: int = 10,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        log_interval: int = 10,
        checkpoint_dir: Optional[str] = None,
        save_best_only: bool = True,
        early_stopping_patience: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: DataLoader for training data
            criterion: Loss function
            eval_loader: DataLoader for evaluation data
            metrics: Dictionary of metric functions
            num_epochs: Number of epochs to train
            gradient_accumulation_steps: Number of steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping
            log_interval: Interval for logging training metrics
            checkpoint_dir: Directory to save checkpoints
            save_best_only: Whether to save only the best model
            early_stopping_patience: Number of epochs to wait for improvement
            
        Returns:
            Dictionary of training metrics
        """
        # Set up checkpoint directory
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize early stopping counter
        early_stopping_counter = 0
        best_val_loss = float("inf")
        
        # Training loop
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train one epoch
            train_metrics = self.train_epoch(
                train_loader=train_loader,
                criterion=criterion,
                metrics=metrics,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_grad_norm=max_grad_norm,
                log_interval=log_interval
            )
            
            # Evaluate on validation set
            if eval_loader is not None:
                val_metrics = self.evaluate(
                    eval_loader=eval_loader,
                    criterion=criterion,
                    metrics=metrics
                )
                
                # Check for early stopping
                if early_stopping_patience is not None:
                    if val_metrics["loss"] < best_val_loss:
                        best_val_loss = val_metrics["loss"]
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1
                        logger.info(f"EarlyStopping counter: {early_stopping_counter} out of {early_stopping_patience}")
                        
                        if early_stopping_counter >= early_stopping_patience:
                            logger.info(f"Early stopping triggered after {epoch+1} epochs")
                            break
                
                # Save checkpoint
                if checkpoint_dir:
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
                    best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
                    
                    # Save current checkpoint
                    if not save_best_only:
                        self.save_checkpoint(checkpoint_path)
                    
                    # Save best model
                    if val_metrics["loss"] == self.metrics["best_val_loss"]:
                        self.save_checkpoint(best_checkpoint_path)
                        logger.info(f"Saved best model checkpoint to {best_checkpoint_path}")
            
            # Save last checkpoint regardless of validation
            elif checkpoint_dir:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
                self.save_checkpoint(checkpoint_path)
        
        # Save final metrics
        if checkpoint_dir:
            metrics_path = os.path.join(checkpoint_dir, "training_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(self.metrics, f, indent=2)
            
            logger.info(f"Saved training metrics to {metrics_path}")
        
        return self.metrics
    
    def save_checkpoint(self, path: str) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "metrics": self.metrics,
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "config": self.config
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str, load_optimizer: bool = True) -> None:
        """
        Load model checkpoint.
        
        Args:
            path: Path to load checkpoint from
            load_optimizer: Whether to load optimizer state
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state if requested
        if load_optimizer and self.optimizer and checkpoint.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler state if available
        if load_optimizer and self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Load metrics and training state
        self.metrics = checkpoint.get("metrics", self.metrics)
        self.current_epoch = checkpoint.get("current_epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        
        # Update configuration
        if checkpoint.get("config"):
            self.config.update(checkpoint["config"])
        
        logger.info(f"Loaded checkpoint from {path}")