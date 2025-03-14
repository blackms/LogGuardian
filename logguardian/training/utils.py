"""
Utility functions for LogGuardian training.
"""
import os
import random
import math
from typing import Dict, Any, Optional, Union, Tuple, List, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, Subset, DataLoader, WeightedRandomSampler
from loguru import logger
from sklearn.utils import resample


def compute_class_weights(labels: Union[List, np.ndarray, torch.Tensor]) -> torch.Tensor:
    """
    Compute class weights inversely proportional to class frequencies.
    
    Args:
        labels: Labels to compute weights for
        
    Returns:
        Tensor of class weights
    """
    # Convert to numpy array if needed
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    elif isinstance(labels, list):
        labels = np.array(labels)
    
    # Count class occurrences
    unique_classes, class_counts = np.unique(labels, return_counts=True)
    
    # Compute weights inversely proportional to class frequencies
    n_samples = len(labels)
    n_classes = len(unique_classes)
    
    # Calculate weights: n_samples / (n_classes * class_count)
    weights = n_samples / (n_classes * class_counts)
    
    # Create weight map
    weight_map = {cls: weight for cls, weight in zip(unique_classes, weights)}
    
    # Map weights to original labels
    sample_weights = np.array([weight_map[label] for label in labels])
    
    return torch.from_numpy(sample_weights).float()


def oversample_minority_class(
    data: Union[List, np.ndarray],
    labels: Union[List, np.ndarray],
    beta: float = 0.3,
    strategy: str = "simple"
) -> Tuple[Union[List, np.ndarray], Union[List, np.ndarray]]:
    """
    Oversample minority class to achieve target proportion (beta).
    
    As described in the LogLLM paper, if the proportion of the minority
    class is α and α < β, and the total number of samples is Sample_num,
    the minority class will be oversampled to:
    
    (β(1-α))/(1-β) * Sample_num
    
    Args:
        data: Data samples (log sequences)
        labels: Class labels
        beta: Target proportion of minority class
        strategy: Oversampling strategy:
            "simple": Basic random oversampling
            "smote": SMOTE-based oversampling (requires sklearn.imblearn)
        
    Returns:
        Tuple of (oversampled_data, oversampled_labels)
    """
    # Convert to numpy arrays if needed
    if isinstance(data, list):
        data = np.array(data, dtype=object)
    if isinstance(labels, list):
        labels = np.array(labels)
    
    # Count class occurrences
    unique_classes, class_counts = np.unique(labels, return_counts=True)
    
    # If only one class is present, return original data
    if len(unique_classes) <= 1:
        logger.warning("Only one class present in the dataset, no oversampling performed")
        return data, labels
    
    # Find the minority class
    minority_class = unique_classes[np.argmin(class_counts)]
    majority_class = unique_classes[np.argmax(class_counts)]
    
    # Calculate current proportion of minority class
    total_samples = len(labels)
    minority_samples = np.sum(labels == minority_class)
    alpha = minority_samples / total_samples
    
    # If minority class proportion is already >= beta, no oversampling needed
    if alpha >= beta:
        logger.info(f"Minority class proportion ({alpha:.4f}) already meets target ({beta:.4f})")
        return data, labels
    
    # Calculate number of samples needed for minority class based on formula
    target_minority_samples = int((beta * (1 - alpha)) / (1 - beta) * total_samples)
    
    logger.info(f"Current minority class proportion: {alpha:.4f}, target: {beta:.4f}")
    logger.info(f"Oversampling minority class ({minority_class}) from {minority_samples} to {target_minority_samples} samples")
    
    # Get indices of minority and majority class samples
    minority_indices = np.where(labels == minority_class)[0]
    
    # Perform oversampling based on selected strategy
    if strategy == "simple":
        # Simple random oversampling with replacement
        oversampling_indices = np.random.choice(
            minority_indices,
            size=target_minority_samples - minority_samples,
            replace=True
        )
        
        # Combine original data with oversampled data
        oversampled_data = np.concatenate([data, data[oversampling_indices]])
        oversampled_labels = np.concatenate([labels, labels[oversampling_indices]])
        
    elif strategy == "smote":
        try:
            from imblearn.over_sampling import SMOTE
            
            # Reshape data for SMOTE if needed
            if isinstance(data, np.ndarray) and data.ndim == 1:
                # Handle 1D object arrays (e.g., strings or lists)
                if data.dtype == object:
                    # Convert to feature representation if possible
                    # This is a simplified approach, assuming data can be vectorized
                    # For real log data, you might need more complex feature extraction
                    logger.warning("SMOTE requires vectorized data; attempting to convert object array")
                    
                    # For now, revert to simple oversampling for object arrays
                    return oversample_minority_class(data, labels, beta, strategy="simple")
            
            # Calculate sampling strategy based on beta
            sampling_ratio = {
                minority_class: target_minority_samples,
                majority_class: np.sum(labels == majority_class)
            }
            
            # Apply SMOTE
            smote = SMOTE(sampling_strategy=sampling_ratio, random_state=42)
            oversampled_data, oversampled_labels = smote.fit_resample(data, labels)
            
        except (ImportError, ValueError, TypeError) as e:
            logger.warning(f"Error applying SMOTE: {e}. Falling back to simple oversampling.")
            return oversample_minority_class(data, labels, beta, strategy="simple")
    else:
        raise ValueError(f"Unknown oversampling strategy: {strategy}")
    
    # Verify the new minority class proportion
    new_minority_samples = np.sum(oversampled_labels == minority_class)
    new_total_samples = len(oversampled_labels)
    new_alpha = new_minority_samples / new_total_samples
    
    logger.info(f"New minority class proportion: {new_alpha:.4f} ({new_minority_samples}/{new_total_samples})")
    
    return oversampled_data, oversampled_labels


def create_class_balanced_sampler(
    labels: Union[List, np.ndarray, torch.Tensor],
    beta: Optional[float] = None
) -> WeightedRandomSampler:
    """
    Create a sampler that oversamples the minority class during training.
    
    This is an alternative approach to oversampling that doesn't modify
    the dataset but instead changes the sampling probabilities.
    
    Args:
        labels: Class labels
        beta: Target proportion of minority class (if None, use balanced sampling)
        
    Returns:
        WeightedRandomSampler that oversamples the minority class
    """
    # Convert to numpy array if needed
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    elif isinstance(labels, list):
        labels = np.array(labels)
    
    # Count class occurrences
    unique_classes, class_counts = np.unique(labels, return_counts=True)
    
    # Calculate class weights
    if beta is None:
        # Simple balanced sampling: weights inversely proportional to class frequency
        weights = 1.0 / class_counts
        weights = weights / weights.sum() * len(unique_classes)  # Normalize
    else:
        # Target specific beta proportion
        majority_class = unique_classes[np.argmax(class_counts)]
        minority_class = unique_classes[np.argmin(class_counts)]
        
        # Set weights to achieve beta ratio
        weights = np.ones_like(class_counts, dtype=float)
        majority_idx = np.where(unique_classes == majority_class)[0][0]
        minority_idx = np.where(unique_classes == minority_class)[0][0]
        
        # Calculate weight for minority class to achieve beta ratio
        # beta = (w_min * n_min) / (w_min * n_min + w_maj * n_maj)
        # Solving for w_min with w_maj = 1:
        # w_min = (beta * n_maj) / ((1 - beta) * n_min)
        majority_count = class_counts[majority_idx]
        minority_count = class_counts[minority_idx]
        
        minority_weight = (beta * majority_count) / ((1 - beta) * minority_count)
        weights[minority_idx] = minority_weight
    
    # Create sample weights: weight for each sample based on its class
    sample_weights = np.array([weights[np.where(unique_classes == label)[0][0]] for label in labels])
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(labels),
        replacement=True
    )
    
    return sampler


def create_balanced_dataloader(
    dataset: Dataset,
    batch_size: int,
    beta: Optional[float] = None,
    labels_attr: str = "labels",
    shuffle: bool = True,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader with balanced sampling.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        beta: Target proportion of minority class (if None, use balanced sampling)
        labels_attr: Attribute name for labels in the dataset
        shuffle: Whether to shuffle the data
        **kwargs: Additional arguments for DataLoader
        
    Returns:
        DataLoader with balanced sampling
    """
    # Get labels from dataset
    if hasattr(dataset, labels_attr):
        labels = getattr(dataset, labels_attr)
    elif isinstance(dataset[0], tuple) and len(dataset[0]) >= 2:
        # Assume dataset returns (data, label) tuples
        labels = [item[1] for item in dataset]
    else:
        raise ValueError(f"Cannot extract labels from dataset. Please specify labels_attr.")
    
    # Create sampler
    if shuffle:
        sampler = create_class_balanced_sampler(labels, beta)
        kwargs.pop("sampler", None)  # Remove existing sampler if present
    else:
        sampler = None
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None and shuffle),
        **kwargs
    )
    
    return dataloader


def get_class_distribution(labels: Union[List, np.ndarray, torch.Tensor]) -> Dict[Any, int]:
    """
    Get the distribution of classes in the dataset.
    
    Args:
        labels: Class labels
        
    Returns:
        Dictionary mapping class labels to counts
    """
    # Convert to numpy array if needed
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    elif isinstance(labels, list):
        labels = np.array(labels)
    
    # Count class occurrences
    unique_classes, class_counts = np.unique(labels, return_counts=True)
    
    # Create distribution dictionary
    distribution = {class_label: count for class_label, count in zip(unique_classes, class_counts)}
    
    return distribution