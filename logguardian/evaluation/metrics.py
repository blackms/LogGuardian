"""
Evaluation metrics for log anomaly detection.
"""
import os
import json
from typing import Dict, Any, Optional, Union, Tuple, List

import numpy as np
import torch
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    f1_score
)


def compute_classification_metrics(
    y_true: Union[np.ndarray, List[int], torch.Tensor],
    y_pred: Union[np.ndarray, List[int], torch.Tensor],
    y_score: Optional[Union[np.ndarray, List[float], torch.Tensor]] = None,
    average: str = "binary"
) -> Dict[str, float]:
    """
    Compute standard classification metrics for anomaly detection.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_score: Prediction scores/probabilities (optional)
        average: Averaging method for multi-class metrics
        
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy arrays if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_score, torch.Tensor) and y_score is not None:
        y_score = y_score.cpu().numpy()
    
    # Ensure arrays are 1D
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Compute precision, recall, F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    
    # Compute accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Initialize metrics dictionary
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }
    
    # Add AUC-ROC and Average Precision if scores are provided
    if y_score is not None:
        y_score = np.asarray(y_score).flatten()
        
        # Compute AUC-ROC
        if len(np.unique(y_true)) > 1:  # Skip if only one class
            try:
                auc_roc = roc_auc_score(y_true, y_score)
                metrics["auc_roc"] = float(auc_roc)
            except ValueError as e:
                # Handle case when there's only one class in y_true
                pass
        
        # Compute Average Precision
        try:
            avg_precision = average_precision_score(y_true, y_score)
            metrics["avg_precision"] = float(avg_precision)
        except ValueError as e:
            # Handle case when there's only one class in y_true
            pass
    
    return metrics


def compute_confusion_matrix(
    y_true: Union[np.ndarray, List[int], torch.Tensor],
    y_pred: Union[np.ndarray, List[int], torch.Tensor],
    normalize: Optional[str] = None
) -> np.ndarray:
    """
    Compute confusion matrix for classification.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        normalize: Normalization method ('true', 'pred', 'all', or None)
        
    Returns:
        Confusion matrix
    """
    # Convert to numpy arrays if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if requested
    if normalize is not None:
        if normalize == 'true':
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        elif normalize == 'pred':
            cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
        elif normalize == 'all':
            cm = cm.astype('float') / cm.sum()
        
        # Replace NaN with 0
        cm = np.nan_to_num(cm)
    
    return cm


def compute_metrics_at_k(
    y_true: Union[np.ndarray, List[int], torch.Tensor],
    y_score: Union[np.ndarray, List[float], torch.Tensor],
    k: Union[int, float]
) -> Dict[str, float]:
    """
    Compute metrics at top-k predictions.
    
    Args:
        y_true: Ground truth labels
        y_score: Prediction scores/probabilities
        k: Either an integer (top k items) or a float (top k percent)
        
    Returns:
        Dictionary of metrics at k
    """
    # Convert to numpy arrays if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_score, torch.Tensor):
        y_score = y_score.cpu().numpy()
    
    # Ensure arrays are 1D
    y_true = np.asarray(y_true).flatten()
    y_score = np.asarray(y_score).flatten()
    
    # Determine k
    if isinstance(k, float) and 0 < k <= 1:
        # k is a percentage
        k = int(len(y_true) * k)
    
    k = min(k, len(y_true))
    
    # Get indices of top-k scores
    top_k_indices = np.argsort(y_score)[-k:]
    
    # Create binary prediction based on top-k
    y_pred = np.zeros_like(y_true)
    y_pred[top_k_indices] = 1
    
    # Compute metrics
    metrics = compute_classification_metrics(y_true, y_pred)
    
    # Rename metrics to indicate they're at k
    metrics_at_k = {f"{key}_at_{k}": value for key, value in metrics.items()}
    
    return metrics_at_k


def compute_threshold_metrics(
    y_true: Union[np.ndarray, List[int], torch.Tensor],
    y_score: Union[np.ndarray, List[float], torch.Tensor],
    thresholds: List[float]
) -> Dict[str, List[float]]:
    """
    Compute metrics at different thresholds.
    
    Args:
        y_true: Ground truth labels
        y_score: Prediction scores/probabilities
        thresholds: List of thresholds to evaluate
        
    Returns:
        Dictionary of metrics at different thresholds
    """
    # Convert to numpy arrays if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_score, torch.Tensor):
        y_score = y_score.cpu().numpy()
    
    # Ensure arrays are 1D
    y_true = np.asarray(y_true).flatten()
    y_score = np.asarray(y_score).flatten()
    
    # Initialize metrics dictionaries
    threshold_metrics = {
        "thresholds": thresholds,
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": []
    }
    
    # Compute metrics at each threshold
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        metrics = compute_classification_metrics(y_true, y_pred)
        
        for key, value in metrics.items():
            if key in threshold_metrics:
                threshold_metrics[key].append(value)
    
    return threshold_metrics


def compute_roc_curve(
    y_true: Union[np.ndarray, List[int], torch.Tensor],
    y_score: Union[np.ndarray, List[float], torch.Tensor]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve.
    
    Args:
        y_true: Ground truth labels
        y_score: Prediction scores/probabilities
        
    Returns:
        Tuple of (fpr, tpr, thresholds)
    """
    # Convert to numpy arrays if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_score, torch.Tensor):
        y_score = y_score.cpu().numpy()
    
    # Ensure arrays are 1D
    y_true = np.asarray(y_true).flatten()
    y_score = np.asarray(y_score).flatten()
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    return fpr, tpr, thresholds


def compute_precision_recall_curve(
    y_true: Union[np.ndarray, List[int], torch.Tensor],
    y_score: Union[np.ndarray, List[float], torch.Tensor]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute precision-recall curve.
    
    Args:
        y_true: Ground truth labels
        y_score: Prediction scores/probabilities
        
    Returns:
        Tuple of (precision, recall, thresholds)
    """
    # Convert to numpy arrays if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_score, torch.Tensor):
        y_score = y_score.cpu().numpy()
    
    # Ensure arrays are 1D
    y_true = np.asarray(y_true).flatten()
    y_score = np.asarray(y_score).flatten()
    
    # Compute precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    
    return precision, recall, thresholds


def find_optimal_threshold(
    y_true: Union[np.ndarray, List[int], torch.Tensor],
    y_score: Union[np.ndarray, List[float], torch.Tensor],
    metric: str = "f1"
) -> Tuple[float, float]:
    """
    Find the optimal threshold that maximizes a given metric.
    
    Args:
        y_true: Ground truth labels
        y_score: Prediction scores/probabilities
        metric: Metric to maximize ('f1', 'precision', 'recall', or 'accuracy')
        
    Returns:
        Tuple of (optimal_threshold, best_metric_value)
    """
    # Convert to numpy arrays if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_score, torch.Tensor):
        y_score = y_score.cpu().numpy()
    
    # Ensure arrays are 1D
    y_true = np.asarray(y_true).flatten()
    y_score = np.asarray(y_score).flatten()
    
    # Generate a range of thresholds to evaluate
    thresholds = np.linspace(0, 1, 100)
    
    # Initialize variables to track best threshold and metric value
    best_threshold = 0
    best_metric_value = 0
    
    # Evaluate each threshold
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        
        if metric == "f1":
            metric_value = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "precision":
            precision, _, _, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
            metric_value = precision
        elif metric == "recall":
            _, recall, _, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
            metric_value = recall
        elif metric == "accuracy":
            metric_value = accuracy_score(y_true, y_pred)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        # Update best threshold if this one is better
        if metric_value > best_metric_value:
            best_threshold = threshold
            best_metric_value = metric_value
    
    return best_threshold, best_metric_value


def save_metrics(metrics: Dict[str, Any], path: str) -> None:
    """
    Save metrics to a JSON file.
    
    Args:
        metrics: Dictionary of metrics
        path: Path to save the metrics
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Convert numpy arrays and other non-serializable objects to lists
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (np.ndarray, list)):
            serializable_metrics[key] = [float(v) if isinstance(v, (np.number, np.ndarray)) else v for v in value]
        elif isinstance(value, dict):
            serializable_metrics[key] = {k: float(v) if isinstance(v, (np.number, np.ndarray)) else v 
                                       for k, v in value.items()}
        elif isinstance(value, (np.number, np.ndarray)):
            serializable_metrics[key] = float(value)
        else:
            serializable_metrics[key] = value
    
    # Save to JSON file
    with open(path, 'w') as f:
        json.dump(serializable_metrics, f, indent=4)


def load_metrics(path: str) -> Dict[str, Any]:
    """
    Load metrics from a JSON file.
    
    Args:
        path: Path to load the metrics from
        
    Returns:
        Dictionary of metrics
    """
    with open(path, 'r') as f:
        metrics = json.load(f)
    
    return metrics