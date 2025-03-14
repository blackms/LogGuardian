"""
Evaluator for log anomaly detection models.
"""
import os
import time
import json
from typing import Dict, Any, Optional, Union, Tuple, List, Callable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from tqdm.auto import tqdm

from logguardian.evaluation.metrics import (
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_roc_curve,
    compute_precision_recall_curve,
    compute_threshold_metrics,
    find_optimal_threshold,
    save_metrics
)


class Evaluator:
    """
    Evaluator for log anomaly detection models.
    
    This class provides methods for evaluating LogGuardian models
    on various log datasets, computing performance metrics, and
    visualizing results.
    """
    
    def __init__(
        self,
        model,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            model: LogGuardian model to evaluate
            config: Configuration parameters
        """
        self.model = model
        self.config = config or {}
        
        # Evaluation results
        self.results = {}
        
        # Evaluation settings
        self.show_progress = self.config.get("show_progress", True)
        self.output_dir = self.config.get("output_dir", "evaluation_results")
        
        # Create output directory if specified
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
    
    def evaluate(
        self,
        test_logs: List[List[str]],
        test_labels: List[int],
        dataset_name: str = "unnamed_dataset",
        batch_size: int = 16,
        raw_output: bool = True,
        save_results: bool = True,
        compute_thresholds: bool = True,
        threshold_steps: int = 100,
        compute_optimal_threshold: bool = True,
        optimal_metric: str = "f1",
        time_evaluation: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate model on test dataset.
        
        Args:
            test_logs: List of log sequences
            test_labels: Ground truth labels
            dataset_name: Name of dataset
            batch_size: Batch size for prediction
            raw_output: Whether to get raw output from model
            save_results: Whether to save evaluation results
            compute_thresholds: Whether to compute metrics at different thresholds
            threshold_steps: Number of threshold steps to evaluate
            compute_optimal_threshold: Whether to find optimal threshold
            optimal_metric: Metric to optimize threshold for
            time_evaluation: Whether to measure inference time
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating model on {dataset_name} dataset ({len(test_logs)} sequences)")
        
        # Detect anomalies
        start_time = time.time()
        
        if self.show_progress:
            logger.info("Running model inference...")
            
        # Get model predictions
        detection_results = self.model.detect(
            test_logs,
            batch_size=batch_size,
            raw_output=True  # Always get raw output for evaluation
        )
        
        # Measure inference time
        inference_time = time.time() - start_time
        inference_time_per_sample = inference_time / len(test_logs)
        
        if time_evaluation:
            logger.info(f"Inference time: {inference_time:.2f}s total, {inference_time_per_sample*1000:.2f}ms per sample")
        
        # Extract predictions and scores
        y_pred = []
        y_score = []
        
        for result in detection_results:
            # Extract label (0=normal, 1=anomalous)
            label = result.get("label_id", 0)
            y_pred.append(label)
            
            # Extract confidence score for anomaly class
            scores = result.get("scores", [0, 0])
            anomaly_score = scores[1] if len(scores) > 1 else scores[0]
            y_score.append(anomaly_score)
        
        # Convert to numpy arrays
        y_true = np.array(test_labels)
        y_pred = np.array(y_pred)
        y_score = np.array(y_score)
        
        # Compute classification metrics
        metrics = compute_classification_metrics(y_true, y_pred, y_score)
        
        # Compute confusion matrix
        conf_matrix = compute_confusion_matrix(y_true, y_pred)
        
        # Compute ROC curve
        fpr, tpr, roc_thresholds = compute_roc_curve(y_true, y_score)
        
        # Compute Precision-Recall curve
        precision_curve, recall_curve, pr_thresholds = compute_precision_recall_curve(y_true, y_score)
        
        # Compute metrics at different thresholds
        threshold_metrics = None
        if compute_thresholds:
            thresholds = np.linspace(0, 1, threshold_steps)
            threshold_metrics = compute_threshold_metrics(y_true, y_score, thresholds)
        
        # Find optimal threshold
        optimal_threshold = None
        if compute_optimal_threshold:
            threshold, value = find_optimal_threshold(y_true, y_score, metric=optimal_metric)
            optimal_threshold = {
                "threshold": float(threshold),
                f"best_{optimal_metric}": float(value)
            }
            
            # Compute metrics at optimal threshold
            y_pred_optimal = (y_score >= threshold).astype(int)
            optimal_metrics = compute_classification_metrics(y_true, y_pred_optimal)
            optimal_threshold["metrics"] = optimal_metrics
        
        # Compile results
        results = {
            "dataset": dataset_name,
            "num_samples": len(test_logs),
            "metrics": metrics,
            "confusion_matrix": conf_matrix.tolist(),
            "roc_curve": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": roc_thresholds.tolist()
            },
            "pr_curve": {
                "precision": precision_curve.tolist(),
                "recall": recall_curve.tolist(),
                "thresholds": pr_thresholds.tolist() if len(pr_thresholds) > 0 else []
            },
            "threshold_metrics": threshold_metrics,
            "optimal_threshold": optimal_threshold,
            "inference_time": {
                "total_seconds": inference_time,
                "per_sample_ms": inference_time_per_sample * 1000
            },
            "timestamp": time.time()
        }
        
        # Save results
        if save_results and self.output_dir:
            results_path = os.path.join(self.output_dir, f"{dataset_name}_results.json")
            save_metrics(results, results_path)
            logger.info(f"Saved evaluation results to {results_path}")
            
            # Save visualizations
            self._save_visualizations(results, dataset_name)
        
        # Store results
        self.results[dataset_name] = results
        
        # Log main metrics
        logger.info(f"Evaluation metrics on {dataset_name}:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return results
    
    def cross_validate(
        self,
        logs: List[List[str]],
        labels: List[int],
        dataset_name: str = "unnamed_dataset",
        n_splits: int = 5,
        stratify: bool = True,
        random_state: Optional[int] = 42,
        time_based: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Args:
            logs: List of log sequences
            labels: Ground truth labels
            dataset_name: Name of dataset
            n_splits: Number of cross-validation splits
            stratify: Whether to stratify splits by label
            random_state: Random seed
            time_based: Whether to use time-based splitting (chronological)
            **kwargs: Additional arguments for evaluate method
            
        Returns:
            Cross-validation results
        """
        logger.info(f"Performing {n_splits}-fold cross-validation on {dataset_name} dataset")
        
        from sklearn.model_selection import KFold, StratifiedKFold
        
        # Convert to numpy arrays
        logs_array = np.array(logs, dtype=object)
        labels_array = np.array(labels)
        
        # Create splitter
        if time_based:
            # For time-based splitting, use regular KFold without shuffling
            splitter = KFold(n_splits=n_splits, shuffle=False)
        elif stratify:
            # For stratified splitting
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        else:
            # For random splitting
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        # Initialize lists to store results
        all_metrics = []
        all_confusion_matrices = []
        fold_results = []
        
        # Perform cross-validation
        for fold, (train_idx, test_idx) in enumerate(splitter.split(logs_array, labels_array if stratify else None)):
            logger.info(f"Evaluating fold {fold+1}/{n_splits}")
            
            # Split data
            train_logs = logs_array[train_idx].tolist()
            train_labels = labels_array[train_idx].tolist()
            test_logs = logs_array[test_idx].tolist()
            test_labels = labels_array[test_idx].tolist()
            
            # Train model on this fold
            # Note: This assumes model has a 'train' method - adjust according to your model
            if hasattr(self.model, 'train'):
                logger.info(f"Training model on fold {fold+1}")
                self.model.train(train_logs, train_labels)
            
            # Evaluate model on this fold
            fold_name = f"{dataset_name}_fold{fold+1}"
            results = self.evaluate(
                test_logs,
                test_labels,
                dataset_name=fold_name,
                **kwargs
            )
            
            # Store results
            all_metrics.append(results["metrics"])
            all_confusion_matrices.append(results["confusion_matrix"])
            fold_results.append(results)
        
        # Compute average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [metrics[key] for metrics in all_metrics]
            avg_metrics[key] = float(np.mean(values))
            avg_metrics[f"{key}_std"] = float(np.std(values))
        
        # Compute average confusion matrix
        avg_confusion_matrix = np.mean([np.array(cm) for cm in all_confusion_matrices], axis=0).tolist()
        
        # Compile cross-validation results
        cv_results = {
            "dataset": dataset_name,
            "n_splits": n_splits,
            "stratify": stratify,
            "time_based": time_based,
            "avg_metrics": avg_metrics,
            "avg_confusion_matrix": avg_confusion_matrix,
            "fold_results": fold_results,
            "timestamp": time.time()
        }
        
        # Save results
        if kwargs.get("save_results", True) and self.output_dir:
            cv_results_path = os.path.join(self.output_dir, f"{dataset_name}_cv_results.json")
            
            # Create a simplified version without fold details to save space
            simplified_cv_results = cv_results.copy()
            simplified_cv_results.pop("fold_results", None)
            
            save_metrics(simplified_cv_results, cv_results_path)
            logger.info(f"Saved cross-validation results to {cv_results_path}")
        
        # Store results
        self.results[f"{dataset_name}_cv"] = cv_results
        
        # Log average metrics
        logger.info(f"Average cross-validation metrics on {dataset_name}:")
        for key, value in avg_metrics.items():
            if not key.endswith("_std"):
                std = avg_metrics.get(f"{key}_std", 0)
                logger.info(f"  {key}: {value:.4f} ± {std:.4f}")
        
        return cv_results
    
    def _save_visualizations(
        self,
        results: Dict[str, Any],
        dataset_name: str
    ) -> None:
        """
        Save visualizations of evaluation results.
        
        Args:
            results: Evaluation results
            dataset_name: Name of dataset
        """
        if not self.output_dir:
            return
        
        # Create visualizations directory
        vis_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Plot confusion matrix
        self._plot_confusion_matrix(
            results["confusion_matrix"],
            os.path.join(vis_dir, f"{dataset_name}_confusion_matrix.png")
        )
        
        # Plot ROC curve
        self._plot_roc_curve(
            results["roc_curve"]["fpr"],
            results["roc_curve"]["tpr"],
            results["metrics"].get("auc_roc", None),
            os.path.join(vis_dir, f"{dataset_name}_roc_curve.png")
        )
        
        # Plot precision-recall curve
        self._plot_pr_curve(
            results["pr_curve"]["precision"],
            results["pr_curve"]["recall"],
            results["metrics"].get("avg_precision", None),
            os.path.join(vis_dir, f"{dataset_name}_pr_curve.png")
        )
        
        # Plot threshold metrics if available
        if results.get("threshold_metrics"):
            self._plot_threshold_metrics(
                results["threshold_metrics"],
                os.path.join(vis_dir, f"{dataset_name}_threshold_metrics.png")
            )
    
    def _plot_confusion_matrix(
        self,
        conf_matrix: List[List[float]],
        output_path: str
    ) -> None:
        """
        Plot confusion matrix.
        
        Args:
            conf_matrix: Confusion matrix
            output_path: Path to save plot
        """
        plt.figure(figsize=(8, 6))
        
        # Convert to numpy array
        cm = np.array(conf_matrix)
        
        # Plot confusion matrix
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        # Add labels
        classes = ['Normal', 'Anomaly']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, f"{cm[i, j]:.0f}",
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curve(
        self,
        fpr: List[float],
        tpr: List[float],
        auc_roc: Optional[float],
        output_path: str
    ) -> None:
        """
        Plot ROC curve.
        
        Args:
            fpr: False positive rates
            tpr: True positive rates
            auc_roc: Area under ROC curve
            output_path: Path to save plot
        """
        plt.figure(figsize=(8, 6))
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_roc:.4f})' if auc_roc else 'ROC curve')
        
        # Plot random guess line
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        # Set plot properties
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        # Save plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pr_curve(
        self,
        precision: List[float],
        recall: List[float],
        avg_precision: Optional[float],
        output_path: str
    ) -> None:
        """
        Plot precision-recall curve.
        
        Args:
            precision: Precision values
            recall: Recall values
            avg_precision: Average precision
            output_path: Path to save plot
        """
        plt.figure(figsize=(8, 6))
        
        # Plot precision-recall curve
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'Precision-Recall curve (AP = {avg_precision:.4f})' if avg_precision else 'Precision-Recall curve')
        
        # Set plot properties
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        
        # Save plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_threshold_metrics(
        self,
        threshold_metrics: Dict[str, List[float]],
        output_path: str
    ) -> None:
        """
        Plot metrics at different thresholds.
        
        Args:
            threshold_metrics: Metrics at different thresholds
            output_path: Path to save plot
        """
        plt.figure(figsize=(10, 6))
        
        # Get data
        thresholds = threshold_metrics["thresholds"]
        metrics_to_plot = ["precision", "recall", "f1", "accuracy"]
        
        # Plot metrics
        for metric in metrics_to_plot:
            if metric in threshold_metrics:
                plt.plot(thresholds, threshold_metrics[metric], lw=2, label=metric.capitalize())
        
        # Set plot properties
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Metrics at Different Thresholds')
        plt.legend(loc="best")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(
        self,
        dataset_results: Optional[Dict[str, Dict[str, Any]]] = None,
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            dataset_results: Dictionary of evaluation results by dataset
            output_file: Path to save report
            
        Returns:
            Report string
        """
        # Use provided results or stored results
        results = dataset_results or self.results
        
        # Create Markdown report
        report = "# LogGuardian Evaluation Report\n\n"
        report += f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Add dataset sections
        for dataset_name, dataset_result in results.items():
            # Skip cross-validation results (handled separately)
            if dataset_name.endswith("_cv"):
                continue
            
            report += f"## Dataset: {dataset_name}\n\n"
            
            # Add main metrics
            if "metrics" in dataset_result:
                report += "### Performance Metrics\n\n"
                report += "| Metric | Value |\n"
                report += "|--------|-------|\n"
                for key, value in dataset_result["metrics"].items():
                    report += f"| {key.capitalize()} | {value:.4f} |\n"
                report += "\n"
            
            # Add confusion matrix
            if "confusion_matrix" in dataset_result:
                report += "### Confusion Matrix\n\n"
                report += "| | Predicted Normal | Predicted Anomaly |\n"
                report += "|-------------------|-------------------|------------------|\n"
                cm = dataset_result["confusion_matrix"]
                report += f"| **Actual Normal** | {cm[0][0]:.0f} | {cm[0][1]:.0f} |\n"
                report += f"| **Actual Anomaly** | {cm[1][0]:.0f} | {cm[1][1]:.0f} |\n\n"
            
            # Add optimal threshold information
            if "optimal_threshold" in dataset_result:
                ot = dataset_result["optimal_threshold"]
                report += "### Optimal Threshold\n\n"
                report += f"- Threshold: {ot['threshold']:.4f}\n"
                metric = [k for k in ot.keys() if k.startswith("best_")][0].replace("best_", "")
                report += f"- Best {metric}: {ot[f'best_{metric}']:.4f}\n\n"
                
                if "metrics" in ot:
                    report += "#### Metrics at Optimal Threshold\n\n"
                    report += "| Metric | Value |\n"
                    report += "|--------|-------|\n"
                    for key, value in ot["metrics"].items():
                        report += f"| {key.capitalize()} | {value:.4f} |\n"
                    report += "\n"
            
            # Add inference time information
            if "inference_time" in dataset_result:
                report += "### Inference Time\n\n"
                report += f"- Total: {dataset_result['inference_time']['total_seconds']:.2f} seconds\n"
                report += f"- Per sample: {dataset_result['inference_time']['per_sample_ms']:.2f} ms\n\n"
            
            report += "---\n\n"
        
        # Add cross-validation results
        cv_results = {k: v for k, v in results.items() if k.endswith("_cv")}
        if cv_results:
            report += "## Cross-Validation Results\n\n"
            
            for cv_name, cv_result in cv_results.items():
                dataset_name = cv_name.replace("_cv", "")
                report += f"### Dataset: {dataset_name}\n\n"
                report += f"- Number of splits: {cv_result['n_splits']}\n"
                report += f"- Stratified: {cv_result['stratify']}\n"
                report += f"- Time-based: {cv_result['time_based']}\n\n"
                
                # Add average metrics
                if "avg_metrics" in cv_result:
                    report += "#### Average Metrics\n\n"
                    report += "| Metric | Value | Std. Dev. |\n"
                    report += "|--------|-------|----------|\n"
                    for key, value in cv_result["avg_metrics"].items():
                        if not key.endswith("_std"):
                            std = cv_result["avg_metrics"].get(f"{key}_std", 0)
                            report += f"| {key.capitalize()} | {value:.4f} | ±{std:.4f} |\n"
                    report += "\n"
                
                report += "---\n\n"
        
        # Add model summary
        if hasattr(self.model, 'summary'):
            report += "## Model Summary\n\n"
            try:
                report += self.model.summary() + "\n\n"
            except:
                report += "*Model summary not available*\n\n"
        
        # Save report if output_file specified
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w') as f:
                f.write(report)
            
            logger.info(f"Saved evaluation report to {output_file}")
        
        return report