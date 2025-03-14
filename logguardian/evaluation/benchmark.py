"""
Benchmark framework for log anomaly detection.
"""
import os
import time
import json
from typing import Dict, Any, Optional, Union, Tuple, List, Callable
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from tqdm.auto import tqdm

from logguardian.evaluation.evaluator import Evaluator
from logguardian.evaluation.metrics import (
    compute_classification_metrics,
    save_metrics
)


class LogAnomalyBenchmark:
    """
    Benchmark framework for log anomaly detection.
    
    This class provides methods for benchmarking multiple log anomaly detection
    models on various datasets, comparing their performance, and generating
    comprehensive reports.
    """
    
    def __init__(
        self,
        methods: Optional[Dict[str, Any]] = None,
        datasets: Optional[Dict[str, Tuple[List[List[str]], List[int]]]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize benchmark.
        
        Args:
            methods: Dictionary of methods to benchmark, mapping method names to models
            datasets: Dictionary of datasets to benchmark on, mapping dataset names to (logs, labels) tuples
            config: Configuration parameters
        """
        self.methods = methods or {}
        self.datasets = datasets or {}
        self.config = config or {}
        
        # Benchmark results
        self.results = {}
        
        # Benchmark settings
        self.show_progress = self.config.get("show_progress", True)
        self.output_dir = self.config.get("output_dir", "benchmark_results")
        
        # Create output directory if specified
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
    
    def add_method(self, name: str, model: Any) -> None:
        """
        Add a method to benchmark.
        
        Args:
            name: Name of the method
            model: Model to benchmark
        """
        self.methods[name] = model
        logger.info(f"Added method: {name}")
    
    def add_dataset(
        self,
        name: str,
        logs: List[List[str]],
        labels: List[int]
    ) -> None:
        """
        Add a dataset to benchmark on.
        
        Args:
            name: Name of the dataset
            logs: List of log sequences
            labels: Ground truth labels
        """
        self.datasets[name] = (logs, labels)
        logger.info(f"Added dataset: {name} with {len(logs)} sequences")
    
    def load_dataset_from_loader(
        self,
        name: str,
        loader,
        split: bool = True,
        test_size: float = 0.2,
        shuffle: bool = False,
        random_state: Optional[int] = 42
    ) -> None:
        """
        Load a dataset from a data loader.
        
        Args:
            name: Name of the dataset
            loader: Data loader instance
            split: Whether to split data into train and test sets
            test_size: Test set size if split is True
            shuffle: Whether to shuffle data before splitting
            random_state: Random seed
        """
        # Load data if not already loaded
        if hasattr(loader, "logs") and loader.logs is not None:
            # Data already loaded
            pass
        else:
            # Load data
            loader.load()
        
        if split:
            # Split data into train and test sets
            train_logs, train_labels, test_logs, test_labels = loader.get_train_test_split(
                test_size=test_size,
                shuffle=shuffle,
                random_state=random_state
            )
            
            # Add train and test datasets
            self.add_dataset(f"{name}_train", train_logs, train_labels)
            self.add_dataset(f"{name}_test", test_logs, test_labels)
        else:
            # Get all data
            if hasattr(loader, "logs") and hasattr(loader, "labels"):
                logs = loader.logs
                labels = loader.labels
            else:
                # Try to get data through get_train_test_split
                train_logs, train_labels, test_logs, test_labels = loader.get_train_test_split(
                    test_size=0.0,
                    shuffle=False
                )
                logs = train_logs
                labels = train_labels
            
            # Add dataset
            self.add_dataset(name, logs, labels)
    
    def run(
        self,
        method_names: Optional[List[str]] = None,
        dataset_names: Optional[List[str]] = None,
        save_results: bool = True,
        generate_report: bool = True,
        train_methods: bool = True,
        **kwargs
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Run benchmark.
        
        Args:
            method_names: List of method names to benchmark
            dataset_names: List of dataset names to benchmark on
            save_results: Whether to save benchmark results
            generate_report: Whether to generate a benchmark report
            train_methods: Whether to train methods on training datasets
            **kwargs: Additional arguments for evaluate method
            
        Returns:
            Benchmark results
        """
        # Use specified methods or all methods
        method_names = method_names or list(self.methods.keys())
        
        # Use specified datasets or all datasets
        dataset_names = dataset_names or list(self.datasets.keys())
        
        # Filter test datasets for evaluation
        test_dataset_names = [name for name in dataset_names if "_test" in name]
        train_dataset_names = [name for name in dataset_names if "_train" in name]
        
        if not test_dataset_names:
            # If no test datasets are found, just use all datasets
            test_dataset_names = dataset_names
        
        # Initialize benchmark results
        benchmark_results = {}
        
        # Run methods on datasets
        for method_name in method_names:
            logger.info(f"Benchmarking method: {method_name}")
            
            # Get method
            method = self.methods[method_name]
            
            # Train method if required
            if train_methods and hasattr(method, 'train'):
                for train_dataset_name in train_dataset_names:
                    logger.info(f"Training {method_name} on {train_dataset_name}")
                    
                    # Get training data
                    train_logs, train_labels = self.datasets[train_dataset_name]
                    
                    # Train method
                    method.train(train_logs, train_labels)
            
            # Initialize method results
            method_results = {}
            
            # Create evaluator for this method
            evaluator = Evaluator(
                model=method,
                config={
                    "output_dir": os.path.join(self.output_dir, method_name) if self.output_dir else None,
                    "show_progress": self.show_progress
                }
            )
            
            # Evaluate method on test datasets
            for dataset_name in test_dataset_names:
                logger.info(f"Evaluating {method_name} on {dataset_name}")
                
                # Get test data
                test_logs, test_labels = self.datasets[dataset_name]
                
                # Evaluate method
                results = evaluator.evaluate(
                    test_logs=test_logs,
                    test_labels=test_labels,
                    dataset_name=dataset_name,
                    save_results=save_results,
                    **kwargs
                )
                
                # Store results
                method_results[dataset_name] = results
            
            # Store method results
            benchmark_results[method_name] = method_results
        
        # Save benchmark results
        if save_results and self.output_dir:
            # Create a simplified version of results (without large arrays) for saving
            simplified_results = {}
            
            for method_name, method_results in benchmark_results.items():
                simplified_results[method_name] = {}
                
                for dataset_name, dataset_results in method_results.items():
                    # Include only metrics and basic information
                    simplified_results[method_name][dataset_name] = {
                        "metrics": dataset_results["metrics"],
                        "dataset": dataset_results["dataset"],
                        "num_samples": dataset_results["num_samples"],
                        "inference_time": dataset_results["inference_time"],
                        "optimal_threshold": dataset_results.get("optimal_threshold"),
                        "timestamp": dataset_results["timestamp"]
                    }
            
            # Save simplified results
            results_path = os.path.join(self.output_dir, "benchmark_results.json")
            save_metrics(simplified_results, results_path)
            logger.info(f"Saved benchmark results to {results_path}")
        
        # Generate benchmark report
        if generate_report and self.output_dir:
            report_path = os.path.join(self.output_dir, "benchmark_report.md")
            self.generate_report(
                benchmark_results=benchmark_results,
                output_file=report_path
            )
        
        # Store benchmark results
        self.results = benchmark_results
        
        return benchmark_results
    
    def generate_report(
        self,
        benchmark_results: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive benchmark report.
        
        Args:
            benchmark_results: Benchmark results
            output_file: Path to save report
            
        Returns:
            Report string
        """
        # Use provided results or stored results
        results = benchmark_results or self.results
        
        # Create Markdown report
        report = "# Log Anomaly Detection Benchmark Report\n\n"
        report += f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Collect metrics for comparison
        metrics_to_compare = ["accuracy", "precision", "recall", "f1"]
        
        # First, get list of all datasets used
        all_datasets = set()
        for method_results in results.values():
            all_datasets.update(method_results.keys())
        
        all_datasets = sorted(list(all_datasets))
        
        # Create comparison tables for each dataset
        for dataset_name in all_datasets:
            report += f"## Dataset: {dataset_name}\n\n"
            
            # Create metrics table
            report += "### Performance Metrics\n\n"
            report += "| Method | " + " | ".join(metric.capitalize() for metric in metrics_to_compare) + " | Inference Time (ms/sample) |\n"
            report += "|--------|" + "|".join(["-------" for _ in metrics_to_compare]) + "|-------------------------|\n"
            
            # Add results for each method
            for method_name, method_results in results.items():
                if dataset_name in method_results:
                    result = method_results[dataset_name]
                    metrics = result["metrics"]
                    
                    # Create table row
                    row = f"| {method_name} |"
                    
                    # Add metrics
                    for metric in metrics_to_compare:
                        value = metrics.get(metric, float('nan'))
                        row += f" {value:.4f} |"
                    
                    # Add inference time
                    if "inference_time" in result:
                        time_per_sample = result["inference_time"]["per_sample_ms"]
                        row += f" {time_per_sample:.2f} |"
                    else:
                        row += " - |"
                    
                    report += row + "\n"
                else:
                    # Method wasn't evaluated on this dataset
                    report += f"| {method_name} | " + " | ".join(["-" for _ in range(len(metrics_to_compare) + 1)]) + " |\n"
            
            report += "\n"
            
            # Highlight best method for each metric
            report += "### Best Methods\n\n"
            
            for metric in metrics_to_compare:
                report += f"#### {metric.capitalize()}\n\n"
                
                # Collect values for this metric
                metric_values = {}
                for method_name, method_results in results.items():
                    if dataset_name in method_results:
                        result = method_results[dataset_name]
                        metrics = result["metrics"]
                        
                        if metric in metrics:
                            metric_values[method_name] = metrics[metric]
                
                # Find best method
                if metric_values:
                    best_method = max(metric_values.items(), key=lambda x: x[1])
                    report += f"- Best method: **{best_method[0]}** with {metric} = {best_method[1]:.4f}\n"
                    
                    # List all methods sorted by this metric
                    report += "- All methods ranked:\n"
                    
                    for method_name, value in sorted(metric_values.items(), key=lambda x: x[1], reverse=True):
                        report += f"  - {method_name}: {value:.4f}\n"
                else:
                    report += f"No data available for {metric} on this dataset.\n"
                
                report += "\n"
            
            report += "---\n\n"
        
        # Add summary section with best method for each dataset and metric
        report += "## Summary\n\n"
        report += "### Best Method by Dataset and Metric\n\n"
        
        # Create summary table
        report += "| Dataset | " + " | ".join(metric.capitalize() for metric in metrics_to_compare) + " |\n"
        report += "|---------|" + "|".join(["-------" for _ in metrics_to_compare]) + "|\n"
        
        for dataset_name in all_datasets:
            row = f"| {dataset_name} |"
            
            for metric in metrics_to_compare:
                # Find best method for this dataset and metric
                best_method = None
                best_value = float('-inf')
                
                for method_name, method_results in results.items():
                    if dataset_name in method_results:
                        result = method_results[dataset_name]
                        metrics = result["metrics"]
                        
                        if metric in metrics and metrics[metric] > best_value:
                            best_value = metrics[metric]
                            best_method = method_name
                
                if best_method:
                    row += f" **{best_method}** ({best_value:.4f}) |"
                else:
                    row += " - |"
            
            report += row + "\n"
        
        report += "\n"
        
        # Add overall best methods across all datasets
        report += "### Overall Best Methods\n\n"
        
        for metric in metrics_to_compare:
            report += f"#### {metric.capitalize()}\n\n"
            
            # Collect average values for this metric across datasets
            avg_values = defaultdict(list)
            
            for method_name, method_results in results.items():
                for dataset_name, result in method_results.items():
                    metrics = result["metrics"]
                    
                    if metric in metrics:
                        avg_values[method_name].append(metrics[metric])
            
            # Calculate averages
            averages = {method: np.mean(values) for method, values in avg_values.items()}
            
            # Find best method
            if averages:
                best_method = max(averages.items(), key=lambda x: x[1])
                report += f"- Best method: **{best_method[0]}** with average {metric} = {best_method[1]:.4f}\n"
                
                # List all methods sorted by this metric
                report += "- All methods ranked by average performance:\n"
                
                for method_name, value in sorted(averages.items(), key=lambda x: x[1], reverse=True):
                    report += f"  - {method_name}: {value:.4f}\n"
            else:
                report += f"No data available for {metric} across datasets.\n"
            
            report += "\n"
        
        # Add visualizations section
        report += "## Visualizations\n\n"
        report += "Detailed visualizations for each method and dataset can be found in the following directories:\n\n"
        
        for method_name in results.keys():
            report += f"- {method_name}: `{self.output_dir}/{method_name}/visualizations/`\n"
        
        report += "\n"
        
        # Save report if output_file specified
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w') as f:
                f.write(report)
            
            logger.info(f"Saved benchmark report to {output_file}")
        
        return report
    
    def create_comparison_visualizations(
        self,
        metric: str = "f1",
        output_dir: Optional[str] = None
    ) -> None:
        """
        Create comparative visualizations for methods.
        
        Args:
            metric: Metric to compare
            output_dir: Directory to save visualizations
        """
        # Use specified output directory or default
        output_dir = output_dir or os.path.join(self.output_dir, "visualizations")
        os.makedirs(output_dir, exist_ok=True)
        
        # Get list of all datasets
        all_datasets = set()
        for method_results in self.results.values():
            all_datasets.update(method_results.keys())
        
        all_datasets = sorted(list(all_datasets))
        
        # Create grouped bar chart for each metric
        plt.figure(figsize=(12, 8))
        
        # Get all method names
        method_names = list(self.results.keys())
        
        # Set up bar positions
        bar_width = 0.8 / len(method_names)
        r = np.arange(len(all_datasets))
        
        # Create bars for each method
        for i, method_name in enumerate(method_names):
            # Collect values for this method
            values = []
            
            for dataset_name in all_datasets:
                if dataset_name in self.results[method_name]:
                    result = self.results[method_name][dataset_name]
                    metrics = result["metrics"]
                    
                    if metric in metrics:
                        values.append(metrics[metric])
                    else:
                        values.append(0)
                else:
                    values.append(0)
            
            # Add bars
            plt.bar(r + i * bar_width, values, width=bar_width, label=method_name)
        
        # Add labels and legend
        plt.xlabel('Dataset')
        plt.ylabel(f'{metric.capitalize()} Score')
        plt.title(f'Comparison of {metric.capitalize()} Scores Across Methods and Datasets')
        plt.xticks(r + bar_width * (len(method_names) - 1) / 2, all_datasets, rotation=45, ha='right')
        plt.legend()
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_comparison.png"), dpi=300)
        plt.close()
        
        # Create radar chart for overall comparison
        # Only include methods with data for all datasets
        complete_methods = []
        
        for method_name in method_names:
            if all(dataset_name in self.results[method_name] for dataset_name in all_datasets):
                complete_methods.append(method_name)
        
        if complete_methods and len(all_datasets) >= 3:
            # Create radar chart
            plt.figure(figsize=(10, 10))
            
            # Number of variables
            N = len(all_datasets)
            
            # Compute angle for each axis
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Initialize radar chart
            ax = plt.subplot(111, polar=True)
            
            # Draw one axis per variable and add labels
            plt.xticks(angles[:-1], all_datasets)
            
            # Draw y-axis labels (0 to 1)
            plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ['0.2', '0.4', '0.6', '0.8', '1.0'], color='grey', size=8)
            plt.ylim(0, 1)
            
            # Plot data for each method
            for method_name in complete_methods:
                values = []
                
                for dataset_name in all_datasets:
                    result = self.results[method_name][dataset_name]
                    metrics = result["metrics"]
                    
                    if metric in metrics:
                        values.append(metrics[metric])
                    else:
                        values.append(0)
                
                # Close the loop
                values += values[:1]
                
                # Plot values
                ax.plot(angles, values, linewidth=2, linestyle='solid', label=method_name)
                ax.fill(angles, values, alpha=0.1)
            
            # Add legend
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            # Add title
            plt.title(f'Comparison of {metric.capitalize()} Across Datasets')
            
            # Save radar chart
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{metric}_radar_comparison.png"), dpi=300)
            plt.close()