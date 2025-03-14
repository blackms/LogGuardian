"""
Example script demonstrating how to use the benchmark framework to evaluate LogGuardian.
"""
import os
import argparse
import numpy as np
import torch
from loguru import logger

from logguardian.pipeline import LogGuardian
from logguardian.data.loaders import (
    HDFSLoader,
    BGLLoader,
    LibertyLoader,
    ThunderbirdLoader
)
from logguardian.data.preprocessors import SystemLogPreprocessor
from logguardian.evaluation.benchmark import LogAnomalyBenchmark

# Import baseline methods
# Replace these with actual implementations or create stub classes for demonstration
class DeepLogModel:
    """Stub for DeepLog model."""
    def __init__(self, name="DeepLog"):
        self.name = name
    
    def detect(self, logs, **kwargs):
        """Stub detection function."""
        import random
        return [{"label_id": random.randint(0, 1), "scores": [0.8, 0.2]} for _ in logs]

class LogAnomalyModel:
    """Stub for LogAnomaly model."""
    def __init__(self, name="LogAnomaly"):
        self.name = name
    
    def detect(self, logs, **kwargs):
        """Stub detection function."""
        import random
        return [{"label_id": random.randint(0, 1), "scores": [0.7, 0.3]} for _ in logs]


def load_datasets(args):
    """
    Load datasets for benchmarking.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Benchmark object with loaded datasets
    """
    # Create benchmark object
    benchmark = LogAnomalyBenchmark(
        config={
            "output_dir": args.output_dir,
            "show_progress": True
        }
    )
    
    # Create preprocessor
    preprocessor = SystemLogPreprocessor()
    
    # Load datasets
    if "hdfs" in args.datasets:
        logger.info("Loading HDFS dataset...")
        
        hdfs_loader = HDFSLoader(
            data_path=os.path.join(args.data_dir, "hdfs"),
            preprocessor=preprocessor
        )
        
        benchmark.load_dataset_from_loader(
            "hdfs",
            hdfs_loader,
            split=True,
            test_size=args.test_size,
            shuffle=args.shuffle,
            random_state=args.seed
        )
    
    if "bgl" in args.datasets:
        logger.info("Loading BGL dataset...")
        
        bgl_loader = BGLLoader(
            data_path=os.path.join(args.data_dir, "bgl"),
            preprocessor=preprocessor,
            config={
                "window_size": args.window_size,
                "step_size": args.step_size
            }
        )
        
        benchmark.load_dataset_from_loader(
            "bgl",
            bgl_loader,
            split=True,
            test_size=args.test_size,
            shuffle=args.shuffle,
            random_state=args.seed
        )
    
    if "liberty" in args.datasets:
        logger.info("Loading Liberty dataset...")
        
        liberty_loader = LibertyLoader(
            data_path=os.path.join(args.data_dir, "liberty"),
            preprocessor=preprocessor,
            config={
                "window_size": args.window_size,
                "step_size": args.step_size
            }
        )
        
        benchmark.load_dataset_from_loader(
            "liberty",
            liberty_loader,
            split=True,
            test_size=args.test_size,
            shuffle=args.shuffle,
            random_state=args.seed
        )
    
    if "thunderbird" in args.datasets:
        logger.info("Loading Thunderbird dataset...")
        
        thunderbird_loader = ThunderbirdLoader(
            data_path=os.path.join(args.data_dir, "thunderbird"),
            preprocessor=preprocessor,
            config={
                "window_size": args.window_size,
                "step_size": args.step_size
            }
        )
        
        benchmark.load_dataset_from_loader(
            "thunderbird",
            thunderbird_loader,
            split=True,
            test_size=args.test_size,
            shuffle=args.shuffle,
            random_state=args.seed
        )
    
    return benchmark


def initialize_methods(args, benchmark):
    """
    Initialize methods for benchmarking.
    
    Args:
        args: Command-line arguments
        benchmark: Benchmark object
        
    Returns:
        Benchmark object with added methods
    """
    # Initialize LogGuardian
    logger.info("Initializing LogGuardian...")
    logguardian = LogGuardian(
        config={
            "feature_extractor": {
                "model_name": args.bert_model,
                "max_length": args.max_length
            },
            "embedding_projector": {
                "input_dim": 768,  # BERT hidden size
                "output_dim": 4096,  # Llama hidden size
                "dropout": 0.1
            },
            "classifier": {
                "model_name_or_path": args.llama_model,
                "max_length": 2048,
                "load_in_8bit": args.load_in_8bit,
                "load_in_4bit": args.load_in_4bit
            }
        }
    )
    
    # Add LogGuardian to benchmark
    benchmark.add_method("LogGuardian", logguardian)
    
    # Add baseline methods if requested
    if args.include_baselines:
        logger.info("Adding baseline methods...")
        
        # Initialize and add DeepLog
        deeplog = DeepLogModel()
        benchmark.add_method("DeepLog", deeplog)
        
        # Initialize and add LogAnomaly
        loganomaly = LogAnomalyModel()
        benchmark.add_method("LogAnomaly", loganomaly)
    
    return benchmark


def main(args):
    """
    Main function.
    
    Args:
        args: Command-line arguments
    """
    # Set up logging
    logger.remove()
    logger.add(lambda msg: print(msg), level=args.log_level)
    
    # Create output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load datasets
    benchmark = load_datasets(args)
    
    # Initialize methods
    benchmark = initialize_methods(args, benchmark)
    
    # Run benchmark
    logger.info("Running benchmark...")
    results = benchmark.run(
        method_names=None,  # Use all methods
        dataset_names=[f"{dataset_name}_test" for dataset_name in args.datasets],  # Use test datasets
        save_results=True,
        generate_report=True,
        train_methods=False,  # Skip training for this example
        batch_size=args.batch_size,
        compute_thresholds=True,
        compute_optimal_threshold=True
    )
    
    # Create comparison visualizations
    logger.info("Creating comparison visualizations...")
    benchmark.create_comparison_visualizations(metric="f1")
    benchmark.create_comparison_visualizations(metric="precision")
    benchmark.create_comparison_visualizations(metric="recall")
    
    logger.info(f"Benchmark completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark LogGuardian against baseline methods")
    
    # Dataset arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing datasets")
    parser.add_argument("--datasets", nargs="+", default=["hdfs"], choices=["hdfs", "bgl", "liberty", "thunderbird"],
                        help="Datasets to benchmark on")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set size")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle data before splitting")
    parser.add_argument("--window_size", type=int, default=100, help="Window size for sliding window")
    parser.add_argument("--step_size", type=int, default=100, help="Step size for sliding window")
    
    # Method arguments
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased", help="BERT model name")
    parser.add_argument("--llama_model", type=str, default="meta-llama/Llama-3-8b", help="Llama model name")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load Llama model in 8-bit mode")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load Llama model in 4-bit mode")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--include_baselines", action="store_true", help="Include baseline methods")
    
    # Benchmark arguments
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="benchmark_results", help="Output directory")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    main(args)