"""
Example script demonstrating how to use dataset loaders for all supported datasets.
"""
import os
import argparse
import numpy as np
from loguru import logger

from logguardian.data.preprocessors import SystemLogPreprocessor
from logguardian.data.loaders import (
    HDFSLoader, 
    BGLLoader, 
    LibertyLoader, 
    ThunderbirdLoader
)


def load_hdfs_dataset(data_path, preprocessor, window_size=None, step_size=None):
    """
    Load and preprocess HDFS dataset.
    
    Args:
        data_path: Path to HDFS dataset
        preprocessor: Log preprocessor
        window_size: Window size for sliding window (not used for HDFS)
        step_size: Step size for sliding window (not used for HDFS)
        
    Returns:
        Tuple of (log_sequences, sequence_labels)
    """
    logger.info("Loading HDFS dataset...")
    
    loader = HDFSLoader(
        data_path=data_path,
        preprocessor=preprocessor
    )
    
    # Load data
    log_sequences, sequence_labels = loader.load()
    
    # Get statistics
    normal_count = len(sequence_labels) - sum(sequence_labels)
    anomaly_count = sum(sequence_labels)
    anomaly_ratio = anomaly_count / len(sequence_labels) if len(sequence_labels) > 0 else 0
    
    logger.info(f"HDFS dataset loaded: {len(log_sequences)} sequences")
    logger.info(f"Normal: {normal_count}, Anomalous: {anomaly_count}, Anomaly ratio: {anomaly_ratio:.4f}")
    
    return log_sequences, sequence_labels


def load_bgl_dataset(data_path, preprocessor, window_size=100, step_size=100):
    """
    Load and preprocess BGL dataset.
    
    Args:
        data_path: Path to BGL dataset
        preprocessor: Log preprocessor
        window_size: Window size for sliding window
        step_size: Step size for sliding window
        
    Returns:
        Tuple of (log_sequences, sequence_labels)
    """
    logger.info("Loading BGL dataset...")
    
    loader = BGLLoader(
        data_path=data_path,
        preprocessor=preprocessor,
        config={
            "window_size": window_size,
            "step_size": step_size,
            "label_type": "sliding_window"
        }
    )
    
    # Load data
    log_sequences, sequence_labels = loader.load()
    
    # Get statistics
    normal_count = len(sequence_labels) - sum(sequence_labels)
    anomaly_count = sum(sequence_labels)
    anomaly_ratio = anomaly_count / len(sequence_labels) if len(sequence_labels) > 0 else 0
    
    logger.info(f"BGL dataset loaded: {len(log_sequences)} sequences")
    logger.info(f"Normal: {normal_count}, Anomalous: {anomaly_count}, Anomaly ratio: {anomaly_ratio:.4f}")
    
    return log_sequences, sequence_labels


def load_liberty_dataset(data_path, preprocessor, window_size=100, step_size=100):
    """
    Load and preprocess Liberty dataset.
    
    Args:
        data_path: Path to Liberty dataset
        preprocessor: Log preprocessor
        window_size: Window size for sliding window
        step_size: Step size for sliding window
        
    Returns:
        Tuple of (log_sequences, sequence_labels)
    """
    logger.info("Loading Liberty dataset...")
    
    loader = LibertyLoader(
        data_path=data_path,
        preprocessor=preprocessor,
        config={
            "window_size": window_size,
            "step_size": step_size,
            "label_type": "sliding_window"
        }
    )
    
    # Load data
    log_sequences, sequence_labels = loader.load()
    
    # Get statistics
    normal_count = len(sequence_labels) - sum(sequence_labels)
    anomaly_count = sum(sequence_labels)
    anomaly_ratio = anomaly_count / len(sequence_labels) if len(sequence_labels) > 0 else 0
    
    logger.info(f"Liberty dataset loaded: {len(log_sequences)} sequences")
    logger.info(f"Normal: {normal_count}, Anomalous: {anomaly_count}, Anomaly ratio: {anomaly_ratio:.4f}")
    
    return log_sequences, sequence_labels


def load_thunderbird_dataset(data_path, preprocessor, window_size=100, step_size=100):
    """
    Load and preprocess Thunderbird dataset.
    
    Args:
        data_path: Path to Thunderbird dataset
        preprocessor: Log preprocessor
        window_size: Window size for sliding window
        step_size: Step size for sliding window
        
    Returns:
        Tuple of (log_sequences, sequence_labels)
    """
    logger.info("Loading Thunderbird dataset...")
    
    loader = ThunderbirdLoader(
        data_path=data_path,
        preprocessor=preprocessor,
        config={
            "window_size": window_size,
            "step_size": step_size,
            "label_type": "sliding_window"
        }
    )
    
    # Load data
    log_sequences, sequence_labels = loader.load()
    
    # Get statistics
    normal_count = len(sequence_labels) - sum(sequence_labels)
    anomaly_count = sum(sequence_labels)
    anomaly_ratio = anomaly_count / len(sequence_labels) if len(sequence_labels) > 0 else 0
    
    logger.info(f"Thunderbird dataset loaded: {len(log_sequences)} sequences")
    logger.info(f"Normal: {normal_count}, Anomalous: {anomaly_count}, Anomaly ratio: {anomaly_ratio:.4f}")
    
    return log_sequences, sequence_labels


def main(args):
    """
    Main function.
    
    Args:
        args: Command-line arguments
    """
    # Set up logging
    logger.remove()
    logger.add(lambda msg: print(msg), level=args.log_level)
    
    # Create log preprocessor
    preprocessor = SystemLogPreprocessor()
    
    if args.dataset_type == "hdfs":
        logs, labels = load_hdfs_dataset(
            args.data_path,
            preprocessor
        )
    elif args.dataset_type == "bgl":
        logs, labels = load_bgl_dataset(
            args.data_path,
            preprocessor,
            window_size=args.window_size,
            step_size=args.step_size
        )
    elif args.dataset_type == "liberty":
        logs, labels = load_liberty_dataset(
            args.data_path,
            preprocessor,
            window_size=args.window_size,
            step_size=args.step_size
        )
    elif args.dataset_type == "thunderbird":
        logs, labels = load_thunderbird_dataset(
            args.data_path,
            preprocessor,
            window_size=args.window_size,
            step_size=args.step_size
        )
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")
    
    # Print sample logs
    if args.print_samples:
        logger.info("\nSample log sequences:")
        
        # Print a few normal examples
        normal_indices = [i for i, label in enumerate(labels) if label == 0]
        if normal_indices:
            logger.info("\nNormal log sequence example:")
            sample_idx = np.random.choice(normal_indices)
            sample_logs = logs[sample_idx]
            
            for i, log in enumerate(sample_logs[:5]):  # Print first 5 logs
                logger.info(f"  [{i+1}] {log}")
            
            if len(sample_logs) > 5:
                logger.info(f"  ... ({len(sample_logs) - 5} more logs)")
        
        # Print a few anomalous examples
        anomaly_indices = [i for i, label in enumerate(labels) if label == 1]
        if anomaly_indices:
            logger.info("\nAnomalous log sequence example:")
            sample_idx = np.random.choice(anomaly_indices)
            sample_logs = logs[sample_idx]
            
            for i, log in enumerate(sample_logs[:5]):  # Print first 5 logs
                logger.info(f"  [{i+1}] {log}")
            
            if len(sample_logs) > 5:
                logger.info(f"  ... ({len(sample_logs) - 5} more logs)")
    
    # Get train/test split
    if args.dataset_type == "hdfs":
        loader = HDFSLoader(data_path=args.data_path, preprocessor=preprocessor)
    elif args.dataset_type == "bgl":
        loader = BGLLoader(
            data_path=args.data_path,
            preprocessor=preprocessor,
            config={"window_size": args.window_size, "step_size": args.step_size}
        )
    elif args.dataset_type == "liberty":
        loader = LibertyLoader(
            data_path=args.data_path,
            preprocessor=preprocessor,
            config={"window_size": args.window_size, "step_size": args.step_size}
        )
    elif args.dataset_type == "thunderbird":
        loader = ThunderbirdLoader(
            data_path=args.data_path,
            preprocessor=preprocessor,
            config={"window_size": args.window_size, "step_size": args.step_size}
        )
    
    # Set load/load_raw based on whether data is already loaded
    if hasattr(loader, "logs") and loader.logs is not None:
        # Data already loaded
        pass
    else:
        # Load data
        loader.load()
    
    # Get train/test split
    train_logs, train_labels, test_logs, test_labels = loader.get_train_test_split(
        test_size=args.test_size,
        shuffle=args.shuffle,
        random_state=args.seed
    )
    
    logger.info(f"\nTrain set: {len(train_logs)} sequences, {sum(train_labels)} anomalies")
    logger.info(f"Test set: {len(test_logs)} sequences, {sum(test_labels)} anomalies")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset loader example")
    
    # Dataset arguments
    parser.add_argument("--dataset_type", type=str, required=True, choices=["hdfs", "bgl", "liberty", "thunderbird"],
                        help="Type of dataset to load")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--window_size", type=int, default=100, help="Window size for sliding window")
    parser.add_argument("--step_size", type=int, default=100, help="Step size for sliding window")
    
    # Split arguments
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set size")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle data before splitting")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Other arguments
    parser.add_argument("--print_samples", action="store_true", help="Print sample logs")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    main(args)