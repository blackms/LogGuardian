"""
Basic usage example for LogGuardian.
"""
import os
import sys
import argparse
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from logguardian.pipeline import LogGuardian
from logguardian.data.loaders import HDFSLoader
from logguardian.data.preprocessors import SystemLogPreprocessor


def load_sample_logs(file_path: Optional[str] = None) -> List[str]:
    """
    Load sample log data for demonstration.
    
    Args:
        file_path: Optional path to log file
        
    Returns:
        List of log messages
    """
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            logs = [line.strip() for line in f]
        logger.info(f"Loaded {len(logs)} logs from {file_path}")
        return logs
    
    # If no file provided or file doesn't exist, use sample logs
    logger.info("Loading sample logs")
    return [
        "2023-02-15 10:12:34 INFO  [server.Main] System startup initiated",
        "2023-02-15 10:12:35 INFO  [server.Config] Loading configuration from /etc/config.json",
        "2023-02-15 10:12:36 INFO  [server.Database] Connected to database at 192.168.1.100:5432",
        "2023-02-15 10:12:37 INFO  [server.Auth] Authentication service started",
        "2023-02-15 10:12:38 INFO  [server.API] API server listening on port 8080",
        "2023-02-15 10:12:40 ERROR [server.Database] Failed to execute query: table 'users' doesn't exist",
        "2023-02-15 10:12:41 WARN  [server.Auth] Suspicious login attempt from 203.0.113.42",
        "2023-02-15 10:12:42 ERROR [server.API] Unhandled exception in request handler: NullPointerException",
        "2023-02-15 10:12:43 ERROR [server.Main] Critical error encountered, attempting recovery",
        "2023-02-15 10:12:44 INFO  [server.Main] Recovery successful, resuming normal operation",
        "2023-02-15 10:12:45 INFO  [server.API] Request processed successfully",
        "2023-02-15 10:12:46 INFO  [server.Database] Query executed successfully",
        "2023-02-15 10:12:47 INFO  [server.Auth] User authenticated successfully",
        "2023-02-15 10:12:48 INFO  [server.API] Request processed successfully",
        "2023-02-15 10:12:49 INFO  [server.API] Request processed successfully",
    ]


def detect_anomalies(logs: List[str], window_size: int = 5, stride: int = 1) -> Dict[str, Any]:
    """
    Detect anomalies in log data using LogGuardian.
    
    Args:
        logs: List of log messages
        window_size: Size of sliding window
        stride: Stride of sliding window
        
    Returns:
        Dictionary with detection results
    """
    logger.info("Initializing LogGuardian")
    
    # Initialize preprocessor
    preprocessor = SystemLogPreprocessor()
    
    # Initialize the pipeline with default components
    # In a real scenario, you would load pre-trained models
    detector = LogGuardian(preprocessor=preprocessor)
    
    # Detect anomalies
    logger.info(f"Detecting anomalies in {len(logs)} logs with window_size={window_size}, stride={stride}")
    results = detector.detect(
        logs=logs,
        window_size=window_size,
        stride=stride,
        raw_output=True
    )
    
    # Process results
    anomaly_indices = []
    confidence_scores = []
    
    for i, result in enumerate(results):
        anomaly_indices.append(i)
        if isinstance(result, dict):
            confidence_scores.append(result.get("confidence", 0.5) if result.get("label_id", 0) == 1 else 0)
        else:
            confidence_scores.append(1.0 if result == 1 else 0)
    
    # Return processed results
    return {
        "logs": logs,
        "anomaly_indices": anomaly_indices,
        "confidence_scores": confidence_scores,
        "raw_results": results
    }


def visualize_results(results: Dict[str, Any], window_size: int = 5) -> None:
    """
    Visualize anomaly detection results.
    
    Args:
        results: Results from detect_anomalies
        window_size: Window size used for detection
    """
    logs = results["logs"]
    anomaly_indices = results["anomaly_indices"]
    confidence_scores = results["confidence_scores"]
    
    # Create a figure
    plt.figure(figsize=(12, 6))
    
    # Plot confidence scores
    plt.subplot(2, 1, 1)
    plt.plot(confidence_scores, 'r-', label="Anomaly Score")
    plt.fill_between(range(len(confidence_scores)), confidence_scores, alpha=0.3)
    plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.3)
    plt.title("Anomaly Detection Results")
    plt.ylabel("Anomaly Score")
    plt.legend()
    
    # Plot log sequence with anomalies
    plt.subplot(2, 1, 2)
    anomaly_positions = []
    
    for i, idx in enumerate(anomaly_indices):
        # Calculate position in the original log sequence
        start_pos = idx * stride
        end_pos = min(start_pos + window_size, len(logs))
        
        if confidence_scores[i] >= 0.5:
            for j in range(start_pos, end_pos):
                anomaly_positions.append(j)
    
    # Create a colormap: green for normal, red for anomalous
    colors = ['green' if i not in set(anomaly_positions) else 'red' for i in range(len(logs))]
    
    # Plot each log message
    plt.scatter(range(len(logs)), [1] * len(logs), c=colors, s=100)
    plt.yticks([])
    plt.xlabel("Log Sequence")
    plt.title("Log Anomalies (Red = Anomalous, Green = Normal)")
    
    plt.tight_layout()
    plt.savefig("anomaly_detection_results.png")
    logger.info("Saved visualization to anomaly_detection_results.png")
    
    # Print detected anomalies
    anomalous_logs = set(anomaly_positions)
    print("\nDetected Anomalies:")
    for i in sorted(anomalous_logs):
        print(f"[{'ANOMALY' if i in anomalous_logs else 'NORMAL'}] {logs[i]}")


def main():
    """
    Main function to run the example.
    """
    parser = argparse.ArgumentParser(description="LogGuardian example")
    parser.add_argument("--log-file", type=str, help="Path to log file")
    parser.add_argument("--window", type=int, default=5, help="Window size")
    parser.add_argument("--stride", type=int, default=1, help="Stride")
    args = parser.parse_args()
    
    # Configure logger
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # Load sample logs
    logs = load_sample_logs(args.log_file)
    
    # Detect anomalies
    results = detect_anomalies(logs, window_size=args.window, stride=args.stride)
    
    # Visualize results
    visualize_results(results, window_size=args.window)


if __name__ == "__main__":
    # Global variable
    stride = 1
    main()