"""
Example script demonstrating three-stage training as described in the LogLLM paper.
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from loguru import logger
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from logguardian.pipeline import LogGuardian
from logguardian.data.loaders import HDFSLoader
from logguardian.data.preprocessors import SystemLogPreprocessor
from logguardian.training.three_stage_trainer import ThreeStageTrainer
from logguardian.training.utils import oversample_minority_class, create_balanced_dataloader


def create_log_dataset(
    logs, 
    labels, 
    bert_tokenizer,
    max_length=128
):
    """
    Create a PyTorch dataset from logs and labels.
    
    Args:
        logs: List of log sequences
        labels: List of labels (0 for normal, 1 for anomalous)
        bert_tokenizer: BERT tokenizer
        max_length: Maximum sequence length
        
    Returns:
        PyTorch dataset
    """
    # Tokenize logs
    encodings = bert_tokenizer(
        logs,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Create tensors
    input_ids = encodings.input_ids
    attention_mask = encodings.attention_mask
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    # Create dataset
    dataset = TensorDataset(input_ids, attention_mask, labels_tensor)
    
    return dataset


def compute_metrics(preds, labels):
    """
    Compute evaluation metrics.
    
    Args:
        preds: Predicted labels
        labels: True labels
        
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy arrays
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    accuracy = accuracy_score(labels, preds)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main(args):
    """Main training function."""
    # Set up logging
    logger.remove()
    logger.add(
        os.path.join(args.output_dir, "training.log") if args.output_dir else "training.log",
        level=args.log_level
    )
    logger.add(lambda msg: print(msg), level=args.log_level)
    
    # Create output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set up data preprocessor
    preprocessor = SystemLogPreprocessor()
    
    # Load dataset
    logger.info(f"Loading dataset from {args.data_path}")
    data_loader = HDFSLoader(
        data_path=args.data_path,
        preprocessor=preprocessor
    )
    logs, labels = data_loader.load()
    
    # Split data into train and test
    train_logs, train_labels, val_logs, val_labels = data_loader.get_train_test_split(
        test_size=args.val_split,
        shuffle=True,
        random_state=args.seed
    )
    
    logger.info(f"Train set: {len(train_logs)} samples")
    logger.info(f"Validation set: {len(val_logs)} samples")
    
    # Apply oversampling if requested
    if args.beta > 0:
        logger.info(f"Applying minority class oversampling with beta={args.beta}")
        train_logs, train_labels = oversample_minority_class(
            train_logs, train_labels, beta=args.beta
        )
        logger.info(f"After oversampling: {len(train_logs)} samples")
    
    # Create LogGuardian model
    logger.info("Initializing LogGuardian model")
    model = LogGuardian(
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
    
    # Move model to device
    model.to(device)
    
    # Create datasets
    logger.info("Creating PyTorch datasets")
    
    # For training data
    train_dataset = create_log_dataset(
        train_logs,
        train_labels,
        model.feature_extractor.tokenizer,
        max_length=args.max_length
    )
    
    # For validation data
    val_dataset = create_log_dataset(
        val_logs,
        val_labels,
        model.feature_extractor.tokenizer,
        max_length=args.max_length
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Define metrics
    metrics = {
        "accuracy": lambda preds, labels: torch.tensor(
            accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
        ),
        "f1": lambda preds, labels: torch.tensor(
            precision_recall_fscore_support(
                labels.cpu().numpy(), preds.cpu().numpy(), 
                average="binary", zero_division=0
            )[2]
        )
    }
    
    # Create checkpoint directories
    if args.output_dir:
        stage1_dir = os.path.join(args.output_dir, "stage1")
        stage2_dir = os.path.join(args.output_dir, "stage2")
        stage3_dir = os.path.join(args.output_dir, "stage3")
        final_dir = os.path.join(args.output_dir, "final")
        
        os.makedirs(stage1_dir, exist_ok=True)
        os.makedirs(stage2_dir, exist_ok=True)
        os.makedirs(stage3_dir, exist_ok=True)
        os.makedirs(final_dir, exist_ok=True)
    else:
        stage1_dir = None
        stage2_dir = None
        stage3_dir = None
        final_dir = None
    
    # Create three-stage trainer
    logger.info("Creating three-stage trainer")
    trainer = ThreeStageTrainer(model, device=device)
    
    # Train the model
    logger.info("Starting three-stage training procedure")
    trainer.train(
        train_loader=train_loader,
        criterion=criterion,
        eval_loader=val_loader,
        metrics=metrics,
        num_epochs_stage1=args.epochs_stage1,
        num_samples_stage1=args.samples_stage1,
        num_epochs_stage2=args.epochs_stage2,
        num_epochs_stage3=args.epochs_stage3,
        learning_rate_stage1=args.lr_stage1,
        learning_rate_stage2=args.lr_stage2,
        learning_rate_stage3=args.lr_stage3,
        checkpoint_dir_stage1=stage1_dir,
        checkpoint_dir_stage2=stage2_dir,
        checkpoint_dir_stage3=stage3_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm
    )
    
    # Save final model
    if final_dir:
        logger.info(f"Saving final model to {final_dir}")
        model.save(final_dir)
    
    logger.info("Training completed!")
    
    # Evaluate on test set
    logger.info("Evaluating on test set")
    test_logs, test_labels = data_loader.get_test_data()
    
    # Preprocess test logs
    test_logs = [preprocessor.preprocess(log) for log in test_logs]
    
    # Detect anomalies
    results = model.detect(test_logs, raw_output=True)
    
    # Extract predictions
    preds = [1 if result["label_id"] == 1 else 0 for result in results]
    
    # Compute metrics
    test_metrics = compute_metrics(preds, test_labels)
    
    # Log results
    logger.info("Test set evaluation:")
    for metric, value in test_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Save metrics
    if args.output_dir:
        import json
        metrics_path = os.path.join(args.output_dir, "test_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(test_metrics, f, indent=2)
        
        logger.info(f"Saved test metrics to {metrics_path}")
    
    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LogGuardian with three-stage procedure")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--beta", type=float, default=0.3, help="Target proportion of minority class")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    
    # Model arguments
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased", help="BERT model name")
    parser.add_argument("--llama_model", type=str, default="meta-llama/Llama-3-8b", help="Llama model name")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load Llama model in 8-bit mode")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load Llama model in 4-bit mode")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--epochs_stage1", type=int, default=1, help="Number of epochs for Stage 1")
    parser.add_argument("--epochs_stage2", type=int, default=2, help="Number of epochs for Stage 2")
    parser.add_argument("--epochs_stage3", type=int, default=2, help="Number of epochs for Stage 3")
    parser.add_argument("--samples_stage1", type=int, default=1000, help="Number of samples for Stage 1")
    parser.add_argument("--lr_stage1", type=float, default=5e-4, help="Learning rate for Stage 1")
    parser.add_argument("--lr_stage2", type=float, default=5e-5, help="Learning rate for Stage 2")
    parser.add_argument("--lr_stage3", type=float, default=5e-5, help="Learning rate for Stage 3")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    
    # Miscellaneous arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    main(args)