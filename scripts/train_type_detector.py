#!/usr/bin/env python3
"""
Train Type Detector for Cutlery Classifier MVP

This script trains a ResNet18 model to classify cutlery types (fork/knife/spoon).
Uses the CutleryTrainer class with configuration from YAML.

Usage:
    python scripts/train_type_detector.py
    python scripts/train_type_detector.py --config config/train_config.yaml
    python scripts/train_type_detector.py --epochs 20 --batch-size 16
"""

import argparse
import logging
import yaml
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import torch

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.training.trainer import CutleryTrainer
from src.evaluation.evaluator import CutleryEvaluator

# Setup logging
logger = logging.getLogger(__name__)


def plot_training_history(history, save_dir):
    """Plot training history and save figures."""
    if not history:
        logger.warning("No training history provided, skipping plot generation")
        return

    # Create plots directory if it doesn't exist
    plots_dir = Path(save_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_losses"], label="Training Loss")
    plt.plot(history["val_losses"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / "loss_history.png")
    plt.close()

    # Plot training and validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_accs"], label="Training Accuracy")
    plt.plot(history["val_accs"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / "accuracy_history.png")
    plt.close()

    logger.info(f"Training plots saved in: {plots_dir}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Create and save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(save_path)
    plt.close()


def evaluate_model(trainer, test_loader, save_dir):
    """Evaluate model on test set and save results."""
    device = trainer.device
    model = trainer.model
    class_names = trainer.class_names

    # Switch to evaluation mode
    model.eval()

    all_preds = []
    all_labels = []

    logging.info("Evaluating model on test set...")

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Calculate metrics
    report = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True
    )

    # Save classification report
    report_path = Path(save_dir) / "test_results.txt"
    with open(report_path, "w") as f:
        f.write("Test Set Evaluation Results\n")
        f.write("==========================\n\n")
        f.write(classification_report(all_labels, all_preds, target_names=class_names))

    # Plot confusion matrix
    cm_path = Path(save_dir) / "confusion_matrix.png"
    plot_confusion_matrix(all_labels, all_preds, class_names, cm_path)

    return report


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_type_detector_config(args) -> dict:
    """Create configuration for type detector training."""

    config = {
        "model": {
            "architecture": "resnet18",
            "num_classes": 3,  # fork, knife, spoon
            "pretrained": True,
            "grayscale": True,
            "freeze_backbone": False,
            "dropout_rate": 0.5,
        },
        "data": {
            "image_size": [320, 320],
            "augmentation": {
                "horizontal_flip": 0.5,
                "rotation_degrees": 15,
                "color_jitter": {
                    "brightness": 0.2,
                    "contrast": 0.2,
                    "saturation": 0.2,
                    "hue": 0.1,
                },
                "gaussian_blur": 0.1,
                "random_crop_scale": [0.8, 1.0],
                "random_crop_ratio": [0.9, 1.1],
            },
        },
        "training": {
            "batch_size": args.batch_size,
            "num_epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": 1e-4,
            "optimizer": "adam",
            "num_workers": 4,
            "scheduler": {
                "enabled": True,
                "type": "step",
                "step_size": 15,
                "gamma": 0.1,
            },
            "early_stopping": {"enabled": True, "patience": 10, "min_delta": 0.001},
        },
    }

    return config


def main():
    parser = argparse.ArgumentParser(description="Train cutlery type detector")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config YAML file"
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--mixed-data",
        action="store_true",
        help="Include mixed cutlery images in training",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Load or create configuration
    if args.config and Path(args.config).exists():
        logger.info(f"Loading config from {args.config}")
        config = load_config(args.config)

        # Override with command line arguments
        if args.epochs != 30:
            config["training"]["num_epochs"] = args.epochs
        if args.batch_size != 32:
            config["training"]["batch_size"] = args.batch_size
        if args.learning_rate != 0.001:
            config["training"]["learning_rate"] = args.learning_rate
    else:
        logger.info("Creating default configuration")
        config = create_type_detector_config(args)

    logger.info("Configuration:")
    logger.info(f"  Model: {config['model']['architecture']}")
    logger.info(f"  Classes: {config['model']['num_classes']}")
    logger.info(f"  Epochs: {config['training']['num_epochs']}")
    logger.info(f"  Batch size: {config['training']['batch_size']}")
    logger.info(f"  Learning rate: {config['training']['learning_rate']}")

    # Check if processed data exists
    data_dir = project_root / "data" / "processed"
    if not data_dir.exists():
        logger.error(f"Processed data directory not found: {data_dir}")
        logger.error("Please run: python scripts/prepare_dataset.py --create-splits")
        return

    # Check for train/val/test directories
    required_dirs = ["train", "val", "test"]
    for dir_name in required_dirs:
        dir_path = data_dir / dir_name
        if not dir_path.exists():
            logger.error(f"Missing directory: {dir_path}")
            logger.error(
                "Please run: python scripts/prepare_dataset.py --create-splits"
            )
            return

    # Create trainer
    trainer = CutleryTrainer(
        config=config, model_name="type_detector", device=args.device
    )

    # Create model and setup training
    trainer.create_model()
    trainer.setup_training()

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    try:
        # Create dataloaders
        train_loader, val_loader, test_loader = trainer.create_dataloaders(
            include_mixed=args.mixed_data
        )

        # Start training
        logger.info("Starting training...")
        history = trainer.train(train_loader, val_loader)

        logger.info("Training completed successfully!")
        logger.info(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")

        # Plot training history
        logger.info("Generating training plots...")
        plot_training_history(history, project_root / "results" / "plots")

        # Evaluate on test set
        logger.info("Evaluating model on test set...")
        results_dir = project_root / "results"
        test_results = evaluate_model(trainer, test_loader, results_dir)

        # Save final model info
        model_info_path = project_root / "models" / "type_detector_info.txt"
        with open(model_info_path, "w") as f:
            f.write(f"Type Detector Model Information\n")
            f.write(f"================================\n\n")
            f.write(f"Architecture: {config['model']['architecture']}\n")
            f.write(f"Classes: {trainer.class_names}\n")
            f.write(f"Best Validation Accuracy: {trainer.best_val_acc:.2f}%\n")
            f.write(f"Test Set Accuracy: {test_results['accuracy']:.2f}%\n")
            f.write(f"Total Epochs: {trainer.current_epoch + 1}\n")
            f.write(f"Device: {trainer.device}\n")
            f.write(f"\nPer-Class Test Metrics:\n")
            for class_name in trainer.class_names:
                metrics = test_results[class_name]
                f.write(f"\n{class_name}:\n")
                f.write(f"  Precision: {metrics['precision']:.3f}\n")
                f.write(f"  Recall: {metrics['recall']:.3f}\n")
                f.write(f"  F1-Score: {metrics['f1-score']:.3f}\n")

        logger.info(f"Model info saved: {model_info_path}")
        logger.info("Training and evaluation completed!")

    except FileNotFoundError as e:
        logger.error(f"Data not found: {e}")
        logger.error("Make sure you have:")
        logger.error("1. Collected images in data/raw/")
        logger.error("2. Run: python scripts/prepare_dataset.py --create-splits")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
