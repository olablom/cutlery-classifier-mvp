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

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import yaml
from sklearn.metrics import confusion_matrix, classification_report

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.training.trainer import CutleryTrainer
from src.models.factory import create_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
CLASSES = ["fork", "knife", "spoon"]
DEFAULT_TEST_IMAGE = "data/simplified/test/fork/IMG_0941[1]_fork_a.jpg"
VERSION = "1.0.0"

# Define data directory
data_dir = Path("data/simplified")


def get_class_names(train_dir: str) -> list:
    """
    Get class names from training directory structure.

    Args:
        train_dir: Path to training directory

    Returns:
        List of class names (sorted)
    """
    train_path = Path(train_dir)
    if not train_path.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    class_names = sorted([d.name for d in train_path.iterdir() if d.is_dir()])
    if not class_names:
        raise ValueError(f"No class directories found in {train_dir}")

    return class_names


def plot_training_history(history, run_dir):
    """Plot and save training/validation curves."""
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_losses"], label="Training Loss")
    plt.plot(history["val_losses"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "loss_history.png")
    plt.close()

    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_accs"], label="Training Accuracy")
    plt.plot(history["val_accs"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "accuracy_history.png")
    plt.close()

    logging.info("Training history plots saved to results directory")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path, labels=None):
    """Create and save a robust confusion matrix plot."""
    logging.info("Generating confusion matrix...")
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if labels is None:
        # Default: use sorted unique labels in y_true
        labels = sorted(np.unique(y_true))
        logging.warning(
            f"No labels provided. Using unique labels from y_true: {labels}"
        )
        used_class_names = [class_names[i] for i in labels]
    else:
        logging.info(f"Using provided labels: {labels}")
        # When labels are provided (as in test), class_names *is already* correct (you passed test_class_names!)
        used_class_names = class_names

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    logging.info(f"Confusion matrix shape: {cm.shape}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=used_class_names,
        yticklabels=used_class_names,
        cbar=True,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    logging.info(f"Confusion matrix saved to: {save_path}")


def evaluate_model(trainer, test_loader, save_dir):
    """Evaluate model on test set and save results, including correct/incorrect examples."""
    device = trainer.device
    model = trainer.model
    test_class_names = trainer.test_classes
    test_labels = list(range(len(trainer.test_classes)))

    # Switch to evaluation mode
    model.eval()

    all_preds = []
    all_labels = []
    all_images = []

    logging.info("Evaluating model on test set...")
    logging.info(f"Test classes: {test_class_names}")
    logging.info(f"Using labels: {test_labels}")

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_images.extend(images.cpu())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Classification report
    report = classification_report(
        all_labels,
        all_preds,
        target_names=test_class_names,
        labels=test_labels,
        output_dict=True,
        zero_division=0,
    )

    # Compute accuracy manually
    accuracy = np.mean(all_preds == all_labels) * 100.0
    logging.info(f"Test Set Accuracy: {accuracy:.2f}%")

    # Save text report
    report_path = save_dir / "test_results.txt"
    with open(report_path, "w") as f:
        f.write("Test Set Evaluation Results\n")
        f.write("==========================\n\n")
        f.write(f"Overall Accuracy: {accuracy:.2f}%\n\n")
        f.write("Detailed Classification Report:\n\n")
        f.write(
            classification_report(
                all_labels,
                all_preds,
                target_names=test_class_names,
                labels=test_labels,
                zero_division=0,
            )
        )

    # Plot confusion matrix
    cm_path = save_dir / "confusion_matrix.png"
    plot_confusion_matrix(
        all_labels, all_preds, test_class_names, cm_path, labels=test_labels
    )

    logging.info("Saved test report to: {}".format(report_path))
    logging.info("Saved confusion matrix to: {}".format(cm_path))

    # Optional sanity print:
    logging.info("Evaluation summary")
    logging.info(f"Unique true labels: {np.unique(all_labels)}")
    logging.info(f"Unique predicted labels: {np.unique(all_preds)}")

    # Save correct/incorrect examples
    import torchvision.transforms.functional as F

    correct_dir = save_dir / "examples_correct"
    incorrect_dir = save_dir / "examples_incorrect"
    correct_dir.mkdir(exist_ok=True)
    incorrect_dir.mkdir(exist_ok=True)

    logging.info("Saving example images (correct/incorrect)...")

    num_saved_correct = 0
    num_saved_incorrect = 0
    max_examples = 5  # Save up to 5 correct and 5 incorrect

    for idx, (image_tensor, true_label, pred_label) in enumerate(
        zip(all_images, all_labels, all_preds)
    ):
        if num_saved_correct >= max_examples and num_saved_incorrect >= max_examples:
            break

        # Convert tensor to image
        img = F.to_pil_image(image_tensor)

        filename = f"img_{idx}_true-{test_class_names[true_label]}_pred-{test_class_names[pred_label]}.png"

        if true_label == pred_label and num_saved_correct < max_examples:
            img.save(correct_dir / filename)
            num_saved_correct += 1
        elif true_label != pred_label and num_saved_incorrect < max_examples:
            img.save(incorrect_dir / filename)
            num_saved_incorrect += 1

    logging.info(f"Saved {num_saved_correct} correct examples to: {correct_dir}")
    logging.info(f"Saved {num_saved_incorrect} incorrect examples to: {incorrect_dir}")

    return {"report": report, "accuracy": accuracy}


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_type_detector_config(args) -> dict:
    """Create configuration dictionary from command line arguments."""
    config = {
        "model": {
            "architecture": "resnet18",
            "num_classes": len(CLASSES),
            "pretrained": True,
        },
        "training": {
            "num_epochs": args.epochs if args.epochs else 30,
            "batch_size": args.batch_size if args.batch_size else 32,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "momentum": 0.9,
            "mixed_data": args.mixed_data,
        },
        "data": {
            "train_dir": str(data_dir / "train"),
            "val_dir": str(data_dir / "val"),
            "test_dir": str(data_dir / "test"),
            "input_size": 224,
            "num_workers": 4,
        },
    }
    return config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train cutlery type detector")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (cuda/cpu)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Training batch size",
    )
    parser.add_argument(
        "--mixed-data",
        action="store_true",
        help="Use mixed real + augmented data",
    )
    args = parser.parse_args()

    # Load or create config
    if args.config:
        config = load_config(args.config)
    else:
        config = create_type_detector_config(args)

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("results") / f"train_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = run_dir / "train_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Initialize trainer
    trainer = CutleryTrainer(config)

    # Create and setup model
    trainer.create_model()
    trainer.setup_training()

    # Create dataloaders
    train_loader, val_loader, test_loader = trainer.create_dataloaders(
        include_mixed=args.mixed_data
    )

    # Train model
    history = trainer.train(train_loader, val_loader)

    # Plot training history
    plot_training_history(history, run_dir)

    # Evaluate on test set
    evaluate_model(trainer, test_loader, run_dir)

    logger.info("✅ Training completed successfully!")
    logger.info(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
