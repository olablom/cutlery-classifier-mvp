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
import sys
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from sklearn.metrics import confusion_matrix, classification_report

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.training.trainer import CutleryTrainer

# Setup logging
logger = logging.getLogger(__name__)

# Define data directory
data_dir = Path("data/processed")


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
        config = yaml.safe_load(f)
    return config


def create_type_detector_config(args) -> dict:
    """Create configuration for type detector training."""
    # Get class names from training directory
    train_dir = args.train_dir if args.train_dir else "data/processed/train"
    class_names = get_class_names(train_dir)
    num_classes = len(class_names)

    logger.info(f"Classes detected: {class_names}")
    logger.info(f"num_classes set to: {num_classes}")

    config = {
        "model": {
            "architecture": "resnet18",
            "num_classes": num_classes,
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
        "classes": class_names,
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
    parser.add_argument(
        "--train-dir",
        type=str,
        default="data/processed/train",
        help="Path to training data directory",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default="data/processed/val",
        help="Path to validation data directory",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="data/processed/test",
        help="Path to test data directory",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)

    logger.info("Starting main()")

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

    # Prepare run output directory
    run_dir = (
        project_root / "results" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "examples").mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving all results to: {run_dir}")

    # Initialize trainer with config and device
    trainer = CutleryTrainer(
        config=config, model_name="type_detector", device=args.device
    )
    logger.info("Created trainer")

    # Create model and setup training
    trainer.create_model()
    logger.info("Created model")

    trainer.setup_training()
    logger.info("Setup training done")

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    try:
        # Create dataloaders
        logger.info("Creating dataloaders...")
        train_loader, val_loader, test_loader = trainer.create_dataloaders(
            include_mixed=args.mixed_data
        )

        # Start training
        logger.info("Starting training...")
        history = trainer.train(train_loader, val_loader, run_dir=run_dir)
        logger.info("Training completed successfully!")
        logger.info(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")

        # Plot training history
        logger.info("Generating training plots...")
        plot_training_history(history, run_dir)

        # Evaluate on test set
        logger.info("Evaluating model on test set...")
        test_results = evaluate_model(trainer, test_loader, run_dir)

        # Save final model info
        model_info_path = run_dir / "type_detector_info.txt"
        with open(model_info_path, "w") as f:
            f.write("Type Detector Model Information\n")
            f.write("================================\n\n")
            f.write(f"Architecture: {config['model']['architecture']}\n")
            f.write(f"Classes: {trainer.class_names}\n")
            f.write(f"Best Validation Accuracy: {trainer.best_val_acc:.2f}%\n")
            f.write(f"Test Set Accuracy: {test_results['accuracy']:.2f}%\n")
            f.write(f"Total Epochs: {trainer.current_epoch + 1}\n")
            f.write(f"Device: {trainer.device}\n")
            f.write("\nPer-Class Test Metrics:\n")
            for class_name in trainer.test_classes:
                metrics = test_results["report"][class_name]
                f.write(f"\n{class_name}:\n")
                f.write(f"  Precision: {metrics['precision']:.3f}\n")
                f.write(f"  Recall: {metrics['recall']:.3f}\n")
                f.write(f"  F1-Score: {metrics['f1-score']:.3f}\n")

        logger.info(f"Model info saved: {model_info_path}")
        logger.info("Training and evaluation completed!")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
