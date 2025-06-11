#!/usr/bin/env python3
"""
Evaluation script for cutlery classification model.

This script evaluates the trained model on the test set and generates
comprehensive metrics and visualizations.

Outputs:
- Accuracy, Precision, Recall, F1-score
- Confusion Matrix visualization
- JSON file with all metrics
"""

import argparse
import json
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm

from cutlery_classifier.inference.inferencer import CutleryInferencer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_test_data(test_dir: Path) -> tuple:
    """
    Load test images and their true labels from directory structure.

    Args:
        test_dir: Path to test directory containing class subdirectories

    Returns:
        Tuple of (image_paths, true_labels, class_names)
    """
    image_paths = []
    true_labels = []
    class_names = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])

    for class_idx, class_name in enumerate(class_names):
        class_dir = test_dir / class_name
        for img_path in class_dir.glob("*.jpg"):
            image_paths.append(img_path)
            true_labels.append(class_idx)

    return image_paths, true_labels, class_names


def create_confusion_matrix_plot(
    conf_matrix: np.ndarray, class_names: list, output_path: Path
) -> None:
    """Create and save confusion matrix visualization."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate cutlery classifier on test set"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/checkpoints/type_detector_best.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="data/simplified/test",
        help="Path to test data directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cpu",
        help="Device to run evaluation on",
    )
    args = parser.parse_args()

    # Setup paths
    test_dir = Path(args.test_dir)
    results_dir = Path("results/evaluation")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load test data
    logger.info("Loading test data...")
    image_paths, true_labels, class_names = load_test_data(test_dir)
    logger.info(
        f"Found {len(image_paths)} test images across {len(class_names)} classes"
    )

    # Create inferencer
    logger.info(f"Loading model from: {args.model}")
    inferencer = CutleryInferencer(model_path=args.model, device=args.device)

    # Run inference on all test images
    logger.info("Running inference on test set...")
    predictions = []
    for img_path in tqdm(image_paths):
        result = inferencer.predict(img_path, top_k=1)
        pred_class_idx = result["predictions"][0]["class_index"]
        predictions.append(pred_class_idx)

    # Calculate metrics
    logger.info("Calculating metrics...")
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average="weighted"
    )

    # Create confusion matrix
    conf_matrix = confusion_matrix(true_labels, predictions)

    # Generate classification report
    class_report = classification_report(
        true_labels, predictions, target_names=class_names, output_dict=True
    )

    # Save confusion matrix plot
    conf_matrix_path = results_dir / "confusion_matrix.png"
    create_confusion_matrix_plot(conf_matrix, class_names, conf_matrix_path)
    logger.info(f"Confusion matrix saved to: {conf_matrix_path}")

    # Save metrics to JSON
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "per_class_metrics": class_report,
        "confusion_matrix": conf_matrix.tolist(),
        "class_names": class_names,
    }

    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to: {metrics_path}")

    # Print results
    logger.info("\nEvaluation Results:")
    logger.info("-" * 50)
    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")
    logger.info("-" * 50)


if __name__ == "__main__":
    main()
