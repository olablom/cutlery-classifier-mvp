import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import json
import os


def compute_metrics(y_true, y_pred, classes):
    """
    Compute comprehensive classification metrics including per-class performance.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        classes: List of class names

    Returns:
        Dictionary containing overall and per-class metrics
    """
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(classes)), average=None
    )

    # Create per-class metrics dictionary
    per_class_metrics = {}
    for i, class_name in enumerate(classes):
        per_class_metrics[class_name] = {
            "precision": precision[i] * 100,
            "recall": recall[i] * 100,
            "f1": f1[i] * 100,
            "support": int(support[i]),
        }

    # Overall metrics (weighted average)
    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(y_true, y_pred, average="weighted")
    )

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "overall": {
            "accuracy": accuracy * 100,
            "weighted_precision": weighted_precision * 100,
            "weighted_recall": weighted_recall * 100,
            "weighted_f1": weighted_f1 * 100,
        },
        "per_class": per_class_metrics,
    }

    return metrics


def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """
    Plot and save an enhanced confusion matrix with percentages and counts.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        classes: List of class names
        save_path: Path to save the confusion matrix plot
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Raw counts
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax1,
    )
    ax1.set_title("Confusion Matrix (Counts)")
    ax1.set_ylabel("True Label")
    ax1.set_xlabel("Predicted Label")

    # Percentages
    sns.heatmap(
        cm_norm * 100,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax2,
    )
    ax2.set_title("Confusion Matrix (Percentages)")
    ax2.set_ylabel("True Label")
    ax2.set_xlabel("Predicted Label")

    plt.tight_layout()

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_metrics(metrics, save_path):
    """
    Save metrics to a JSON file with timestamp.

    Args:
        metrics: Dictionary containing the metrics
        save_path: Path to save the metrics JSON file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)

    logging.info(f"Metrics saved to: {save_path}")

    # Log key metrics
    logging.info(f"Overall Accuracy: {metrics['overall']['accuracy']:.2f}%")
    logging.info(f"Weighted F1-Score: {metrics['overall']['weighted_f1']:.2f}%")


def plot_confusion_matrix_vg(y_true, y_pred, class_names, save_path):
    """
    Plot and save a VG-grade confusion matrix with both counts and percentages in each cell.
    Args:
        y_true: Ground truth labels (list or np.array)
        y_pred: Predicted labels (list or np.array)
        class_names: List of class names (strings)
        save_path: Path to save the confusion matrix plot (PNG)
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    cm_sum = cm.sum(axis=1, keepdims=True)
    # Avoid division by zero
    with np.errstate(all="ignore"):
        cm_perc = np.divide(cm, cm_sum, where=cm_sum != 0) * 100
        cm_perc = np.nan_to_num(cm_perc)

    n_classes = len(class_names)
    annot = np.empty_like(cm).astype(str)
    for i in range(n_classes):
        for j in range(n_classes):
            count = cm[i, j]
            perc = cm_perc[i, j]
            if cm_sum[i, 0] == 0:
                annot[i, j] = f"0 (0%)"
            else:
                annot[i, j] = f"{count} ({perc:.0f}%)"

    plt.figure(figsize=(15, 12))
    ax = sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=0.5,
        linecolor="gray",
        square=True,
    )
    ax.set_xlabel("Predicted Label", fontsize=16)
    ax.set_ylabel("True Label", fontsize=16)
    ax.set_title("Confusion Matrix (Counts and % per class)", fontsize=18, pad=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
