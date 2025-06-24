#!/usr/bin/env python3
"""
Generate professional visualizations for real-world metrics analysis.
Creates bar charts and comparison plots for the cutlery classifier performance.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style for professional plots
plt.style.use("default")
sns.set_palette("husl")


def create_metrics_comparison():
    """Create comprehensive metrics comparison visualization."""

    # Data from real-world test analysis
    classes = ["Fork", "Knife", "Spoon"]
    accuracy = [40.0, 66.7, 91.7]
    precision = [61.5, 59.7, 74.3]
    recall = [40.0, 66.7, 91.7]
    f1_score = [48.5, 63.0, 82.1]

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        "Cutlery Classifier: Real-World Performance Metrics",
        fontsize=16,
        fontweight="bold",
    )

    # Plot 1: Accuracy comparison
    bars1 = ax1.bar(
        classes, accuracy, color=["#FF6B6B", "#4ECDC4", "#45B7D1"], alpha=0.8
    )
    ax1.set_title("Accuracy by Class", fontweight="bold")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars1, accuracy):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Plot 2: Precision, Recall, F1 comparison
    x = np.arange(len(classes))
    width = 0.25

    bars2 = ax2.bar(
        x - width, [p * 100 for p in precision], width, label="Precision", alpha=0.8
    )
    bars3 = ax2.bar(x, [r * 100 for r in recall], width, label="Recall", alpha=0.8)
    bars4 = ax2.bar(
        x + width, [f * 100 for f in f1_score], width, label="F1-Score", alpha=0.8
    )

    ax2.set_title("Precision, Recall, and F1-Score", fontweight="bold")
    ax2.set_ylabel("Score (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    # Add value labels
    for bars in [bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Plot 3: Error analysis
    errors = [36, 20, 5]  # Number of errors per class
    total = [60, 60, 60]  # Total samples per class
    error_rate = [e / t * 100 for e, t in zip(errors, total)]

    bars5 = ax3.bar(
        classes, error_rate, color=["#FF6B6B", "#FFA07A", "#98FB98"], alpha=0.8
    )
    ax3.set_title("Error Rate by Class", fontweight="bold")
    ax3.set_ylabel("Error Rate (%)")
    ax3.grid(True, alpha=0.3)

    # Add value labels
    for bar, rate, error in zip(bars5, error_rate, errors):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{rate:.1f}%\n({error} errors)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Plot 4: Overall performance summary
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    overall_scores = [66.1, 65.2, 66.1, 64.5]  # Macro averages

    bars6 = ax4.bar(
        metrics,
        overall_scores,
        color=["#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"],
        alpha=0.8,
    )
    ax4.set_title("Overall Performance (Macro Average)", fontweight="bold")
    ax4.set_ylabel("Score (%)")
    ax4.set_ylim(0, 100)
    ax4.grid(True, alpha=0.3)

    # Add value labels
    for bar, score in zip(bars6, overall_scores):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{score:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()

    # Save the plot
    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_dir / "real_world_metrics_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(
        f"✅ Metrics comparison visualization saved to {output_dir / 'real_world_metrics_comparison.png'}"
    )


def create_confusion_matrix_visualization():
    """Create a professional confusion matrix visualization."""

    # Confusion matrix data (estimated from analysis)
    cm_data = np.array(
        [
            [24, 25, 11],  # Fork predictions
            [12, 40, 8],  # Knife predictions
            [3, 2, 55],  # Spoon predictions
        ]
    )

    classes = ["Fork", "Knife", "Spoon"]

    # Create confusion matrix plot
    plt.figure(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(
        cm_data,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={"label": "Number of Predictions"},
    )

    plt.title(
        "Confusion Matrix: Real-World Test Set", fontsize=16, fontweight="bold", pad=20
    )
    plt.xlabel("Predicted Class", fontsize=12, fontweight="bold")
    plt.ylabel("True Class", fontsize=12, fontweight="bold")

    # Add performance metrics as text
    accuracy = np.sum(np.diag(cm_data)) / np.sum(cm_data) * 100
    plt.figtext(
        0.5,
        0.02,
        f"Overall Accuracy: {accuracy:.1f}%",
        ha="center",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
    )

    plt.tight_layout()

    # Save the plot
    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_dir / "real_world_confusion_matrix.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(
        f"✅ Confusion matrix visualization saved to {output_dir / 'real_world_confusion_matrix.png'}"
    )


def main():
    """Generate all visualizations."""
    print("📊 Generating professional visualizations for real-world metrics...")

    create_metrics_comparison()
    create_confusion_matrix_visualization()

    print("✅ All visualizations completed!")


if __name__ == "__main__":
    main()
