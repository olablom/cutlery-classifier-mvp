#!/usr/bin/env python3
"""
Image Analysis Script for Cutlery Dataset

This script helps analyze and categorize existing images in the dataset.
It creates a montage of images for each class and saves them for review.
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def create_montage(image_paths, output_path, title, rows=4):
    """Create a montage of images for visualization."""
    n_images = len(image_paths)
    cols = (n_images + rows - 1) // rows

    # Create figure
    fig = plt.figure(figsize=(15, 15))
    plt.suptitle(title, fontsize=16)

    # Add each image as a subplot
    for idx, img_path in enumerate(image_paths):
        img = Image.open(img_path)
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(np.array(img))
        plt.title(f"Image {idx + 1}\n{Path(img_path).name}", fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def analyze_dataset():
    """Analyze the existing dataset and create visualizations."""
    # Setup paths
    data_root = Path("data")
    raw_dir = data_root / "raw"
    results_dir = Path("results/dataset_analysis")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Process each class
    for class_name in ["fork", "knife", "spoon"]:
        # Get all images for this class
        class_dir = raw_dir / class_name
        images = list(class_dir.glob("*.jpg"))

        # Create visualization
        output_path = results_dir / f"{class_name}_montage.png"
        create_montage(
            images, output_path, f"{class_name.upper()} Images (Total: {len(images)})"
        )

        print(f"\n{class_name.upper()} Analysis:")
        print(f"Total images: {len(images)}")
        print("Image files:")
        for img in sorted(images):
            print(f"  - {img.name}")


if __name__ == "__main__":
    analyze_dataset()
