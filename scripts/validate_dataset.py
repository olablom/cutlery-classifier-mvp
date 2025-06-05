#!/usr/bin/env python3
"""
Dataset Validation Script for Cutlery Classifier MVP

This script validates the dataset structure and image quality.
"""

import os
from pathlib import Path
import argparse
import logging
from PIL import Image
import sys

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_dataset(data_dir: Path, quick_check: bool = False) -> bool:
    """
    Validate dataset structure and image quality.

    Args:
        data_dir: Path to data directory
        quick_check: If True, only check structure and count

    Returns:
        bool: True if validation passes
    """
    print("🔍 Dataset Validation")
    print("=" * 50 + "\n")

    categories = ["fork", "knife", "spoon"]
    total_images = 0
    category_counts = {}

    for category in categories:
        category_dir = data_dir / category
        if not category_dir.exists():
            print(f"❌ Missing directory: {category}")
            continue

        # Count images
        image_files = [f for f in category_dir.glob("*.jpg")]
        count = len(image_files)
        category_counts[category] = count
        total_images += count

        print(f"{category.capitalize()}:")
        print(f"  Images found: {count}")

        if not quick_check:
            # Validate each image
            for img_path in image_files:
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        if width < 300 or height < 300:
                            print(f"⚠️ Low resolution image: {img_path.name}")
                except Exception as e:
                    print(f"❌ Error reading {img_path.name}: {e}")

    print("\n📊 Summary:")
    print(f"Total images: {total_images}")

    # Validation criteria
    min_images_per_class = 10
    max_ratio = (
        max(category_counts.values()) / min(category_counts.values())
        if category_counts
        else 0
    )

    if total_images < min_images_per_class * len(categories):
        print("❌ Need more images before training")
        return False
    elif max_ratio > 1.5:
        print("⚠️ Class imbalance detected")
        return False
    else:
        print("✅ Dataset validation passed")
        return True


def main():
    parser = argparse.ArgumentParser(description="Validate cutlery dataset")
    parser.add_argument(
        "--quick-check", action="store_true", help="Only check structure and count"
    )
    args = parser.parse_args()

    data_dir = project_root / "data" / "raw"
    validate_dataset(data_dir, args.quick_check)


if __name__ == "__main__":
    main()
