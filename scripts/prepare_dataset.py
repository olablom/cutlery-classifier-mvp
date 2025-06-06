#!/usr/bin/env python3
"""
Dataset Preparation Script for Cutlery Classifier MVP

This script prepares the dataset by:
1. Creating train/validation/test splits
2. Copying images to processed directory
3. Validating the splits

Usage:
    python scripts/prepare_dataset.py --create-splits
"""

import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import shutil
from collections import defaultdict
import random
import logging
import sys

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Dataset configuration for MVP
DATASET_CONFIG = {
    "types": ["fork_a", "fork_b", "knife_a", "knife_b", "spoon_a", "spoon_b"],
    "manufacturers": ["ikea", "obh"],  # Easy to find, distinct styles
    "target_images_per_class": 40,
    "min_images_per_class": 20,
    "train_split": 0.7,
    "val_split": 0.2,
    "test_split": 0.1,
}


def setup_directory_structure():
    """Create the complete directory structure for data collection."""

    print("🏗️  Setting up directory structure for data collection...")

    # Raw data directories
    raw_dir = PROJECT_ROOT / "data" / "raw"
    processed_dir = PROJECT_ROOT / "data" / "processed"

    # Create type-based structure for MVP (type detection)
    for cutlery_type in DATASET_CONFIG["types"]:
        for manufacturer in DATASET_CONFIG["manufacturers"]:
            # Raw directories
            type_dir = raw_dir / cutlery_type / manufacturer
            type_dir.mkdir(parents=True, exist_ok=True)

            # Create a README in each directory
            readme_path = type_dir / "README.md"
            with open(readme_path, "w") as f:
                f.write(f"# {cutlery_type.title()} - {manufacturer.upper()}\n\n")
                f.write(f"Target: {DATASET_CONFIG['target_images_per_class']} images\n")
                f.write(f"Minimum: {DATASET_CONFIG['min_images_per_class']} images\n\n")
                f.write("## Photo Guidelines:\n")
                f.write("- Use mobile camera\n")
                f.write("- Light, neutral background (white plate/tray)\n")
                f.write("- Various angles and distances\n")
                f.write("- Natural lighting preferred\n")
                f.write("- Name files: img_001.jpg, img_002.jpg, etc.\n")

    # Processed data directories
    for split in ["train", "val", "test"]:
        for cutlery_type in DATASET_CONFIG["types"]:
            split_dir = processed_dir / split / cutlery_type
            split_dir.mkdir(parents=True, exist_ok=True)

    print("✅ Directory structure created!")
    print(f"📁 Raw data: {raw_dir}")
    print(f"📁 Processed data: {processed_dir}")

    # Create collection checklist
    create_collection_checklist()


def create_collection_checklist():
    """Create a checklist for photo collection."""

    checklist_path = PROJECT_ROOT / "data" / "collection_checklist.md"

    with open(checklist_path, "w") as f:
        f.write("# Photo Collection Checklist\n\n")
        f.write("## Target: 240 images total (40 per class)\n\n")

        for cutlery_type in DATASET_CONFIG["types"]:
            f.write(f"### {cutlery_type.title()}\n\n")
            for manufacturer in DATASET_CONFIG["manufacturers"]:
                f.write(
                    f"- [ ] **{manufacturer.upper()}**: 0/{DATASET_CONFIG['target_images_per_class']} images\n"
                )
                f.write(f"  - Location: `data/raw/{cutlery_type}/{manufacturer}/`\n")
                f.write(f"  - Files: `img_001.jpg` to `img_040.jpg`\n\n")

        f.write("## Photo Guidelines\n\n")
        f.write("### Setup\n")
        f.write("- 📱 Use mobile camera\n")
        f.write("- 🟫 Light, neutral background (white plate/tray)\n")
        f.write("- 💡 Natural lighting (near window, daytime)\n")
        f.write("- 📏 Fill frame but leave some margin\n\n")

        f.write("### Variation (per manufacturer)\n")
        f.write("- 🔄 **Angles**: Top-down, 45°, side view\n")
        f.write("- 📐 **Orientations**: Horizontal, vertical, diagonal\n")
        f.write("- 📏 **Distances**: Close-up, medium, slightly farther\n")
        f.write("- 🎯 **Positions**: Center, slightly off-center\n\n")

        f.write("### Quick Tips\n")
        f.write("- Take 5-10 photos, then move the cutlery slightly\n")
        f.write("- Rotate the piece between shots\n")
        f.write("- Change your position/height\n")
        f.write("- Keep background consistent within each session\n")

    print(f"📋 Collection checklist created: {checklist_path}")


def validate_dataset():
    """Validate the collected dataset and provide statistics."""

    print("🔍 Validating dataset...")

    raw_dir = PROJECT_ROOT / "data" / "raw"
    stats = defaultdict(lambda: defaultdict(int))
    total_images = 0

    for cutlery_type in DATASET_CONFIG["types"]:
        for manufacturer in DATASET_CONFIG["manufacturers"]:
            type_dir = raw_dir / cutlery_type / manufacturer

            if type_dir.exists():
                # Count images (jpg, jpeg, png)
                image_files = (
                    list(type_dir.glob("*.jpg"))
                    + list(type_dir.glob("*.jpeg"))
                    + list(type_dir.glob("*.png"))
                )

                count = len(image_files)
                stats[cutlery_type][manufacturer] = count
                total_images += count

    # Print statistics
    print(f"\n📊 Dataset Statistics (Total: {total_images} images)")
    print("=" * 50)

    for cutlery_type in DATASET_CONFIG["types"]:
        print(f"\n{cutlery_type.title()}:")
        type_total = 0
        for manufacturer in DATASET_CONFIG["manufacturers"]:
            count = stats[cutlery_type][manufacturer]
            type_total += count
            status = "✅" if count >= DATASET_CONFIG["min_images_per_class"] else "⚠️"
            target = DATASET_CONFIG["target_images_per_class"]
            print(f"  {manufacturer.upper()}: {count:2d}/{target} {status}")
        print(f"  Subtotal: {type_total}")

    # Check readiness
    ready_classes = 0
    total_classes = len(DATASET_CONFIG["types"]) * len(DATASET_CONFIG["manufacturers"])

    for cutlery_type in DATASET_CONFIG["types"]:
        for manufacturer in DATASET_CONFIG["manufacturers"]:
            if (
                stats[cutlery_type][manufacturer]
                >= DATASET_CONFIG["min_images_per_class"]
            ):
                ready_classes += 1

    print(f"\n🎯 Training Readiness: {ready_classes}/{total_classes} classes ready")

    if ready_classes >= total_classes:
        print("✅ Dataset ready for training!")
    elif ready_classes >= total_classes * 0.5:
        print("⚠️  Dataset partially ready - can start with available classes")
    else:
        print("❌ Need more images before training")

    return stats


def create_train_val_test_splits(
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> bool:
    """
    Create train/validation/test splits from raw data.

    Args:
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        seed: Random seed for reproducibility

    Returns:
        bool: True if successful
    """
    print("\n📂 Creating train/val/test splits...")

    # Setup paths
    raw_dir = PROJECT_ROOT / "data" / "raw"
    processed_dir = PROJECT_ROOT / "data" / "processed"

    # Create processed directory structure
    splits = ["train", "val", "test"]
    categories = DATASET_CONFIG["types"]

    # Remove existing processed directory if it exists
    if processed_dir.exists():
        try:
            shutil.rmtree(processed_dir)
        except PermissionError:
            print(
                "⚠️ Could not remove existing processed directory. Please close any open files and try again."
            )
            return False
        except Exception as e:
            print(f"⚠️ Error removing processed directory: {e}")
            return False

    # Create directory structure
    for split in splits:
        for category in categories:
            (processed_dir / split / category).mkdir(parents=True, exist_ok=True)

    # Process each category
    for category in categories:
        print(f"\nProcessing {category}...")

        # Get all images
        category_dir = raw_dir / category
        if not category_dir.exists():
            print(f"⚠️ Category directory not found: {category}")
            continue

        image_files = list(category_dir.glob("*.jpg"))
        if not image_files:
            print(f"⚠️ No images found in {category}")
            continue

        random.shuffle(image_files)

        # Calculate split sizes
        total = len(image_files)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)

        # Split files
        train_files = image_files[:train_size]
        val_files = image_files[train_size : train_size + val_size]
        test_files = image_files[train_size + val_size :]

        # Copy files to respective directories
        for files, split in [
            (train_files, "train"),
            (val_files, "val"),
            (test_files, "test"),
        ]:
            target_dir = processed_dir / split / category
            print(f"  {split}: {len(files)} images")

            for src in files:
                dst = target_dir / src.name
                try:
                    shutil.copy2(src, dst)
                except Exception as e:
                    print(f"⚠️ Error copying {src.name}: {e}")

    # Verify splits
    print("\n📊 Final Split Summary:")
    total_processed = 0

    for split in splits:
        split_total = 0
        print(f"\n{split.upper()}:")

        for category in categories:
            count = len(list((processed_dir / split / category).glob("*.jpg")))
            split_total += count
            print(f"  {category}: {count} images")

        print(f"  Total: {split_total}")
        total_processed += split_total

    print(f"\nTotal processed images: {total_processed}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Prepare cutlery dataset")
    parser.add_argument(
        "--create-splits", action="store_true", help="Create train/val/test splits"
    )
    args = parser.parse_args()

    if args.create_splits:
        create_train_val_test_splits()


if __name__ == "__main__":
    main()
