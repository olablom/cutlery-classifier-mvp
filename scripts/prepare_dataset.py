#!/usr/bin/env python3
"""
Dataset Preparation Script for Cutlery Classifier MVP

This script helps organize photo collection with proper directory structure
and provides utilities for dataset management.

Usage:
    python scripts/prepare_dataset.py --setup-dirs
    python scripts/prepare_dataset.py --validate-dataset
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

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Dataset configuration for MVP
DATASET_CONFIG = {
    "types": ["fork", "knife", "spoon"],
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


def create_train_val_test_splits():
    """Create train/validation/test splits from raw data."""

    print("📂 Creating train/val/test splits...")

    raw_dir = PROJECT_ROOT / "data" / "raw"
    processed_dir = PROJECT_ROOT / "data" / "processed"

    # Clear existing processed data
    if processed_dir.exists():
        shutil.rmtree(processed_dir)

    split_info = defaultdict(lambda: defaultdict(list))

    for cutlery_type in DATASET_CONFIG["types"]:
        for manufacturer in DATASET_CONFIG["manufacturers"]:
            type_dir = raw_dir / cutlery_type / manufacturer

            if not type_dir.exists():
                continue

            # Get all image files
            image_files = (
                list(type_dir.glob("*.jpg"))
                + list(type_dir.glob("*.jpeg"))
                + list(type_dir.glob("*.png"))
            )

            if len(image_files) < DATASET_CONFIG["min_images_per_class"]:
                print(
                    f"⚠️  Skipping {cutlery_type}/{manufacturer}: only {len(image_files)} images"
                )
                continue

            # Shuffle for random splits
            random.shuffle(image_files)

            # Calculate split sizes
            n_total = len(image_files)
            n_train = int(n_total * DATASET_CONFIG["train_split"])
            n_val = int(n_total * DATASET_CONFIG["val_split"])
            n_test = n_total - n_train - n_val

            # Split files
            train_files = image_files[:n_train]
            val_files = image_files[n_train : n_train + n_val]
            test_files = image_files[n_train + n_val :]

            # Copy files to processed directories
            for split, files in [
                ("train", train_files),
                ("val", val_files),
                ("test", test_files),
            ]:
                split_dir = processed_dir / split / cutlery_type
                split_dir.mkdir(parents=True, exist_ok=True)

                for i, src_file in enumerate(files):
                    # Create new filename with manufacturer info
                    dst_name = f"{manufacturer}_{i + 1:03d}{src_file.suffix}"
                    dst_path = split_dir / dst_name
                    shutil.copy2(src_file, dst_path)

                    split_info[split][cutlery_type].append(str(dst_path))

            print(
                f"✅ {cutlery_type}/{manufacturer}: {n_train} train, {n_val} val, {n_test} test"
            )

    # Save split information
    split_info_path = processed_dir / "split_info.json"
    with open(split_info_path, "w") as f:
        json.dump(dict(split_info), f, indent=2)

    print(f"📄 Split information saved: {split_info_path}")
    print("✅ Dataset splits created successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for cutlery classifier"
    )
    parser.add_argument(
        "--setup-dirs",
        action="store_true",
        help="Setup directory structure for data collection",
    )
    parser.add_argument(
        "--validate-dataset",
        action="store_true",
        help="Validate collected dataset and show statistics",
    )
    parser.add_argument(
        "--create-splits",
        action="store_true",
        help="Create train/val/test splits from raw data",
    )
    parser.add_argument(
        "--all", action="store_true", help="Run all operations in sequence"
    )

    args = parser.parse_args()

    if args.all:
        setup_directory_structure()
        print("\n" + "=" * 50)
        validate_dataset()
        print("\n" + "=" * 50)
        create_train_val_test_splits()
    elif args.setup_dirs:
        setup_directory_structure()
    elif args.validate_dataset:
        validate_dataset()
    elif args.create_splits:
        create_train_val_test_splits()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
