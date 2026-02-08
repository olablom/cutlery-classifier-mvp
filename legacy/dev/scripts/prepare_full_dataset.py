#!/usr/bin/env python3
"""
Complete Dataset Preparation Pipeline v3.0

This script handles the entire dataset preparation process:
1. Reorganizes raw images into 3 classes
2. Applies augmentation to training data
3. Applies preprocessing pipeline
4. Creates train/val/test splits

Input: Raw images (60 images: 3 types × 2 manufacturers × 10 images)
Output: Processed dataset ready for training

Version: 3.0 - Production Ready Edition™
Author: Ola Blom
Date: 2024-06-08
"""

import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime
import time

import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constants
SPLITS = ["train", "val", "test"]
CLASS_MAPPING = {
    "manufacturer_a/fork": "fork",
    "manufacturer_b/fork": "fork",
    "manufacturer_a/knife": "knife",
    "manufacturer_b/knife": "knife",
    "manufacturer_a/spoon": "spoon",
    "manufacturer_b/spoon": "spoon",
}

# Directory structure
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# Preprocessing parameters
INPUT_SIZE = (320, 320)
CROP_SIZE = (224, 224)
NORMALIZATION_PARAMS = {"mean": [0.5], "std": [0.5]}

# Augmentation parameters
AUGMENTATION_CONFIG = {
    "rotation_range": (-30, 30),
    "scale_range": (0.8, 1.2),
    "brightness_range": (0.7, 1.3),
    "noise_std": 0.01,
    "num_variations": 3,
}

# Version info
VERSION = "3.0"
PIPELINE_NAME = "Production Ready Edition™"


def clean_directories() -> None:
    """Clean and create output directories."""
    logger.info("Cleaning output directories...")

    # Remove existing processed directory if it exists
    if PROCESSED_DIR.exists():
        try:
            logger.info(f"Removing existing directory: {PROCESSED_DIR}")
            shutil.rmtree(PROCESSED_DIR, ignore_errors=True)
            # Windows workaround: wait a bit after deletion
            time.sleep(1)
        except Exception as e:
            logger.warning(f"Could not fully remove directory: {e}")
            logger.info("Continuing anyway...")

    # Create new directory structure
    for split in SPLITS:
        split_dir = PROCESSED_DIR / split
        logger.info(f"Creating directory structure for {split} split...")

        # Create class directories
        for class_name in set(CLASS_MAPPING.values()):
            class_dir = split_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {class_dir}")


def create_preprocessing_pipeline() -> T.Compose:
    """Create the standard preprocessing pipeline."""
    return T.Compose(
        [
            T.Grayscale(),
            T.Resize(INPUT_SIZE),
            T.CenterCrop(CROP_SIZE),
            T.ToTensor(),
            T.Normalize(**NORMALIZATION_PARAMS),
        ]
    )


def create_augmentation_pipeline() -> T.Compose:
    """Create the augmentation pipeline for training data."""
    return T.Compose(
        [
            T.RandomRotation(AUGMENTATION_CONFIG["rotation_range"]),
            T.RandomResizedCrop(
                INPUT_SIZE, scale=AUGMENTATION_CONFIG["scale_range"], ratio=(0.9, 1.1)
            ),
            T.ColorJitter(brightness=AUGMENTATION_CONFIG["brightness_range"][1]),
            T.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
        ]
    )


def process_image(
    img_path: Path,
    preprocess_pipeline: Any,
    augment_pipeline: Any = None,
    num_aug: int = 0,
) -> List[torch.Tensor]:
    """Process a single image and its augmentations."""
    try:
        # Read image using PIL
        img = Image.open(str(img_path)).convert("L")

        # Apply preprocessing
        processed = preprocess_pipeline(img)
        processed = processed.unsqueeze(0)  # Add batch dimension
        results = [processed]

        # Generate augmentations if requested
        if augment_pipeline and num_aug > 0:
            for _ in range(num_aug):
                aug_img = augment_pipeline(img)
                aug_processed = preprocess_pipeline(aug_img)
                aug_processed = aug_processed.unsqueeze(0)
                results.append(aug_processed)

        return results
    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
        return []


def organize_dataset() -> Dict[str, Dict[str, int]]:
    """Organize and process the dataset."""
    logger.info("Starting dataset organization...")
    stats = {
        split: {class_name: 0 for class_name in set(CLASS_MAPPING.values())}
        for split in SPLITS
    }

    # First, collect all images per class
    class_images = {class_name: [] for class_name in set(CLASS_MAPPING.values())}
    for source_path, target_class in CLASS_MAPPING.items():
        source_dir = RAW_DATA_DIR / source_path
        if not source_dir.exists():
            logger.warning(f"Source directory not found: {source_dir}")
            continue

        class_images[target_class].extend(list(source_dir.glob("*.jpg")))

    # Calculate minimum number of images per class
    min_images = min(len(imgs) for imgs in class_images.values())
    logger.info(f"Balancing classes to {min_images} images each")

    # Balance classes and split data
    for target_class, images in class_images.items():
        # Randomly select equal number of images per class
        selected_images = np.random.choice(images, size=min_images, replace=False)
        np.random.shuffle(selected_images)

        # Split into train/val/test (60/20/20)
        n_train = int(0.6 * min_images)
        n_val = int(0.2 * min_images)

        splits = {
            "train": selected_images[:n_train],
            "val": selected_images[n_train : n_train + n_val],
            "test": selected_images[n_train + n_val :],
        }

        # Process each split
        for split_name, split_images in splits.items():
            target_dir = PROCESSED_DIR / split_name / target_class

            for img_path in split_images:
                # Generate a unique filename that's safe for Windows
                source_path = (
                    str(img_path.parent).replace(str(RAW_DATA_DIR), "").strip("/\\")
                )
                safe_name = f"{source_path.replace('/', '_')}_{img_path.stem}"
                safe_name = "".join(c for c in safe_name if c.isalnum() or c in "._- ")

                # Process image and get tensors
                processed_images = process_image(
                    img_path,
                    create_preprocessing_pipeline(),
                    create_augmentation_pipeline() if split_name == "train" else None,
                    AUGMENTATION_CONFIG["num_variations"]
                    if split_name == "train"
                    else 0,
                )

                if not processed_images:  # Skip if processing failed
                    continue

                # Save original processed image
                torch.save(processed_images[0], target_dir / f"{safe_name}.pt")
                stats[split_name][target_class] += 1

                # If this is a training image, save augmented versions
                if split_name == "train" and len(processed_images) > 1:
                    for i, aug_img in enumerate(processed_images[1:], 1):
                        aug_name = f"{safe_name}_aug{i}"
                        torch.save(aug_img, target_dir / f"{aug_name}.pt")
                        stats[split_name][target_class] += 1

    return stats


def validate_dataset_balance(
    stats: Dict[str, Dict[str, int]],
) -> Tuple[bool, List[str]]:
    """Validate dataset balance and generate warnings."""
    warnings = []
    is_valid = True

    # Check class balance
    for split in SPLITS:
        split_stats = stats[split]
        counts = list(split_stats.values())

        # Check if all classes have same count
        if len(set(counts)) > 1:
            is_valid = False
            warnings.append(f"Unbalanced classes in {split}: {split_stats}")

        # Check minimum images per class
        min_required = 5 if split != "train" else 10
        if any(count < min_required for count in counts):
            is_valid = False
            warnings.append(
                f"Too few images in {split} (min {min_required}): {split_stats}"
            )

    return is_valid, warnings


def generate_report(stats: Dict[str, Dict[str, int]]) -> str:
    """Generate a detailed report of the dataset preparation."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = [
        "Dataset Preparation Report",
        "========================",
        f"Version: {VERSION} - {PIPELINE_NAME}",
        f"Generated: {timestamp}",
        "",
        "Pipeline Configuration:",
        "---------------------",
        f"Input Size: {INPUT_SIZE}",
        f"Crop Size: {CROP_SIZE}",
        f"Normalization: mean={NORMALIZATION_PARAMS['mean'][0]}, std={NORMALIZATION_PARAMS['std'][0]}",
        "",
        "Augmentation Settings:",
        "--------------------",
        f"Rotation Range: {AUGMENTATION_CONFIG['rotation_range']}°",
        f"Scale Range: {AUGMENTATION_CONFIG['scale_range']}",
        f"Brightness Range: {AUGMENTATION_CONFIG['brightness_range']}",
        f"Noise STD: {AUGMENTATION_CONFIG['noise_std']}",
        f"Variations per Image: {AUGMENTATION_CONFIG['num_variations']}",
        "",
        "Dataset Statistics:",
        "------------------",
    ]

    total_images = sum(sum(split.values()) for split in stats.values())
    total_original = sum(stats["train"].values()) // (
        AUGMENTATION_CONFIG["num_variations"] + 1
    )
    report.append(f"Total processed images: {total_images}")
    report.append(f"Original images: {total_original}")
    report.append(f"Augmented images: {total_images - total_original}")
    report.append("")

    for split in SPLITS:
        report.append(f"{split.capitalize()} Split:")
        split_total = sum(stats[split].values())
        for class_name, count in stats[split].items():
            report.append(f"  - {class_name}: {count} images")
        report.append(f"  Total: {split_total} images")
        report.append("")

    # Add validation results
    is_valid, warnings = validate_dataset_balance(stats)
    report.append("Validation Results:")
    report.append("------------------")
    report.append(f"Dataset Balance: {'✅ Valid' if is_valid else '❌ Invalid'}")
    if warnings:
        report.append("\nWarnings:")
        for warning in warnings:
            report.append(f"- {warning}")

    return "\n".join(report)


def main():
    """Main execution function."""
    logger.info(f"Starting dataset preparation pipeline v{VERSION} - {PIPELINE_NAME}")

    # Validate raw data directory
    if not RAW_DATA_DIR.exists():
        logger.error(f"Raw data directory not found: {RAW_DATA_DIR}")
        sys.exit(1)

    # Clean and create directory structure
    clean_directories()

    # Process dataset
    stats = organize_dataset()

    # Generate and save report
    report = generate_report(stats)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed report with UTF-8 encoding
    report_path = PROCESSED_DIR / f"preparation_report_{timestamp}.txt"
    report_path.write_text(report, encoding="utf-8")

    # Save stats as JSON for easy parsing
    stats_path = PROCESSED_DIR / f"dataset_stats_{timestamp}.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "version": VERSION,
                "pipeline_name": PIPELINE_NAME,
                "timestamp": timestamp,
                "stats": stats,
                "config": {
                    "preprocessing": {
                        "input_size": INPUT_SIZE,
                        "crop_size": CROP_SIZE,
                        "normalization": NORMALIZATION_PARAMS,
                    },
                    "augmentation": AUGMENTATION_CONFIG,
                },
            },
            f,
            indent=2,
        )

    logger.info("Dataset preparation completed successfully!")
    logger.info("\n" + report)


if __name__ == "__main__":
    main()
