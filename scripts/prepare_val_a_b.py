#!/usr/bin/env python3
"""
Validation data preparation script for creating manufacturer-specific class splits.

Input structure:
    data/raw/manufacturer_a/{fork,knife,spoon}/
    data/raw/manufacturer_b/{fork,knife,spoon}/

Output structure:
    data/processed/val/fork_a/
    data/processed/val/fork_b/
    data/processed/val/knife_a/
    data/processed/val/knife_b/
    data/processed/val/spoon_a/
    data/processed/val/spoon_b/

Each output folder will contain N randomly selected images (default: N=5).
"""

import os
import time
import logging
import shutil
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
RAW_DATA_DIR = Path("data/raw")
VAL_DIR = Path("data/processed/val")
MANUFACTURERS = ["manufacturer_a", "manufacturer_b"]
CLASSES = ["fork", "knife", "spoon"]


def safe_rmtree(path: Path, max_retries: int = 3, retry_delay: float = 1.0) -> None:
    """
    Safely remove a directory tree with retries for Windows filesystem locks.

    Args:
        path: Directory path to remove
        max_retries: Maximum number of removal attempts
        retry_delay: Delay between retries in seconds
    """
    if not path.exists():
        return

    for attempt in range(max_retries):
        try:
            shutil.rmtree(path)
            break
        except PermissionError as e:
            if attempt == max_retries - 1:
                logger.error(
                    f"Failed to remove {path} after {max_retries} attempts: {e}"
                )
                raise
            logger.warning(
                f"Permission error removing {path}, retrying in {retry_delay}s..."
            )
            time.sleep(retry_delay)


def setup_output_dirs() -> None:
    """Create output directories, safely removing old ones if they exist."""
    # Ensure base validation directory exists
    VAL_DIR.mkdir(parents=True, exist_ok=True)

    # Create/recreate class-specific directories
    for class_name in CLASSES:
        for suffix in ["a", "b"]:
            target_dir = VAL_DIR / f"{class_name}_{suffix}"

            # Safely remove if exists
            if target_dir.exists():
                logger.info(f"Removing existing directory: {target_dir}")
                safe_rmtree(target_dir)

            # Create fresh directory
            target_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {target_dir}")


def get_image_files(source_dir: Path) -> List[Path]:
    """
    Get list of image files from directory.

    Args:
        source_dir: Source directory containing images

    Returns:
        List of image file paths
    """
    if not source_dir.exists():
        return []

    image_files = []
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        image_files.extend(list(source_dir.glob(f"*{ext}")))
    return image_files


def copy_random_images(source_dir: Path, target_dir: Path, num_images: int) -> int:
    """
    Copy random subset of images from source to target directory.

    Args:
        source_dir: Source directory containing images
        target_dir: Target directory to copy images to
        num_images: Number of images to randomly select and copy

    Returns:
        Number of images actually copied
    """
    if not source_dir.exists():
        logger.warning(f"Source directory does not exist: {source_dir}")
        return 0

    # Get all image files
    image_files = get_image_files(source_dir)

    # Randomly select images
    num_to_copy = min(num_images, len(image_files))
    if num_to_copy < num_images:
        logger.warning(
            f"Only {num_to_copy} images available in {source_dir} "
            f"(requested {num_images})"
        )

    selected_files = random.sample(image_files, num_to_copy)

    # Copy selected images
    for img_path in selected_files:
        shutil.copy2(img_path, target_dir / img_path.name)

    return num_to_copy


def process_validation_data(num_images: int) -> Dict[str, int]:
    """
    Process raw data into manufacturer-specific validation splits.

    Args:
        num_images: Number of images to select per class

    Returns:
        Dictionary with copy statistics
    """
    stats = {}

    # Process each manufacturer
    for i, manufacturer in enumerate(MANUFACTURERS):
        suffix = "a" if i == 0 else "b"
        manufacturer_dir = RAW_DATA_DIR / manufacturer

        if not manufacturer_dir.exists():
            logger.error(f"Manufacturer directory not found: {manufacturer_dir}")
            continue

        # Process each class
        for class_name in CLASSES:
            source_dir = manufacturer_dir / class_name
            target_dir = VAL_DIR / f"{class_name}_{suffix}"

            # Copy random subset of images
            count = copy_random_images(source_dir, target_dir, num_images)
            stats[f"{class_name}_{suffix}"] = count

            logger.info(
                f"Copied {count} images for validation: {source_dir} → {target_dir}"
            )

    return stats


def verify_validation_split(stats: Dict[str, int]) -> None:
    """
    Verify that validation folders contain the expected images.

    Args:
        stats: Dictionary with copy statistics
    """
    logger.info("\nVerifying validation split:")

    for class_name in CLASSES:
        for suffix in ["a", "b"]:
            val_dir = VAL_DIR / f"{class_name}_{suffix}"
            actual_count = len(list(get_image_files(val_dir)))
            expected_count = stats[f"{class_name}_{suffix}"]

            if actual_count == expected_count:
                logger.info(
                    f"✓ {class_name}_{suffix}: {actual_count} images (as expected)"
                )
            else:
                logger.error(
                    f"✗ {class_name}_{suffix}: Found {actual_count} images, "
                    f"expected {expected_count}"
                )


def main():
    """Main function to prepare manufacturer-specific validation split."""
    parser = argparse.ArgumentParser(
        description="Prepare validation split with manufacturer-specific classes"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=5,
        help="Number of images to select per class (default: 5)",
    )

    args = parser.parse_args()

    try:
        logger.info("Starting validation data preparation...")

        # Verify raw data directory exists
        if not RAW_DATA_DIR.exists():
            raise FileNotFoundError(f"Raw data directory not found: {RAW_DATA_DIR}")

        # Set random seed for reproducibility
        random.seed(42)

        # Setup output directories
        setup_output_dirs()

        # Process validation data
        stats = process_validation_data(args.num_images)

        # Verify the split
        verify_validation_split(stats)

        # Print summary
        logger.info("\nProcessing complete! Summary:")
        for class_name, count in stats.items():
            logger.info(f"{class_name}: {count} images")

    except Exception as e:
        logger.error(f"Error during validation data preparation: {str(e)}")
        raise


if __name__ == "__main__":
    main()
