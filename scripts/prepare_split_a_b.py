#!/usr/bin/env python3
"""
Data preparation script for creating manufacturer-specific class splits.

Input structure:
    data/raw/manufacturer_a/fork/
    data/raw/manufacturer_a/knife/
    data/raw/manufacturer_a/spoon/
    data/raw/manufacturer_b/fork/
    data/raw/manufacturer_b/knife/
    data/raw/manufacturer_b/spoon/

Output structure:
    data/processed/train/fork_a/
    data/processed/train/fork_b/
    data/processed/train/knife_a/
    data/processed/train/knife_b/
    data/processed/train/spoon_a/
    data/processed/train/spoon_b/
"""

import os
import time
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed/train")
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
    # Ensure base processed directory exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Create/recreate class-specific directories
    for class_name in CLASSES:
        for suffix in ["a", "b"]:
            target_dir = PROCESSED_DIR / f"{class_name}_{suffix}"

            # Safely remove if exists
            if target_dir.exists():
                logger.info(f"Removing existing directory: {target_dir}")
                safe_rmtree(target_dir)

            # Create fresh directory
            target_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {target_dir}")


def copy_images(source_dir: Path, target_dir: Path) -> int:
    """
    Copy images from source to target directory.

    Args:
        source_dir: Source directory containing images
        target_dir: Target directory to copy images to

    Returns:
        Number of images copied
    """
    if not source_dir.exists():
        logger.warning(f"Source directory does not exist: {source_dir}")
        return 0

    count = 0
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        for img_path in source_dir.glob(f"*{ext}"):
            shutil.copy2(img_path, target_dir / img_path.name)
            count += 1

    return count


def process_data() -> Dict[str, int]:
    """
    Process raw data into manufacturer-specific class splits.

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
            target_dir = PROCESSED_DIR / f"{class_name}_{suffix}"

            # Copy images
            count = copy_images(source_dir, target_dir)
            stats[f"{class_name}_{suffix}"] = count

            logger.info(f"Copied {count} images: {source_dir} → {target_dir}")

    return stats


def main():
    """Main function to prepare manufacturer-specific data split."""
    try:
        logger.info("Starting data preparation...")

        # Verify raw data directory exists
        if not RAW_DATA_DIR.exists():
            raise FileNotFoundError(f"Raw data directory not found: {RAW_DATA_DIR}")

        # Setup output directories
        setup_output_dirs()

        # Process data
        stats = process_data()

        # Print summary
        logger.info("\nProcessing complete! Summary:")
        for class_name, count in stats.items():
            logger.info(f"{class_name}: {count} images")

    except Exception as e:
        logger.error(f"Error during data preparation: {str(e)}")
        raise


if __name__ == "__main__":
    main()
