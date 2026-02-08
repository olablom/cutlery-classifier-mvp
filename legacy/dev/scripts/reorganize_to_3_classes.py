#!/usr/bin/env python3
"""
Reorganize Dataset to 3 Classes

This script consolidates the 6-class dataset (fork_a/b, knife_a/b, spoon_a/b)
into a simplified 3-class structure (fork, knife, spoon).

The script:
1. Creates a new directory structure for simplified classes
2. Copies images from A/B variants into consolidated classes
3. Maintains train/val/test splits
4. Generates a report of the reorganization

Usage:
    python scripts/reorganize_to_3_classes.py

Author: Claude for Cutlery Classifier MVP
Date: 2024-06-07
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

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
    "fork_a": "fork",
    "fork_b": "fork",
    "knife_a": "knife",
    "knife_b": "knife",
    "spoon_a": "spoon",
    "spoon_b": "spoon",
}
SOURCE_DIR = Path("data/processed")
TARGET_DIR = Path("data/simplified")


def setup_directory_structure() -> None:
    """Create the simplified directory structure."""
    logger.info("Setting up directory structure...")

    # Create main splits
    for split in SPLITS:
        split_dir = TARGET_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)

        # Create class directories
        for simplified_class in set(CLASS_MAPPING.values()):
            class_dir = split_dir / simplified_class
            class_dir.mkdir(exist_ok=True)
            logger.debug(f"Created directory: {class_dir}")


def copy_and_reorganize_images() -> Dict[str, Dict[str, int]]:
    """
    Copy images from original structure to simplified structure.
    Returns statistics about the reorganization.
    """
    stats = {
        split: {new_class: 0 for new_class in set(CLASS_MAPPING.values())}
        for split in SPLITS
    }

    logger.info("Copying and reorganizing images...")

    for split in SPLITS:
        split_dir = SOURCE_DIR / split
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            continue

        for old_class, new_class in CLASS_MAPPING.items():
            old_class_dir = split_dir / old_class
            if not old_class_dir.exists():
                logger.warning(f"Class directory not found: {old_class_dir}")
                continue

            new_class_dir = TARGET_DIR / split / new_class

            # Copy all images
            for img_path in old_class_dir.glob("*"):
                if img_path.is_file():
                    # Create new filename with old class as suffix
                    new_name = f"{img_path.stem}_{old_class}{img_path.suffix}"
                    target_path = new_class_dir / new_name
                    shutil.copy2(img_path, target_path)
                    stats[split][new_class] += 1

    return stats


def generate_report(stats: Dict[str, Dict[str, int]]) -> str:
    """Generate a detailed report of the reorganization."""
    report = ["Dataset Reorganization Report", "=========================", ""]

    # Overall statistics
    total_images = sum(sum(class_stats.values()) for class_stats in stats.values())
    report.append(f"Total images processed: {total_images}")
    report.append("")

    # Per-split statistics
    for split in SPLITS:
        report.append(f"{split.capitalize()} Split:")
        split_total = sum(stats[split].values())
        for class_name, count in stats[split].items():
            report.append(f"  - {class_name}: {count} images")
        report.append(f"  Total: {split_total} images")
        report.append("")

    return "\n".join(report)


def main():
    """Main execution function."""
    logger.info("Starting dataset reorganization...")

    # Create new directory structure
    setup_directory_structure()

    # Copy and reorganize images
    stats = copy_and_reorganize_images()

    # Generate and save report
    report = generate_report(stats)
    report_path = TARGET_DIR / "reorganization_report.txt"
    report_path.write_text(report)

    logger.info(f"Reorganization complete. Report saved to {report_path}")
    logger.info("\n" + report)


if __name__ == "__main__":
    main()
