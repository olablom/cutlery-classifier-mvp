#!/usr/bin/env python3
"""
Organize images into manufacturer-specific subfolders.

This script moves images from raw/{manufacturer_a,manufacturer_b}/{fork,knife,spoon}/
to processed/{test,val}/{fork_a,fork_b,knife_a,knife_b,spoon_a,spoon_b}/
according to predefined mappings.
"""

import os
import shutil
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Define the mapping of source directories to target classes
MANUFACTURER_MAPPING = {
    "manufacturer_a": {"fork": "fork_a", "knife": "knife_a", "spoon": "spoon_a"},
    "manufacturer_b": {"fork": "fork_b", "knife": "knife_b", "spoon": "spoon_b"},
}


def setup_directories():
    """Create necessary directory structure."""
    for split in ["test", "val"]:
        for class_name in [
            "fork_a",
            "fork_b",
            "knife_a",
            "knife_b",
            "spoon_a",
            "spoon_b",
        ]:
            (PROCESSED_DIR / split / class_name).mkdir(parents=True, exist_ok=True)


def process_images():
    """Process and organize images according to manufacturer mapping."""
    for manufacturer, type_mapping in MANUFACTURER_MAPPING.items():
        manufacturer_dir = RAW_DIR / manufacturer
        if not manufacturer_dir.exists():
            logger.warning(f"Manufacturer directory not found: {manufacturer}")
            continue

        for src_type, target_class in type_mapping.items():
            src_dir = manufacturer_dir / src_type
            if not src_dir.exists():
                logger.warning(f"Source directory not found: {src_dir}")
                continue

            # Process images
            images = (
                list(src_dir.glob("*.jpg"))
                + list(src_dir.glob("*.jpeg"))
                + list(src_dir.glob("*.png"))
            )
            if not images:
                logger.warning(f"No images found in {src_dir}")
                continue

            # Copy images to test and val directories
            for img_path in images:
                # Decide whether to put in test or val (you can modify this logic)
                target_split = (
                    "test"
                    if len(list((PROCESSED_DIR / "test" / target_class).glob("*.jpg")))
                    < 5
                    else "val"
                )
                target_dir = PROCESSED_DIR / target_split / target_class

                try:
                    shutil.copy2(img_path, target_dir / img_path.name)
                    logger.info(
                        f"Copied {img_path.name} to {target_split}/{target_class}/"
                    )
                except Exception as e:
                    logger.error(f"Error copying {img_path}: {e}")


def verify_organization():
    """Verify the organization of images."""
    logger.info("\nVerifying image organization:")

    for split in ["test", "val"]:
        logger.info(f"\n{split.upper()} Split:")
        split_dir = PROCESSED_DIR / split

        for class_name in [
            "fork_a",
            "fork_b",
            "knife_a",
            "knife_b",
            "spoon_a",
            "spoon_b",
        ]:
            class_dir = split_dir / class_name
            if class_dir.exists():
                images = (
                    list(class_dir.glob("*.jpg"))
                    + list(class_dir.glob("*.jpeg"))
                    + list(class_dir.glob("*.png"))
                )
                logger.info(f"  {class_name}: {len(images)} images")
            else:
                logger.warning(f"  {class_name}: Directory not found")


def main():
    """Main function to organize images."""
    logger.info("Starting image organization...")

    # Create directory structure
    setup_directories()

    # Process and organize images
    process_images()

    # Verify organization
    verify_organization()

    logger.info("\nImage organization completed!")


if __name__ == "__main__":
    main()
