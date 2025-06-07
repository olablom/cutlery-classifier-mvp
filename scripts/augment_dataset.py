#!/usr/bin/env python3
"""
Augment dataset using diffusion models.

This script:
1. Takes processed images from input directory
2. Generates augmented versions using diffusion models
3. Saves augmented images to output directory
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.augment.generate_diffusion_images import main as diffusion_main

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main function to augment dataset using diffusion models."""
    parser = argparse.ArgumentParser(
        description="Augment dataset using diffusion-based image generation."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Input directory containing processed images",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for augmented images",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["fork_a", "fork_b", "knife_a", "knife_b", "spoon_a", "spoon_b"],
        help="Classes to augment (default: fork_a fork_b knife_a knife_b spoon_a spoon_b)",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=10,
        help="Number of augmented images to generate per original (default: 10)",
    )
    args = parser.parse_args()

    logger.info("Starting dataset augmentation...")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Classes to augment: {args.classes}")
    logger.info(f"Images per original: {args.num_images}")

    try:
        diffusion_main(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            classes=args.classes,
            num_images=args.num_images,
        )
        logger.info("✅ Dataset augmentation completed successfully!")
    except Exception as e:
        logger.error(f"❌ Error during augmentation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
