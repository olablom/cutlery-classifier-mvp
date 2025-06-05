#!/usr/bin/env python3
"""
Utility script to count augmented images per class.

This script walks through the data/augmented directory and prints
a summary of how many images are in each class folder.
"""

import os
from pathlib import Path
from collections import defaultdict
import argparse


def count_images_in_dir(directory: Path) -> int:
    """Count number of image files (jpg, png) in a directory."""
    return len(list(directory.glob("*.jpg"))) + len(list(directory.glob("*.png")))


def print_class_counts(augmented_dir: Path):
    """Print nicely formatted counts of images per class."""
    if not augmented_dir.exists():
        print(f"\nError: Directory not found: {augmented_dir}")
        return

    # Get counts for each class
    class_counts = {}
    for class_dir in sorted(augmented_dir.iterdir()):
        if class_dir.is_dir():
            class_counts[class_dir.name] = count_images_in_dir(class_dir)

    if not class_counts:
        print(f"\nNo class directories found in: {augmented_dir}")
        return

    # Calculate padding for nice formatting
    max_class_len = max(len(class_name) for class_name in class_counts.keys())
    max_count_len = max(len(str(count)) for count in class_counts.values())

    # Print header
    print("\nAugmented Image Counts")
    print("=" * (max_class_len + max_count_len + 7))  # +7 for spacing and │ chars

    # Print counts
    total_images = 0
    for class_name, count in class_counts.items():
        print(f"{class_name:<{max_class_len}} │ {count:>{max_count_len}} images")
        total_images += count

    # Print footer with total
    print("-" * (max_class_len + max_count_len + 7))
    print(f"{'Total':<{max_class_len}} │ {total_images:>{max_count_len}} images")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Count augmented images per class in data/augmented directory."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="data/augmented",
        help="Directory containing augmented images (default: data/augmented)",
    )

    args = parser.parse_args()
    print_class_counts(Path(args.dir))


if __name__ == "__main__":
    main()
