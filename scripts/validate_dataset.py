#!/usr/bin/env python3
"""
Simple Dataset Validation Script

Counts images in each directory and shows collection progress.
"""

import os
from pathlib import Path


def validate_dataset():
    """Count images and show progress."""

    project_root = Path(__file__).parent.parent
    raw_dir = project_root / "data" / "raw"

    types = ["fork", "knife", "spoon"]
    manufacturers = ["ikea", "obh"]
    target_per_class = 40
    min_per_class = 20

    print("🔍 Dataset Validation")
    print("=" * 50)

    total_images = 0
    ready_classes = 0
    total_classes = len(types) * len(manufacturers)

    for cutlery_type in types:
        print(f"\n{cutlery_type.title()}:")
        type_total = 0

        for manufacturer in manufacturers:
            type_dir = raw_dir / cutlery_type / manufacturer

            if type_dir.exists():
                # Count image files
                image_files = (
                    list(type_dir.glob("*.jpg"))
                    + list(type_dir.glob("*.jpeg"))
                    + list(type_dir.glob("*.png"))
                )
                count = len(image_files)
            else:
                count = 0

            type_total += count
            total_images += count

            # Status indicator
            if count >= min_per_class:
                status = "✅"
                ready_classes += 1
            else:
                status = "⚠️"

            print(f"  {manufacturer.upper()}: {count:2d}/{target_per_class} {status}")

        print(f"  Subtotal: {type_total}")

    print(f"\n📊 Summary:")
    print(f"Total images: {total_images}")
    print(f"Ready classes: {ready_classes}/{total_classes}")

    if ready_classes >= total_classes:
        print("✅ Dataset ready for training!")
    elif ready_classes >= total_classes * 0.5:
        print("⚠️  Dataset partially ready - can start training")
    else:
        print("❌ Need more images before training")


if __name__ == "__main__":
    validate_dataset()
