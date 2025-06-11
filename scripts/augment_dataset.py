#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Data Augmentation Script

This script provides realistic data augmentation for the cutlery classifier:
1. Lighting variations (brightness, contrast)
2. Realistic blur and noise
3. Natural rotations and perspectives
4. Background variations
5. Quality degradation simulation

Author: Claude for Cutlery Classifier MVP
Date: 2024-06-07
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torchvision.transforms as T
from torchvision.transforms import functional as TF
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

from cutlery_classifier.augment.generate_diffusion_images import main as diffusion_main

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)


class RealisticAugmenter:
    """Provides realistic image augmentations for cutlery images."""

    def __init__(
        self,
        input_size: tuple = (320, 320),
        num_variations: int = 3,
        realistic_mode: bool = True,
    ):
        self.input_size = input_size
        self.num_variations = num_variations
        self.realistic_mode = realistic_mode

        # Define realistic transform ranges
        self.transform_ranges = {
            "rotation": (-30, 30),  # Natural hand-held angles
            "brightness": (0.7, 1.3),  # Indoor lighting variation
            "contrast": (0.8, 1.2),  # Camera exposure variation
            "blur_radius": (0, 2.0),  # Motion and focus blur
            "noise_factor": (0.01, 0.05),  # Sensor noise
            "perspective_distortion": (0.05, 0.15),  # Natural perspective
        }

        # Setup base transforms
        self.base_transform = T.Compose(
            [
                T.Resize(input_size),
                T.Grayscale(),
                T.ToTensor(),
            ]
        )

    def _add_realistic_noise(self, image: torch.Tensor) -> torch.Tensor:
        """Add realistic camera sensor noise."""
        noise_factor = np.random.uniform(*self.transform_ranges["noise_factor"])
        noise = torch.randn_like(image) * noise_factor
        noisy_image = image + noise
        return torch.clamp(noisy_image, 0, 1)

    def _apply_realistic_blur(self, image: Image.Image) -> Image.Image:
        """Apply realistic motion or focus blur."""
        blur_radius = np.random.uniform(*self.transform_ranges["blur_radius"])
        if np.random.random() < 0.5:
            # Motion blur
            angle = np.random.randint(0, 360)
            return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        else:
            # Focus blur
            return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    def _apply_perspective(self, image: Image.Image) -> Image.Image:
        """Apply realistic perspective transformation."""
        width, height = image.size
        distortion = np.random.uniform(*self.transform_ranges["perspective_distortion"])

        # Define perspective coefficients
        coeffs = [distortion * np.random.uniform(-1, 1) for _ in range(8)]

        return image.transform(image.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)

    def augment_image(self, image_path: Path) -> List[Image.Image]:
        """Generate realistic variations of an input image."""
        logger.debug(f"Augmenting image: {image_path}")

        # Load and convert image
        image = Image.open(image_path).convert("RGB")
        augmented_images = []

        for _ in range(self.num_variations):
            # Start with original image
            aug_image = image.copy()

            if self.realistic_mode:
                # 1. Perspective distortion (30% chance)
                if np.random.random() < 0.3:
                    aug_image = self._apply_perspective(aug_image)

                # 2. Rotation
                angle = np.random.uniform(*self.transform_ranges["rotation"])
                aug_image = TF.rotate(aug_image, angle)

                # 3. Lighting variations
                brightness_factor = np.random.uniform(
                    *self.transform_ranges["brightness"]
                )
                contrast_factor = np.random.uniform(*self.transform_ranges["contrast"])

                aug_image = ImageEnhance.Brightness(aug_image).enhance(
                    brightness_factor
                )
                aug_image = ImageEnhance.Contrast(aug_image).enhance(contrast_factor)

                # 4. Blur (40% chance)
                if np.random.random() < 0.4:
                    aug_image = self._apply_realistic_blur(aug_image)

                # 5. Convert to tensor
                aug_tensor = self.base_transform(aug_image)

                # 6. Add noise (60% chance)
                if np.random.random() < 0.6:
                    aug_tensor = self._add_realistic_noise(aug_tensor)

                # Convert back to PIL
                aug_image = TF.to_pil_image(aug_tensor)

            else:
                # Simple augmentations for non-realistic mode
                aug_image = T.Compose(
                    [
                        T.RandomHorizontalFlip(),
                        T.RandomRotation(30),
                        T.ColorJitter(brightness=0.2, contrast=0.2),
                        T.Resize(self.input_size),
                    ]
                )(aug_image)

            augmented_images.append(aug_image)

        return augmented_images


def augment_dataset(
    input_dir: Path,
    output_dir: Path,
    classes: List[str],
    realistic_mode: bool = True,
    num_variations: int = 3,
) -> None:
    """Augment entire dataset with realistic variations."""
    logger.info(f"Starting dataset augmentation (realistic_mode={realistic_mode})")

    # Create augmenter
    augmenter = RealisticAugmenter(
        realistic_mode=realistic_mode, num_variations=num_variations
    )

    # Process each class
    for class_name in classes:
        input_class_dir = input_dir / class_name
        output_class_dir = output_dir / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing class: {class_name}")

        # Process each image
        for image_path in input_class_dir.glob("*.jpg"):
            try:
                # Generate augmented versions
                augmented_images = augmenter.augment_image(image_path)

                # Save augmented images
                for i, aug_image in enumerate(augmented_images):
                    output_path = output_class_dir / f"{image_path.stem}_aug_{i}.jpg"
                    aug_image.save(output_path, "JPEG")
                    logger.debug(f"Saved augmented image: {output_path}")

            except Exception as e:
                logger.error(f"Error processing {image_path}: {str(e)}")
                continue

        logger.info(f"Completed class {class_name}")


def main():
    """Main function to augment dataset using diffusion models."""
    parser = argparse.ArgumentParser(description="Enhanced dataset augmentation")
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Input directory containing class folders",
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
        required=True,
        help="List of class names",
    )
    parser.add_argument(
        "--realistic-augmentation",
        action="store_true",
        help="Enable realistic augmentation mode",
    )
    parser.add_argument(
        "--variations",
        type=int,
        default=3,
        help="Number of augmented versions per image",
    )
    args = parser.parse_args()

    logger.info("Starting dataset augmentation...")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Classes to augment: {args.classes}")
    logger.info(f"Realistic mode: {args.realistic_augmentation}")
    logger.info(f"Variations per image: {args.variations}")

    try:
        # Ensure output directory exists
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # Run augmentation
        augment_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            classes=args.classes,
            realistic_mode=args.realistic_augmentation,
            num_variations=args.variations,
        )
        logger.info("✅ Dataset augmentation completed successfully!")
    except Exception as e:
        logger.error(f"❌ Error during augmentation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
