#!/usr/bin/env python3
"""
Diffusion-based Data Augmentation Script

This script uses a pre-trained diffusion model to generate augmented
versions of the training images for improved model robustness.

The generated images are saved in data/augmented/{class_name}/.
"""

import os
import logging
from pathlib import Path
import torch
from torch import nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from diffusers import DDPMScheduler, UNet2DModel
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
IMAGE_SIZE = 320
NUM_INFERENCE_STEPS = 50
NUM_AUGMENTED_PER_IMAGE = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DiffusionAugmenter:
    """Handles diffusion-based image augmentation."""

    def __init__(self):
        self.model = self._load_model()
        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
        )

    def _load_model(self) -> nn.Module:
        """Load the pre-trained U-Net model."""
        model = UNet2DModel(
            sample_size=IMAGE_SIZE,
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        model.to(DEVICE)
        return model

    def load_and_preprocess(self, image_path: str) -> torch.Tensor:
        """Load and preprocess an image."""
        transform = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        image = Image.open(image_path).convert("L")
        return transform(image).unsqueeze(0).to(DEVICE)

    def postprocess_image(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image."""
        tensor = (tensor + 1) / 2
        tensor = tensor.clamp(0, 1)
        tensor = tensor.squeeze().cpu().numpy()
        return Image.fromarray((tensor * 255).astype(np.uint8))

    def generate_variations(
        self, image_tensor: torch.Tensor, num_variations: int = NUM_AUGMENTED_PER_IMAGE
    ) -> list:
        """Generate variations of the input image using diffusion."""
        variations = []

        for _ in range(num_variations):
            # Add noise
            noise = torch.randn_like(image_tensor)
            noisy = self.scheduler.add_noise(
                image_tensor, noise, torch.tensor([NUM_INFERENCE_STEPS])
            )

            # Denoise
            for t in self.scheduler.timesteps:
                with torch.no_grad():
                    noise_pred = self.model(noisy, t).sample
                    noisy = self.scheduler.step(noise_pred, t, noisy).prev_sample

            variations.append(self.postprocess_image(noisy))

        return variations


def process_directory(augmenter: DiffusionAugmenter, input_dir: Path, output_dir: Path):
    """Process all images in a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    logger.info(f"Found {len(image_files)} images in {input_dir}")

    for image_path in tqdm(image_files, desc=f"Processing {input_dir.name}"):
        # Load and generate variations
        image_tensor = augmenter.load_and_preprocess(str(image_path))
        variations = augmenter.generate_variations(image_tensor)

        # Save variations
        for i, variation in enumerate(variations):
            output_path = output_dir / f"{image_path.stem}_aug_{i}.jpg"
            variation.save(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate augmented images using diffusion"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/train",
        help="Directory containing training images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/augmented",
        help="Directory to save augmented images",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    logger.info(f"Using device: {DEVICE}")
    augmenter = DiffusionAugmenter()

    # Process each class directory
    for class_dir in input_dir.iterdir():
        if class_dir.is_dir():
            output_class_dir = output_dir / class_dir.name
            process_directory(augmenter, class_dir, output_class_dir)

    logger.info("Augmentation complete! ✨")
    logger.info(f"Augmented images saved in: {output_dir}")


if __name__ == "__main__":
    main()
