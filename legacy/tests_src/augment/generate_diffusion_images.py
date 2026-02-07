#!/usr/bin/env python3
"""
Stable Diffusion Image-to-Image Augmentation Script

This script uses the Stable Diffusion v1.5 model to generate augmented
versions of the training images for improved model robustness.

The generated images are saved in data/augmented/{class_name}/.
"""

import os
import stat
import shutil
import logging
from pathlib import Path
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionImg2ImgPipeline
import argparse
import errno

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
FINAL_SIZE = 320  # Size for saved images
IMAGE_SIZE = 512  # Stable Diffusion works best with 512x512
NUM_INFERENCE_STEPS = 50
NUM_AUGMENTED_PER_IMAGE = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID = "runwayml/stable-diffusion-v1-5"


class StableDiffusionAugmenter:
    """Handles Stable Diffusion image-to-image augmentation."""

    def __init__(self):
        self.pipeline = self._load_pipeline()

    def _load_pipeline(self) -> StableDiffusionImg2ImgPipeline:
        """Load the Stable Diffusion pipeline."""
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
        )
        pipeline.to(DEVICE)
        pipeline.enable_attention_slicing()  # Reduce memory usage

        # Disable safety checker for reproducible training data
        pipeline.safety_checker = lambda images, clip_input: (
            images,
            [False] * len(images),
        )
        logger.info(
            "Safety checker disabled for reproducible training data generation (MVP mode)"
        )

        return pipeline

    def load_and_preprocess(self, image_path: str) -> Image.Image:
        """Load and preprocess an image."""
        # Load image and convert to RGB (Stable Diffusion requires RGB)
        image = Image.open(image_path).convert("RGB")

        # Resize maintaining aspect ratio
        aspect_ratio = image.width / image.height
        if aspect_ratio > 1:
            new_width = IMAGE_SIZE
            new_height = int(IMAGE_SIZE / aspect_ratio)
        else:
            new_height = IMAGE_SIZE
            new_width = int(IMAGE_SIZE * aspect_ratio)

        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return image

    def postprocess_image(self, image: Image.Image) -> Image.Image:
        """Resize the generated image to final size."""
        return image.resize((FINAL_SIZE, FINAL_SIZE), Image.Resampling.LANCZOS)

    def generate_variations(
        self,
        image: Image.Image,
        prompt: str,
        num_variations: int = NUM_AUGMENTED_PER_IMAGE,
    ) -> list:
        """Generate variations of the input image using Stable Diffusion."""
        variations = []

        for _ in range(num_variations):
            result = self.pipeline(
                prompt=prompt,
                image=image,
                strength=0.3,
                guidance_scale=7.5,
                num_inference_steps=NUM_INFERENCE_STEPS,
            ).images[0]
            # Resize to final size before adding to variations
            result = self.postprocess_image(result)
            variations.append(result)

        return variations


def handle_remove_readonly(func, path, exc):
    """
    Error handler for shutil.rmtree to handle read-only files and access errors.

    This is particularly important on Windows where files might be marked read-only
    or locked by other processes.
    """
    excvalue = exc[1]
    if func in (os.rmdir, os.remove, os.unlink) and excvalue.errno == errno.EACCES:
        # Ensure parent directory is writable
        parent_dir = Path(path).parent
        try:
            parent_dir.chmod(stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
        except Exception:
            pass

        # Make file writable and try again
        os.chmod(path, stat.S_IWRITE)
        try:
            func(path)
        except Exception as e:
            logger.warning(f"Failed to remove {path}: {str(e)}")
    else:
        logger.warning(f"Failed operation {func.__name__} on {path}: {str(excvalue)}")


def safe_rmtree(path: Path):
    """
    Safely remove a directory tree, handling permission errors gracefully.
    """
    if not path.exists():
        return

    logger.info(f"Clearing output directory: {path}")
    try:
        # First attempt: normal removal
        shutil.rmtree(path)
    except (PermissionError, OSError) as e:
        logger.warning(f"Initial removal failed, retrying with force: {str(e)}")
        try:
            # Second attempt: force removal with error handler
            shutil.rmtree(path, onerror=handle_remove_readonly)
        except Exception as e:
            # If both attempts fail, log but continue
            logger.error(f"Could not fully clear directory {path}: {str(e)}")
            logger.info("Continuing with existing directory...")


def process_directory(
    augmenter: StableDiffusionAugmenter,
    input_dir: Path,
    output_dir: Path,
    class_name: str,
):
    """Process all images in a directory."""
    # Clear existing augmented images with robust error handling
    safe_rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare prompt based on class name - using a more descriptive, safe format
    prompt = f"high quality product photo of a {class_name} on white background, tableware, kitchenware, professional studio lighting"
    logger.info(f"Using prompt: '{prompt}'")

    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    logger.info(f"Found {len(image_files)} images in {input_dir}")

    for image_path in tqdm(image_files, desc=f"Processing {input_dir.name}"):
        # Load and generate variations
        image = augmenter.load_and_preprocess(str(image_path))
        variations = augmenter.generate_variations(image, prompt)

        # Save variations
        for i, variation in enumerate(variations):
            output_path = output_dir / f"{image_path.stem}_aug_{i}.jpg"
            variation.save(output_path, "JPEG", quality=95)


def main():
    parser = argparse.ArgumentParser(
        description="Generate augmented images using Stable Diffusion"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/processed/train",
        help="Directory containing training images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/augmented",
        help="Directory to save augmented images",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=str,
        help="Specific classes to process (e.g. --classes fork knife). If not provided, all classes will be processed.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Loading Stable Diffusion model: {MODEL_ID}")
    augmenter = StableDiffusionAugmenter()

    # Get available classes
    available_classes = [d.name for d in input_dir.iterdir() if d.is_dir()]

    # Determine which classes to process
    if args.classes:
        # Validate requested classes exist
        invalid_classes = set(args.classes) - set(available_classes)
        if invalid_classes:
            raise ValueError(
                f"Invalid class(es): {', '.join(invalid_classes)}. "
                f"Available classes: {', '.join(available_classes)}"
            )
        classes_to_process = args.classes
        logger.info(f"Processing specified classes: {', '.join(classes_to_process)}")
    else:
        classes_to_process = available_classes
        logger.info(f"Processing all classes: {', '.join(classes_to_process)}")

    # Process each requested class directory
    for class_name in classes_to_process:
        class_dir = input_dir / class_name
        output_class_dir = output_dir / class_name
        process_directory(augmenter, class_dir, output_class_dir, class_name)

    logger.info("Augmentation complete! ✨")
    logger.info(f"Augmented images saved in: {output_dir}")


if __name__ == "__main__":
    main()
