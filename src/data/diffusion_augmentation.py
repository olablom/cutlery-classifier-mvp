import torch
from torch import nn
from diffusers import StableDiffusionImg2ImgPipeline
from pathlib import Path
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class DiffusionAugmenter:
    def __init__(
        self,
        model_id="stabilityai/stable-diffusion-2-1",
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)

    def augment_image(
        self,
        image_path: str,
        num_variations: int = 1,
        strength: float = 0.75,
        guidance_scale: float = 7.5,
        prompt: str = "high quality photo of cutlery on white background",
    ):
        """
        Generate variations of an input image using Stable Diffusion.

        Args:
            image_path: Path to the input image
            num_variations: Number of variations to generate
            strength: How much to transform the image (0-1)
            guidance_scale: How closely to follow the prompt
            prompt: Text prompt to guide the generation

        Returns:
            List of PIL Images
        """
        try:
            # Load and preprocess image
            init_image = Image.open(image_path).convert("RGB")

            # Generate variations
            variations = []
            for _ in range(num_variations):
                result = self.pipeline(
                    prompt=prompt,
                    image=init_image,
                    num_inference_steps=50,
                    strength=strength,
                    guidance_scale=guidance_scale,
                ).images[0]
                variations.append(result)

            return variations

        except Exception as e:
            logger.error(f"Error augmenting image {image_path}: {str(e)}")
            return []

    def augment_directory(
        self, input_dir: str, output_dir: str, variations_per_image: int = 5
    ):
        """
        Augment all images in a directory and save results.

        Args:
            input_dir: Directory containing original images
            output_dir: Directory to save augmented images
            variations_per_image: Number of variations to generate per image
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))

        for img_path in image_files:
            variations = self.augment_image(
                str(img_path), num_variations=variations_per_image
            )

            # Save variations
            for i, var_img in enumerate(variations):
                output_file = output_path / f"{img_path.stem}_var{i}{img_path.suffix}"
                var_img.save(output_file)
                logger.info(f"Saved augmented image: {output_file}")


def augment_dataset(input_root: str, output_root: str, variations_per_image: int = 5):
    """
    Augment an entire dataset maintaining the class structure.

    Args:
        input_root: Root directory of original dataset
        output_root: Root directory for augmented dataset
        variations_per_image: Number of variations per image
    """
    augmenter = DiffusionAugmenter()

    input_path = Path(input_root)
    output_path = Path(output_root)

    # Process each class directory
    for class_dir in input_path.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name

            # Create output directory for this class
            output_class_dir = output_path / class_name
            output_class_dir.mkdir(parents=True, exist_ok=True)

            # Augment images for this class
            augmenter.augment_directory(
                str(class_dir),
                str(output_class_dir),
                variations_per_image=variations_per_image,
            )

            logger.info(f"Completed augmentation for class: {class_name}")
