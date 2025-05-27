"""
Image Transforms for Cutlery Classifier MVP

This module provides standardized image preprocessing and augmentation transforms
for the hierarchical cutlery classification system.

Features:
- Grayscale conversion for embedded optimization
- Standardized resizing to 320x320
- ImageNet normalization adapted for grayscale
- Data augmentation for training
- Consistent preprocessing for inference
"""

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class GrayscaleNormalize:
    """
    Custom normalization for grayscale images using ImageNet statistics.

    Since we convert RGB pretrained models to grayscale by averaging weights,
    we use the average of ImageNet RGB means and stds for normalization.
    """

    def __init__(self):
        # ImageNet RGB: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        # Average for grayscale: mean=0.449, std=0.226
        self.mean = 0.449
        self.std = 0.226

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize grayscale tensor."""
        return (tensor - self.mean) / self.std


def get_base_transforms(image_size: Tuple[int, int] = (320, 320)) -> transforms.Compose:
    """
    Get base transforms for inference and validation.

    Args:
        image_size: Target image size (height, width)

    Returns:
        Composed transforms for basic preprocessing
    """
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            GrayscaleNormalize(),
        ]
    )


def get_training_transforms(
    image_size: Tuple[int, int] = (320, 320),
    augmentation_config: Optional[Dict[str, Any]] = None,
) -> transforms.Compose:
    """
    Get transforms for training with data augmentation.

    Args:
        image_size: Target image size (height, width)
        augmentation_config: Dictionary with augmentation parameters

    Returns:
        Composed transforms for training
    """
    if augmentation_config is None:
        augmentation_config = {
            "horizontal_flip": 0.5,
            "rotation_degrees": 15,
            "color_jitter": {
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.2,
                "hue": 0.1,
            },
            "gaussian_blur": 0.1,
            "random_crop_scale": (0.8, 1.0),
            "random_crop_ratio": (0.9, 1.1),
        }

    transform_list = []

    # Initial resize (slightly larger for random crop)
    initial_size = (int(image_size[0] * 1.1), int(image_size[1] * 1.1))
    transform_list.append(transforms.Resize(initial_size))

    # Color augmentation (before grayscale conversion)
    if "color_jitter" in augmentation_config:
        cj_config = augmentation_config["color_jitter"]
        transform_list.append(
            transforms.ColorJitter(
                brightness=cj_config.get("brightness", 0.2),
                contrast=cj_config.get("contrast", 0.2),
                saturation=cj_config.get("saturation", 0.2),
                hue=cj_config.get("hue", 0.1),
            )
        )

    # Random crop to target size
    if "random_crop_scale" in augmentation_config:
        transform_list.append(
            transforms.RandomResizedCrop(
                size=image_size,
                scale=augmentation_config["random_crop_scale"],
                ratio=augmentation_config["random_crop_ratio"],
            )
        )
    else:
        transform_list.append(transforms.Resize(image_size))

    # Horizontal flip
    if "horizontal_flip" in augmentation_config:
        transform_list.append(
            transforms.RandomHorizontalFlip(p=augmentation_config["horizontal_flip"])
        )

    # Rotation
    if "rotation_degrees" in augmentation_config:
        transform_list.append(
            transforms.RandomRotation(
                degrees=augmentation_config["rotation_degrees"],
                fill=255,  # White fill for cutlery on white background
            )
        )

    # Convert to grayscale
    transform_list.append(transforms.Grayscale(num_output_channels=1))

    # Convert to tensor
    transform_list.append(transforms.ToTensor())

    # Gaussian blur (after tensor conversion)
    if "gaussian_blur" in augmentation_config:
        transform_list.append(
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],
                p=augmentation_config["gaussian_blur"],
            )
        )

    # Normalization
    transform_list.append(GrayscaleNormalize())

    return transforms.Compose(transform_list)


def get_validation_transforms(
    image_size: Tuple[int, int] = (320, 320),
) -> transforms.Compose:
    """
    Get transforms for validation (same as base transforms).

    Args:
        image_size: Target image size (height, width)

    Returns:
        Composed transforms for validation
    """
    return get_base_transforms(image_size)


def get_test_time_augmentation_transforms(
    image_size: Tuple[int, int] = (320, 320), num_augmentations: int = 5
) -> List[transforms.Compose]:
    """
    Get multiple transforms for test-time augmentation.

    Args:
        image_size: Target image size (height, width)
        num_augmentations: Number of different augmentations

    Returns:
        List of transform compositions for TTA
    """
    tta_transforms = []

    # Base transform (no augmentation)
    tta_transforms.append(get_base_transforms(image_size))

    # Horizontal flip
    tta_transforms.append(
        transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.Grayscale(num_output_channels=1),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                GrayscaleNormalize(),
            ]
        )
    )

    # Small rotations
    for angle in [-10, 10]:
        tta_transforms.append(
            transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.RandomRotation(degrees=(angle, angle), fill=255),
                    transforms.ToTensor(),
                    GrayscaleNormalize(),
                ]
            )
        )

    # Slight scale variations
    tta_transforms.append(
        transforms.Compose(
            [
                transforms.Resize(
                    (int(image_size[0] * 1.05), int(image_size[1] * 1.05))
                ),
                transforms.CenterCrop(image_size),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                GrayscaleNormalize(),
            ]
        )
    )

    return tta_transforms[:num_augmentations]


def create_transform_from_config(
    config: Dict[str, Any], mode: str = "train"
) -> transforms.Compose:
    """
    Create transforms based on configuration and mode.

    Args:
        config: Configuration dictionary
        mode: 'train', 'val', or 'test'

    Returns:
        Appropriate transform composition
    """
    image_size = tuple(config.get("image_size", [320, 320]))

    if mode == "train":
        augmentation_config = config.get("augmentation", {})
        return get_training_transforms(image_size, augmentation_config)
    elif mode in ["val", "validation"]:
        return get_validation_transforms(image_size)
    elif mode == "test":
        return get_base_transforms(image_size)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'train', 'val', or 'test'")


def denormalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize a tensor for visualization.

    Args:
        tensor: Normalized tensor

    Returns:
        Denormalized tensor in [0, 1] range
    """
    normalizer = GrayscaleNormalize()
    return tensor * normalizer.std + normalizer.mean


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a tensor to PIL Image for visualization.

    Args:
        tensor: Tensor of shape (1, H, W) or (H, W)

    Returns:
        PIL Image
    """
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)

    # Denormalize if needed (check if values are roughly in normalized range)
    if tensor.min() < -1 or tensor.max() > 2:
        pass  # Already denormalized
    else:
        tensor = denormalize_tensor(tensor)

    # Clamp to [0, 1] and convert to [0, 255]
    tensor = torch.clamp(tensor, 0, 1)
    array = (tensor * 255).byte().numpy()

    return Image.fromarray(array, mode="L")


def test_transforms():
    """Test function to verify transforms work correctly."""

    print("Testing image transforms...")

    # Create a dummy image
    dummy_image = Image.new("RGB", (400, 300), color="white")

    # Test base transforms
    base_transform = get_base_transforms()
    base_output = base_transform(dummy_image)
    print(f"Base transform output shape: {base_output.shape}")
    print(
        f"Base transform value range: [{base_output.min():.3f}, {base_output.max():.3f}]"
    )

    # Test training transforms
    train_transform = get_training_transforms()
    train_output = train_transform(dummy_image)
    print(f"Training transform output shape: {train_output.shape}")
    print(
        f"Training transform value range: [{train_output.min():.3f}, {train_output.max():.3f}]"
    )

    # Test config-based creation
    config = {
        "image_size": [320, 320],
        "augmentation": {
            "horizontal_flip": 0.5,
            "rotation_degrees": 10,
            "color_jitter": {"brightness": 0.1, "contrast": 0.1},
        },
    }

    config_transform = create_transform_from_config(config, mode="train")
    config_output = config_transform(dummy_image)
    print(f"Config-based transform output shape: {config_output.shape}")

    # Test TTA transforms
    tta_transforms = get_test_time_augmentation_transforms(num_augmentations=3)
    print(f"Created {len(tta_transforms)} TTA transforms")

    # Test denormalization and PIL conversion
    pil_image = tensor_to_pil(base_output)
    print(f"PIL conversion successful: {pil_image.size}, mode: {pil_image.mode}")

    print("✅ Transform tests passed!")


if __name__ == "__main__":
    # Run test when script is executed directly
    logging.basicConfig(level=logging.INFO)
    test_transforms()
