#!/usr/bin/env python3
"""
Inference script for cutlery classification model.

This script runs inference on a single image using the trained ResNet18 model,
with optional Grad-CAM visualization support.

Dependencies:
    pytorch-grad-cam>=1.4.0
    matplotlib>=3.3.0
"""

import os
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def count_parameters(model):
    """Count total and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def load_model(model_path: str, device: str) -> Tuple[nn.Module, List[str]]:
    """Load the model.

    Args:
        model_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        tuple: (model, class_names)
    """
    # Load model checkpoint
    checkpoint_path = Path("models/checkpoints/type_detector_best.pth")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    logger.info(f"Loading model from: {checkpoint_path}")
    logger.info("Using 3 simplified classes: fork, knife, spoon")

    # Create model
    logger.info("Creating resnet18 model with 3 classes")
    logger.info("Pretrained: True, Grayscale: True, Freeze: False")

    model = models.resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(model.fc.in_features, 3))

    # Count parameters
    total_params, trainable_params = count_parameters(model)
    logger.info(
        f"Model created: {total_params:,} total params, {trainable_params:,} trainable"
    )

    # Load checkpoint with proper device mapping
    # Always load to CPU first, then move to target device to avoid CUDA issues
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    class_names = ["fork", "knife", "spoon"]
    logger.info(f"Model loaded successfully with {len(class_names)} output classes")
    return model, class_names


def preprocess_image(image_path: str) -> torch.Tensor:
    """
    Preprocess image for model inference.

    Args:
        image_path: path to input image

    Returns:
        preprocessed image tensor
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.449], std=[0.226]),
        ]
    )

    image = Image.open(image_path)
    return transform(image).unsqueeze(0)  # Add batch dimension


def get_predictions(
    model: nn.Module,
    image_tensor: torch.Tensor,
    class_names: List[str],
    device: torch.device,
) -> Dict[str, float]:
    """
    Get model predictions and class probabilities.

    Args:
        model: trained model
        image_tensor: preprocessed input image
        class_names: list of class names
        device: torch device

    Returns:
        dictionary of class probabilities
    """
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    return {
        class_name: prob.item() for class_name, prob in zip(class_names, probabilities)
    }


def generate_grad_cam(
    model: nn.Module,
    image_path: str,
    image_tensor: torch.Tensor,
    predicted_class: str,
    device: torch.device,
) -> None:
    """
    Generate and save Grad-CAM visualization.

    Args:
        model: trained model
        image_path: path to original image
        image_tensor: preprocessed image tensor
        predicted_class: predicted class name
        device: torch device
    """
    # Configure GradCAM
    target_layer = model.layer4[-1]  # Last layer of ResNet
    cam = GradCAM(model=model, target_layers=[target_layer])

    # Generate CAM
    grayscale_cam = cam(input_tensor=image_tensor)

    # Load and preprocess original image for visualization
    rgb_img = Image.open(image_path).convert("RGB")
    rgb_img = rgb_img.resize((320, 320))
    rgb_img = np.array(rgb_img) / 255.0

    # Create visualization
    visualization = show_cam_on_image(rgb_img, grayscale_cam[0])

    # Save result
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results/grad_cam")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{predicted_class}_{timestamp}.jpg"
    plt.imsave(str(output_path), visualization)
    logger.info(f"Grad-CAM visualization saved to: {output_path}")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description="Run inference on an image using trained cutlery classifier"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to run inference on (auto=automatically detect)",
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--model",
        type=str,
        default="models/checkpoints/type_detector_best.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--grad-cam", action="store_true", help="Generate Grad-CAM visualization"
    )

    args = parser.parse_args()

    try:
        # Setup device with automatic detection
        if args.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info("CUDA detected, using GPU acceleration")
            else:
                device = torch.device("cpu")
                logger.info("CUDA not available, falling back to CPU")
        else:
            # Manual device selection with validation
            if args.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                device = torch.device("cpu")
            else:
                device = torch.device(args.device)

        logger.info(f"Using device: {device}")

        # Load model
        model, class_names = load_model(args.model, device)

        # Preprocess image
        image_tensor = preprocess_image(args.image)

        # Get predictions
        predictions = get_predictions(model, image_tensor, class_names, device)

        # Find predicted class
        predicted_class = max(predictions, key=predictions.get)

        # Print results
        logger.info("\nPrediction Results:")
        logger.info(f"Predicted class: {predicted_class}")
        logger.info("Class probabilities:")
        for class_name, prob in predictions.items():
            logger.info(f"{class_name}: {prob:.4f}")

        # Generate Grad-CAM if requested
        if args.grad_cam:
            generate_grad_cam(model, args.image, image_tensor, predicted_class, device)

    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise


if __name__ == "__main__":
    main()
