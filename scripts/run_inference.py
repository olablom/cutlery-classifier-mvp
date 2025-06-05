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


def load_model(device: torch.device) -> Tuple[nn.Module, List[str]]:
    """
    Load the trained ResNet18 model and class names.

    Args:
        device: torch device to load model on

    Returns:
        model: loaded and configured model
        class_names: list of class names
    """
    model_path = Path("models/checkpoints/best_model.pt")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    # Try to load class names from training data directory
    data_dirs = [Path("data/processed/train"), Path("data/augmented")]

    class_names = None
    for data_dir in data_dirs:
        if data_dir.exists():
            class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
            if class_names:
                break

    if not class_names:
        raise RuntimeError(
            "Could not find class names in data/processed/train or data/augmented"
        )

    num_classes = len(class_names)
    logger.info(f"Found {num_classes} classes: {', '.join(class_names)}")

    # Create and configure model
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    logger.info("Model loaded successfully")
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
        choices=["cuda", "cpu"],
        required=True,
        help="Device to run inference on",
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--grad-cam", action="store_true", help="Generate Grad-CAM visualization"
    )

    args = parser.parse_args()

    # Set up device
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Using CPU instead.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    try:
        # Load model and preprocess image
        model, class_names = load_model(device)
        image_tensor = preprocess_image(args.image)

        # Get predictions
        predictions = get_predictions(model, image_tensor, class_names, device)
        predicted_class = max(predictions.items(), key=lambda x: x[1])[0]

        # Log results
        logger.info("\nPrediction Results:")
        logger.info(f"Predicted class: {predicted_class}")
        logger.info("Class probabilities:")
        for class_name, probability in sorted(
            predictions.items(), key=lambda x: x[1], reverse=True
        ):
            logger.info(f"{class_name}: {probability:.4f}")

        # Generate Grad-CAM if requested
        if args.grad_cam:
            generate_grad_cam(model, args.image, image_tensor, predicted_class, device)

    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise


if __name__ == "__main__":
    main()
