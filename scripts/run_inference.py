#!/usr/bin/env python3
"""
Unified Inference Script for Cutlery Classifier

This script supports:
1. Regular inference on single images
2. Grad-CAM visualization using pytorch_grad_cam
3. Both CPU and CUDA execution

Usage:
    python run_inference.py --device [cuda|cpu] --image path/to/image.jpg [--grad-cam]
"""

import argparse
import logging
from pathlib import Path
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from datetime import datetime

from src.models.factory import create_model
from src.utils.device import get_device
from src.visualization.grad_cam_utils import generate_grad_cam

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
CLASS_LABELS = ["fork", "knife", "spoon"]
IMAGE_SIZE = 320
CHECKPOINT_PATH = "models/checkpoints/type_detector_best.pth"


def create_model_config():
    """Create default model configuration."""
    return {
        "architecture": "resnet18",
        "num_classes": len(CLASS_LABELS),
        "pretrained": True,
        "grayscale": True,
        "freeze_backbone": False,
    }


def load_model(device: str) -> torch.nn.Module:
    """Load the trained model from checkpoint."""
    # Create model using factory with config
    config = create_model_config()
    model = create_model(config)

    if not Path(CHECKPOINT_PATH).exists():
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


def preprocess_image(image_path: str, device: str) -> tuple:
    """Load and preprocess an image for inference and visualization."""
    # Load image
    image = Image.open(image_path).convert("L")

    # Prepare normalized tensor for inference
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229]),  # ImageNet stats for grayscale
        ]
    )
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Prepare normalized image for Grad-CAM visualization
    image_for_cam = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_for_cam = cv2.resize(image_for_cam, (IMAGE_SIZE, IMAGE_SIZE))
    image_for_cam = image_for_cam / 255.0  # Normalize to [0, 1]

    return image_tensor, image_for_cam


def run_inference(model: torch.nn.Module, image_tensor: torch.Tensor) -> tuple:
    """Run inference and return prediction and probabilities."""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    return CLASS_LABELS[predicted_class], probabilities[0].tolist()


def main():
    parser = argparse.ArgumentParser(description="Run inference on a single image")
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

    # Validate device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    # Load model
    model = load_model(device)
    logger.info("Model loaded successfully")

    # Preprocess image
    image_tensor, image_for_cam = preprocess_image(args.image, device)

    # Run inference
    predicted_class, probabilities = run_inference(model, image_tensor)

    # Print results
    logger.info(f"\nPrediction Results:")
    logger.info(f"Predicted class: {predicted_class}")
    logger.info("\nClass probabilities:")
    for label, prob in zip(CLASS_LABELS, probabilities):
        logger.info(f"{label}: {prob:.4f}")

    # Generate Grad-CAM if requested
    if args.grad_cam:
        logger.info("\nGenerating Grad-CAM visualization...")
        generate_grad_cam(model, image_tensor, image_for_cam, predicted_class, device)


if __name__ == "__main__":
    main()
