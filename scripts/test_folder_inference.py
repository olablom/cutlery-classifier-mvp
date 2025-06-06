#!/usr/bin/env python3
"""
Inference script for running cutlery classification on a folder of images.

This script:
1. Loads the trained ResNet18 model
2. Runs inference on all images in the input directory
3. Generates Grad-CAM visualizations
4. Prints predictions to console
"""

import os
import logging
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_model(device: torch.device) -> Tuple[nn.Module, List[str]]:
    """
    Load the trained ResNet18 model.

    Args:
        device: torch device to load model on

    Returns:
        model: loaded model
        class_names: list of class names
    """
    model_path = Path("models/checkpoints/best_model.pt")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    # Create and configure model
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Load checkpoint to get number of classes
    checkpoint = torch.load(model_path, map_location=device)
    num_classes = checkpoint["model_state_dict"]["fc.weight"].size(0)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Load weights and prepare model
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Define class names
    class_names = ["fork", "knife", "spoon"]

    logger.info("Model loaded successfully")
    return model, class_names


def get_transform() -> transforms.Compose:
    """Get image preprocessing transforms."""
    return transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.449], std=[0.226]),
        ]
    )


def generate_grad_cam(
    model: nn.Module,
    image_path: str,
    pred_label: str,
    device: torch.device,
) -> None:
    """
    Generate and save Grad-CAM visualization.

    Args:
        model: trained model
        image_path: path to input image
        pred_label: predicted class name
        device: torch device
    """
    transform = get_transform()

    image = Image.open(image_path)
    input_tensor = transform(image).unsqueeze(0).to(device)
    input_tensor.requires_grad_()

    # Configure GradCAM
    target_layer = model.layer4[-1]
    model.eval()
    model.requires_grad_(True)
    cam = GradCAM(model=model, target_layers=[target_layer])

    # Set target class for GradCAM
    target_class_idx = ["fork", "knife", "spoon"].index(pred_label)
    targets = [ClassifierOutputTarget(target_class_idx)]

    # Generate CAM
    with torch.enable_grad():
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # Load and preprocess original image for visualization
    rgb_img = Image.open(image_path).convert("RGB")
    rgb_img = rgb_img.resize((320, 320))
    rgb_img = np.array(rgb_img) / 255.0

    # Create visualization
    visualization = show_cam_on_image(rgb_img, grayscale_cam[0])

    # Save result
    output_dir = Path("results/folder_inference_grad_cam")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_name = Path(image_path).name
    output_path = output_dir / f"pred-{pred_label}_{image_name}"
    plt.imsave(str(output_path), visualization)


def process_image(
    model: nn.Module,
    image_path: str,
    transform: transforms.Compose,
    class_names: List[str],
    device: torch.device,
) -> str:
    """
    Process a single image and return prediction.

    Args:
        model: trained model
        image_path: path to image
        transform: preprocessing transforms
        class_names: list of class names
        device: torch device

    Returns:
        predicted class name
    """
    # Load and preprocess image
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Get prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        predicted_idx = torch.argmax(probabilities).item()
        predicted_class = class_names[predicted_idx]

    return predicted_class


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Run inference on a folder of images")
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        required=True,
        help="Device to run inference on",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to input directory containing images",
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
        # Load model
        model, class_names = load_model(device)
        transform = get_transform()

        # Get all images
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        image_paths = []
        for ext in [".jpg", ".jpeg", ".png"]:
            image_paths.extend(input_dir.glob(f"*{ext}"))
            image_paths.extend(input_dir.glob(f"*{ext.upper()}"))

        if not image_paths:
            logger.error(f"No images found in {input_dir}")
            return

        logger.info(f"Found {len(image_paths)} images")

        # Process each image
        for image_path in image_paths:
            try:
                predicted_class = process_image(
                    model, str(image_path), transform, class_names, device
                )
                print(f"{image_path.name} → {predicted_class}")

                # Generate Grad-CAM
                generate_grad_cam(model, str(image_path), predicted_class, device)

            except Exception as e:
                logger.error(f"Error processing {image_path.name}: {str(e)}")
                continue

        logger.info("Processing complete")
        logger.info(
            f"Grad-CAM visualizations saved in: results/folder_inference_grad_cam/"
        )

    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise


if __name__ == "__main__":
    main()
