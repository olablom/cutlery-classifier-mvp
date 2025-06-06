#!/usr/bin/env python3
"""
Test script for evaluating cutlery classifier performance.

This script runs inference on a test dataset and reports per-class and overall accuracy,
with optional Grad-CAM visualization for misclassified images.
"""

import os
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.models.factory import create_model


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_model(model_path: str, device: torch.device) -> nn.Module:
    """
    Load the trained ResNet18 model.

    Args:
        model_path: path to model checkpoint
        device: torch device to load model on

    Returns:
        loaded and configured model
    """
    model_path = Path(model_path)
    abs_model_path = model_path.resolve()
    logger.info(f"Loading model from: {abs_model_path}")

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

    # Create model using factory with same config as training
    model = create_model(
        model_config="resnet18",
        num_classes=num_classes,
        pretrained=True,
        grayscale=True,
        freeze_backbone=False,
    )

    # Load trained weights from checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded successfully with {num_classes} output classes")
    return model, class_names


def create_data_loader(test_dir: str) -> Tuple[DataLoader, List[str]]:
    """
    Create test data loader and get class names.

    Args:
        test_dir: path to test dataset directory

    Returns:
        data loader and list of class names
    """
    transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.449], std=[0.226]),
        ]
    )

    dataset = ImageFolder(test_dir, transform=transform)
    class_names = dataset.classes
    logger.info(f"Found classes: {class_names}")

    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )

    return loader, class_names


def generate_grad_cam(
    model: nn.Module,
    image_path: str,
    true_label: str,
    pred_label: str,
    class_names: List[str],
    device: torch.device,
) -> None:
    """
    Generate and save Grad-CAM visualization for misclassified image.

    Args:
        model: trained model
        image_path: path to original image
        true_label: true class name
        pred_label: predicted class name
        class_names: list of class names
        device: torch device
    """
    # Prepare image
    transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.449], std=[0.226]),
        ]
    )

    image = Image.open(image_path)
    input_tensor = transform(image).unsqueeze(0).to(device)
    input_tensor.requires_grad_()

    # Configure GradCAM
    target_layer = model.layer4[-1]
    model.eval()
    model.requires_grad_(True)
    cam = GradCAM(model=model, target_layers=[target_layer])

    # Set target class for GradCAM
    target_class_idx = class_names.index(pred_label)
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
    output_dir = Path("results/misclassified_grad_cam")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_name = Path(image_path).name
    output_path = output_dir / f"{true_label}_pred-{pred_label}_{image_name}"
    plt.imsave(str(output_path), visualization)


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    class_names: List[str],
    device: torch.device,
    save_misclassified: bool = False,
) -> Tuple[Dict[str, int], Dict[str, int], List[Tuple[str, str, str]]]:
    """
    Evaluate model on test dataset.

    Args:
        model: trained model
        data_loader: test data loader
        class_names: list of class names
        device: torch device
        save_misclassified: whether to save Grad-CAM for misclassified images

    Returns:
        correct counts per class, total counts per class, and misclassified examples
    """
    correct_per_class = defaultdict(int)
    total_per_class = defaultdict(int)
    misclassified = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Get image path and true/predicted classes
            image_path = data_loader.dataset.samples[len(misclassified)][0]
            true_class = class_names[targets.item()]
            pred_class = class_names[predicted.item()]

            total_per_class[true_class] += 1

            if predicted == targets:
                correct_per_class[true_class] += 1
            else:
                misclassified.append((image_path, true_class, pred_class))
                if save_misclassified:
                    generate_grad_cam(
                        model, image_path, true_class, pred_class, class_names, device
                    )

    return correct_per_class, total_per_class, misclassified


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(
        description="Evaluate cutlery classifier on test dataset"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        required=True,
        help="Device to run inference on",
    )
    parser.add_argument(
        "--test_dir", type=str, required=True, help="Path to test dataset directory"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/checkpoints/type_detector_best.pth",
        help="Path to trained model (.pth). Defaults to models/checkpoints/type_detector_best.pth",
    )
    parser.add_argument(
        "--save-misclassified",
        action="store_true",
        help="Generate Grad-CAM visualizations for misclassified images",
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
        # Load model and create data loader
        model, class_names = load_model(args.model, device)
        data_loader, class_names = create_data_loader(args.test_dir)

        logger.info(f"Running inference on {len(data_loader)} test images...")

        # Evaluate model
        correct_per_class, total_per_class, misclassified = evaluate_model(
            model, data_loader, class_names, device, args.save_misclassified
        )

        # Print results
        print("\nResults:")
        total_correct = 0
        total_images = 0

        for class_name in class_names:
            correct = correct_per_class[class_name]
            total = total_per_class[class_name]
            accuracy = (correct / total * 100) if total > 0 else 0
            print(f"Class '{class_name}': {correct}/{total} correct ({accuracy:.2f}%)")

            total_correct += correct
            total_images += total

        overall_accuracy = total_correct / total_images * 100
        print(
            f"\nOverall accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_images})"
        )

        if misclassified:
            print("\nMisclassified images:")
            for image_path, true_label, pred_label in misclassified:
                rel_path = Path(image_path).relative_to(args.test_dir)
                print(f"{rel_path} → predicted as {pred_label}")

            if args.save_misclassified:
                print(
                    "\nGrad-CAM saved for misclassified images in: results/misclassified_grad_cam/"
                )
        else:
            print("\nNo misclassified images!")

    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        raise


if __name__ == "__main__":
    main()
