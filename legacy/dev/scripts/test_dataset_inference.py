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
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import shutil
from cutlery_classifier.evaluation.metrics import plot_confusion_matrix_vg

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
        Tuple of (loaded model, class names)
    """
    model_path = Path(model_path)
    abs_model_path = model_path.resolve()
    logger.info(f"Loading model from: {abs_model_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    # Try to load class names from training data directory
    data_dirs = [Path("data/simplified/train"), Path("data/augmented/train")]

    class_names = None
    for data_dir in data_dirs:
        if data_dir.exists():
            class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
            if class_names:
                break

    if not class_names:
        raise RuntimeError(
            "Could not find class names in data/simplified/train or data/augmented/train"
        )

    num_classes = len(class_names)
    logger.info(f"Found {num_classes} classes: {', '.join(class_names)}")

    # Create model with same architecture as training
    model = models.resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5), nn.Linear(model.fc.in_features, num_classes)
    )

    # Load trained weights from checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Detect checkpoint format
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        logger.info("Detected trainer checkpoint format")
        state_dict = checkpoint["model_state_dict"]
    else:
        logger.info("Detected raw state_dict format")
        state_dict = checkpoint

    # Load state dict and prepare model
    model.load_state_dict(state_dict)
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
    confidence_analysis: bool = False,
    stress_test: bool = False,
) -> Tuple[Dict[str, int], Dict[str, int], List[Tuple[str, str, str]], Optional[Dict]]:
    """
    Evaluate model on test dataset.

    Args:
        model: trained model
        data_loader: test data loader
        class_names: list of class names
        device: torch device
        save_misclassified: whether to save Grad-CAM for misclassified images
        confidence_analysis: whether to perform confidence analysis
        stress_test: whether to perform stress tests

    Returns:
        correct counts per class, total counts per class, misclassified examples, and analysis results
    """
    correct_per_class = defaultdict(int)
    total_per_class = defaultdict(int)
    misclassified = []
    confidences = []
    analysis_results = {}

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)

            # Run inference
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            # Get image path and true/predicted classes
            image_path = data_loader.dataset.samples[len(misclassified)][0]
            true_class = class_names[targets.item()]
            pred_class = class_names[predicted.item()]

            total_per_class[true_class] += 1
            confidences.append(confidence.item())

            if predicted == targets:
                correct_per_class[true_class] += 1
            else:
                misclassified.append((image_path, true_class, pred_class))
                if save_misclassified:
                    generate_grad_cam(
                        model, image_path, true_class, pred_class, class_names, device
                    )

    if confidence_analysis:
        analysis_results["confidence"] = {
            "mean": np.mean(confidences),
            "min": np.min(confidences),
            "max": np.max(confidences),
            "std": np.std(confidences),
        }

        # Plot confidence distribution
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=20)
        plt.title("Model Confidence Distribution")
        plt.xlabel("Confidence")
        plt.ylabel("Count")
        plt.savefig("results/confidence_distribution.png")
        plt.close()

    if stress_test:
        # Perform stress tests with various perturbations
        noise_results = stress_test_noise(model, data_loader, device)
        blur_results = stress_test_blur(model, data_loader, device)
        rotation_results = stress_test_rotation(model, data_loader, device)

        analysis_results["stress_test"] = {
            "noise": noise_results,
            "blur": blur_results,
            "rotation": rotation_results,
        }

    return correct_per_class, total_per_class, misclassified, analysis_results


def stress_test_noise(
    model: nn.Module, data_loader: DataLoader, device: torch.device
) -> float:
    """Run stress test with Gaussian noise."""
    correct = 0
    total = 0
    noise_level = 0.1

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)

            # Add Gaussian noise
            noise = torch.randn_like(images) * noise_level
            noisy_images = images + noise
            noisy_images = torch.clamp(noisy_images, 0, 1)

            outputs = model(noisy_images)
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    return correct / total


def stress_test_blur(
    model: nn.Module, data_loader: DataLoader, device: torch.device
) -> float:
    """Run stress test with Gaussian blur."""
    correct = 0
    total = 0
    kernel_size = 5
    sigma = 2.0

    blur = transforms.GaussianBlur(kernel_size, sigma)

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)

            # Apply blur
            blurred_images = blur(images)

            outputs = model(blurred_images)
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    return correct / total


def stress_test_rotation(
    model: nn.Module, data_loader: DataLoader, device: torch.device
) -> float:
    """Run stress test with random rotations."""
    correct = 0
    total = 0
    max_angle = 30

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)

            # Apply random rotation
            angle = torch.randint(-max_angle, max_angle + 1, (1,)).item()
            rotated_images = transforms.functional.rotate(images, angle)

            outputs = model(rotated_images)
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    return correct / total


def create_output_folder(base_dir="outputs"):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / f"run_{now}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        required=True,
        help="Device to run on",
    )
    parser.add_argument(
        "--test_dir", type=str, required=True, help="Path to test dataset directory"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/checkpoints/type_detector_best.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--save-misclassified",
        action="store_true",
        help="Save Grad-CAM visualizations for misclassified images",
    )
    parser.add_argument(
        "--confidence-analysis", action="store_true", help="Perform confidence analysis"
    )
    parser.add_argument(
        "--stress-test", action="store_true", help="Perform stress tests"
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    # Skapa unik output-folder
    output_dir = create_output_folder()
    logger.info(f"Output folder for this run: {output_dir}")

    # Load model and data
    model, class_names = load_model(args.model, device)
    data_loader, dataset_classes = create_data_loader(args.test_dir)

    # Evaluate model
    correct_per_class, total_per_class, misclassified, analysis_results = (
        evaluate_model(
            model,
            data_loader,
            class_names,
            device,
            args.save_misclassified,
            args.confidence_analysis,
            args.stress_test,
        )
    )

    # Samla alla true/predicted labels för confusion matrix
    y_true = []
    y_pred = []
    for i, (img_path, true_label, pred_label) in enumerate(misclassified):
        y_true.append(class_names.index(true_label))
        y_pred.append(class_names.index(pred_label))
    # Lägg till korrekt klassificerade också
    for class_name in class_names:
        n_correct = correct_per_class[class_name]
        y_true.extend([class_names.index(class_name)] * n_correct)
        y_pred.extend([class_names.index(class_name)] * n_correct)

    # Plotta och spara confusion matrix (VG-version)
    cm_path = output_dir / "confusion_matrix_vg.png"
    plot_confusion_matrix_vg(y_true, y_pred, class_names, str(cm_path))
    logger.info(f"VG confusion matrix saved to: {cm_path}")

    # --- VISUELL UTVÄRDERING & SUMMERING ---
    # 1. Hämta paths till alla testbilder och deras prediktioner
    all_samples = list(data_loader.dataset.samples)
    pred_labels = []
    true_labels = []
    img_paths = []
    idx = 0
    for class_idx, class_name in enumerate(class_names):
        n_total = total_per_class[class_name]
        n_correct = correct_per_class[class_name]
        for i in range(n_total):
            img_path, true_class = all_samples[idx]
            img_paths.append(img_path)
            true_labels.append(class_name)
            if n_correct > 0:
                pred_labels.append(class_name)
                n_correct -= 1
            else:
                # Leta upp felklassificering för denna klass
                for m in misclassified:
                    if m[1] == class_name:
                        pred_labels.append(m[2])
                        break
            idx += 1

    # 2. Skapa listor för rätt/fel exempel (med variation mellan klasser)
    correct_examples = []
    incorrect_examples = []
    for i, (img_path, true_label, pred_label) in enumerate(
        zip(img_paths, true_labels, pred_labels)
    ):
        if true_label == pred_label and len(correct_examples) < 6:
            correct_examples.append((img_path, pred_label))
        elif true_label != pred_label and len(incorrect_examples) < 6:
            incorrect_examples.append((img_path, pred_label, true_label))
        if len(correct_examples) >= 6 and len(incorrect_examples) >= 6:
            break

    # 3. Plotta rätt/fel exempel
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    for i, (img_path, pred_label) in enumerate(correct_examples):
        img = Image.open(img_path).convert("L")
        axes[0, i].imshow(img, cmap="gray")
        axes[0, i].set_title(f"✓ Pred: {pred_label}", color="green")
        axes[0, i].axis("off")
    for i, (img_path, pred_label, true_label) in enumerate(incorrect_examples):
        img = Image.open(img_path).convert("L")
        axes[1, i].imshow(img, cmap="gray")
        axes[1, i].set_title(f"✗ Pred: {pred_label}\nTrue: {true_label}", color="red")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Korrekt", fontsize=14)
    axes[1, 0].set_ylabel("Fel", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "correct_vs_incorrect.png")
    plt.close()

    # 4. Spara summary.txt
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Total testbilder: {sum(total_per_class.values())}\n")
        f.write(f"Korrekt: {sum(correct_per_class.values())}\n")
        f.write(
            f"Felaktiga: {sum(total_per_class.values()) - sum(correct_per_class.values())}\n"
        )
        f.write(
            f"Overall accuracy: {(sum(correct_per_class.values()) / sum(total_per_class.values())) * 100:.2f}%\n\n"
        )
        for class_name in class_names:
            acc = (correct_per_class[class_name] / total_per_class[class_name]) * 100
            f.write(
                f"{class_name}: {acc:.2f}% ({correct_per_class[class_name]}/{total_per_class[class_name]})\n"
            )
    logger.info(f"Summary sparad till: {summary_path}")

    print(f"\n📊 Visualisering + summary sparad i {output_dir}\n")


if __name__ == "__main__":
    main()
