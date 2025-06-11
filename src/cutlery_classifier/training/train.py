#!/usr/bin/env python3
"""
Training script for cutlery classification model.

This script trains a ResNet18 model for cutlery classification using
configuration from a YAML file. It supports both original and augmented datasets.
"""

import os
import yaml
import random
import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms, datasets
import numpy as np


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_config(config_path: str) -> Dict:
    """Load and validate configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Validate required fields
    required_fields = [
        "data.processed_data_path",
        "data.batch_size",
        "training.epochs",
        "training.learning_rate",
        "paths.model_save_dir",
        "paths.export_dir",
    ]

    for field in required_fields:
        parts = field.split(".")
        current = config
        for part in parts:
            if part not in current:
                raise ValueError(f"Missing required config field: {field}")
            current = current[part]

    return config


def create_data_loaders(
    data_dir: str, batch_size: int, val_split: float = 0.2
) -> Tuple[DataLoader, DataLoader, int]:
    """Create train and validation data loaders."""
    # Define transforms
    transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Match dataset preparation
        ]
    )

    # Load dataset
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    logger.info(f"Found {len(dataset)} images in {len(dataset.classes)} classes")
    logger.info(f"Classes: {dataset.classes}")

    # Split dataset
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, len(dataset.classes)


def create_model(num_classes: int, device: torch.device) -> nn.Module:
    """Create and initialize the ResNet18 model."""
    model = models.resnet18(pretrained=False)

    # Modify first conv layer for grayscale input
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Modify final layer for our number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model.to(device)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            logger.info(
                f"Training batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}"
            )

    return total_loss / len(train_loader)


def validate(
    model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float]:
    """Validate the model and return loss and accuracy."""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    val_loss /= len(val_loader)
    accuracy = 100.0 * correct / total

    return val_loss, accuracy


def export_to_onnx(
    model: nn.Module,
    export_path: str,
    input_shape: Tuple[int, ...],
    device: torch.device,
) -> None:
    """Export the model to ONNX format."""
    dummy_input = torch.randn(input_shape, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        export_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    logger.info(f"Model exported to ONNX: {export_path}")


def main():
    """Main training function."""
    import argparse

    parser = argparse.ArgumentParser(description="Train cutlery classifier model")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Set random seeds
    set_random_seeds(42)

    # Create data loaders
    train_loader, val_loader, num_classes = create_data_loaders(
        config["data"]["processed_data_path"],
        config["data"]["batch_size"],
        config["data"]["val_split"],
    )

    # Create model, criterion, optimizer
    model = create_model(num_classes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    # Training loop
    best_accuracy = 0
    epochs = config["training"]["epochs"]

    for epoch in range(1, epochs + 1):
        logger.info(f"\nEpoch {epoch}/{epochs}")

        # Train and validate
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, accuracy = validate(model, val_loader, criterion, device)

        logger.info(
            f"Train loss: {train_loss:.4f}, "
            f"Val loss: {val_loss:.4f}, "
            f"Val accuracy: {accuracy:.2f}%"
        )

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model_path = Path(config["paths"]["model_save_dir"]) / "best_model.pt"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved best model to {model_path}")

    # Export to ONNX
    onnx_path = Path(config["paths"]["export_dir"]) / "model.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_onnx(
        model,
        str(onnx_path),
        (1, 1, 320, 320),  # (batch_size, channels, height, width)
        device,
    )

    logger.info(f"\nTraining completed!")
    logger.info(f"Best validation accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main()
