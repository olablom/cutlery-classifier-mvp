#!/usr/bin/env python3
"""
Training script for cutlery classification model.

This script trains a ResNet18 model for cutlery classification using the
configured data pipeline and training parameters.
"""

import argparse
import logging
from pathlib import Path
import yaml
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as T

from cutlery_classifier.training.trainer import CutleryTrainer
from cutlery_classifier.data.transforms import create_transform_from_config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_dataloaders(config: dict) -> tuple:
    """Create train and validation dataloaders."""
    # Get data paths
    data_config = config.get("data", {})
    processed_data_path = Path(
        data_config.get("processed_data_path", "data/simplified")
    )
    train_dir = processed_data_path / "train"
    val_dir = processed_data_path / "val"

    # Create transforms
    transform = create_transform_from_config(data_config)

    # Create datasets
    train_dataset = ImageFolder(train_dir, transform=transform)
    val_dataset = ImageFolder(val_dir, transform=transform)

    # Create dataloaders
    batch_size = config.get("training", {}).get("batch_size", 32)
    num_workers = data_config.get("num_workers", 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train cutlery classification model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_config.yaml",
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        required=True,
        help="Device to train on",
    )
    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)
    logger.info(
        f"Created dataloaders: {len(train_loader)} training batches, {len(val_loader)} validation batches"
    )

    # Create trainer
    trainer = CutleryTrainer(
        config=config, model_name="type_detector", device=args.device
    )

    # Train model
    logger.info("Starting training...")
    trainer.train(train_loader, val_loader)
    logger.info("Training complete!")

    # Save final model
    trainer.save_checkpoint("final")
    logger.info("Final model saved!")


if __name__ == "__main__":
    main()
