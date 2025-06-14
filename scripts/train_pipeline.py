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
import os
import matplotlib.pyplot as plt
from datetime import datetime

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


def create_output_folder(base_dir="outputs"):
    now = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = Path(base_dir) / f"run_{now}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def create_dataloaders(config: dict, mode: str = "train") -> tuple:
    """Create train and validation dataloaders with correct transforms."""
    data_config = config.get("data", {})
    processed_data_path = Path(
        data_config.get("processed_data_path", "data/simplified")
    )
    train_dir = processed_data_path / "train"
    val_dir = processed_data_path / "val"
    train_transform = create_transform_from_config(data_config, mode="train")
    val_transform = create_transform_from_config(data_config, mode="val")
    train_dataset = ImageFolder(train_dir, transform=train_transform)
    val_dataset = ImageFolder(val_dir, transform=val_transform)
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
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Finetune from best checkpoint with low LR",
    )
    args = parser.parse_args()

    logger.info(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Skapa output-folder
    output_dir = create_output_folder()
    logger.info(f"Output folder for this run: {output_dir}")

    # Skapa dataloaders
    train_loader, val_loader = create_dataloaders(config)
    logger.info(
        f"Created dataloaders: {len(train_loader)} training batches, {len(val_loader)} validation batches"
    )

    # Skapa trainer
    trainer = CutleryTrainer(
        config=config, model_name="type_detector", device=args.device
    )

    if args.finetune:
        # Ladda modell från best checkpoint
        best_ckpt = "models/checkpoints/type_detector_best.pth"
        logger.info(f"Finetuning from checkpoint: {best_ckpt}")
        trainer.create_model()
        trainer.setup_training()
        trainer.load_checkpoint(best_ckpt)
        # Sätt låg LR och optimizer, och skapa om optimizer/scheduler
        config["training"]["learning_rate"] = 1e-4
        config["training"]["optimizer"] = "adam"
        trainer.setup_training()
        config["training"]["num_epochs"] = 5
        model_save_path = "models/checkpoints/type_detector_finetuned.pth"
    else:
        model_save_path = "models/checkpoints/type_detector_best.pth"

    # Träna
    logger.info("Starting training...")
    history = trainer.train(train_loader, val_loader)
    logger.info("Training complete!")

    # Spara modell
    trainer.save_checkpoint(model_save_path)
    logger.info(f"Model saved to: {model_save_path}")

    # Spara träningskurvor
    if history:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history["train_losses"], label="Train Loss")
        plt.plot(history["val_losses"], label="Val Loss")
        plt.legend()
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.subplot(1, 2, 2)
        plt.plot(history["train_accs"], label="Train Acc")
        plt.plot(history["val_accs"], label="Val Acc")
        plt.legend()
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.tight_layout()
        plt.savefig(output_dir / "accuracy_loss_curves.png")
        plt.close()
        logger.info(
            f"Saved accuracy/loss curves to: {output_dir / 'accuracy_loss_curves.png'}"
        )

    logger.info(f"✅ Träning/finetuning klar. Output-folder: {output_dir}")


if __name__ == "__main__":
    main()
