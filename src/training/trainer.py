"""
Training Pipeline for Cutlery Classifier MVP

This module provides a comprehensive training pipeline that integrates:
- Model factory for architecture creation
- Transform pipeline for data preprocessing
- Training loop with validation and checkpointing
- Metrics tracking and logging
- Early stopping and learning rate scheduling

Features:
- Modular design for easy experimentation
- Automatic checkpointing and model saving
- Comprehensive metrics logging
- Support for both type detection and manufacturer classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from ..models.factory import create_model, get_model_info
from ..data.transforms import create_transform_from_config

logger = logging.getLogger(__name__)


class CutleryTrainer:
    """
    Comprehensive trainer for cutlery classification models.

    Supports both type detection (fork/knife/spoon) and manufacturer classification.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model_name: str = "type_detector",
        device: Optional[str] = None,
    ):
        """
        Initialize trainer with configuration.

        Args:
            config: Training configuration dictionary
            model_name: Name for saving models and logs
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.config = config
        self.model_name = model_name

        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

        # Paths
        self.setup_paths()

    def setup_paths(self):
        """Setup paths for saving models, logs, and results."""
        self.project_root = Path(__file__).parent.parent.parent
        self.models_dir = self.project_root / "models" / "checkpoints"
        self.results_dir = self.project_root / "results"
        self.logs_dir = self.project_root / "results" / "logs"

        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def create_model(self):
        """Create model based on configuration."""
        model_config = self.config.get("model", {})
        self.model = create_model(model_config)
        self.model.to(self.device)

        # Log model info
        model_info = get_model_info(self.model)
        logger.info(f"Model created: {model_info}")

        return self.model

    def create_dataloaders(
        self, include_mixed: bool = False
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, validation, and test dataloaders.

        Args:
            include_mixed: Whether to include mixed cutlery images in training

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        data_config = self.config.get("data", {})
        data_dir = self.project_root / "data" / "processed"
        mixed_dir = self.project_root / "data" / "mixed"

        # Create transforms
        train_transform = create_transform_from_config(data_config, mode="train")
        val_transform = create_transform_from_config(data_config, mode="val")
        test_transform = create_transform_from_config(data_config, mode="test")

        # Create datasets
        train_dataset = ImageFolder(root=data_dir / "train", transform=train_transform)

        # Add mixed images to training if requested
        if include_mixed and mixed_dir.exists():
            mixed_dataset = ImageFolder(root=mixed_dir, transform=train_transform)
            train_dataset.samples.extend(mixed_dataset.samples)
            train_dataset.targets.extend(mixed_dataset.targets)
            logger.info(f"Added {len(mixed_dataset)} mixed images to training set")

        val_dataset = ImageFolder(root=data_dir / "val", transform=val_transform)
        test_dataset = ImageFolder(root=data_dir / "test", transform=test_transform)

        # Log class information
        self.class_names = train_dataset.classes
        self.num_classes = len(self.class_names)
        logger.info(f"Classes: {self.class_names}")
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Val samples: {len(val_dataset)}")
        logger.info(f"Test samples: {len(test_dataset)}")

        # Create dataloaders
        batch_size = self.config.get("training", {}).get("batch_size", 32)
        num_workers = self.config.get("training", {}).get("num_workers", 4)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        return train_loader, val_loader, test_loader

    def setup_training(self):
        """Setup optimizer, scheduler, and loss function."""
        training_config = self.config.get("training", {})

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        optimizer_name = training_config.get("optimizer", "adam").lower()
        lr = training_config.get("learning_rate", 0.001)
        weight_decay = training_config.get("weight_decay", 1e-4)

        if optimizer_name == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == "sgd":
            momentum = training_config.get("momentum", 0.9)
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Scheduler
        scheduler_config = training_config.get("scheduler", {})
        if scheduler_config.get("enabled", True):
            scheduler_type = scheduler_config.get("type", "step")

            if scheduler_type == "step":
                step_size = scheduler_config.get("step_size", 10)
                gamma = scheduler_config.get("gamma", 0.1)
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=step_size, gamma=gamma
                )
            elif scheduler_type == "plateau":
                patience = scheduler_config.get("patience", 5)
                factor = scheduler_config.get("factor", 0.5)
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode="max", patience=patience, factor=factor
                )

        logger.info(f"Optimizer: {optimizer_name}, LR: {lr}")
        logger.info(f"Scheduler: {scheduler_config}")

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            pbar.set_postfix(
                {
                    "Loss": f"{running_loss / (batch_idx + 1):.3f}",
                    "Acc": f"{100.0 * correct / total:.2f}%",
                }
            )

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc

    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate for one epoch.

        Args:
            val_loader: Validation data loader

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accs": self.train_accs,
            "val_accs": self.val_accs,
            "config": self.config,
            "class_names": self.class_names,
        }

        # Save latest checkpoint
        checkpoint_path = self.models_dir / f"{self.model_name}_latest.pth"
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.models_dir / f"{self.model_name}_best.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved: {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_acc = checkpoint["best_val_acc"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        self.train_accs = checkpoint["train_accs"]
        self.val_accs = checkpoint["val_accs"]
        self.class_names = checkpoint["class_names"]

        logger.info(f"Checkpoint loaded from {checkpoint_path}")

    def plot_training_curves(self):
        """Plot and save training curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Loss curves
        ax1.plot(self.train_losses, label="Train Loss")
        ax1.plot(self.val_losses, label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.legend()
        ax1.grid(True)

        # Accuracy curves
        ax2.plot(self.train_accs, label="Train Acc")
        ax2.plot(self.val_accs, label="Val Acc")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Training and Validation Accuracy")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        # Save plot
        plot_path = self.results_dir / f"{self.model_name}_training_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Training curves saved: {plot_path}")

    def train(
        self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int = None
    ):
        """
        Main training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train

        Returns:
            dict: Training history containing losses and accuracies
        """
        if num_epochs is None:
            num_epochs = self.config.get("training", {}).get("num_epochs", 50)

        # Early stopping
        early_stopping_config = self.config.get("training", {}).get(
            "early_stopping", {}
        )
        early_stopping_enabled = early_stopping_config.get("enabled", True)
        patience = early_stopping_config.get("patience", 10)
        min_delta = early_stopping_config.get("min_delta", 0.001)

        epochs_without_improvement = 0

        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Early stopping: {early_stopping_enabled}, patience: {patience}")

        start_time = time.time()

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader)

            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()

            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            # Check for best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Save checkpoint
            self.save_checkpoint(is_best)

            # Log progress
            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - "
                f"LR: {current_lr:.6f}"
            )

            # Early stopping
            if early_stopping_enabled and epochs_without_improvement >= patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")

        # Plot training curves
        self.plot_training_curves()

        # Save training log
        self.save_training_log()

        # Return training history
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accs": self.train_accs,
            "val_accs": self.val_accs,
        }

    def save_training_log(self):
        """Save training log with metrics and configuration."""
        log_data = {
            "model_name": self.model_name,
            "config": self.config,
            "final_epoch": self.current_epoch,
            "best_val_acc": self.best_val_acc,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accs": self.train_accs,
            "val_accs": self.val_accs,
            "class_names": self.class_names,
            "device": str(self.device),
        }

        log_path = self.logs_dir / f"{self.model_name}_training_log.json"
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

        logger.info(f"Training log saved: {log_path}")


def test_trainer_creation():
    """Test function to verify trainer can be created."""

    # Test configuration
    config = {
        "model": {
            "architecture": "resnet18",
            "num_classes": 3,
            "pretrained": True,
            "grayscale": True,
            "freeze_backbone": False,
            "dropout_rate": 0.5,
        },
        "data": {
            "image_size": [320, 320],
            "augmentation": {"horizontal_flip": 0.5, "rotation_degrees": 15},
        },
        "training": {
            "batch_size": 16,
            "num_epochs": 2,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "scheduler": {
                "enabled": True,
                "type": "step",
                "step_size": 10,
                "gamma": 0.1,
            },
            "early_stopping": {"enabled": True, "patience": 5},
        },
    }

    print("Testing trainer creation...")

    # Create trainer
    trainer = CutleryTrainer(config, model_name="test_model")

    # Create model
    model = trainer.create_model()
    trainer.setup_training()

    print(f"✅ Trainer created successfully!")
    print(f"Model: {trainer.config['model']['architecture']}")
    print(f"Device: {trainer.device}")
    print(f"Optimizer: {trainer.config['training']['optimizer']}")

    return trainer


if __name__ == "__main__":
    # Run test when script is executed directly
    logging.basicConfig(level=logging.INFO)
    test_trainer_creation()
