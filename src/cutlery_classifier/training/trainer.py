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
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader
import numpy as np
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from ..models.factory import create_model, get_model_info
from ..data.transforms import create_transform_from_config

logger = logging.getLogger(__name__)


class ConfigDatasetFolder(DatasetFolder):
    """Custom DatasetFolder that uses classes from config."""

    def __init__(
        self,
        root: Union[str, Path],
        classes: List[str],
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        loader: callable = default_loader,
        is_valid_file: Optional[callable] = None,
    ):
        self.config_classes = classes
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Override to use classes from config."""
        if not self.config_classes:
            raise ValueError("No classes defined in config")

        classes = sorted(self.config_classes)
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


class CutleryTrainer:
    """
    Comprehensive trainer for cutlery classification models.

    Supports both type detection with manufacturer variants (fork_a/b, knife_a/b, spoon_a/b)
    and future manufacturer-specific classification.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model_name: str = "type_detector",
        device: Optional[str] = None,
        train_dir: Optional[str] = None,
        val_dir: Optional[str] = None,
        test_dir: Optional[str] = None,
    ):
        """
        Initialize trainer with configuration.

        Args:
            config: Training configuration dictionary
            model_name: Name for saving models and logs
            device: Device to use ('cuda', 'cpu', or None for auto)
            train_dir: Optional custom training data directory
            val_dir: Optional custom validation data directory
            test_dir: Optional custom test directory
        """
        self.config = config
        self.model_name = model_name
        self.project_root = Path(__file__).parent.parent.parent
        logger.info(f"Project root set to: {self.project_root}")

        # Use data paths from config if not explicitly provided
        data_config = config.get("data", {})
        self.processed_data_path = Path(
            data_config.get("processed_data_path", "data/processed")
        )

        self.train_dir = (
            Path(train_dir) if train_dir else self.processed_data_path / "train"
        )
        self.val_dir = Path(val_dir) if val_dir else self.processed_data_path / "val"
        self.test_dir = (
            Path(test_dir) if test_dir else self.processed_data_path / "test"
        )

        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")
        logger.info(f"Using train directory: {self.train_dir}")
        logger.info(f"Using validation directory: {self.val_dir}")
        logger.info(f"Using test directory: {self.test_dir}")

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

        # Classes from config
        self.classes = config.get("classes", [])
        if not self.classes:
            raise ValueError("No classes defined in config")

        # Paths
        self.setup_paths()

    def setup_paths(self):
        """Setup paths for saving models, logs, and results."""
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

    def update_model_output_layer(self, num_classes: int) -> None:
        """
        Update the model's output layer to match the number of classes.
        Works with both Linear and Sequential heads.
        """
        if not hasattr(self.model, "fc"):
            logger.warning("Model has no fc layer, skipping output layer update")
            return

        if isinstance(self.model.fc, nn.Linear):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
        elif isinstance(self.model.fc, nn.Sequential):
            # Find and replace the last Linear layer
            last_linear = None
            for layer in reversed(self.model.fc):
                if isinstance(layer, nn.Linear):
                    last_linear = layer
                    break

            if last_linear is None:
                logger.warning(
                    "No Linear layer found in Sequential fc, skipping update"
                )
                return

            # Create new Sequential with updated final layer
            new_layers = []
            for layer in self.model.fc:
                if layer is last_linear:
                    new_layers.append(nn.Linear(last_linear.in_features, num_classes))
                else:
                    new_layers.append(layer)
            self.model.fc = nn.Sequential(*new_layers)

        # Move updated layer to correct device
        self.model.to(self.device)

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
        data_dir = self.processed_data_path
        mixed_dir = self.project_root / "data" / "mixed"

        # Use custom directories if provided, otherwise use defaults
        train_dir = self.train_dir if self.train_dir else data_dir / "train"
        val_dir = self.val_dir if self.val_dir else data_dir / "val"
        test_dir = self.test_dir if self.test_dir else data_dir / "test"

        logger.info(f"Using test directory: {test_dir}")
        print("DEBUG: Listing test_dir contents NOW:")
        for entry in test_dir.iterdir():
            print(f"  - {entry.name} (dir={entry.is_dir()})")

        # Verify directories exist
        if not train_dir.exists():
            raise FileNotFoundError(f"Training directory not found: {train_dir}")
        if not val_dir.exists():
            raise FileNotFoundError(f"Validation directory not found: {val_dir}")
        if not test_dir.exists():
            raise FileNotFoundError(f"Test directory not found: {test_dir}")

        # Get expected classes from config
        expected_classes = self.config.get("classes", [])
        if not expected_classes:
            raise ValueError("No classes defined in config")

        # Verify class directories and images exist
        for class_name in expected_classes:
            for dir_path in [
                train_dir / class_name,
                val_dir / class_name,
                test_dir / class_name,
            ]:
                if not dir_path.exists():
                    raise FileNotFoundError(f"Class directory not found: {dir_path}")

                # Check for valid images
                valid_images = (
                    list(dir_path.glob("*.jpg"))
                    + list(dir_path.glob("*.jpeg"))
                    + list(dir_path.glob("*.png"))
                )
                if not valid_images:
                    raise FileNotFoundError(f"No valid images found in {dir_path}")
                logger.info(f"Found {len(valid_images)} images in {dir_path}")

        # Create transforms
        train_transform = create_transform_from_config(data_config, mode="train")
        val_transform = create_transform_from_config(data_config, mode="val")
        test_transform = create_transform_from_config(data_config, mode="test")

        # Create datasets with explicit class names
        train_dataset = ConfigDatasetFolder(
            root=train_dir, classes=expected_classes, transform=train_transform
        )

        val_dataset = ConfigDatasetFolder(
            root=val_dir, classes=expected_classes, transform=val_transform
        )

        test_dataset = ConfigDatasetFolder(
            root=test_dir, classes=expected_classes, transform=test_transform
        )

        # Add mixed images to training if requested
        if include_mixed and mixed_dir.exists():
            mixed_dataset = ConfigDatasetFolder(
                root=mixed_dir, classes=expected_classes, transform=train_transform
            )
            train_dataset.samples.extend(mixed_dataset.samples)
            train_dataset.targets.extend(mixed_dataset.targets)
            logger.info(f"Added {len(mixed_dataset)} mixed images to training set")

        # Store class information
        self.train_classes = train_dataset.classes
        self.val_classes = val_dataset.classes
        self.test_classes = test_dataset.classes
        self.class_names = self.train_classes
        self.num_classes = len(self.class_names)

        # Log class information
        logger.info(f"Train classes: {self.train_classes}")
        logger.info(f"Val classes: {self.val_classes}")
        logger.info(f"Test classes: {self.test_classes}")

        # Update model's num_classes if it doesn't match
        if self.model is not None and hasattr(self.model, "fc"):
            current_out_features = None
            if isinstance(self.model.fc, nn.Linear):
                current_out_features = self.model.fc.out_features
            elif isinstance(self.model.fc, nn.Sequential):
                # Find the last Linear layer in the Sequential
                for layer in reversed(self.model.fc):
                    if isinstance(layer, nn.Linear):
                        current_out_features = layer.out_features
                        break

            if current_out_features != self.num_classes:
                self.update_model_output_layer(self.num_classes)
                logger.info(f"Updated model head to {self.num_classes} classes")

        # Log dataset information
        logger.info(f"Train samples: {len(train_dataset)} ({train_dir})")
        logger.info(f"Val samples: {len(val_dataset)} ({val_dir})")
        logger.info(f"Test samples: {len(test_dataset)} ({test_dir})")

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
            "classes": self.classes,
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
        # Fallback för classes
        if "classes" in checkpoint:
            self.classes = checkpoint["classes"]
        else:
            logger.warning(
                "Checkpoint saknar 'classes', använder self.classes från config."
            )

        logger.info(f"Checkpoint loaded from {checkpoint_path}")

    def plot_training_curves(self, run_dir: Optional[Path] = None):
        """
        Plot and save training curves.

        Args:
            run_dir: Optional directory for the current run. If provided, saves per-run plots there.
        """
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

        # Save per-run plot if run_dir is provided
        if run_dir is not None:
            run_plot_path = run_dir / f"{self.model_name}_training_curves.png"
            plt.savefig(run_plot_path, dpi=300, bbox_inches="tight")
            logger.info(f"Training curves saved to run directory: {run_plot_path}")

        # Save latest plot in top-level results directory if configured
        if self.config.get("training", {}).get("save_latest_training_curve", True):
            latest_plot_path = (
                self.results_dir / f"{self.model_name}_latest_training_curves.png"
            )
            plt.savefig(latest_plot_path, dpi=300, bbox_inches="tight")
            logger.info(f"Latest training curves saved: {latest_plot_path}")

        plt.close()

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = None,
        run_dir: Optional[Path] = None,
    ):
        """
        Train the model.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Optional override for number of epochs
            run_dir: Optional directory for saving run-specific results
        """
        # Create model if not already created
        if self.model is None:
            self.create_model()
            self.setup_training()

        # Get number of epochs from config if not provided
        if num_epochs is None:
            num_epochs = self.config.get("training", {}).get("num_epochs", 30)

        logger.info(f"Starting training for {num_epochs} epochs")

        # Early stopping setup
        early_stopping_patience = self.config.get("training", {}).get(
            "early_stopping_patience", 10
        )
        early_stopping_counter = 0
        logger.info(f"Early stopping: True, patience: {early_stopping_patience}")

        # Training loop
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate_epoch(val_loader)

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            # Save metrics
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            # Log progress
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

            # Save checkpoint if best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(is_best=True)
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            # Early stopping
            if early_stopping_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

            # Save regular checkpoint every N epochs
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint()

        # Save final model and plots
        self.save_checkpoint()
        self.plot_training_curves(run_dir)
        self.save_training_log()

        logger.info("Training complete!")
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accs": self.train_accs,
            "val_accs": self.val_accs,
            "best_val_acc": self.best_val_acc,
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
            "classes": self.classes,
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
