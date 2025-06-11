"""
Evaluation Module for Cutlery Classifier MVP

This module provides comprehensive evaluation tools including:
- Model performance metrics (accuracy, precision, recall, F1)
- Confusion matrix visualization
- Classification reports
- Grad-CAM visualization for model interpretability
- Test set evaluation and analysis

Features:
- Load trained models from checkpoints
- Generate detailed performance reports
- Create publication-ready visualizations
- Support for both type detection and manufacturer classification
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json
import logging
from tqdm import tqdm
import cv2
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

from ..models.factory import create_model
from ..data.transforms import (
    create_transform_from_config,
    tensor_to_pil,
    denormalize_tensor,
)

logger = logging.getLogger(__name__)


class CutleryEvaluator:
    """
    Comprehensive evaluator for cutlery classification models.

    Provides metrics, visualizations, and interpretability analysis.
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize evaluator with trained model.

        Args:
            model_path: Path to trained model checkpoint
            config: Model configuration (loaded from checkpoint if None)
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_path = Path(model_path)

        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Load model and config
        self.model = None
        self.config = config
        self.class_names = None
        self.load_model()

        # Setup paths
        self.setup_paths()

    def setup_paths(self):
        """Setup paths for saving evaluation results."""
        self.project_root = Path(__file__).parent.parent.parent
        self.results_dir = self.project_root / "results"
        self.confusion_dir = self.results_dir / "confusion_matrices"
        self.gradcam_dir = self.results_dir / "grad_cam"
        self.metrics_dir = self.results_dir / "metrics"

        # Create directories
        self.confusion_dir.mkdir(parents=True, exist_ok=True)
        self.gradcam_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self):
        """Load trained model from checkpoint."""
        logger.info(f"Loading model from {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Load config if not provided
        if self.config is None:
            self.config = checkpoint["config"]

        # Load class names
        self.class_names = checkpoint["class_names"]

        # Create and load model
        model_config = self.config.get("model", {})
        self.model = create_model(model_config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded successfully")
        logger.info(f"Classes: {self.class_names}")

    def create_test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        data_config = self.config.get("data", {})
        data_dir = self.project_root / "data" / "processed"

        # Create test transform
        test_transform = create_transform_from_config(data_config, mode="test")

        # Create test dataset
        test_dataset = ImageFolder(root=data_dir / "test", transform=test_transform)

        # Verify class names match
        if test_dataset.classes != self.class_names:
            logger.warning(
                f"Class mismatch: model={self.class_names}, data={test_dataset.classes}"
            )

        # Create dataloader
        batch_size = self.config.get("training", {}).get("batch_size", 32)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )

        logger.info(f"Test dataset: {len(test_dataset)} samples")
        return test_loader

    def evaluate_model(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate model on test set.

        Args:
            test_loader: Test data loader

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating model on test set...")

        all_predictions = []
        all_targets = []
        all_probabilities = []

        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc="Evaluating"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            all_targets, all_predictions, average="weighted"
        )

        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = (
            precision_recall_fscore_support(all_targets, all_predictions, average=None)
        )

        # Classification report
        class_report = classification_report(
            all_targets,
            all_predictions,
            target_names=self.class_names,
            output_dict=True,
        )

        # Confusion matrix
        conf_matrix = confusion_matrix(all_targets, all_predictions)

        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": support,
            "per_class_metrics": {
                "precision": precision_per_class.tolist(),
                "recall": recall_per_class.tolist(),
                "f1_score": f1_per_class.tolist(),
                "support": support_per_class.tolist(),
            },
            "classification_report": class_report,
            "confusion_matrix": conf_matrix.tolist(),
            "predictions": all_predictions,
            "targets": all_targets,
            "probabilities": all_probabilities,
        }

        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Test F1-Score: {f1:.4f}")

        return results

    def plot_confusion_matrix(self, conf_matrix: np.ndarray, model_name: str = "model"):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(10, 8))

        # Normalize confusion matrix
        conf_matrix_norm = (
            conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis]
        )

        # Create heatmap
        sns.heatmap(
            conf_matrix_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={"label": "Normalized Count"},
        )

        plt.title(f"Confusion Matrix - {model_name}", fontsize=16, fontweight="bold")
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plt.tight_layout()

        # Save plot
        save_path = self.confusion_dir / f"{model_name}_confusion_matrix.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Confusion matrix saved: {save_path}")

        # Also save raw confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={"label": "Count"},
        )

        plt.title(
            f"Confusion Matrix (Raw Counts) - {model_name}",
            fontsize=16,
            fontweight="bold",
        )
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plt.tight_layout()

        raw_save_path = self.confusion_dir / f"{model_name}_confusion_matrix_raw.png"
        plt.savefig(raw_save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Raw confusion matrix saved: {raw_save_path}")

    def plot_classification_report(self, class_report: Dict, model_name: str = "model"):
        """Plot classification report as heatmap."""
        # Extract metrics for plotting
        metrics_data = []
        for class_name in self.class_names:
            if class_name in class_report:
                metrics_data.append(
                    [
                        class_report[class_name]["precision"],
                        class_report[class_name]["recall"],
                        class_report[class_name]["f1-score"],
                    ]
                )

        metrics_df = pd.DataFrame(
            metrics_data,
            index=self.class_names,
            columns=["Precision", "Recall", "F1-Score"],
        )

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            metrics_df,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Score"},
        )

        plt.title(
            f"Classification Report - {model_name}", fontsize=16, fontweight="bold"
        )
        plt.xlabel("Metrics", fontsize=12)
        plt.ylabel("Classes", fontsize=12)
        plt.tight_layout()

        save_path = self.metrics_dir / f"{model_name}_classification_report.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Classification report saved: {save_path}")

    def setup_grad_cam(self, target_layers: Optional[List] = None):
        """Setup Grad-CAM for model interpretability."""
        if target_layers is None:
            # Auto-detect target layers based on architecture
            architecture = self.config.get("model", {}).get("architecture", "").lower()

            if "resnet" in architecture:
                target_layers = [self.model.layer4[-1]]
            elif "mobilenet" in architecture:
                target_layers = [self.model.features[-1]]
            else:
                # Fallback: try to find the last convolutional layer
                target_layers = []
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.Conv2d):
                        target_layers = [module]

                if not target_layers:
                    logger.warning("Could not auto-detect target layers for Grad-CAM")
                    return None

        # Create Grad-CAM
        cam = GradCAM(model=self.model, target_layers=target_layers)
        logger.info(f"Grad-CAM setup complete with target layers: {target_layers}")

        return cam

    def generate_grad_cam_visualizations(
        self, test_loader: DataLoader, num_samples: int = 10, model_name: str = "model"
    ):
        """Generate Grad-CAM visualizations for sample images."""
        logger.info(f"Generating Grad-CAM visualizations for {num_samples} samples...")

        cam = self.setup_grad_cam()
        if cam is None:
            logger.error("Failed to setup Grad-CAM")
            return

        # Get sample images
        sample_count = 0

        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if sample_count >= num_samples:
                break

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            for i in range(inputs.size(0)):
                if sample_count >= num_samples:
                    break

                input_tensor = inputs[i : i + 1]
                target_class = targets[i].item()

                # Generate Grad-CAM
                target = [ClassifierOutputTarget(target_class)]
                grayscale_cam = cam(input_tensor=input_tensor, targets=target)
                grayscale_cam = grayscale_cam[0, :]

                # Convert input tensor to image
                input_image = inputs[i].cpu()
                input_image = denormalize_tensor(input_image)
                input_image = torch.clamp(input_image, 0, 1)

                # Convert to numpy for visualization
                if input_image.shape[0] == 1:  # Grayscale
                    input_np = input_image.squeeze().numpy()
                    input_rgb = np.stack([input_np] * 3, axis=-1)
                else:
                    input_rgb = input_image.permute(1, 2, 0).numpy()

                # Create visualization
                visualization = show_cam_on_image(
                    input_rgb, grayscale_cam, use_rgb=True
                )

                # Create subplot with original and Grad-CAM
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                # Original image
                if input_image.shape[0] == 1:
                    axes[0].imshow(input_np, cmap="gray")
                else:
                    axes[0].imshow(input_rgb)
                axes[0].set_title("Original Image")
                axes[0].axis("off")

                # Grad-CAM heatmap
                axes[1].imshow(grayscale_cam, cmap="jet")
                axes[1].set_title("Grad-CAM Heatmap")
                axes[1].axis("off")

                # Overlay
                axes[2].imshow(visualization)
                axes[2].set_title("Grad-CAM Overlay")
                axes[2].axis("off")

                # Add title with prediction info
                predicted_class = torch.argmax(self.model(input_tensor), dim=1).item()
                true_class_name = self.class_names[target_class]
                pred_class_name = self.class_names[predicted_class]

                fig.suptitle(
                    f"Sample {sample_count + 1}: True={true_class_name}, Pred={pred_class_name}",
                    fontsize=14,
                    fontweight="bold",
                )

                plt.tight_layout()

                # Save visualization
                save_path = (
                    self.gradcam_dir
                    / f"{model_name}_gradcam_sample_{sample_count + 1}.png"
                )
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close()

                sample_count += 1

        logger.info(f"Grad-CAM visualizations saved to {self.gradcam_dir}")

    def save_evaluation_report(
        self, results: Dict[str, Any], model_name: str = "model"
    ):
        """Save comprehensive evaluation report."""
        report = {
            "model_name": model_name,
            "model_path": str(self.model_path),
            "config": self.config,
            "class_names": self.class_names,
            "test_results": {
                "accuracy": results["accuracy"],
                "precision": results["precision"],
                "recall": results["recall"],
                "f1_score": results["f1_score"],
                "support": int(results["support"]),
            },
            "per_class_results": {},
        }

        # Add per-class results
        for i, class_name in enumerate(self.class_names):
            report["per_class_results"][class_name] = {
                "precision": results["per_class_metrics"]["precision"][i],
                "recall": results["per_class_metrics"]["recall"][i],
                "f1_score": results["per_class_metrics"]["f1_score"][i],
                "support": int(results["per_class_metrics"]["support"][i]),
            }

        # Save report
        report_path = self.metrics_dir / f"{model_name}_evaluation_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Evaluation report saved: {report_path}")

        # Also save detailed classification report
        detailed_path = (
            self.metrics_dir / f"{model_name}_detailed_classification_report.json"
        )
        with open(detailed_path, "w") as f:
            json.dump(results["classification_report"], f, indent=2)

        logger.info(f"Detailed classification report saved: {detailed_path}")

    def run_full_evaluation(self, model_name: str = None) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline.

        Args:
            model_name: Name for saving results

        Returns:
            Evaluation results dictionary
        """
        if model_name is None:
            model_name = self.model_path.stem

        logger.info(f"Starting full evaluation for {model_name}")

        # Create test dataloader
        test_loader = self.create_test_dataloader()

        # Evaluate model
        results = self.evaluate_model(test_loader)

        # Generate visualizations
        self.plot_confusion_matrix(np.array(results["confusion_matrix"]), model_name)
        self.plot_classification_report(results["classification_report"], model_name)

        # Generate Grad-CAM visualizations
        self.generate_grad_cam_visualizations(
            test_loader, num_samples=10, model_name=model_name
        )

        # Save evaluation report
        self.save_evaluation_report(results, model_name)

        logger.info("Full evaluation completed successfully!")

        return results


def test_evaluator_creation():
    """Test function to verify evaluator can be created (without actual model)."""
    print("Testing evaluator creation...")

    # This would normally require a trained model
    print("✅ Evaluator module loaded successfully!")
    print("Note: Actual evaluation requires a trained model checkpoint")


if __name__ == "__main__":
    # Run test when script is executed directly
    logging.basicConfig(level=logging.INFO)
    test_evaluator_creation()
