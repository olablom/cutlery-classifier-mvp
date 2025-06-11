"""
Production Validation Module for Cutlery Classifier

Enhanced for airport security deployment with:
1. Conveyor belt simulation
2. Multi-item detection
3. Enhanced confidence analysis
4. Industrial performance metrics
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from PIL import Image, ImageDraw
import cv2
import numpy as np
import matplotlib.pyplot as plt

from cutlery_classifier.models.factory import create_model
from cutlery_classifier.data.dataset import CutleryDataset

logger = logging.getLogger(__name__)


class ConveyorSimulator:
    """Simulates airport conveyor belt conditions."""

    def __init__(
        self,
        belt_speed: float = 0.5,  # meters per second
        fps: int = 30,
        image_size: Tuple[int, int] = (320, 320),
    ):
        self.belt_speed = belt_speed
        self.fps = fps
        self.image_size = image_size

        # Calculate motion blur parameters
        self.pixels_per_frame = int((belt_speed * image_size[0]) / fps)

    def apply_motion_blur(self, image: Image.Image) -> Image.Image:
        """Apply directional blur simulating conveyor movement."""
        # Convert PIL to cv2
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Create motion blur kernel
        kernel_size = int(self.pixels_per_frame * 2)
        kernel_size = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size

        # Apply motion blur
        blurred = cv2.filter2D(cv_image, -1, kernel)

        return Image.fromarray(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))

    def simulate_multi_item(
        self,
        primary_image: Image.Image,
        secondary_images: List[Image.Image],
        max_items: int = 3,
    ) -> Image.Image:
        """Simulate multiple items in frame with realistic placement."""
        # Create base canvas
        canvas = Image.new("RGB", (self.image_size[0], self.image_size[1]))

        # Resize primary image to fit
        primary_size = (self.image_size[0] // 3, self.image_size[1] // 3)
        primary_image = primary_image.resize(primary_size)

        # Place primary item
        primary_pos = (
            np.random.randint(0, self.image_size[0] - primary_size[0]),
            np.random.randint(0, self.image_size[1] - primary_size[1]),
        )
        canvas.paste(primary_image, primary_pos)

        # Add random secondary items
        num_extra = np.random.randint(1, max_items)
        for i in range(min(num_extra, len(secondary_images))):
            sec_img = secondary_images[i]
            # Resize secondary image
            sec_size = (self.image_size[0] // 4, self.image_size[1] // 4)
            sec_img = sec_img.resize(sec_size)

            # Calculate non-overlapping position
            for _ in range(10):  # Try 10 times to find non-overlapping spot
                pos_x = np.random.randint(0, self.image_size[0] - sec_size[0])
                pos_y = np.random.randint(0, self.image_size[1] - sec_size[1])
                # TODO: Add overlap detection
                canvas.paste(sec_img, (pos_x, pos_y))
                break

        return canvas

    def simulate_occlusion(
        self, image: Image.Image, occlusion_ratio: float = 0.3
    ) -> Image.Image:
        """Simulate partial occlusion by luggage/bags."""
        # Create occlusion mask
        mask = Image.new("L", image.size, 255)

        # Random occlusion patterns
        num_shapes = np.random.randint(1, 4)
        for _ in range(num_shapes):
            shape_size = (
                int(image.size[0] * np.random.uniform(0.1, occlusion_ratio)),
                int(image.size[1] * np.random.uniform(0.1, occlusion_ratio)),
            )
            shape_pos = (
                np.random.randint(0, image.size[0] - shape_size[0]),
                np.random.randint(0, image.size[1] - shape_size[1]),
            )
            # Draw random dark region
            ImageDraw.Draw(mask).rectangle(
                [
                    shape_pos,
                    (shape_pos[0] + shape_size[0], shape_pos[1] + shape_size[1]),
                ],
                fill=0,
            )

        return Image.composite(image, Image.new("RGB", image.size, "black"), mask)


class SecurityConfidenceAnalyzer:
    """Enhanced confidence analysis for security applications."""

    def __init__(
        self,
        cost_matrix: Optional[Dict[Tuple[str, str], float]] = None,
        confidence_threshold: float = 0.95,
        uncertainty_threshold: float = 0.2,
    ):
        # Default cost matrix (can be customized)
        self.cost_matrix = cost_matrix or {
            ("knife", "fork"): 10.0,  # High cost for missing dangerous items
            ("knife", "spoon"): 10.0,
            ("fork", "knife"): 1.0,  # Lower cost for false alarms
            ("spoon", "knife"): 1.0,
            ("fork", "spoon"): 0.1,  # Low cost for confusion between safe items
            ("spoon", "fork"): 0.1,
        }

        self.confidence_threshold = confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold

    def analyze_prediction(
        self, probabilities: torch.Tensor, true_label: Optional[int] = None
    ) -> Dict[str, float]:
        """Analyze a single prediction with security-focused metrics."""
        sorted_probs, indices = torch.sort(probabilities, descending=True)

        # Calculate metrics
        metrics = {
            "top1_confidence": sorted_probs[0].item(),
            "top2_confidence": sorted_probs[1].item(),
            "confidence_margin": (sorted_probs[0] - sorted_probs[1]).item(),
            "entropy": (-probabilities * torch.log(probabilities)).sum().item(),
            "requires_manual_check": False,
            "risk_score": 0.0,
        }

        # Uncertainty check
        if metrics["confidence_margin"] < self.uncertainty_threshold:
            metrics["requires_manual_check"] = True

        # Risk analysis
        if true_label is not None:
            pred_label = indices[0].item()
            metrics["risk_score"] = self.cost_matrix.get(
                (str(pred_label), str(true_label)), 0.0
            )

        return metrics


class ProductionValidator:
    """Validates model performance in production-like conditions."""

    def __init__(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        device: torch.device,
        class_names: List[str],
    ):
        """Initialize validator.

        Args:
            model: trained model
            data_loader: test data loader
            device: torch device
            class_names: list of class names
        """
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.class_names = class_names
        self.results: Dict = {}

    def run_validation(self) -> Dict:
        """Run full validation suite.

        Returns:
            Dictionary containing all validation results
        """
        logger.info("Running production validation suite...")

        # Basic accuracy test
        accuracy = self._test_accuracy()
        self.results["accuracy"] = accuracy

        # Confidence analysis
        confidence_stats = self._analyze_confidence()
        self.results["confidence"] = confidence_stats

        # Stress tests
        stress_results = self._run_stress_tests()
        self.results["stress_tests"] = stress_results

        # Throughput test
        throughput = self._measure_throughput()
        self.results["throughput"] = throughput

        logger.info("Validation complete!")
        return self.results

    def _test_accuracy(self) -> float:
        """Test basic model accuracy.

        Returns:
            Accuracy as percentage
        """
        correct = 0
        total = 0

        with torch.no_grad():
            for images, targets in self.data_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)

                correct += (predicted == targets).sum().item()
                total += targets.size(0)

        accuracy = (correct / total) * 100
        logger.info(f"Base accuracy: {accuracy:.2f}%")
        return accuracy

    def _analyze_confidence(self) -> Dict[str, float]:
        """Analyze model confidence.

        Returns:
            Dictionary with confidence statistics
        """
        confidences = []

        with torch.no_grad():
            for images, _ in self.data_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confidence, _ = torch.max(probs, dim=1)
                confidences.extend(confidence.cpu().numpy())

        stats = {
            "mean": float(np.mean(confidences)),
            "min": float(np.min(confidences)),
            "max": float(np.max(confidences)),
            "std": float(np.std(confidences)),
        }

        logger.info(
            f"Confidence stats - Mean: {stats['mean']:.2f}, "
            f"Min: {stats['min']:.2f}, Max: {stats['max']:.2f}"
        )
        return stats

    def _run_stress_tests(self) -> Dict[str, float]:
        """Run stress tests with various perturbations.

        Returns:
            Dictionary with stress test results
        """
        results = {}

        # Test with noise
        noise_acc = self._test_with_noise()
        results["noise"] = noise_acc

        # Test with blur
        blur_acc = self._test_with_blur()
        results["blur"] = blur_acc

        # Test with rotation
        rotation_acc = self._test_with_rotation()
        results["rotation"] = rotation_acc

        logger.info(
            f"Stress test results - "
            f"Noise: {noise_acc:.2f}%, "
            f"Blur: {blur_acc:.2f}%, "
            f"Rotation: {rotation_acc:.2f}%"
        )
        return results

    def _test_with_noise(self, noise_level: float = 0.1) -> float:
        """Test model with Gaussian noise.

        Args:
            noise_level: standard deviation of noise

        Returns:
            Accuracy under noise as percentage
        """
        correct = 0
        total = 0

        with torch.no_grad():
            for images, targets in self.data_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                # Add Gaussian noise
                noise = torch.randn_like(images) * noise_level
                noisy_images = images + noise
                noisy_images = torch.clamp(noisy_images, 0, 1)

                outputs = self.model(noisy_images)
                _, predicted = torch.max(outputs, 1)

                correct += (predicted == targets).sum().item()
                total += targets.size(0)

        return (correct / total) * 100

    def _test_with_blur(self, kernel_size: int = 5, sigma: float = 2.0) -> float:
        """Test model with Gaussian blur.

        Args:
            kernel_size: size of Gaussian kernel
            sigma: standard deviation of Gaussian kernel

        Returns:
            Accuracy under blur as percentage
        """
        correct = 0
        total = 0
        blur = T.GaussianBlur(kernel_size, sigma)

        with torch.no_grad():
            for images, targets in self.data_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                # Apply blur
                blurred_images = blur(images)

                outputs = self.model(blurred_images)
                _, predicted = torch.max(outputs, 1)

                correct += (predicted == targets).sum().item()
                total += targets.size(0)

        return (correct / total) * 100

    def _test_with_rotation(self, max_angle: int = 30) -> float:
        """Test model with random rotations.

        Args:
            max_angle: maximum rotation angle in degrees

        Returns:
            Accuracy under rotation as percentage
        """
        correct = 0
        total = 0

        with torch.no_grad():
            for images, targets in self.data_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                # Apply random rotation
                angle = torch.randint(-max_angle, max_angle + 1, (1,)).item()
                rotated_images = T.functional.rotate(images, angle)

                outputs = self.model(rotated_images)
                _, predicted = torch.max(outputs, 1)

                correct += (predicted == targets).sum().item()
                total += targets.size(0)

        return (correct / total) * 100

    def _measure_throughput(self, batch_size: int = 8, num_batches: int = 100) -> float:
        """Measure model throughput.

        Args:
            batch_size: batch size for inference
            num_batches: number of batches to test

        Returns:
            Average items processed per minute
        """
        # Create dummy data for throughput testing
        dummy_input = torch.randn(batch_size, 1, 320, 320).to(self.device)

        # Warmup
        for _ in range(10):
            _ = self.model(dummy_input)

        # Measure inference time
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        start_time.record()
        for _ in range(num_batches):
            _ = self.model(dummy_input)
        end_time.record()

        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds

        items_per_second = (batch_size * num_batches) / elapsed_time
        items_per_minute = items_per_second * 60

        logger.info(f"Throughput: {items_per_minute:.0f} items/minute")
        return items_per_minute

    def generate_report(
        self,
        benchmark_results: Dict[str, float],
        confidence_results: Dict[str, Dict[str, float]],
        stress_results: Dict[str, float],
        output_dir: Path,
    ) -> None:
        """Generate comprehensive validation report."""
        logger.info("Generating validation report...")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Format results
        report_lines = [
            "=================================",
            "AIRPORT SECURITY VALIDATION REPORT",
            "=================================",
            "",
            "1. Inference Performance",
            "----------------------",
        ]

        # Add benchmark results
        for metric, value in benchmark_results.items():
            report_lines.append(f"{metric}: {value:.2f}")

        report_lines.extend(
            [
                "",
                "2. Confidence Analysis",
                "--------------------",
            ]
        )

        # Add confidence results
        for class_name, metrics in confidence_results.items():
            report_lines.append(f"\n{class_name}:")
            for metric, value in metrics.items():
                report_lines.append(f"  {metric}: {value:.2f}")

        report_lines.extend(
            [
                "",
                "3. Stress Test Results",
                "-------------------",
            ]
        )

        # Add stress test results
        for condition, accuracy in stress_results.items():
            report_lines.append(f"{condition}: {accuracy:.2f}%")

        # Write report
        report_path = output_dir / "airport_validation_report.txt"
        report_path.write_text("\n".join(report_lines))
