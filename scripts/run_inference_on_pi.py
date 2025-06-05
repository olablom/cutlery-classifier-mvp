#!/usr/bin/env python3
"""
Raspberry Pi Inference Script for Cutlery Classifier

This script:
1. Loads the optimized ONNX model
2. Runs inference on test images
3. Measures performance metrics
4. Provides benchmarking capabilities
5. Auto-detects ARM/Raspberry Pi environment

Usage:
    python run_inference_on_pi.py [--test_dir path/to/images] [--benchmark]
"""

import os
import sys
import time
import logging
import platform
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import json

import numpy as np
import cv2
import onnxruntime as ort
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
SUPPORTED_ARM_PLATFORMS = ["aarch64", "armv7l"]
MODEL_PATH = Path("models/exports/cutlery_classifier_edge.onnx")
DEFAULT_TEST_DIR = Path("test_images")
IMAGE_SIZE = (320, 320)
CLASS_LABELS = ["fork", "knife", "spoon"]
BENCHMARK_ITERATIONS = 50


class InferenceEnvironment:
    """Manages the inference environment and hardware detection."""

    def __init__(self):
        self.platform = platform.machine()
        self.is_arm = self.platform in SUPPORTED_ARM_PLATFORMS
        self.cpu_count = os.cpu_count() or 4
        self.providers = ort.get_available_providers()

    def get_session_options(self) -> ort.SessionOptions:
        """Configure ONNX Runtime session options optimized for the platform."""
        options = ort.SessionOptions()

        if self.is_arm:
            # Raspberry Pi optimizations
            options.intra_op_num_threads = self.cpu_count
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.enable_cpu_mem_arena = False  # Reduce memory usage
        else:
            # Desktop/development optimizations
            options.intra_op_num_threads = self.cpu_count // 2
            options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            )

        return options

    def print_environment_info(self):
        """Print information about the execution environment."""
        logger.info("\n=== Environment Information ===")
        logger.info(f"Platform: {self.platform}")
        logger.info(
            f"Running on ARM: {'Yes' if self.is_arm else 'No (Simulation Mode)'}"
        )
        logger.info(f"CPU Cores: {self.cpu_count}")
        logger.info(f"Available Providers: {', '.join(self.providers)}")
        if not self.is_arm:
            logger.warning(
                "⚠️ Not running on ARM platform. Operating in simulation mode. "
                "Performance metrics will not be representative of Raspberry Pi."
            )


class ImagePreprocessor:
    """Handles image loading and preprocessing."""

    @staticmethod
    def load_and_preprocess(image_path: Path) -> np.ndarray:
        """Load and preprocess an image for inference."""
        # Read image in grayscale
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Resize and normalize
        image = cv2.resize(image, IMAGE_SIZE)
        image = image.astype(np.float32) / 255.0

        # Add batch and channel dimensions
        image = image[np.newaxis, np.newaxis, :, :]

        return image


class InferenceEngine:
    """Manages model loading and inference execution."""

    def __init__(self, model_path: Path, env: InferenceEnvironment):
        self.env = env
        self.session = self._load_model(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def _load_model(self, model_path: Path) -> ort.InferenceSession:
        """Load the ONNX model with optimized settings."""
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Please run export_for_raspberry.py first."
            )

        try:
            session = ort.InferenceSession(
                str(model_path),
                self.env.get_session_options(),
                providers=["CPUExecutionProvider"],
            )
            return session
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def run_inference(self, image: np.ndarray) -> Tuple[str, float, List[float]]:
        """
        Run inference on a single preprocessed image.

        Returns:
            Tuple of (predicted_class, inference_time, probabilities)
        """
        start_time = time.perf_counter()

        output = self.session.run(None, {self.input_name: image})
        probabilities = output[0][0].tolist()

        inference_time = (time.perf_counter() - start_time) * 1000  # ms
        predicted_class = CLASS_LABELS[np.argmax(probabilities)]

        return predicted_class, inference_time, probabilities


class BenchmarkResults:
    """Collects and analyzes benchmark results."""

    def __init__(self):
        self.inference_times = []
        self.results_per_image = {}

    def add_result(self, image_name: str, inference_time: float, prediction: str):
        """Add a benchmark result."""
        self.inference_times.append(inference_time)
        if image_name not in self.results_per_image:
            self.results_per_image[image_name] = []
        self.results_per_image[image_name].append(
            {"time": inference_time, "prediction": prediction}
        )

    def get_statistics(self) -> Dict:
        """Calculate benchmark statistics."""
        times = np.array(self.inference_times)
        return {
            "mean_time": float(np.mean(times)),
            "std_time": float(np.std(times)),
            "min_time": float(np.min(times)),
            "max_time": float(np.max(times)),
            "p95_time": float(np.percentile(times, 95)),
            "total_inferences": len(times),
        }

    def save_results(self, output_path: Path):
        """Save benchmark results to JSON file."""
        results = {
            "statistics": self.get_statistics(),
            "per_image_results": self.results_per_image,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to: {output_path}")


def run_benchmark(
    engine: InferenceEngine, test_dir: Path, benchmark_mode: bool = False
) -> BenchmarkResults:
    """Run inference benchmark on test images."""
    results = BenchmarkResults()

    # Get list of test images
    image_files = [
        f for f in test_dir.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    if not image_files:
        raise ValueError(f"No images found in {test_dir}")

    # Setup progress bar
    total_iterations = len(image_files)
    if benchmark_mode:
        total_iterations *= BENCHMARK_ITERATIONS

    with tqdm(total=total_iterations, desc="Running inference") as pbar:
        # Process each image
        for image_path in image_files:
            image = ImagePreprocessor.load_and_preprocess(image_path)

            # Run inference (multiple times if benchmarking)
            iterations = BENCHMARK_ITERATIONS if benchmark_mode else 1
            for _ in range(iterations):
                prediction, inference_time, probabilities = engine.run_inference(image)
                results.add_result(image_path.name, inference_time, prediction)
                pbar.update(1)

            # Print result for first run
            if not benchmark_mode:
                logger.info(
                    f"\nImage: {image_path.name}\n"
                    f"Prediction: {prediction}\n"
                    f"Inference Time: {inference_time:.2f}ms\n"
                    f"Probabilities: "
                    + ", ".join(
                        [f"{c}: {p:.4f}" for c, p in zip(CLASS_LABELS, probabilities)]
                    )
                )

    return results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Run inference on Raspberry Pi")
    parser.add_argument(
        "--test_dir",
        type=Path,
        default=DEFAULT_TEST_DIR,
        help="Directory containing test images",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark mode with multiple iterations",
    )
    args = parser.parse_args()

    try:
        # Initialize environment
        env = InferenceEnvironment()
        env.print_environment_info()

        # Create inference engine
        engine = InferenceEngine(MODEL_PATH, env)

        # Run inference/benchmark
        results = run_benchmark(engine, args.test_dir, args.benchmark)

        # Print summary
        stats = results.get_statistics()
        logger.info("\n=== Performance Summary ===")
        logger.info(f"Total images processed: {stats['total_inferences']}")
        logger.info(f"Mean inference time: {stats['mean_time']:.2f}ms")
        logger.info(f"95th percentile: {stats['p95_time']:.2f}ms")
        logger.info(
            f"Min/Max time: {stats['min_time']:.2f}ms / {stats['max_time']:.2f}ms"
        )

        # Save results if benchmarking
        if args.benchmark:
            results.save_results(Path("results/benchmarks/raspberry_pi_benchmark.json"))

    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
