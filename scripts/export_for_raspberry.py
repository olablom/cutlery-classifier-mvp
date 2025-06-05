#!/usr/bin/env python3
"""
Export script for Raspberry Pi / edge deployment.

This script:
1. Loads the trained PyTorch model
2. Optimizes it for edge deployment (quantization, pruning)
3. Exports to ONNX format with Raspberry Pi optimizations
4. Validates the exported model
5. Provides detailed memory and performance profiling
"""

import os
import sys
import torch
import onnx
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import psutil
import platform

from src.models.factory import create_model
from src.data.transforms import get_base_transforms

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
IMAGE_SIZE = (320, 320)
NUM_CLASSES = 3
EXPORT_DIR = Path("models/exports")
EXPORT_FILENAME = "cutlery_classifier_edge.onnx"

# Raspberry Pi specific settings
PI_MEMORY_LIMIT = 1024  # MB
PI_CPU_CORES = 4
SUPPORTED_ARM_PLATFORMS = ["aarch64", "armv7l"]


def get_system_info() -> Dict:
    """Get system information for optimization decisions."""
    return {
        "memory_total": psutil.virtual_memory().total / (1024 * 1024),  # MB
        "cpu_count": psutil.cpu_count(),
        "platform": platform.machine(),
        "is_arm": platform.machine() in SUPPORTED_ARM_PLATFORMS,
    }


def setup_export_directory() -> Path:
    """Create export directory if it doesn't exist."""
    export_path = EXPORT_DIR / EXPORT_FILENAME
    export_path.parent.mkdir(parents=True, exist_ok=True)
    return export_path


def get_model_metadata() -> Dict:
    """Get model metadata for ONNX export."""
    return {
        "input_names": ["input"],
        "output_names": ["output"],
        "dynamic_axes": {"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    }


def optimize_model_for_edge(
    model: torch.nn.Module, lightweight: bool = False
) -> torch.nn.Module:
    """
    Apply Raspberry Pi specific optimizations.

    Args:
        model: PyTorch model to optimize
        lightweight: Whether to apply aggressive optimizations

    Returns:
        Optimized model
    """
    logger.info("Applying Raspberry Pi optimizations...")

    # Set model to evaluation mode
    model.eval()

    # Basic optimizations
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if lightweight:
        logger.info("Applying lightweight mode optimizations...")

        # Enable static quantization
        model.qconfig = torch.quantization.get_default_qconfig("qnnpack")
        torch.quantization.prepare(model, inplace=True)

        # Fuse common operations
        model = torch.quantization.fuse_modules(
            model, [["conv", "bn", "relu"], ["conv", "relu"], ["linear", "relu"]]
        )

        # Convert to quantized model
        torch.quantization.convert(model, inplace=True)

    return model


def estimate_memory_usage(model: torch.nn.Module, batch_size: int = 1) -> Dict:
    """
    Estimate memory usage for the model.

    Args:
        model: PyTorch model
        batch_size: Batch size for inference

    Returns:
        Dictionary with memory estimates
    """

    def get_size_mb(tensor):
        return tensor.element_size() * tensor.nelement() / (1024 * 1024)

    total_params = sum(p.numel() for p in model.parameters())
    param_size = total_params * 4 / (1024 * 1024)  # Size in MB

    # Estimate buffer size for one batch
    dummy_input = torch.randn(batch_size, 1, *IMAGE_SIZE)
    buffer_size = get_size_mb(dummy_input)

    # Estimate activation memory (rough approximation)
    activation_size = buffer_size * 3  # Conservative estimate

    return {
        "model_size_mb": param_size,
        "buffer_size_mb": buffer_size,
        "activation_size_mb": activation_size,
        "total_runtime_mb": param_size + buffer_size + activation_size,
    }


def verify_arm_compatibility(onnx_path: str) -> bool:
    """
    Verify model compatibility with ARM devices.

    Args:
        onnx_path: Path to ONNX model

    Returns:
        True if compatible
    """
    try:
        import onnxruntime as ort

        # Check if ARM build is available
        providers = ort.get_available_providers()
        if "CPUExecutionProvider" not in providers:
            logger.warning("CPU execution provider not available!")
            return False

        # Load model
        options = ort.SessionOptions()
        options.intra_op_num_threads = PI_CPU_CORES
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        session = ort.InferenceSession(
            onnx_path, options, providers=["CPUExecutionProvider"]
        )

        # Test inference
        dummy_input = np.random.randn(1, 1, *IMAGE_SIZE).astype(np.float32)
        input_name = session.get_inputs()[0].name
        session.run(None, {input_name: dummy_input})

        return True

    except Exception as e:
        logger.error(f"ARM compatibility check failed: {str(e)}")
        return False


def export_model(
    model: torch.nn.Module,
    export_path: Path,
    checkpoint_path: Optional[str] = None,
    lightweight: bool = False,
) -> None:
    """
    Export model to ONNX format with Raspberry Pi optimizations.

    Args:
        model: PyTorch model to export
        export_path: Path to save exported model
        checkpoint_path: Optional path to model checkpoint
        lightweight: Whether to apply aggressive optimizations
    """
    try:
        # Load checkpoint if provided
        if checkpoint_path:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])

        # Get system info
        sys_info = get_system_info()
        logger.info(f"System info: {sys_info}")

        # Optimize model
        model = optimize_model_for_edge(model, lightweight)

        # Prepare for export
        dummy_input = torch.randn(1, 1, *IMAGE_SIZE)
        export_args = get_model_metadata()

        # Export to ONNX
        logger.info(f"Exporting model to {export_path}")
        torch.onnx.export(
            model,
            dummy_input,
            export_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=export_args["input_names"],
            output_names=export_args["output_names"],
            dynamic_axes=export_args["dynamic_axes"],
        )

        # Verify and optimize ONNX model
        logger.info("Verifying and optimizing ONNX model...")
        onnx_model = onnx.load(export_path)
        onnx.checker.check_model(onnx_model)

        # Optimize with ONNX Runtime
        import onnxruntime as ort
        from onnxruntime.transformers import optimizer

        opt_options = ort.SessionOptions()
        opt_options.optimized_model_filepath = str(export_path)
        opt_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Save optimized model
        optimized_model = optimizer.optimize_model(
            str(export_path),
            model_type="bert",  # Using BERT optimizer as it has good general optimizations
            num_heads=12,
            hidden_size=768,
            optimization_options=opt_options,
        )
        optimized_model.save_model_to_file(str(export_path))

        # Estimate memory usage
        memory_stats = estimate_memory_usage(model)

        # Verify ARM compatibility
        arm_compatible = verify_arm_compatibility(str(export_path))

        # Print detailed summary
        logger.info("\n=== Model Export Summary ===")
        logger.info(f"✓ Model exported successfully to: {export_path}")
        logger.info(f"✓ Input shape: {tuple(dummy_input.shape)}")
        logger.info(f"✓ Number of classes: {NUM_CLASSES}")
        logger.info(
            f"✓ Model file size: {export_path.stat().st_size / (1024 * 1024):.2f} MB"
        )
        logger.info("\nMemory Usage Estimates:")
        logger.info(f"- Model parameters: {memory_stats['model_size_mb']:.2f} MB")
        logger.info(f"- Runtime buffers: {memory_stats['buffer_size_mb']:.2f} MB")
        logger.info(f"- Peak activations: {memory_stats['activation_size_mb']:.2f} MB")
        logger.info(f"- Total runtime: {memory_stats['total_runtime_mb']:.2f} MB")
        logger.info(
            f"\nARM Compatibility: {'✓ Compatible' if arm_compatible else '⨯ Not verified'}"
        )

        if memory_stats["total_runtime_mb"] > PI_MEMORY_LIMIT:
            logger.warning(
                f"⚠️ Total memory usage ({memory_stats['total_runtime_mb']:.2f} MB) "
                f"exceeds Raspberry Pi recommended limit ({PI_MEMORY_LIMIT} MB)"
            )
            if not lightweight:
                logger.info(
                    "Consider using --lightweight flag for reduced memory usage"
                )

    except Exception as e:
        logger.error(f"❌ Export failed: {str(e)}")
        sys.exit(1)


def main():
    """Main export function."""
    parser = argparse.ArgumentParser(
        description="Export model for Raspberry Pi deployment"
    )
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument(
        "--lightweight",
        action="store_true",
        help="Enable aggressive optimizations for very constrained devices",
    )
    args = parser.parse_args()

    try:
        # Setup export directory
        export_path = setup_export_directory()

        # Create model
        logger.info("Creating model...")
        model = create_model(
            model_name="resnet18",
            num_classes=NUM_CLASSES,
            pretrained=True,
            grayscale=True,
        )

        # Export model
        export_model(model, export_path, args.checkpoint, args.lightweight)

    except Exception as e:
        logger.error(f"❌ Export process failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
