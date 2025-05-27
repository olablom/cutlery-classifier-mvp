#!/usr/bin/env python3
"""
Model Export Script for Cutlery Classifier MVP

This script exports trained PyTorch models to various formats for deployment:
- ONNX format for cross-platform inference
- TorchScript for PyTorch mobile deployment
- Optimized versions for edge devices

Usage:
    # Export to ONNX
    python scripts/export_model.py --model models/checkpoints/type_detector_best.pth --format onnx

    # Export to TorchScript
    python scripts/export_model.py --model models/checkpoints/type_detector_best.pth --format torchscript

    # Export both formats
    python scripts/export_model.py --model models/checkpoints/type_detector_best.pth --format all

    # Export with optimization
    python scripts/export_model.py --model models/checkpoints/type_detector_best.pth --format onnx --optimize
"""

import argparse
import logging
import sys
import torch
import torch.onnx
from pathlib import Path
import time

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.factory import create_model


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    input_size: tuple = (1, 1, 320, 320),
    optimize: bool = False,
    opset_version: int = 11,
):
    """
    Export model to ONNX format.

    Args:
        model: PyTorch model to export
        output_path: Path to save ONNX model
        input_size: Input tensor size (batch, channels, height, width)
        optimize: Whether to optimize the ONNX model
        opset_version: ONNX opset version
    """
    logger = logging.getLogger(__name__)

    model.eval()

    # Create dummy input
    dummy_input = torch.randn(input_size)

    logger.info(f"Exporting to ONNX: {output_path}")
    logger.info(f"Input size: {input_size}")
    logger.info(f"Opset version: {opset_version}")

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=optimize,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    # Verify the exported model
    try:
        import onnx

        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        logger.info("✅ ONNX model verification passed")

        # Get model size
        file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
        logger.info(f"📦 ONNX model size: {file_size:.2f} MB")

    except ImportError:
        logger.warning("ONNX package not available for verification")
    except Exception as e:
        logger.error(f"ONNX model verification failed: {e}")


def export_to_torchscript(
    model: torch.nn.Module,
    output_path: str,
    input_size: tuple = (1, 1, 320, 320),
    method: str = "trace",
):
    """
    Export model to TorchScript format.

    Args:
        model: PyTorch model to export
        output_path: Path to save TorchScript model
        input_size: Input tensor size for tracing
        method: Export method ('trace' or 'script')
    """
    logger = logging.getLogger(__name__)

    model.eval()

    logger.info(f"Exporting to TorchScript: {output_path}")
    logger.info(f"Method: {method}")

    if method == "trace":
        # Create dummy input for tracing
        dummy_input = torch.randn(input_size)

        # Trace the model
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(output_path)

    elif method == "script":
        # Script the model
        scripted_model = torch.jit.script(model)
        scripted_model.save(output_path)

    else:
        raise ValueError(f"Unsupported TorchScript method: {method}")

    # Get model size
    file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
    logger.info(f"📦 TorchScript model size: {file_size:.2f} MB")
    logger.info("✅ TorchScript export completed")


def test_exported_model(
    original_model: torch.nn.Module,
    exported_path: str,
    format_type: str,
    input_size: tuple = (1, 1, 320, 320),
    tolerance: float = 1e-5,
):
    """
    Test exported model against original model.

    Args:
        original_model: Original PyTorch model
        exported_path: Path to exported model
        format_type: Export format ('onnx' or 'torchscript')
        input_size: Input tensor size for testing
        tolerance: Numerical tolerance for comparison
    """
    logger = logging.getLogger(__name__)

    logger.info(f"Testing exported {format_type} model...")

    # Create test input
    test_input = torch.randn(input_size)

    # Get original model output
    original_model.eval()
    with torch.no_grad():
        original_output = original_model(test_input)

    # Test exported model
    if format_type == "onnx":
        try:
            import onnxruntime as ort

            # Create ONNX Runtime session
            ort_session = ort.InferenceSession(exported_path)

            # Run inference
            ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
            ort_outputs = ort_session.run(None, ort_inputs)
            exported_output = torch.tensor(ort_outputs[0])

        except ImportError:
            logger.warning("ONNX Runtime not available for testing")
            return

    elif format_type == "torchscript":
        # Load TorchScript model
        loaded_model = torch.jit.load(exported_path)
        loaded_model.eval()

        with torch.no_grad():
            exported_output = loaded_model(test_input)

    else:
        raise ValueError(f"Unsupported format for testing: {format_type}")

    # Compare outputs
    max_diff = torch.max(torch.abs(original_output - exported_output)).item()

    if max_diff < tolerance:
        logger.info(f"✅ Model test passed (max diff: {max_diff:.2e})")
    else:
        logger.warning(
            f"⚠️ Model test failed (max diff: {max_diff:.2e}, tolerance: {tolerance:.2e})"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Export trained cutlery classifier models to deployment formats"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["onnx", "torchscript", "all"],
        default="onnx",
        help="Export format (default: onnx)",
    )
    parser.add_argument(
        "--output", type=str, help="Output directory (default: models/exports/)"
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Custom name for exported model (default: use model filename)",
    )
    parser.add_argument(
        "--optimize", action="store_true", help="Apply optimizations during export"
    )
    parser.add_argument(
        "--test", action="store_true", help="Test exported model against original"
    )
    parser.add_argument(
        "--opset-version", type=int, default=11, help="ONNX opset version (default: 11)"
    )
    parser.add_argument(
        "--torchscript-method",
        type=str,
        choices=["trace", "script"],
        default="trace",
        help="TorchScript export method (default: trace)",
    )
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return

    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = project_root / "models" / "exports"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine model name
    if args.name:
        model_name = args.name
    else:
        model_name = model_path.stem

    logger.info(f"Loading model: {model_path}")

    try:
        # Load model
        checkpoint = torch.load(model_path, map_location="cpu")
        config = checkpoint["config"]
        class_names = checkpoint["class_names"]

        # Create model
        model_config = config.get("model", {})
        model = create_model(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Get input size from config
        data_config = config.get("data", {})
        image_size = data_config.get("image_size", [320, 320])
        input_size = (1, 1, image_size[0], image_size[1])  # Grayscale

        logger.info(f"Model loaded successfully")
        logger.info(f"Architecture: {model_config.get('architecture', 'unknown')}")
        logger.info(f"Classes: {len(class_names)} ({', '.join(class_names)})")
        logger.info(f"Input size: {input_size}")

        # Export based on format
        exported_files = []

        if args.format in ["onnx", "all"]:
            onnx_path = output_dir / f"{model_name}.onnx"

            start_time = time.time()
            export_to_onnx(
                model,
                str(onnx_path),
                input_size=input_size,
                optimize=args.optimize,
                opset_version=args.opset_version,
            )
            export_time = time.time() - start_time

            logger.info(f"⏱️ ONNX export time: {export_time:.2f}s")
            exported_files.append(("onnx", str(onnx_path)))

            # Test ONNX model if requested
            if args.test:
                test_exported_model(model, str(onnx_path), "onnx", input_size)

        if args.format in ["torchscript", "all"]:
            ts_path = output_dir / f"{model_name}.pt"

            start_time = time.time()
            export_to_torchscript(
                model,
                str(ts_path),
                input_size=input_size,
                method=args.torchscript_method,
            )
            export_time = time.time() - start_time

            logger.info(f"⏱️ TorchScript export time: {export_time:.2f}s")
            exported_files.append(("torchscript", str(ts_path)))

            # Test TorchScript model if requested
            if args.test:
                test_exported_model(model, str(ts_path), "torchscript", input_size)

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("📦 EXPORT SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Original model: {model_path}")
        logger.info(f"Output directory: {output_dir}")

        for format_type, file_path in exported_files:
            file_size = Path(file_path).stat().st_size / (1024 * 1024)
            logger.info(f"✅ {format_type.upper()}: {file_path} ({file_size:.2f} MB)")

        logger.info("=" * 60)

        # Usage examples
        if exported_files:
            logger.info("\n🚀 Usage Examples:")
            for format_type, file_path in exported_files:
                if format_type == "onnx":
                    logger.info(f"  ONNX Runtime: ort.InferenceSession('{file_path}')")
                elif format_type == "torchscript":
                    logger.info(f"  TorchScript: torch.jit.load('{file_path}')")

    except Exception as e:
        logger.error(f"Export failed: {e}")
        if args.verbose:
            raise


if __name__ == "__main__":
    main()
