#!/usr/bin/env python3
"""
Image Inference Script for Cutlery Classifier MVP

This script performs inference on single images using trained models.
Supports both terminal output and visual overlay modes.

Usage:
    # Basic inference with terminal output
    python scripts/infer_image.py --model models/checkpoints/type_detector_best.pth --image test.jpg

    # Inference with visual overlay
    python scripts/infer_image.py --model models/checkpoints/type_detector_best.pth --image test.jpg --visualize --output result.jpg

    # Batch inference on multiple images
    python scripts/infer_image.py --model models/checkpoints/type_detector_best.pth --batch data/test_images/ --output results/

    # Show top-3 predictions
    python scripts/infer_image.py --model models/checkpoints/type_detector_best.pth --image test.jpg --top-k 3
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import List

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.inference.inferencer import CutleryInferencer


def print_prediction_results(results: dict, show_details: bool = False):
    """Print prediction results in a formatted way."""
    print("\n" + "=" * 60)
    print("🔍 CUTLERY CLASSIFICATION RESULTS")
    print("=" * 60)

    # Main prediction
    top_pred = results["top_prediction"]
    print(f"🏆 Prediction: {top_pred['class_name'].upper()}")
    print(f"📊 Confidence: {top_pred['percentage']:.1f}%")
    print(f"⚡ Inference Time: {results['inference_time_ms']:.1f}ms")

    # Additional predictions if available
    if len(results["predictions"]) > 1:
        print(f"\n📋 Top {len(results['predictions'])} Predictions:")
        for i, pred in enumerate(results["predictions"], 1):
            confidence_bar = "█" * int(pred["percentage"] / 5)  # Scale to 20 chars max
            print(
                f"  {i}. {pred['class_name']:<15} {pred['percentage']:>6.1f}% {confidence_bar}"
            )

    if show_details:
        print(f"\n🔧 Technical Details:")
        print(f"  Model: {Path(results['model_path']).name}")
        print(f"  Device: {results['device']}")

    print("=" * 60)


def process_single_image(args):
    """Process a single image."""
    logger = logging.getLogger(__name__)

    # Check if image exists
    image_path = Path(args.image)
    if not image_path.exists():
        logger.error(f"Image file not found: {image_path}")
        return

    logger.info(f"Processing image: {image_path}")

    try:
        # Create inferencer
        inferencer = CutleryInferencer(model_path=args.model, device=args.device)

        # Show model info if requested
        if args.info:
            model_info = inferencer.get_model_info()
            print("\n📋 Model Information:")
            print(f"  Architecture: {model_info['architecture']}")
            print(
                f"  Classes: {model_info['num_classes']} ({', '.join(model_info['class_names'])})"
            )
            print(f"  Parameters: {model_info['total_parameters']:,}")
            print(f"  Input Size: {model_info['input_size']}")
            print(f"  Device: {model_info['device']}")

        # Make prediction
        if args.visualize:
            # Create visualization
            output_path = (
                args.output if args.output else f"result_{image_path.stem}.jpg"
            )
            results, vis_image = inferencer.predict_with_visualization(
                image_path,
                output_path=output_path,
                top_k=args.top_k,
                font_size=args.font_size,
            )

            print_prediction_results(results, show_details=args.verbose)
            print(f"\n💾 Visualization saved: {output_path}")

            # Show image if requested
            if args.show:
                try:
                    vis_image.show()
                except Exception as e:
                    logger.warning(f"Could not display image: {e}")

        else:
            # Terminal output only
            results = inferencer.predict(image_path, top_k=args.top_k)
            print_prediction_results(results, show_details=args.verbose)

        # Save results as JSON if requested
        if args.json:
            json_path = (
                args.json if args.json != True else f"result_{image_path.stem}.json"
            )
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"📄 Results saved: {json_path}")

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        if args.verbose:
            raise


def process_batch_images(args):
    """Process multiple images in batch."""
    logger = logging.getLogger(__name__)

    # Get image paths
    batch_path = Path(args.batch)
    if batch_path.is_file():
        # Single file provided
        image_paths = [batch_path]
    elif batch_path.is_dir():
        # Directory provided - find all images
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(batch_path.glob(f"*{ext}"))
            image_paths.extend(batch_path.glob(f"*{ext.upper()}"))
    else:
        logger.error(f"Batch path not found: {batch_path}")
        return

    if not image_paths:
        logger.error(f"No images found in: {batch_path}")
        return

    logger.info(f"Found {len(image_paths)} images for batch processing")

    try:
        # Create inferencer
        inferencer = CutleryInferencer(model_path=args.model, device=args.device)

        # Process batch
        results = inferencer.batch_predict(image_paths, top_k=args.top_k)

        # Print summary
        print("\n" + "=" * 80)
        print("📊 BATCH PROCESSING RESULTS")
        print("=" * 80)

        successful = [r for r in results if "error" not in r]
        failed = [r for r in results if "error" in r]

        print(f"✅ Successful: {len(successful)}/{len(results)}")
        if failed:
            print(f"❌ Failed: {len(failed)}")

        # Show individual results
        for i, result in enumerate(successful, 1):
            image_name = Path(result["image_path"]).name
            top_pred = result["top_prediction"]
            print(
                f"{i:3d}. {image_name:<30} → {top_pred['class_name']:<15} ({top_pred['percentage']:>5.1f}%)"
            )

        # Show errors if any
        if failed and args.verbose:
            print(f"\n❌ Failed Images:")
            for result in failed:
                image_name = Path(result["image_path"]).name
                print(f"  {image_name}: {result['error']}")

        # Save batch results
        if args.output:
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save JSON results
            json_path = output_path / "batch_results.json"
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n📄 Batch results saved: {json_path}")

            # Create visualizations if requested
            if args.visualize:
                print(f"\n🎨 Creating visualizations...")
                for result in successful:
                    try:
                        image_path = result["image_path"]
                        image_name = Path(image_path).stem
                        vis_output = output_path / f"{image_name}_result.jpg"

                        _, vis_image = inferencer.predict_with_visualization(
                            image_path,
                            output_path=str(vis_output),
                            top_k=args.top_k,
                            font_size=args.font_size,
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to create visualization for {image_path}: {e}"
                        )

                print(f"💾 Visualizations saved to: {output_path}")

        print("=" * 80)

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        if args.verbose:
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Perform inference on cutlery images using trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference
  python scripts/infer_image.py --model models/checkpoints/type_detector_best.pth --image test.jpg

  # With visualization
  python scripts/infer_image.py --model models/checkpoints/type_detector_best.pth --image test.jpg --visualize --output result.jpg

  # Batch processing
  python scripts/infer_image.py --model models/checkpoints/type_detector_best.pth --batch data/test_images/ --output results/

  # Show top-3 predictions
  python scripts/infer_image.py --model models/checkpoints/type_detector_best.pth --image test.jpg --top-k 3
        """,
    )

    # Required arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth file)",
    )

    # Input arguments (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str, help="Path to single image file")
    input_group.add_argument(
        "--batch",
        type=str,
        help="Path to directory containing images or text file with image paths",
    )

    # Output arguments
    parser.add_argument(
        "--output", type=str, help="Output path for visualization or batch results"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create visualization with prediction overlay",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display visualization image (requires --visualize)",
    )
    parser.add_argument(
        "--json",
        nargs="?",
        const=True,
        help="Save results as JSON file (optional: specify path)",
    )

    # Prediction arguments
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of top predictions to return (default: 1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto)",
    )

    # Visualization arguments
    parser.add_argument(
        "--font-size",
        type=int,
        default=24,
        help="Font size for text overlay (default: 24)",
    )

    # Information arguments
    parser.add_argument("--info", action="store_true", help="Show model information")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output and technical information",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Validate arguments
    if args.show and not args.visualize:
        logger.error("--show requires --visualize")
        return

    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        logger.error("Make sure you have trained a model first:")
        logger.error("  python scripts/train_type_detector.py")
        return

    # Process based on input type
    if args.image:
        process_single_image(args)
    elif args.batch:
        process_batch_images(args)


if __name__ == "__main__":
    main()
