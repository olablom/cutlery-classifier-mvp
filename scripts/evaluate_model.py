#!/usr/bin/env python3
"""
Evaluate Trained Model for Cutlery Classifier MVP

This script evaluates a trained model and generates comprehensive analysis including:
- Confusion matrix
- Classification report
- Grad-CAM visualizations
- Performance metrics

Usage:
    python scripts/evaluate_model.py --model models/checkpoints/type_detector_best.pth
    python scripts/evaluate_model.py --model models/checkpoints/type_detector_best.pth --samples 20
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.evaluation.evaluator import CutleryEvaluator


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained cutlery classifier model"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of samples for Grad-CAM visualization",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--name", type=str, default=None, help="Custom name for saving results"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Check if model file exists
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        logger.error("Make sure you have trained a model first:")
        logger.error("  python scripts/train_type_detector.py")
        return

    # Check if test data exists
    data_dir = project_root / "data" / "processed" / "test"
    if not data_dir.exists():
        logger.error(f"Test data directory not found: {data_dir}")
        logger.error("Please run: python scripts/prepare_dataset.py --create-splits")
        return

    logger.info(f"Evaluating model: {model_path}")
    logger.info(f"Grad-CAM samples: {args.samples}")

    try:
        # Create evaluator
        evaluator = CutleryEvaluator(model_path=str(model_path), device=args.device)

        # Determine model name for saving results
        model_name = args.name if args.name else model_path.stem

        # Run full evaluation
        results = evaluator.run_full_evaluation(model_name=model_name)

        # Print summary
        logger.info("\n" + "=" * 50)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Model: {model_name}")
        logger.info(f"Classes: {evaluator.class_names}")
        logger.info(
            f"Test Accuracy: {results['accuracy']:.4f} ({results['accuracy'] * 100:.2f}%)"
        )
        logger.info(f"Test Precision: {results['precision']:.4f}")
        logger.info(f"Test Recall: {results['recall']:.4f}")
        logger.info(f"Test F1-Score: {results['f1_score']:.4f}")

        logger.info("\nPer-class results:")
        for i, class_name in enumerate(evaluator.class_names):
            precision = results["per_class_metrics"]["precision"][i]
            recall = results["per_class_metrics"]["recall"][i]
            f1 = results["per_class_metrics"]["f1_score"][i]
            support = results["per_class_metrics"]["support"][i]
            logger.info(
                f"  {class_name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, Support={support}"
            )

        logger.info("\nResults saved to:")
        logger.info(f"  Confusion matrices: results/confusion_matrices/")
        logger.info(f"  Grad-CAM visualizations: results/grad_cam/")
        logger.info(f"  Metrics and reports: results/metrics/")

        # Check if accuracy meets target
        target_accuracy = 0.80  # 80% target for MVP
        if results["accuracy"] >= target_accuracy:
            logger.info(
                f"\n🎉 SUCCESS: Model achieves target accuracy of {target_accuracy * 100:.0f}%!"
            )
        else:
            logger.info(
                f"\n⚠️  Model accuracy ({results['accuracy'] * 100:.1f}%) below target ({target_accuracy * 100:.0f}%)"
            )
            logger.info("Consider:")
            logger.info("  - Collecting more training data")
            logger.info("  - Adjusting hyperparameters")
            logger.info("  - Using data augmentation")

    except FileNotFoundError as e:
        logger.error(f"Required files not found: {e}")
        logger.error("Make sure you have:")
        logger.error("1. Trained a model: python scripts/train_type_detector.py")
        logger.error(
            "2. Prepared test data: python scripts/prepare_dataset.py --create-splits"
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
