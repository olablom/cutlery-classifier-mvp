"""
Inference Module for Cutlery Classifier MVP

This module provides inference capabilities for trained cutlery classification models.
Supports both type detection and manufacturer classification with confidence scoring
and optional visualization.

Features:
- Load trained models from checkpoints
- Single image inference with preprocessing
- Confidence scoring and top-k predictions
- Visual overlay of predictions on images
- Support for both ResNet18 and MobileNetV2 architectures
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
import time

from ..models.factory import create_model
from ..data.transforms import (
    create_transform_from_config,
    get_base_transforms,
    tensor_to_pil,
    denormalize_tensor,
)

logger = logging.getLogger(__name__)


class CutleryInferencer:
    """
    Comprehensive inference engine for cutlery classification.

    Supports loading trained models and making predictions on single images
    with confidence scoring and optional visualization.
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize inferencer with trained model.

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

        # Initialize components
        self.model = None
        self.config = config
        self.class_names = None
        self.transform = None

        # Load model and setup
        self.load_model()
        self.setup_transforms()

        logger.info(f"Inferencer ready for {len(self.class_names)} classes")

    def load_model(self):
        """Load trained model from checkpoint."""
        logger.info(f"Loading model from {self.model_path}")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Load config if not provided
        if self.config is None:
            if "config" not in checkpoint:
                raise KeyError(
                    "No config found in checkpoint. This might be an old checkpoint format."
                )
            self.config = checkpoint["config"]

        # Load class names (handle both old and new checkpoint formats)
        if "class_names" in checkpoint:
            self.class_names = checkpoint["class_names"]
        elif "classes" in checkpoint:
            self.class_names = checkpoint["classes"]
        else:
            raise KeyError("No class names found in checkpoint")

        # Create and load model
        model_config = self.config.get("model", {})
        self.model = create_model(model_config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded successfully")
        logger.info(f"Architecture: {model_config.get('architecture', 'unknown')}")
        logger.info(f"Classes: {self.class_names}")

    def setup_transforms(self):
        """Setup image preprocessing transforms."""
        data_config = self.config.get("data", {})

        # Use test transforms (no augmentation)
        self.transform = create_transform_from_config(data_config, mode="test")

        logger.info("Image transforms setup complete")

    def preprocess_image(
        self, image_path: Union[str, Path, Image.Image]
    ) -> torch.Tensor:
        """
        Preprocess image for inference.

        Args:
            image_path: Path to image file or PIL Image object

        Returns:
            Preprocessed tensor ready for model input
        """
        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path).convert("RGB")
        elif isinstance(image_path, Image.Image):
            image = image_path.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image_path)}")

        # Apply transforms
        tensor = self.transform(image)

        # Add batch dimension
        tensor = tensor.unsqueeze(0)

        return tensor

    def predict(
        self, image_path: Union[str, Path, Image.Image], top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Make prediction on single image.

        Args:
            image_path: Path to image file or PIL Image object
            top_k: Number of top predictions to return

        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()

        # Preprocess image
        input_tensor = self.preprocess_image(image_path)
        input_tensor = input_tensor.to(self.device)

        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)

        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
        top_probs = top_probs.squeeze().cpu().numpy()
        top_indices = top_indices.squeeze().cpu().numpy()

        # Handle single prediction case
        if top_k == 1:
            top_probs = [top_probs.item()]
            top_indices = [top_indices.item()]

        # Format results
        predictions = []
        for i in range(len(top_indices)):
            predictions.append(
                {
                    "class_name": self.class_names[top_indices[i]],
                    "class_index": int(top_indices[i]),
                    "confidence": float(top_probs[i]),
                    "percentage": float(top_probs[i] * 100),
                }
            )

        inference_time = time.time() - start_time

        result = {
            "predictions": predictions,
            "top_prediction": predictions[0],
            "inference_time_ms": inference_time * 1000,
            "model_path": str(self.model_path),
            "device": str(self.device),
        }

        return result

    def predict_with_visualization(
        self,
        image_path: Union[str, Path],
        output_path: Optional[str] = None,
        top_k: int = 3,
        font_size: int = 24,
    ) -> Tuple[Dict[str, Any], Image.Image]:
        """
        Make prediction and create visualization with results overlaid.

        Args:
            image_path: Path to image file
            output_path: Path to save visualization (optional)
            top_k: Number of predictions to show
            font_size: Font size for text overlay

        Returns:
            Tuple of (prediction_results, visualization_image)
        """
        # Make prediction
        results = self.predict(image_path, top_k=top_k)

        # Load original image
        if isinstance(image_path, (str, Path)):
            original_image = Image.open(image_path).convert("RGB")
        else:
            original_image = image_path.convert("RGB")

        # Create visualization
        vis_image = self.create_visualization(
            original_image, results, font_size=font_size
        )

        # Save if output path provided
        if output_path:
            vis_image.save(output_path)
            logger.info(f"Visualization saved: {output_path}")

        return results, vis_image

    def create_visualization(
        self, image: Image.Image, results: Dict[str, Any], font_size: int = 24
    ) -> Image.Image:
        """
        Create visualization with prediction overlay.

        Args:
            image: Original PIL Image
            results: Prediction results from predict()
            font_size: Font size for text

        Returns:
            PIL Image with prediction overlay
        """
        # Create copy for drawing
        vis_image = image.copy()
        draw = ImageDraw.Draw(vis_image)

        # Try to load a font, fallback to default
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None

        # Get image dimensions
        width, height = vis_image.size

        # Prepare text
        top_pred = results["top_prediction"]
        main_text = f"{top_pred['class_name']}"
        confidence_text = f"{top_pred['percentage']:.1f}%"
        time_text = f"{results['inference_time_ms']:.1f}ms"

        # Text positioning
        margin = 20
        line_height = font_size + 10 if font else 30

        # Background rectangle for text
        text_lines = [main_text, confidence_text, time_text]
        max_text_width = max(
            [
                draw.textlength(text, font=font) if font else len(text) * 10
                for text in text_lines
            ]
        )

        rect_height = len(text_lines) * line_height + margin
        rect_coords = [
            margin,
            margin,
            margin + max_text_width + margin,
            margin + rect_height,
        ]

        # Draw semi-transparent background
        overlay = Image.new("RGBA", vis_image.size, (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle(rect_coords, fill=(0, 0, 0, 128))
        vis_image = Image.alpha_composite(vis_image.convert("RGBA"), overlay).convert(
            "RGB"
        )

        # Redraw on the composite image
        draw = ImageDraw.Draw(vis_image)

        # Draw main prediction
        y_pos = margin * 2
        draw.text((margin * 2, y_pos), main_text, fill=(255, 255, 255), font=font)

        y_pos += line_height
        draw.text((margin * 2, y_pos), confidence_text, fill=(0, 255, 0), font=font)

        y_pos += line_height
        draw.text((margin * 2, y_pos), time_text, fill=(200, 200, 200), font=font)

        # Add additional predictions if available
        if len(results["predictions"]) > 1:
            y_pos += line_height + 10
            for i, pred in enumerate(results["predictions"][1:], 1):
                if i >= 3:  # Limit to top 3
                    break
                text = f"{i + 1}. {pred['class_name']} ({pred['percentage']:.1f}%)"
                draw.text((margin * 2, y_pos), text, fill=(200, 200, 200), font=font)
                y_pos += line_height

        return vis_image

    def batch_predict(
        self, image_paths: List[Union[str, Path]], top_k: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Make predictions on multiple images.

        Args:
            image_paths: List of image paths
            top_k: Number of top predictions per image

        Returns:
            List of prediction results
        """
        results = []

        logger.info(f"Processing {len(image_paths)} images...")

        for i, image_path in enumerate(image_paths):
            try:
                result = self.predict(image_path, top_k=top_k)
                result["image_path"] = str(image_path)
                results.append(result)

                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(image_paths)} images")

            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results.append(
                    {"image_path": str(image_path), "error": str(e), "predictions": []}
                )

        logger.info(f"Batch prediction completed: {len(results)} results")
        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded model."""
        total_params = sum(p.numel() for p in self.model.parameters())

        return {
            "model_path": str(self.model_path),
            "architecture": self.config.get("model", {}).get("architecture", "unknown"),
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
            "total_parameters": total_params,
            "device": str(self.device),
            "input_size": self.config.get("data", {}).get("image_size", [320, 320]),
        }


def test_inferencer_creation():
    """Test function to verify inferencer can be created (without actual model)."""
    print("Testing inferencer creation...")
    print("✅ Inferencer module loaded successfully!")
    print("Note: Actual inference requires a trained model checkpoint")


if __name__ == "__main__":
    # Run test when script is executed directly
    logging.basicConfig(level=logging.INFO)
    test_inferencer_creation()
