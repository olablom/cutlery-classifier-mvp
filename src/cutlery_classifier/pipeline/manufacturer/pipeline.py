"""
Manufacturer-specific Pipeline for Cutlery Classification

This module provides a pipeline for manufacturer-specific cutlery classification,
integrating preprocessing, inference, and result handling.
"""

import torch
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging

from ...inference.inferencer import CutleryInferencer
from ...data.transforms import create_transform_from_config

logger = logging.getLogger(__name__)


class CutleryPipeline:
    """
    Pipeline for manufacturer-specific cutlery classification.

    Integrates preprocessing, model inference, and result handling
    for manufacturer-specific cutlery classification tasks.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize pipeline.

        Args:
            model_path: Path to trained model checkpoint
            config: Model configuration
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_path = Path(model_path) if model_path else None
        self.config = config

        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Initialize components
        self.inferencer = None
        if self.model_path:
            self.setup_inferencer()

    def setup_inferencer(self):
        """Setup inference engine."""
        if not self.model_path:
            raise ValueError("Model path not provided")

        self.inferencer = CutleryInferencer(
            str(self.model_path), config=self.config, device=str(self.device)
        )
        logger.info("Inference engine initialized")

    def process_image(
        self, image_path: Union[str, Path], top_k: int = 3, visualize: bool = False
    ) -> Dict[str, Any]:
        """
        Process single image through pipeline.

        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return
            visualize: Whether to generate visualization

        Returns:
            Dictionary with prediction results
        """
        if not self.inferencer:
            raise RuntimeError("Pipeline not initialized. Call setup_inferencer first.")

        if visualize:
            result, viz = self.inferencer.predict_with_visualization(
                image_path, top_k=top_k
            )
            return {"predictions": result, "visualization": viz}
        else:
            result = self.inferencer.predict(image_path, top_k=top_k)
            return {"predictions": result}

    def batch_process(
        self, image_paths: List[Union[str, Path]], top_k: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Process multiple images through pipeline.

        Args:
            image_paths: List of image paths
            top_k: Number of top predictions per image

        Returns:
            List of prediction results
        """
        if not self.inferencer:
            raise RuntimeError("Pipeline not initialized. Call setup_inferencer first.")

        return self.inferencer.batch_predict(image_paths, top_k=top_k)

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about pipeline components."""
        info = {
            "device": str(self.device),
            "model_path": str(self.model_path) if self.model_path else None,
        }

        if self.inferencer:
            info.update(self.inferencer.get_model_info())

        return info
