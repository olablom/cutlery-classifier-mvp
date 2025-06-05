"""
Grad-CAM visualization utilities for model interpretability.
"""

import logging
from pathlib import Path
import numpy as np
import cv2
from datetime import datetime
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

logger = logging.getLogger(__name__)

# Constants
CLASS_LABELS = ["fork", "knife", "spoon"]


def generate_grad_cam(
    model, image_tensor, image_for_cam, predicted_class, device
) -> None:
    """Generate and save Grad-CAM visualization."""
    # Get the final convolutional layer
    target_layer = model.layer4[-1]

    # Get target class index
    target_class = CLASS_LABELS.index(predicted_class)
    targets = [ClassifierOutputTarget(target_class)]

    # Initialize and run GradCAM using context manager
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        # Generate grayscale CAM
        cam = cam(input_tensor=image_tensor.to(device), targets=targets)
        cam = cam[0]  # Get first (only) image in batch

    # Handle grayscale → RGB conversion for show_cam_on_image
    if image_for_cam.ndim == 2:
        image_for_cam_rgb = np.stack([image_for_cam] * 3, axis=-1)
    else:
        image_for_cam_rgb = image_for_cam

    # Create heatmap overlay
    visualization = show_cam_on_image(image_for_cam_rgb, cam, use_rgb=True)

    # Save visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"results/grad_cam_{predicted_class}_{timestamp}.jpg"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    logger.info(f"Grad-CAM visualization saved to: {output_path}")
