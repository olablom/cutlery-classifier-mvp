"""
Grad-CAM implementation for model interpretability.
Based on the paper "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
https://arxiv.org/abs/1610.02391
"""

import logging
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Grad-CAM implementation for model interpretability.
    Generates class activation maps for CNN predictions.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """Initialize GradCAM.

        Args:
            model: The model to analyze
            target_layer: The target layer to compute GradCAM for
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Save activations during forward pass."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Save gradients during backward pass."""
        self.gradients = grad_output[0].detach()

    def __call__(
        self, input_tensor: torch.Tensor, target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for the input image.

        Args:
            input_tensor: Input image tensor (N, C, H, W)
            target_class: Target class index. If None, uses the predicted class.

        Returns:
            Numpy array containing the Grad-CAM heatmap
        """
        # Forward pass
        model_output = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(model_output)

        # Zero gradients
        self.model.zero_grad()

        # Backward pass
        class_loss = model_output[0, target_class]
        class_loss.backward()

        # Generate GradCAM
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = torch.relu(heatmap)
        heatmap = heatmap / torch.max(heatmap)

        return heatmap.cpu().numpy()

    def overlay_heatmap(self, heatmap, original_image, alpha=0.5):
        """Overlay heatmap on original image.

        Args:
            heatmap: GradCAM heatmap
            original_image: Original image
            alpha: Transparency of heatmap overlay

        Returns:
            PIL.Image: Image with overlaid heatmap
        """
        # Resize heatmap to match image size
        heatmap = Image.fromarray(np.uint8(255 * heatmap))
        heatmap = heatmap.resize(original_image.size, Image.LANCZOS)
        heatmap = np.array(heatmap)

        # Apply colormap
        colormap = plt.get_cmap("jet")
        heatmap = colormap(heatmap)[:, :, :3]
        heatmap = np.uint8(255 * heatmap)

        # Convert original image to numpy array
        original_array = np.array(original_image)

        # Overlay heatmap
        overlaid = np.uint8(original_array * (1 - alpha) + heatmap * alpha)

        return Image.fromarray(overlaid)
