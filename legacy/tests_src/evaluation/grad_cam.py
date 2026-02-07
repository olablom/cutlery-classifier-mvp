"""
Grad-CAM implementation for model interpretability.
Based on the paper "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
https://arxiv.org/abs/1610.02391
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple


class GradCAM:
    """
    Grad-CAM implementation for model interpretability.
    Generates class activation maps for CNN predictions.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: List[torch.Tensor] = []
        self.activations: List[torch.Tensor] = []

        # Register hooks
        self.register_hooks()

    def register_hooks(self):
        """Register forward and backward hooks on the target layer."""

        def forward_hook(
            module: torch.nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor
        ):
            self.activations.append(output)

        def backward_hook(
            module: torch.nn.Module,
            grad_input: Tuple[torch.Tensor],
            grad_output: Tuple[torch.Tensor],
        ):
            self.gradients.append(grad_output[0])

        # Register the hooks
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

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
        # Clear any previous gradients and activations
        self.gradients = []
        self.activations = []

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero all gradients
        self.model.zero_grad()

        # Backward pass for target class
        target = output[0, target_class]
        target.backward()

        # Get gradients and activations
        gradients = self.gradients[0]
        activations = self.activations[0]

        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

        # Weight the activations by the gradients
        cam = torch.sum(weights * activations, dim=1, keepdim=True)

        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = F.interpolate(
            cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False
        )

        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Convert to numpy array
        return cam[0, 0].detach().cpu().numpy()
