"""
Model Factory for Cutlery Classifier MVP

This module provides a factory function to create different model architectures
with configurable parameters for the hierarchical cutlery classification system.

Supported architectures:
- ResNet18 (for type detection)
- MobileNetV2 (for manufacturer classification)

Features:
- Grayscale input support (1 channel)
- Transfer learning with pretrained weights
- Configurable number of output classes
- Layer freezing for fine-tuning
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


def create_model(
    model_config, num_classes=3, pretrained=True, grayscale=True, freeze_backbone=False
):
    """Create a model with specified configuration.

    Args:
        model_config (Union[str, Dict]): Either model name as string or config dict
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        grayscale (bool): Whether input images are grayscale
        freeze_backbone (bool): Whether to freeze backbone layers

    Returns:
        torch.nn.Module: The configured model
    """
    # Handle dict input
    if isinstance(model_config, dict):
        model_name = model_config.get("architecture")
        num_classes = model_config.get("num_classes", num_classes)
        pretrained = model_config.get("pretrained", pretrained)
        grayscale = model_config.get("grayscale", grayscale)
        freeze_backbone = model_config.get("freeze_backbone", freeze_backbone)
    else:
        model_name = model_config

    logger.info(f"Creating {model_name} model with {num_classes} classes")
    logger.info(
        f"Pretrained: {pretrained}, Grayscale: {grayscale}, Freeze: {freeze_backbone}"
    )

    # Create model with pretrained weights if specified
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)

        # Modify first conv layer for grayscale if needed
        if grayscale:
            original_conv = model.conv1
            model.conv1 = torch.nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            if pretrained:
                model.conv1.weight.data = original_conv.weight.data.sum(
                    dim=1, keepdim=True
                )

        # Modify final layer for our number of classes
        in_features = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5), torch.nn.Linear(in_features, num_classes)
        )

        # Freeze backbone if specified
        if freeze_backbone:
            for param in list(model.parameters())[:-2]:
                param.requires_grad = False
    elif model_name == "mobilenet_v2":
        model = _create_mobilenet_v2(
            num_classes=num_classes,
            pretrained=pretrained,
            grayscale=grayscale,
            freeze_backbone=freeze_backbone,
        )
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")

    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model created: {total_params:,} total params, {trainable_params:,} trainable"
    )

    return model


def _create_mobilenet_v2(
    num_classes: int,
    pretrained: bool = True,
    grayscale: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Create MobileNetV2 model with custom configuration."""

    # Load pretrained MobileNetV2
    model = models.mobilenet_v2(pretrained=pretrained)

    # Modify first layer for grayscale input if needed
    if grayscale:
        # Replace first conv layer to accept 1 channel instead of 3
        original_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias,
        )

        # If pretrained, initialize new conv layer with averaged weights
        if pretrained:
            with torch.no_grad():
                # Average the RGB weights to create grayscale weights
                model.features[0][0].weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )

    # Freeze backbone if requested
    if freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:  # Don't freeze final classifier
                param.requires_grad = False
        logger.info("Backbone layers frozen for fine-tuning")

    # Replace final classifier
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5), nn.Linear(num_features, num_classes)
    )

    return model


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get information about a model.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": total_params - trainable_params,
        "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
    }


def test_model_creation():
    """Test function to verify model creation works correctly."""

    # Test ResNet18 for type detection
    type_config = {
        "model_name": "resnet18",
        "num_classes": 3,
        "pretrained": True,
        "grayscale": True,
        "freeze_backbone": False,
    }

    # Test MobileNetV2 for manufacturer classification
    manufacturer_config = {
        "model_name": "mobilenet_v2",
        "num_classes": 3,
        "pretrained": True,
        "grayscale": True,
        "freeze_backbone": False,
    }

    print("Testing model creation...")

    # Create models
    type_model = create_model(**type_config)
    manufacturer_model = create_model(**manufacturer_config)

    # Test forward pass with dummy data
    dummy_input = torch.randn(1, 1, 320, 320)  # Batch=1, Channels=1, H=320, W=320

    with torch.no_grad():
        type_output = type_model(dummy_input)
        manufacturer_output = manufacturer_model(dummy_input)

    print(f"Type model output shape: {type_output.shape}")
    print(f"Manufacturer model output shape: {manufacturer_output.shape}")

    # Print model info
    type_info = get_model_info(type_model)
    manufacturer_info = get_model_info(manufacturer_model)

    print(
        f"\nType model (ResNet18): {type_info['trainable_parameters']:,} trainable params"
    )
    print(
        f"Manufacturer model (MobileNetV2): {manufacturer_info['trainable_parameters']:,} trainable params"
    )

    print("✅ Model creation test passed!")


if __name__ == "__main__":
    # Run test when script is executed directly
    logging.basicConfig(level=logging.INFO)
    test_model_creation()
