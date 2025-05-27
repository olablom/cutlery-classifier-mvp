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
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def create_model(config: Dict[str, Any]) -> nn.Module:
    """
    Create a model based on configuration parameters.

    Args:
        config: Dictionary containing model configuration with keys:
            - architecture: str, model architecture ('resnet18' or 'mobilenet_v2')
            - num_classes: int, number of output classes
            - pretrained: bool, whether to use pretrained weights
            - grayscale: bool, whether input is grayscale (1 channel)
            - freeze_backbone: bool, whether to freeze backbone layers
            - dropout_rate: float, dropout rate for classifier

    Returns:
        nn.Module: Configured PyTorch model

    Raises:
        ValueError: If unsupported architecture is specified
    """
    architecture = config.get("architecture", "resnet18").lower()
    num_classes = config.get("num_classes", 3)
    pretrained = config.get("pretrained", True)
    grayscale = config.get("grayscale", True)
    freeze_backbone = config.get("freeze_backbone", False)
    dropout_rate = config.get("dropout_rate", 0.5)

    logger.info(f"Creating {architecture} model with {num_classes} classes")
    logger.info(
        f"Pretrained: {pretrained}, Grayscale: {grayscale}, Freeze: {freeze_backbone}"
    )

    if architecture == "resnet18":
        model = _create_resnet18(
            num_classes=num_classes,
            pretrained=pretrained,
            grayscale=grayscale,
            freeze_backbone=freeze_backbone,
            dropout_rate=dropout_rate,
        )
    elif architecture == "mobilenet_v2":
        model = _create_mobilenet_v2(
            num_classes=num_classes,
            pretrained=pretrained,
            grayscale=grayscale,
            freeze_backbone=freeze_backbone,
            dropout_rate=dropout_rate,
        )
    else:
        raise ValueError(
            f"Unsupported architecture: {architecture}. "
            f"Supported: ['resnet18', 'mobilenet_v2']"
        )

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model created: {total_params:,} total params, {trainable_params:,} trainable"
    )

    return model


def _create_resnet18(
    num_classes: int,
    pretrained: bool = True,
    grayscale: bool = True,
    freeze_backbone: bool = False,
    dropout_rate: float = 0.5,
) -> nn.Module:
    """Create ResNet18 model with custom configuration."""

    # Load pretrained ResNet18
    model = models.resnet18(pretrained=pretrained)

    # Modify first layer for grayscale input if needed
    if grayscale:
        # Replace first conv layer to accept 1 channel instead of 3
        original_conv = model.conv1
        model.conv1 = nn.Conv2d(
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
                model.conv1.weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )

    # Freeze backbone if requested
    if freeze_backbone:
        for name, param in model.named_parameters():
            if "fc" not in name:  # Don't freeze final classifier
                param.requires_grad = False
        logger.info("Backbone layers frozen for fine-tuning")

    # Replace final classifier
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate), nn.Linear(num_features, num_classes)
    )

    return model


def _create_mobilenet_v2(
    num_classes: int,
    pretrained: bool = True,
    grayscale: bool = True,
    freeze_backbone: bool = False,
    dropout_rate: float = 0.5,
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
        nn.Dropout(dropout_rate), nn.Linear(num_features, num_classes)
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
        "architecture": "resnet18",
        "num_classes": 3,
        "pretrained": True,
        "grayscale": True,
        "freeze_backbone": False,
        "dropout_rate": 0.5,
    }

    # Test MobileNetV2 for manufacturer classification
    manufacturer_config = {
        "architecture": "mobilenet_v2",
        "num_classes": 3,
        "pretrained": True,
        "grayscale": True,
        "freeze_backbone": False,
        "dropout_rate": 0.3,
    }

    print("Testing model creation...")

    # Create models
    type_model = create_model(type_config)
    manufacturer_model = create_model(manufacturer_config)

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
