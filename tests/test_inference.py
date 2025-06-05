"""
Test suite for model inference and ONNX export validation.

This module tests:
1. Model export to ONNX format
2. ONNX model validation
3. Inference using ONNX Runtime
4. Input/output shape consistency
5. Performance benchmarking
"""

import os
import pytest
import torch
import numpy as np
import onnx
import onnxruntime
from pathlib import Path
from PIL import Image
from typing import Tuple, Dict

from src.models.factory import create_model
from src.data.transforms import get_base_transforms
from src.inference.inferencer import CutleryClassifier

# Constants
TEST_IMAGE_SIZE = (320, 320)
NUM_CLASSES = 3
ONNX_EXPORT_PATH = "models/exports/cutlery_classifier.onnx"
TEST_BATCH_SIZE = 1


def get_test_input() -> Tuple[torch.Tensor, Dict]:
    """Create a test input tensor and sample metadata."""
    # Create dummy input (black image)
    dummy_input = torch.zeros((TEST_BATCH_SIZE, 1, *TEST_IMAGE_SIZE))

    # Create sample metadata
    input_names = ["input"]
    output_names = ["output"]
    dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

    return dummy_input, {
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_axes": dynamic_axes,
    }


def test_model_creation():
    """Test that the model can be created successfully."""
    model = create_model(
        model_name="resnet18", num_classes=NUM_CLASSES, pretrained=True, grayscale=True
    )
    assert model is not None, "Model creation failed"

    # Verify model architecture
    dummy_input = torch.randn(1, 1, *TEST_IMAGE_SIZE)
    output = model(dummy_input)
    assert output.shape == (1, NUM_CLASSES), (
        f"Expected output shape (1, {NUM_CLASSES}), got {output.shape}"
    )


def test_model_export():
    """Test ONNX export and validation."""
    # Create and prepare model
    model = create_model(
        model_name="resnet18", num_classes=NUM_CLASSES, pretrained=True, grayscale=True
    )
    model.eval()

    # Get test input and export parameters
    dummy_input, export_params = get_test_input()

    # Ensure export directory exists
    export_path = Path(ONNX_EXPORT_PATH)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    # Export model to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_EXPORT_PATH,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        **export_params,
    )

    # Verify ONNX model
    onnx_model = onnx.load(ONNX_EXPORT_PATH)
    onnx.checker.check_model(onnx_model)

    # Test with ONNX Runtime
    ort_session = onnxruntime.InferenceSession(
        ONNX_EXPORT_PATH, providers=["CPUExecutionProvider"]
    )

    # Run inference
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)

    # Verify output shape
    assert ort_outputs[0].shape == (TEST_BATCH_SIZE, NUM_CLASSES), (
        f"Invalid ONNX output shape: {ort_outputs[0].shape}"
    )


def test_end_to_end_inference():
    """Test end-to-end inference pipeline with ONNX model."""
    # Create dummy test image
    test_image = Image.new("RGB", TEST_IMAGE_SIZE, color="white")

    # Initialize transforms
    transform = get_base_transforms(image_size=TEST_IMAGE_SIZE)

    # Prepare input
    img_tensor = transform(test_image).unsqueeze(0)

    # Load ONNX model
    ort_session = onnxruntime.InferenceSession(
        ONNX_EXPORT_PATH, providers=["CPUExecutionProvider"]
    )

    # Run inference
    ort_inputs = {ort_session.get_inputs()[0].name: img_tensor.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)

    # Verify output
    predictions = ort_outputs[0]
    assert predictions.shape == (1, NUM_CLASSES), (
        f"Invalid prediction shape: {predictions.shape}"
    )

    # Check if probabilities sum to approximately 1
    probabilities = torch.nn.functional.softmax(torch.from_numpy(predictions), dim=1)
    assert abs(1.0 - probabilities.sum().item()) < 1e-6, "Probabilities do not sum to 1"


def test_model_performance():
    """Test model inference performance."""
    # Load ONNX model
    ort_session = onnxruntime.InferenceSession(
        ONNX_EXPORT_PATH, providers=["CPUExecutionProvider"]
    )

    # Create batch of test inputs
    batch_size = 8
    dummy_batch = torch.zeros((batch_size, 1, *TEST_IMAGE_SIZE))

    # Warm-up run
    _ = ort_session.run(None, {ort_session.get_inputs()[0].name: dummy_batch.numpy()})

    # Measure inference time
    import time

    start_time = time.time()
    num_iterations = 10

    for _ in range(num_iterations):
        _ = ort_session.run(
            None, {ort_session.get_inputs()[0].name: dummy_batch.numpy()}
        )

    avg_time = (time.time() - start_time) / num_iterations

    # Assert performance requirements (adjust threshold as needed)
    assert avg_time < 0.1, f"Inference too slow: {avg_time:.3f}s per batch"


def test_model_memory():
    """Test model memory usage."""
    import psutil

    process = psutil.Process()

    # Record initial memory
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Load model and run inference
    ort_session = onnxruntime.InferenceSession(
        ONNX_EXPORT_PATH, providers=["CPUExecutionProvider"]
    )

    dummy_input = torch.zeros((1, 1, *TEST_IMAGE_SIZE))
    _ = ort_session.run(None, {ort_session.get_inputs()[0].name: dummy_input.numpy()})

    # Check memory usage
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory

    # Assert reasonable memory usage (adjust threshold as needed)
    assert memory_increase < 500, f"Memory usage too high: {memory_increase:.1f}MB"


if __name__ == "__main__":
    pytest.main([__file__])
