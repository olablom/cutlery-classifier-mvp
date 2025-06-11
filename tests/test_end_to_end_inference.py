"""
End-to-end tests for the cutlery classification pipeline.

These tests verify that the entire pipeline works correctly,
from loading the model to making predictions on known images.
"""

import pytest
from pathlib import Path
import torch

from cutlery_classifier.inference.inferencer import CutleryInferencer


@pytest.fixture
def model_path():
    """Get path to trained model."""
    return Path("models/checkpoints/type_detector_best.pth")


@pytest.fixture
def test_images():
    """Get paths to test images, one per class."""
    return {
        "fork": Path("data/simplified/test/fork/IMG_0974[1]_fork_b.jpg"),
        "knife": Path("data/simplified/test/knife/IMG_1042[1]_knife_a.jpg"),
        "spoon": Path("data/simplified/test/spoon/IMG_1002[1]_spoon_b.jpg"),
    }


def test_model_exists(model_path):
    """Test that the trained model file exists."""
    assert model_path.exists(), f"Model file not found at {model_path}"


def test_inferencer_creation(model_path):
    """Test that we can create an inferencer with the trained model."""
    inferencer = CutleryInferencer(model_path=model_path, device="cpu")
    assert inferencer.model is not None, "Model not loaded"
    assert inferencer.class_names == ["fork", "knife", "spoon"], "Incorrect class names"


def test_model_inference_on_known_images(model_path, test_images):
    """Test model predictions on known test images."""
    inferencer = CutleryInferencer(model_path=model_path, device="cpu")

    for true_class, img_path in test_images.items():
        # Verify image exists
        assert img_path.exists(), f"Test image not found: {img_path}"

        # Get prediction
        result = inferencer.predict(img_path)
        pred_class = result["predictions"][0]["class_name"]
        confidence = result["predictions"][0]["confidence"]

        # Verify prediction
        assert pred_class == true_class, (
            f"Wrong prediction for {true_class}: got {pred_class}"
        )
        assert confidence > 0.99, f"Low confidence ({confidence:.2f}) for {true_class}"


def test_model_architecture(model_path):
    """Test that the model has the correct architecture."""
    checkpoint = torch.load(model_path, map_location="cpu")
    config = checkpoint["config"]

    assert config["model"]["architecture"] == "resnet18", "Wrong architecture"
    assert config["model"]["num_classes"] == 3, "Wrong number of classes"
    assert config["model"]["input_channels"] == 1, "Wrong number of input channels"


def test_model_output_format(model_path, test_images):
    """Test that model output has the correct format and structure."""
    inferencer = CutleryInferencer(model_path=model_path, device="cpu")
    result = inferencer.predict(test_images["fork"])

    # Check result structure
    assert "predictions" in result, "Missing predictions in result"
    assert "inference_time_ms" in result, "Missing inference time"
    assert len(result["predictions"]) == 3, "Wrong number of predictions"

    # Check prediction structure
    pred = result["predictions"][0]
    assert "class_name" in pred, "Missing class name"
    assert "confidence" in pred, "Missing confidence"
    assert "class_index" in pred, "Missing class index"
    assert "percentage" in pred, "Missing percentage"


def test_model_reproducibility(model_path, test_images):
    """Test that predictions are consistent across multiple runs."""
    inferencer = CutleryInferencer(model_path=model_path, device="cpu")
    img_path = test_images["fork"]

    # Get multiple predictions
    results = [
        inferencer.predict(img_path)["predictions"][0]["class_name"] for _ in range(3)
    ]

    # Verify all predictions are the same
    assert len(set(results)) == 1, "Inconsistent predictions across runs"
