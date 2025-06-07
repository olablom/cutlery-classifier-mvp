import pytest
from src.inference.predictor import CutleryPredictor
from src.models.factory import create_model


def test_predictor_creation():
    """Test that predictor can be created with default settings"""
    predictor = CutleryPredictor()
    assert predictor is not None


def test_inference_pipeline():
    """Test that inference pipeline components can be initialized"""
    model = create_model("resnet18", num_classes=6)
    assert model is not None
