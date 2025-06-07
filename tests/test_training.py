import pytest
from src.training.trainer import CutleryTrainer
from src.models.factory import create_model


def test_trainer_creation():
    """Test that trainer can be created with default settings"""
    trainer = CutleryTrainer()
    assert trainer is not None


def test_model_creation():
    """Test that model can be created with correct number of classes"""
    model = create_model("resnet18", num_classes=6)
    assert model is not None
