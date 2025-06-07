import pytest
from pathlib import Path
from src.pipeline.manufacturer import CutleryPipeline


def test_pipeline_creation():
    """Test that pipeline can be created with default settings"""
    pipeline = CutleryPipeline()
    assert pipeline is not None


def test_pipeline_config():
    """Test that pipeline can load configuration"""
    config_path = Path("config/train_config.yaml")
    assert config_path.exists(), "Training config must exist"
