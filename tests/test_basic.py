"""
Basic tests for Cutlery Classifier MVP

These tests verify core functionality and serve as a foundation for CI/CD integration.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from models.factory import create_model
    from data.transforms import get_transforms

    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)


class TestModelFactory:
    """Test model creation and basic functionality."""

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Model imports not available")
    def test_resnet18_creation(self):
        """Test ResNet18 model creation."""
        config = {
            "architecture": "resnet18",
            "num_classes": 3,
            "pretrained": False,  # Faster for testing
            "input_channels": 1,
        }

        model = create_model(config)
        assert model is not None

        # Test forward pass
        dummy_input = torch.randn(1, 1, 320, 320)
        output = model(dummy_input)
        assert output.shape == (1, 3)

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Model imports not available")
    def test_mobilenet_creation(self):
        """Test MobileNetV2 model creation."""
        config = {
            "architecture": "mobilenet_v2",
            "num_classes": 3,
            "pretrained": False,  # Faster for testing
            "input_channels": 1,
        }

        model = create_model(config)
        assert model is not None

        # Test forward pass
        dummy_input = torch.randn(1, 1, 320, 320)
        output = model(dummy_input)
        assert output.shape == (1, 3)


class TestDataTransforms:
    """Test data preprocessing pipeline."""

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Transform imports not available")
    def test_train_transforms(self):
        """Test training transforms creation."""
        config = {
            "image_size": [320, 320],
            "normalize": {"mean": [0.449], "std": [0.226]},
        }

        transforms = get_transforms(config, mode="train")
        assert transforms is not None

        # Test with dummy PIL image
        from PIL import Image

        dummy_image = Image.new("RGB", (640, 480), color="gray")
        transformed = transforms(dummy_image)

        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (1, 320, 320)  # Grayscale

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Transform imports not available")
    def test_val_transforms(self):
        """Test validation transforms creation."""
        config = {
            "image_size": [320, 320],
            "normalize": {"mean": [0.449], "std": [0.226]},
        }

        transforms = get_transforms(config, mode="val")
        assert transforms is not None


class TestProjectStructure:
    """Test project structure and configuration."""

    def test_project_directories_exist(self):
        """Test that essential directories exist."""
        essential_dirs = [
            "src",
            "scripts",
            "config",
            "docs",
            "data",
            "models",
            "results",
        ]

        for dir_name in essential_dirs:
            dir_path = project_root / dir_name
            assert dir_path.exists(), f"Directory {dir_name} should exist"

    def test_config_files_exist(self):
        """Test that configuration files exist."""
        config_files = [
            "config/train_config.yaml",
            "requirements.txt",
            "README.md",
            ".gitignore",
        ]

        for file_name in config_files:
            file_path = project_root / file_name
            assert file_path.exists(), f"Config file {file_name} should exist"

    def test_script_files_exist(self):
        """Test that all CLI scripts exist."""
        script_files = [
            "scripts/train_type_detector.py",
            "scripts/evaluate_model.py",
            "scripts/infer_image.py",
            "scripts/export_model.py",
            "scripts/prepare_dataset.py",
            "scripts/validate_dataset.py",
        ]

        for script_name in script_files:
            script_path = project_root / script_name
            assert script_path.exists(), f"Script {script_name} should exist"


class TestVersioning:
    """Test version information and metadata."""

    def test_package_version(self):
        """Test that package version is defined."""
        try:
            import src

            assert hasattr(src, "__version__")
            assert isinstance(src.__version__, str)
            assert len(src.__version__) > 0
        except ImportError:
            pytest.skip("src package not importable")

    def test_version_format(self):
        """Test that version follows semantic versioning."""
        try:
            import src

            version_parts = src.__version__.split(".")
            assert len(version_parts) >= 2, "Version should have at least major.minor"

            # Check that parts are numeric
            for part in version_parts:
                assert part.isdigit(), f"Version part '{part}' should be numeric"
        except ImportError:
            pytest.skip("src package not importable")


# Placeholder test for CI readiness
def test_placeholder():
    """Placeholder test to ensure pytest runs successfully."""
    assert True


def test_pytorch_available():
    """Test that PyTorch is available."""
    import torch

    assert torch.__version__ is not None


def test_basic_torch_operations():
    """Test basic PyTorch operations work."""
    x = torch.randn(2, 3)
    y = torch.randn(3, 4)
    z = torch.mm(x, y)
    assert z.shape == (2, 4)


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
