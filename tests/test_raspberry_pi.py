"""
Test suite for Raspberry Pi deployment and optimization.

This module tests:
1. ARM compatibility
2. Memory constraints
3. Performance optimization
4. System integration
"""

import os
import pytest
import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path
import psutil
import platform
import time

from src.models.factory import create_model
from scripts.run_inference_on_pi import InferenceEnvironment, ImagePreprocessor

# Constants
TEST_IMAGE_SIZE = (320, 320)
NUM_CLASSES = 3
ONNX_EXPORT_PATH = Path("models/exports/cutlery_classifier_edge.onnx")
MEMORY_LIMIT = 512 * 1024 * 1024  # 512MB
INFERENCE_TIME_LIMIT = 200  # ms


def is_arm_platform():
    """Check if running on ARM platform."""
    return platform.machine() in ["aarch64", "armv7l"]


def get_test_image():
    """Create a test image."""
    return np.random.randint(0, 255, (320, 320), dtype=np.uint8)


class TestArmCompatibility:
    """Test ARM platform compatibility."""

    def test_platform_detection(self):
        """Test platform detection logic."""
        env = InferenceEnvironment()
        assert hasattr(env, "is_arm"), "Platform detection not implemented"
        assert isinstance(env.is_arm, bool), "Invalid platform detection result"

    def test_onnx_providers(self):
        """Test ONNX Runtime providers."""
        providers = ort.get_available_providers()
        assert "CPUExecutionProvider" in providers, "CPU provider not available"

    @pytest.mark.skipif(not is_arm_platform(), reason="Not running on ARM")
    def test_arm_optimizations(self):
        """Test ARM-specific optimizations."""
        env = InferenceEnvironment()
        options = env.get_session_options()

        # Verify ARM optimizations
        assert options.intra_op_num_threads == env.cpu_count
        assert options.inter_op_num_threads == 1
        assert options.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL
        assert not options.enable_cpu_mem_arena


class TestMemoryConstraints:
    """Test behavior under memory constraints."""

    def test_memory_monitoring(self):
        """Test memory monitoring functionality."""
        initial_memory = psutil.Process().memory_info().rss

        # Load model
        env = InferenceEnvironment()
        session = ort.InferenceSession(
            str(ONNX_EXPORT_PATH),
            env.get_session_options(),
            providers=["CPUExecutionProvider"],
        )

        # Check memory increase
        final_memory = psutil.Process().memory_info().rss
        memory_increase = final_memory - initial_memory

        assert memory_increase < MEMORY_LIMIT, (
            f"Memory usage too high: {memory_increase / 1024 / 1024:.1f}MB"
        )

    def test_image_preprocessing_memory(self):
        """Test memory efficiency of image preprocessing."""
        preprocessor = ImagePreprocessor()
        initial_memory = psutil.Process().memory_info().rss

        # Process test image
        test_image = get_test_image()
        processed = preprocessor.load_and_preprocess(test_image)

        # Check memory usage
        final_memory = psutil.Process().memory_info().rss
        memory_increase = final_memory - initial_memory

        assert memory_increase < 10 * 1024 * 1024, (
            "Image preprocessing uses too much memory"
        )

    @pytest.mark.skipif(not is_arm_platform(), reason="Not running on ARM")
    def test_low_memory_handling(self):
        """Test behavior under low memory conditions."""
        env = InferenceEnvironment()
        session = ort.InferenceSession(
            str(ONNX_EXPORT_PATH),
            env.get_session_options(),
            providers=["CPUExecutionProvider"],
        )

        # Simulate low memory by allocating arrays
        arrays = []
        try:
            while psutil.virtual_memory().available > MEMORY_LIMIT:
                arrays.append(np.zeros((1024, 1024), dtype=np.float32))

            # Try inference
            test_image = get_test_image()
            processed = ImagePreprocessor.load_and_preprocess(test_image)

            # Should handle low memory gracefully
            output = session.run(None, {session.get_inputs()[0].name: processed})
            assert output is not None

        finally:
            # Clean up
            arrays.clear()


class TestPerformanceOptimization:
    """Test performance optimization features."""

    def test_inference_speed(self):
        """Test inference speed meets requirements."""
        env = InferenceEnvironment()
        session = ort.InferenceSession(
            str(ONNX_EXPORT_PATH),
            env.get_session_options(),
            providers=["CPUExecutionProvider"],
        )

        # Prepare test image
        test_image = get_test_image()
        processed = ImagePreprocessor.load_and_preprocess(test_image)

        # Warmup
        _ = session.run(None, {session.get_inputs()[0].name: processed})

        # Measure inference time
        start_time = time.perf_counter()
        output = session.run(None, {session.get_inputs()[0].name: processed})
        inference_time = (time.perf_counter() - start_time) * 1000

        assert inference_time < INFERENCE_TIME_LIMIT, (
            f"Inference too slow: {inference_time:.1f}ms"
        )

    @pytest.mark.skipif(not is_arm_platform(), reason="Not running on ARM")
    def test_thread_optimization(self):
        """Test thread optimization settings."""
        env = InferenceEnvironment()
        options = env.get_session_options()

        assert options.intra_op_num_threads == env.cpu_count
        assert options.inter_op_num_threads == 1

    def test_model_optimization(self):
        """Test model optimization settings."""
        env = InferenceEnvironment()
        options = env.get_session_options()

        assert (
            options.graph_optimization_level
            == ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )


class TestSystemIntegration:
    """Test system integration features."""

    def test_error_handling(self):
        """Test error handling and recovery."""
        env = InferenceEnvironment()

        # Test invalid model path
        with pytest.raises(FileNotFoundError):
            session = ort.InferenceSession(
                "invalid_model.onnx",
                env.get_session_options(),
                providers=["CPUExecutionProvider"],
            )

        # Test invalid input
        session = ort.InferenceSession(
            str(ONNX_EXPORT_PATH),
            env.get_session_options(),
            providers=["CPUExecutionProvider"],
        )

        with pytest.raises(RuntimeError):
            invalid_input = np.zeros((1, 3, 32, 32))  # Wrong shape
            session.run(None, {session.get_inputs()[0].name: invalid_input})

    @pytest.mark.skipif(not is_arm_platform(), reason="Not running on ARM")
    def test_thermal_monitoring(self):
        """Test thermal monitoring on Raspberry Pi."""
        if os.path.exists("/sys/class/thermal/thermal_zone0/temp"):
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                temp = float(f.read()) / 1000
                assert temp < 85, f"Temperature too high: {temp}°C"

    def test_graceful_shutdown(self):
        """Test graceful shutdown behavior."""
        env = InferenceEnvironment()
        session = ort.InferenceSession(
            str(ONNX_EXPORT_PATH),
            env.get_session_options(),
            providers=["CPUExecutionProvider"],
        )

        # Run inference
        test_image = get_test_image()
        processed = ImagePreprocessor.load_and_preprocess(test_image)
        output = session.run(None, {session.get_inputs()[0].name: processed})

        # Cleanup should not raise errors
        del session
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
