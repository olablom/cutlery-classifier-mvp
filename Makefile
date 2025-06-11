# Cutlery Classifier MVP - Development Makefile
# Professional commands for development, testing, and benchmarking

.PHONY: help install test benchmark inference clean lint format

# Default target
help:
	@echo "Cutlery Classifier MVP - Development Commands"
	@echo "============================================="
	@echo ""
	@echo "Setup & Installation:"
	@echo "  install          Install package in development mode"
	@echo "  install-deps     Install dependencies only"
	@echo ""
	@echo "Development & Testing:"
	@echo "  test             Run full test suite"
	@echo "  test-quick       Run quick tests only"
	@echo "  lint             Run code linting"
	@echo "  format           Format code with black"
	@echo ""
	@echo "Performance & Benchmarking:"
	@echo "  benchmark        Run RTX 5090 performance benchmark"
	@echo "  benchmark-cpu    Run CPU benchmark for comparison"
	@echo "  stress-test      Run stress testing on model"
	@echo ""
	@echo "Inference & Demo:"
	@echo "  inference        Run demo inference on sample image"
	@echo "  inference-all    Test inference on all demo classes"
	@echo "  gradcam          Generate Grad-CAM visualization"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean            Clean temporary files and cache"
	@echo "  clean-results    Clean generated results (be careful!)"

# Installation
install:
	@echo "Installing Cutlery Classifier in development mode..."
	pip install -e .

install-deps:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

# Testing
test:
	@echo "Running full test suite..."
	python -m pytest tests/ -v --cov=src/cutlery_classifier

test-quick:
	@echo "Running quick tests..."
	python -m pytest tests/ -v -k "not slow"

# Performance Benchmarking
benchmark:
	@echo "Running RTX 5090 Performance Benchmark..."
	@echo "========================================="
	python benchmark_rtx5090.py

benchmark-cpu:
	@echo "Running CPU Benchmark for comparison..."
	python -c "import benchmark_rtx5090; benchmark_rtx5090.benchmark_rtx5090()" --device cpu || \
	python scripts/run_inference.py --device cpu --image data/raw/fork/IMG_0941[1].jpg

stress-test:
	@echo "Running stress testing..."
	python scripts/test_dataset_inference.py --device cuda --test_dir data/simplified/test --stress-test

# Inference & Demo
inference:
	@echo "Running demo inference..."
	python scripts/run_inference.py --device cuda --image data/raw/fork/IMG_0941[1].jpg

inference-all:
	@echo "Testing inference on all classes..."
	@echo "Fork:"
	@python scripts/run_inference.py --device cuda --image data/raw/fork/IMG_0941[1].jpg --quiet || echo "Fork test failed"
	@echo "Knife:" 
	@python scripts/run_inference.py --device cuda --image data/raw/knife/IMG_0959[1].jpg --quiet || echo "Knife test failed"
	@echo "Spoon:"
	@python scripts/run_inference.py --device cuda --image data/raw/spoon/IMG_0962[1].jpg --quiet || echo "Spoon test failed"

gradcam:
	@echo "Generating Grad-CAM visualization..."
	python scripts/run_inference.py --device cuda --image data/raw/fork/IMG_0941[1].jpg --grad-cam

# Code Quality
lint:
	@echo "Running code linting..."
	flake8 src/ scripts/ --max-line-length=88 --ignore=E203,W503

format:
	@echo "Formatting code..."
	black src/ scripts/ --line-length=88

# Maintenance
clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete
	rm -rf .pytest_cache/

clean-results:
	@echo "WARNING: This will delete generated results!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo ""; \
		echo "Cleaning results..."; \
		rm -rf results/grad_cam/*.jpg; \
		rm -rf results/optuna_plots/; \
		rm -f results/production_validation/*.txt; \
	else \
		echo ""; \
		echo "Cancelled."; \
	fi

# Version and Release
version:
	@echo "Current version info:"
	@git describe --tags --abbrev=0 2>/dev/null || echo "No tags found"
	@echo "Latest commit: $$(git log -1 --format='%h %s')"

# Quick validation that everything works
validate:
	@echo "Quick validation of installation..."
	@python -c "import torch; print(f'PyTorch: {torch.__version__}')"
	@python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
	@python -c "from src.cutlery_classifier.models.model_factory import create_model; print('Model factory works')" 2>/dev/null || echo "Model factory needs PYTHONPATH or install -e ."
	@echo "Validation complete!" 