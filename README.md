# Cutlery Classifier MVP

## Project Overview

An advanced deep learning system for automated cutlery classification, achieving production-ready performance with ResNet18 architecture. The system demonstrates perfect accuracy on the test set while maintaining high throughput and robustness to real-world variations.

### Key Features

- High-accuracy cutlery classification (fork, knife, spoon)
- Production-ready validation pipeline
- Comprehensive stress testing suite
- Real-time inference capabilities
- Explainable AI with Grad-CAM visualizations

## Technical Architecture

- **Framework**: PyTorch
- **Base Model**: ResNet18 (transfer learning)
- **Input**: 224x224 grayscale images
- **Output**: 3-class classification
- **Performance**: 120,000+ predictions/minute
- **Export Format**: ONNX for edge deployment

## Setup Instructions

### Prerequisites

```bash
# Required system packages
python >= 3.8
CUDA >= 11.0 (for GPU support)
```

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/cutlery-classifier-mvp.git
cd cutlery-classifier-mvp
```

2. Create and activate virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
# Run full training pipeline
python scripts/full_pipeline.py --device cuda

# Train with specific configuration
python scripts/train_type_detector.py --device cuda --config config/model_config.yaml
```

### Inference

```bash
# Single image inference
python scripts/run_inference.py --device cuda --image path/to/image.jpg

# With Grad-CAM visualization
python scripts/run_inference.py --device cuda --image path/to/image.jpg --grad-cam
```

### Testing

```bash
# Run complete test suite
pytest tests/

# Run specific test category
pytest tests/test_inference.py
```

## Project Structure

```
cutlery-classifier-mvp/
├── config/                  # Configuration files
├── scripts/                 # Training and inference scripts
│   ├── run_inference.py    # Production inference script
│   ├── train_type_detector.py  # Training script
│   └── test_dataset_inference.py  # Dataset validation
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model architecture and factory
│   ├── training/          # Training utilities
│   ├── evaluation/        # Evaluation metrics
│   └── inference/         # Inference pipeline
└── tests/                 # Unit tests
```

## Documentation

Comprehensive documentation is available covering:

- Technical Implementation Guide
- Performance Analysis
- Testing Reports
- Grad-CAM Visualizations
- Production Deployment Guide

For access to the complete documentation package, please contact the project maintainer.

## Performance Metrics

### Model Performance

- **Test Accuracy**: 100%
- **Validation Accuracy**: 100%
- **Inference Speed**: <10ms per prediction
- **Model Size**: 43MB

### Stress Test Results

- **Noise Tolerance**: 93.33% accuracy
- **Blur Resistance**: 100% accuracy
- **Rotation Invariance**: 100% accuracy

## Development Process

The project follows professional software development practices:

- Modern Python packaging (pyproject.toml)
- Comprehensive testing suite
- Clear code documentation
- Regular performance benchmarking
- Explainable AI integration

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- ResNet authors for the backbone architecture
