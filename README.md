# Cutlery Classifier MVP

A computer vision system that classifies cutlery (forks, knives, spoons) using deep learning. The system uses a ResNet18 architecture trained on grayscale images to achieve high-accuracy classification.

## Project Overview

**Purpose**: Develop an AI system to classify cutlery by type (fork, knife, or spoon) using computer vision.

**Current MVP implements**:

- Fork/knife/spoon classification using ResNet18
- Grayscale image preprocessing pipeline
- GPU-accelerated inference with CUDA support
- Grad-CAM visualization for model interpretability
- Clean, modular architecture for easy extension
- Tested on NVIDIA RTX 5090

## System Requirements

- Python 3.11 or higher
- CUDA 11.8 (for GPU acceleration)
- NVIDIA GPU with CUDA support
- 8GB RAM minimum (16GB recommended for training)

## Setup Instructions

1. **Environment Setup**

```bash
# Clone repository
git clone <repository-url>
cd cutlery-classifier-mvp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install PyTorch with CUDA support
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

2. **Configuration**

```bash
# Copy environment template
cp env.example .env

# Edit .env with your settings
nano .env
```

## Running Inference

The model classifies individual cutlery images into three categories: fork, knife, or spoon.

### Basic Inference

To run inference on a single image:

```bash
python scripts/run_inference.py --device cuda --image "data/processed/test/fork/IMG_0941[1].jpg"
```

Example output:

```
INFO - Using device: cuda
INFO - Model loaded successfully
INFO - Prediction Results:
INFO - Predicted class: fork
INFO - Class probabilities:
INFO - fork: 1.0000
INFO - knife: 0.0000
INFO - spoon: 0.0000
```

### Grad-CAM Visualization

To understand which parts of the image influenced the model's decision, you can generate a Grad-CAM visualization:

```bash
python scripts/run_inference.py --device cuda --image "data/processed/test/fork/IMG_0941[1].jpg" --grad-cam
```

This will:

1. Run normal inference and show predictions
2. Generate a heatmap visualization highlighting important regions
3. Save the visualization to: `results/grad_cam_{class}_{timestamp}.jpg`

Example output with Grad-CAM:

```
INFO - Using device: cuda
INFO - Model loaded successfully
INFO - Prediction Results:
INFO - Predicted class: fork
INFO - Class probabilities:
INFO - fork: 1.0000
INFO - knife: 0.0000
INFO - spoon: 0.0000
INFO - Generating Grad-CAM visualization...
INFO - Grad-CAM visualization saved to: results/grad_cam_fork_20240315_143022.jpg
```

## Project Structure

```
cutlery-classifier-mvp/
├── data/                    # Dataset directory
│   ├── processed/          # Processed datasets
│   └── test/              # Test images
├── models/                 # Model files
│   └── checkpoints/       # Training checkpoints
├── results/               # Results and analysis
│   └── grad_cam/         # Grad-CAM visualizations
├── scripts/              # Utility scripts
├── src/                  # Source code
│   ├── data/            # Data processing
│   ├── evaluation/      # Model evaluation
│   ├── inference/       # Inference pipeline
│   ├── models/          # Model architectures
│   ├── training/        # Training code
│   ├── utils/           # Utilities
│   └── visualization/   # Visualization tools
└── tests/               # Test suite
```

## Performance Metrics

| Metric         | Value  |
| -------------- | ------ |
| Accuracy       | 98.5%  |
| Inference Time | <200ms |
| Memory Usage   | <512MB |

## Data Augmentation

This project uses Stable Diffusion v1.5 (image-to-image pipeline) for advanced data augmentation.

Key points:

- Model: `runwayml/stable-diffusion-v1-5`
- Augmentation script: `src/augment/generate_diffusion_images.py`
- Safety checker: disabled for reproducible training data (MVP mode)
- Augmentation CLI example:

```bash
python src/augment/generate_diffusion_images.py --classes fork knife spoon


## License

MIT License - See LICENSE file for details.

## Acknowledgments

- ResNet architecture: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- Grad-CAM visualization: [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)
```
