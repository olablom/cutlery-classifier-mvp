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

## ⚠️ Datasets Disclaimer

This repository does not include training or test datasets to keep the repo lightweight.  
Please provide your own dataset under:

- `data/raw/` → original photos
- `data/processed/` → preprocessed dataset for training and testing
- `data/augmented/` → (optional) diffusion-augmented dataset

See the "Data Preparation" section for guidance on collecting and preprocessing your own dataset.

## 🔍 Inference and Testing

### Run inference on a single image:

```bash
python scripts/run_inference.py --device cuda --image path/to/image.jpg --grad-cam
```

### Run full dataset test with per-class accuracy:

```bash
python scripts/test_dataset_inference.py --device cuda --test_dir data/processed/test --save-misclassified
```

Results will be saved in `results/misclassified_grad_cam/` if there are misclassified images.

A detailed test report template is available in `docs/test_report_template.md`. Use this template to document your test results and model performance.

Note: Since datasets are not included in the repository, you'll need to:

1. Collect your own cutlery images
2. Process them using the data preparation pipeline
3. Or use the provided data augmentation script to generate synthetic training data

## 📁 Project Structure

```
cutlery-classifier-mvp/
├── data/                    # Dataset directory
│   ├── processed/          # Processed datasets
│   └── augmented/         # Augmented datasets
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

## 📊 Performance Metrics

| Metric         | Value              |
| -------------- | ------------------ |
| Accuracy       | 100.0%             |
| Inference Time | <200ms on RTX 5090 |
| Memory Usage   | <512MB             |

## 🔄 Data Augmentation

This project uses Stable Diffusion v1.5 (image-to-image pipeline) for advanced data augmentation.

Key points:

- Model: `runwayml/stable-diffusion-v1-5`
- Augmentation script: `src/augment/generate_diffusion_images.py`
- Safety checker: disabled for reproducible training data (MVP mode)
- Augmentation CLI example:

```bash
python src/augment/generate_diffusion_images.py --classes fork knife spoon
```

## 📜 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

- ResNet architecture: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- Grad-CAM visualization: [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)

```

```
