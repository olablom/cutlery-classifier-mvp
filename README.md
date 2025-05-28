# Cutlery Classifier MVP

An AI-based cutlery classification system that can identify and sort cutlery (knives, forks, spoons). This MVP serves as the foundation for an automated sorting system for commercial kitchens and food courts.

## Project Overview

**Purpose**: Develop an AI system to classify washed cutlery by type, enabling automated sorting in commercial kitchen environments.

**Goal**: Create a working classification system using mobile phone images (~20 per class) with at least 80% accuracy, optimized for embedded deployment with industrial cameras.

## Project Scope

### MVP Scope (Academic Project)

This MVP demonstrates the core classification pipeline with:

- **Type detection** (fork/knife/spoon) using grayscale ResNet18
- **Complete training/evaluation/export pipeline**
- **Modular architecture** for easy expansion
- **Optimized for edge deployment**

### Production Roadmap (Future Development)

The modular architecture enables straightforward expansion to:

- Manufacturer classification for each cutlery type
- Real-time embedded deployment on ARM processors
- Industrial global shutter camera integration
- Advanced optimization techniques (quantization, pruning)

## Features

- **Classification pipeline:**
  - Type classification (fork/knife/spoon) using grayscale images
  - Input size: 320x320 pixels
  - ResNet18 backbone with transfer learning
- **Complete ML pipeline** with training, evaluation, and inference
- **Visual inference** with prediction overlays and confidence scores
- **Model export** to ONNX format
- **CLI interfaces** for all major operations

## Project Structure

```
cutlery-classifier-mvp/
├── data/
│   ├── raw/                    # Original images by type
│   │   ├── fork/              # Fork images
│   │   ├── knife/             # Knife images
│   │   └── spoon/             # Spoon images
│   └── processed/             # Train/val/test splits
│       ├── train/
│       ├── val/
│       └── test/
├── src/
│   ├── data/                  # Data loading and preprocessing
│   ├── models/                # Model architectures
│   ├── training/              # Training scripts
│   ├── evaluation/            # Evaluation and metrics
│   ├── inference/             # Inference pipeline
│   └── utils/                 # Utility functions
├── models/
│   ├── checkpoints/           # Training checkpoints
│   └── exports/              # Exported models (.onnx)
├── results/
│   ├── plots/                # Training plots
│   └── metrics/              # Performance metrics
├── config/                   # Configuration files
├── tests/                    # Unit tests
└── scripts/                  # Utility scripts
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- PIL/Pillow
- NumPy
- ONNX (for model export)

## Quick Start

1. **Setup environment**

   ```bash
   git clone <repository-url>
   cd cutlery-classifier-mvp
   pip install -r requirements.txt
   ```

2. **Prepare your data**

   ```bash
   # Place images in data/raw/{fork,knife,spoon}/
   # Validate dataset
   python scripts/validate_dataset.py

   # Create train/val/test splits
   python scripts/prepare_dataset.py --create-splits
   ```

3. **Train the model**

   ```bash
   # Train type detector
   python scripts/train_type_detector.py --epochs 30
   ```

4. **Evaluate the model**

   ```bash
   # Run evaluation
   python scripts/evaluate_model.py --model models/checkpoints/type_detector_best.pth
   ```

5. **Run inference**

   ```bash
   # Single image inference
   python scripts/infer_image.py --model models/checkpoints/type_detector_best.pth --image test.jpg
   ```

## Dataset Structure

The MVP uses a simple dataset structure:

- ~20 images per class (fork/knife/spoon)
- Split ratio: 70% train, 20% validation, 10% test
- Image size: 320x320 pixels (grayscale)
- Data augmentation during training:
  - Random rotation (±15°)
  - Random horizontal flip
  - Color jitter (brightness/contrast)

## Model Architecture

- **Backbone**: ResNet18 (pretrained on ImageNet)
- **Input**: 320x320 grayscale
- **Output**: 3 classes (fork/knife/spoon)
- **Training**:
  - Optimizer: Adam
  - Learning rate: 0.001
  - Epochs: 30
  - Early stopping

## Results

Results will be updated after training with:

- Training/validation curves
- Confusion matrix
- Per-class accuracy
- Example predictions
