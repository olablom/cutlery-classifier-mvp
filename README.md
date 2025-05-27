# Cutlery Classifier MVP

An AI-based cutlery classification system that can identify and sort cutlery (knives, forks, spoons) by manufacturer. This MVP serves as the foundation for an automated sorting system for commercial kitchens and food courts.

## Project Overview

**Purpose**: Develop an AI system to classify washed cutlery by manufacturer, enabling automated sorting in commercial kitchen environments to ensure proper cutlery distribution and reduce manual labor.

**Goal**: Create a working hierarchical classification system using mobile phone images (30-50 per class) with at least 80% accuracy, optimized for embedded deployment with industrial cameras.

## Project Scope

### MVP Scope (Academic Project)

This MVP demonstrates the core classification pipeline with:

- **Type detection** (fork/knife/spoon) using grayscale ResNet18
- **Proof-of-concept manufacturer classification** for forks
- Complete training/evaluation/export pipeline
- Modular architecture for easy expansion

### Production Roadmap (Future Development)

The modular architecture enables straightforward expansion to:

- Full manufacturer models for all cutlery types (knife, spoon)
- Real-time embedded deployment on ARM processors
- Industrial global shutter camera integration
- Advanced optimization techniques (quantization, pruning)

## Features

- **Hierarchical classification pipeline:**
  - Type classification (fork/knife/spoon) using grayscale images
  - Manufacturer classification for specific cutlery types
- **Modular architecture** enabling independent model training and deployment
- **Complete ML pipeline** with training, evaluation, inference, and export
- **Visual inference** with prediction overlays and confidence scores
- **Batch processing** for multiple images
- **Model export** to ONNX and TorchScript formats
- **Grad-CAM visualization** for model interpretability
- **CLI interfaces** for all major operations
- Production-ready model architecture with edge optimization

## Project Structure

```
cutlery-classifier-mvp/
├── data/
│   ├── raw/                    # Original images organized by class
│   │   ├── manufacturer_a/
│   │   │   ├── knife/
│   │   │   ├── fork/
│   │   │   └── spoon/
│   │   ├── manufacturer_b/
│   │   └── manufacturer_c/
│   ├── processed/              # Preprocessed images
│   └── augmented/              # Augmented training data
├── src/
│   ├── data/                   # Data loading and preprocessing
│   ├── models/                 # Model architectures
│   ├── pipeline/               # Classification pipeline components
│   │   ├── type_detector.py    # Type classification (fork/knife/spoon)
│   │   └── manufacturer/       # Manufacturer-specific models
│   │       ├── fork_classifier.py    # Fork manufacturer classification
│   │       ├── knife_classifier.py   # (Future development)
│   │       └── spoon_classifier.py   # (Future development)
│   ├── training/               # Training scripts and utilities
│   ├── evaluation/             # Evaluation and metrics
│   ├── inference/              # Inference pipeline
│   └── utils/                  # Utility functions
├── notebooks/                  # Jupyter notebooks for exploration
├── experiments/                # Experiment tracking and configs
├── models/
│   ├── checkpoints/            # Training checkpoints
│   └── exports/                # Exported models (.pt, .onnx)
├── results/
│   ├── plots/                  # Training plots and visualizations
│   ├── metrics/                # Performance metrics
│   ├── confusion_matrices/     # Confusion matrix outputs
│   └── grad_cam/               # Grad-CAM visualizations
├── config/                     # Configuration files
├── tests/                      # Unit tests
├── scripts/                    # Utility scripts
└── docs/                       # Documentation
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- OpenCV
- PIL/Pillow
- NumPy
- Matplotlib
- scikit-learn
- ONNX (for model export)
- Grad-CAM (for interpretability)

## Quick Start

1. **Clone and setup**

   ```bash
   git clone <repository-url>
   cd cutlery-classifier-mvp
   pip install -r requirements.txt
   ```

2. **Prepare your data**

   ```bash
   # Place images in data/raw/manufacturer_x/cutlery_type/
   # Validate dataset
   python scripts/validate_dataset.py

   # Create train/val/test splits
   python scripts/prepare_dataset.py --create-splits
   ```

3. **Train the model**

   ```bash
   # Train type detector (fork/knife/spoon)
   python scripts/train_type_detector.py --epochs 30 --batch-size 32
   ```

4. **Evaluate the model**

   ```bash
   # Comprehensive evaluation with Grad-CAM
   python scripts/evaluate_model.py --model models/checkpoints/type_detector_best.pth
   ```

5. **Run inference**

   ```bash
   # Basic inference
   python scripts/infer_image.py --model models/checkpoints/type_detector_best.pth --image test.jpg

   # With visualization
   python scripts/infer_image.py --model models/checkpoints/type_detector_best.pth --image test.jpg --visualize --output result.jpg
   ```

6. **Export for deployment**

   ```bash
   # Export to ONNX
   python scripts/export_model.py --model models/checkpoints/type_detector_best.pth --format onnx
   ```

## Usage Examples

### Training

```bash
# Train with custom parameters
python scripts/train_type_detector.py --epochs 50 --batch-size 16 --learning-rate 0.0005

# Resume from checkpoint
python scripts/train_type_detector.py --resume models/checkpoints/type_detector_latest.pth
```

### Evaluation

```bash
# Full evaluation with 20 Grad-CAM samples
python scripts/evaluate_model.py --model models/checkpoints/type_detector_best.pth --samples 20

# Custom evaluation name
python scripts/evaluate_model.py --model models/checkpoints/type_detector_best.pth --name "final_evaluation"
```

### Inference

```bash
# Single image with top-3 predictions
python scripts/infer_image.py --model models/checkpoints/type_detector_best.pth --image test.jpg --top-k 3

# Batch processing with visualizations
python scripts/infer_image.py --model models/checkpoints/type_detector_best.pth --batch data/test_images/ --visualize --output results/

# Show model information
python scripts/infer_image.py --model models/checkpoints/type_detector_best.pth --image test.jpg --info

# Save results as JSON
python scripts/infer_image.py --model models/checkpoints/type_detector_best.pth --image test.jpg --json results.json
```

### Model Export

```bash
# Export to ONNX with optimization
python scripts/export_model.py --model models/checkpoints/type_detector_best.pth --format onnx --optimize

# Export to both ONNX and TorchScript
python scripts/export_model.py --model models/checkpoints/type_detector_best.pth --format all

# Export with testing
python scripts/export_model.py --model models/checkpoints/type_detector_best.pth --format onnx --test
```

## Model Architecture

The MVP uses a hierarchical classification approach:

### Type Detection (Stage 1)

- **Model**: ResNet18 with grayscale input (1 channel)
- **Classes**: 3 (fork, knife, spoon)
- **Parameters**: ~11.2M
- **Purpose**: Fast, accurate type classification

### Manufacturer Classification (Stage 2)

- **Model**: MobileNetV2 (optimized for edge deployment)
- **Parameters**: ~2.2M per model
- **Implementation**: Separate model per cutlery type
- **MVP Scope**: Fork classifier only (proof-of-concept)

### Pipeline Flow

```
Input Image → Grayscale → Type Detector → Load Specific Model → Manufacturer Prediction
```

## Data Collection Guidelines

- **Image Quality**: Clear, well-lit images
- **Angles**: Multiple angles per cutlery piece
- **Background**: Varied backgrounds to improve generalization
- **Lighting**: Different lighting conditions
- **Format**: JPEG, PNG supported
- **Target**: 40 images per class (minimum 20)

## Performance Targets

- **Accuracy**: ≥80% on validation set
- **Inference Speed**: <100ms per image
- **Model Size**: <50MB for deployment
- **Memory Usage**: <2GB VRAM for training

## Export Formats

- **PyTorch (.pth)**: Training checkpoints with full state
- **ONNX (.onnx)**: Cross-platform deployment
- **TorchScript (.pt)**: PyTorch mobile and C++ integration

## Hardware Requirements

### Training

- **GPU**: RTX 3050 or better (8GB+ VRAM recommended)
- **RAM**: 16GB+ system memory
- **Storage**: 10GB+ free space

### Inference

- **CPU**: Any modern processor
- **GPU**: Optional (CUDA-capable for acceleration)
- **RAM**: 4GB+ system memory

## Future Development

- Integration with industrial global shutter cameras
- Real-time classification pipeline
- Raspberry Pi deployment optimization
- Domain adaptation techniques
- Extended manufacturer support

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license here]

## Contact

[Add contact information]
