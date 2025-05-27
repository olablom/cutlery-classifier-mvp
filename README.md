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
- Mobile image training pipeline optimized for embedded systems
- Model export capabilities (.pt and .onnx formats)
- Grad-CAM visualization for model interpretability
- Inference pipeline with GUI/CLI options
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
│   └── confusion_matrices/     # Confusion matrix outputs
├── config/                     # Configuration files
├── tests/                      # Unit tests
├── scripts/                    # Utility scripts
└── docs/                       # Documentation
```

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- OpenCV
- PIL
- NumPy
- Matplotlib
- scikit-learn

## Quick Start

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd cutlery-classifier-mvp
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data**

   - Place images in `data/raw/manufacturer_x/cutlery_type/`
   - Ensure at least 30 images per class
   - Recommended image size: 320x320 pixels

4. **Train the model**

   ```bash
   python scripts/train.py --config config/train_config.yaml
   ```

5. **Run inference**
   ```bash
   python scripts/inference.py --model models/exports/best_model.pt --image path/to/test/image.jpg
   ```

## Model Architecture

The MVP uses a hierarchical classification approach:

### Type Detection (Stage 1)

- **Model**: ResNet18 with grayscale input (1 channel)
- **Classes**: 3 (fork, knife, spoon)
- **Purpose**: Fast, accurate type classification

### Manufacturer Classification (Stage 2)

- **Model**: MobileNetV2 (optimized for edge deployment)
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

## Performance Targets

- **Accuracy**: ≥80% on validation set
- **Inference Speed**: <100ms per image
- **Model Size**: <50MB for deployment

## Export Formats

- **PyTorch (.pt)**: For Python deployment
- **ONNX (.onnx)**: For cross-platform deployment
- **TorchScript**: For C++ integration

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
