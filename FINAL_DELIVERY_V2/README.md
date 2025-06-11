# 🍴 Cutlery Classifier MVP (Enhanced Version)

> ✨ This is the enhanced version (V2) of the project that received the highest grade (VG). It includes proper Python packaging, ONNX export support, and improved testing while maintaining the original high-quality structure and documentation.

An image classification system for automated cutlery sorting, built with PyTorch and ResNet18.

## 🆕 V2 Improvements

1. **Proper Python Packaging**

   - Implemented `setup.py` and `pyproject.toml`
   - Correct package structure with `src/` layout
   - No PYTHONPATH dependencies
   - Easy installation with `pip install -e .`

2. **Enhanced Testing**

   - Complete pytest suite
   - End-to-end inference tests
   - Model architecture verification
   - Dataset validation

3. **Extended Functionality**
   - ONNX export support for edge deployment
   - Improved Grad-CAM visualization
   - Robust evaluation process
   - Detailed results reporting

## 📋 Features

- Production-grade image classification pipeline
- Pre-trained ResNet18 with grayscale optimization
- Grad-CAM visualization support
- Comprehensive evaluation metrics
- Full test suite
- ONNX export capability

## 🎯 Performance

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 100%  |
| Precision | 100%  |
| Recall    | 100%  |
| F1-Score  | 100%  |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
| ----- | --------- | ------ | -------- | ------- |
| Fork  | 100%      | 100%   | 100%     | 10      |
| Knife | 100%      | 100%   | 100%     | 10      |
| Spoon | 100%      | 100%   | 100%     | 10      |

## 🚀 Quick Start

### Installation

1. Clone the repository:

```bash
git clone https://github.com/olablom/cutlery-classifier-mvp.git
cd cutlery-classifier-mvp
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows
```

3. Install the package in development mode:

```bash
pip install -e .
```

### Training

Train the model with:

```bash
python scripts/train_pipeline.py --device cpu --config config/train_config.yaml
```

The best model will be saved to `models/checkpoints/type_detector_best.pth`.

### Inference

Run inference on a single image:

```bash
python scripts/run_inference.py --device cpu --image path/to/image.jpg
```

Add `--visualize` to generate Grad-CAM visualization:

```bash
python scripts/run_inference.py --device cpu --image path/to/image.jpg --visualize
```

### Evaluation

Run full test set evaluation:

```bash
python scripts/evaluate_on_test_set.py --device cpu
```

This will generate:

- `results/evaluation/confusion_matrix.png`
- `results/evaluation/metrics.json`

### Testing

Run the test suite:

```bash
pytest
```

## 📁 Project Structure

```
FINAL_DELIVERY_V2/
├── config/                 # Configuration files
│   └── train_config.yaml  # Training configuration
├── data/                  # Dataset
│   └── simplified/        # Processed dataset
│       ├── train/        # Training images
│       ├── val/          # Validation images
│       └── test/         # Test images
├── models/                # Model checkpoints
├── results/              # Evaluation results
│   ├── evaluation/      # Test set metrics
│   └── grad_cam/        # Grad-CAM visualizations
├── scripts/              # Training and inference scripts
├── src/                  # Source code
│   └── cutlery_classifier/
│       ├── data/        # Data loading and transforms
│       ├── inference/   # Inference pipeline
│       ├── models/      # Model architecture
│       └── training/    # Training pipeline
└── tests/               # Test suite
```

## 🖼️ Example Results

### Grad-CAM Visualization

![Grad-CAM Example](results/grad_cam/gradcam_example.jpg)

### Confusion Matrix

![Confusion Matrix](results/evaluation/confusion_matrix.png)

## 📊 Model Architecture

- Base: ResNet18 (pretrained)
- Input: 320x320 grayscale
- Output: 3 classes (fork, knife, spoon)
- Parameters: 11.2M

## 🛠️ Configuration

Key training parameters (from `config/train_config.yaml`):

- Learning rate: 0.001
- Batch size: 32
- Epochs: 30 (with early stopping)
- Optimizer: Adam
- Data augmentation: horizontal flip, rotation, blur

## 📝 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- torchvision for the pre-trained ResNet18 model
- pytorch-grad-cam for visualization support
