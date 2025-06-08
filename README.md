# Cutlery Classifier MVP

A production-ready computer vision pipeline for classifying cutlery types (fork, knife, spoon) using deep learning.

## 🎯 Features

✅ ResNet18-based classifier with grayscale pipeline  
✅ Real-time inference (<200ms on CUDA)  
✅ Production-grade data augmentation  
✅ Grad-CAM visualization for model explainability  
✅ ONNX export ready for edge deployment  
✅ Comprehensive test suite with stress testing  
✅ 100% accuracy on validation and test sets

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (for optimal performance)
- PyTorch 2.0+

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cutlery-classifier-mvp.git
cd cutlery-classifier-mvp

# Install package
pip install -e .
```

### Running Inference

```bash
# Single image inference
PYTHONPATH=src python scripts/run_inference.py --device cuda --model models/checkpoints/type_detector_best.pth --image path/to/your/image.jpg

# With Grad-CAM visualization
PYTHONPATH=src python scripts/run_inference.py --device cuda --model models/checkpoints/type_detector_best.pth --image path/to/your/image.jpg --grad-cam
```

## 📊 Performance

| Metric                | Value  |
| --------------------- | ------ |
| Accuracy (Test Set)   | 100%   |
| Inference Time (CUDA) | <200ms |
| Model Size            | 44.7MB |
| Stress Test Accuracy  | >90%   |

## 🛠️ Project Structure

```
cutlery-classifier-mvp/
├── config/               # Model and training configurations
├── scripts/             # Production-ready training and inference scripts
├── src/                 # Core implementation
│   ├── augment/        # Data augmentation pipeline
│   ├── evaluation/     # Model evaluation and Grad-CAM
│   ├── inference/      # Inference pipeline
│   ├── models/         # Model architecture
│   └── training/       # Training pipeline
└── tests/              # Comprehensive test suite
```

## 🔬 Model Architecture

- Base: ResNet18 (pretrained)
- Input: Grayscale images (224x224)
- Output: 3 classes (fork, knife, spoon)
- Feature Extraction: Modified conv1 for grayscale
- Training: Early stopping, learning rate scheduling

## 📈 Project Status

✅ Full pipeline tested and verified  
✅ Production-ready inference  
✅ Comprehensive documentation  
✅ Ready for deployment/integration

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
