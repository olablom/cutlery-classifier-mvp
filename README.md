# Cutlery Classifier MVP

A computer vision system that classifies cutlery by type and manufacturer variant using deep learning. The system uses a ResNet18 architecture trained on grayscale images to achieve high-accuracy classification between different types (fork_a, fork_b, knife_a, knife_b, spoon_a, spoon_b).

## Project Overview

**Purpose**: Develop an AI system to classify cutlery by type (fork, knife, or spoon) using computer vision.

**Current MVP implements**:

- Fork/knife/spoon classification using ResNet18
- Grayscale image preprocessing pipeline
- GPU-accelerated inference with CUDA support
- Grad-CAM visualization for model interpretability
- Clean, modular architecture for easy extension
- Tested on NVIDIA RTX 5090

## Project Checklist

- [x] All tests pass (`pytest`)
- [x] Documentation complete (`README.md`)
- [x] Inference scripts verified (`run_inference.py`, `test_dataset_inference.py`)
- [x] Demo images present (`demo_images/`)
- [x] Required plots present (`results/plots/`)
- [x] Grad-CAM visualizations included (`results/grad_cam/`)
- [x] ONNX export verified (`models/exports/`)
- [x] No redundant or temporary files in repository
- [x] `.gitignore` properly configured
- [x] Project structure matches documented layout
- [x] Full pipeline is reproducible (clean clone → train → test → export → inference)

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

# Install dependencies
pip install -r requirements.txt
```

2. **Configuration**

```bash
# Copy environment template
cp env.example .env

# Edit .env with your settings
nano .env
```

## Training the Model

1. **Prepare your dataset**:

```bash
# Organize raw images into processed format
python scripts/prepare_dataset.py --raw_dir data/raw --output_dir data/processed

# (Optional) Generate augmented data
python scripts/augment_dataset.py --input_dir data/processed --output_dir data/augmented
```

2. **Start training**:

```bash
# Train with default parameters
python scripts/train.py --data_dir data/processed --epochs 50 --batch_size 32 --device cuda

# Train with augmented data
python scripts/train.py --data_dir data/augmented --epochs 50 --batch_size 32 --device cuda
```

Training outputs will be saved to:

- Model checkpoints: `models/checkpoints/type_detector_best.pth` (best model)
- Training plots: `results/plots/`
- Logs: `results/logs/`

## Evaluation and Testing

1. **Run inference on test dataset**:

```bash
python scripts/test_dataset_inference.py --test_dir data/processed/test --model models/checkpoints/type_detector_best.pth
```

2. **Generate Grad-CAM visualizations**:

```bash
python scripts/grad_cam.py --image path/to/image.jpg --model models/checkpoints/type_detector_best.pth
```

3. **Run full test suite**:

```bash
pytest tests/
```

Test outputs will be saved to:

- Confusion matrices: `results/confusion_matrices/`
- Grad-CAM visualizations: `results/grad_cam/`
- Test reports: `results/test_reports/`

## 📤 Outputs

After training and evaluation, the following outputs will be generated:

- `results/plots/` (Required for presentation)
  - `confusion_matrix.png` - Model performance visualization
  - `training_loss.png` - Training convergence plot
  - `training_accuracy.png` - Accuracy progression
- `results/grad_cam/`
  - Grad-CAM visualizations of selected images
  - Interpretability examples for presentation
- `results/test_reports/`
  - Detailed test report (classification report)
  - Per-class accuracy metrics
- `demo_images/`
  - One example image per class (fork_a, fork_b, knife_a, etc.)
  - Selected Grad-CAM examples for interpretability
  - Required for project presentation
- `models/checkpoints/`
  - `type_detector_best.pth` - Best model checkpoint for inference
  - `type_detector_latest.pth` - Latest model checkpoint (backup)
- `models/exports/`
  - ONNX exported model for future deployment
  - Optimized for edge device deployment

### Export model to ONNX (optional for future deployment):

```bash
# Export to ONNX format
python scripts/export_model.py --model models/checkpoints/type_detector_best.pth --format onnx --optimize

# Export to both ONNX and TorchScript (recommended)
python scripts/export_model.py --model models/checkpoints/type_detector_best.pth --format all --optimize

# Verify exported model
python scripts/export_model.py --model models/checkpoints/type_detector_best.pth --format onnx --test
```

The ONNX export enables:

- Deployment to edge devices
- Framework-independent inference
- Runtime optimization
- Quantization support

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

| Metric         | Value               |
| -------------- | ------------------- |
| Accuracy       | 90.00% (latest run) |
| Inference Time | <200ms on RTX 5090  |
| Memory Usage   | <512MB              |

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

## Hyperparameter Tuning

The project includes Optuna-based hyperparameter optimization. To run the tuning process:

```bash
python scripts/tune_type_detector.py --n_trials 50
```

This will:

1. Run 50 trials of hyperparameter optimization
2. Save the best model as `models/checkpoints/type_detector_best_tuned.pth`
3. Generate visualization plots in `results/optuna_plots/`
4. Save the best configuration to `results/best_tuning_config.json`

Parameters tuned:

- Learning rate
- Weight decay
- Dropout rate
- Optimizer choice (Adam vs SGD)
- Momentum (when using SGD)

## Plotting Optuna Results

After running hyperparameter optimization, you can generate visualization plots using:

```bash
python scripts/plot_optuna_results.py
```

The script uses Matplotlib to generate the following plots in `results/optuna_plots/`:

- `optimization_history.png`: Shows the optimization progress
- `param_importances.png`: Shows the relative importance of each parameter
- `parallel_coordinates.png`: Shows parameter relationships (if available)

You can customize the input/output paths:

```bash
python scripts/plot_optuna_results.py --study_db path/to/study.db --output_dir path/to/plots/
```

```

```
