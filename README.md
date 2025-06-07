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

## 🚀 Full Pipeline Example

This is the complete pipeline for training, testing and running inference with the cutlery classifier:

### 1️⃣ (Optional) Data Augmentation with Diffusion

```bash
PYTHONPATH=. python scripts/augment_dataset.py --input_dir data/processed/train --output_dir data/augmented/train --classes fork_a fork_b knife_a knife_b spoon_a spoon_b
```

### 2️⃣ Training with Mixed Real + Augmented Data

```bash
PYTHONPATH=. python scripts/train_type_detector.py --device cuda --config results/best_tuning_config.json --mixed-data
```

### 3️⃣ Testing on Test Set

```bash
PYTHONPATH=. python scripts/test_dataset_inference.py --device cuda --test_dir data/processed/test --model models/checkpoints/type_detector_best.pth --save-misclassified
```

### 4️⃣ Single Image Inference with Grad-CAM

```bash
PYTHONPATH=. python scripts/run_inference.py --device cuda --image "data/processed/test/fork_a/IMG_0941[1].jpg" --grad-cam
```

### 💾 Dataset Structure

```
data/processed/
├── train/
│   ├── fork_a/
│   ├── fork_b/
│   ├── knife_a/
│   ├── knife_b/
│   ├── spoon_a/
│   ├── spoon_b/
├── val/
│   ├── fork_a/
│   ├── fork_b/
│   ├── knife_a/
│   ├── knife_b/
│   ├── spoon_a/
│   ├── spoon_b/
└── test/
    ├── fork_a/
    ├── fork_b/
    ├── knife_a/
    ├── knife_b/
    ├── spoon_a/
    ├── spoon_b/
```

### 📝 Notes

- Data augmentation is optional but improves model performance
- The classifier is based on a ResNet18 architecture with grayscale preprocessing
- The pipeline supports ONNX export for future deployment

## System Requirements

- Python 3.11 or higher
- CUDA 11.8 (for GPU acceleration)
- NVIDIA GPU with CUDA support
- 8GB RAM minimum (16GB recommended for training)

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

## Project Structure Note

Some helper scripts have been moved to `scripts/archive/` for cleaner organization. The core pipeline consists of:

- `train_type_detector.py` - Training
- `test_dataset_inference.py` - Testing
- `tune_type_detector.py` - Hyperparameter tuning
- `run_inference.py` - Single image inference

All results are saved in timestamped directories:
`results/run_YYYYMMDD_HHMMSS/`

## Training the Model

### Dataset Structure

The project expects data to be organized in the following structure:

```
data/
  processed/
    train/
      fork/
      knife/
      spoon/
    val/
      fork/
      knife/
      spoon/
    test/
      fork/
      knife/
      spoon/
```

Place your images in these directories before proceeding with training.

### Optional: Data Augmentation

To improve model generalization, you can generate additional training data:

```bash
# Generate augmented training data (optional but recommended)
python scripts/augment_dataset.py --input_dir data/processed/train --output_dir data/augmented --classes fork_a fork_b knife_a knife_b spoon_a spoon_b

# Or use default classes (same as above)
python scripts/augment_dataset.py --input_dir data/processed/train --output_dir data/augmented
```

Note: The script will process all cutlery variants (fork_a, fork_b, knife_a, knife_b, spoon_a, spoon_b) by default.

### Start Training

```bash
# Train with default configuration
python scripts/train_type_detector.py --device cuda

# Or train with tuned hyperparameters
python scripts/train_type_detector.py --device cuda --config results/best_tuning_config.json
```

Training outputs will be saved to:

- Model checkpoints: `models/checkpoints/type_detector_best.pth`
- Training plots and logs: `results/run_YYYYMMDD_HHMMSS/`

### Legacy Note

> ⚠️ The `prepare_dataset.py` script mentioned in older versions is no longer maintained.
> The current pipeline expects a pre-organized dataset structure as shown above.
> If you need to split your own dataset, please organize the images manually into train/val/test folders.

## Evaluation and Testing

1. **Run inference on test dataset**:

```bash
python scripts/test_dataset_inference.py --device cuda --test_dir data/processed/test --model models/checkpoints/type_detector_best.pth --save-misclassified
```

2. **Run inference with Grad-CAM visualization**:

```bash
python scripts/run_inference.py --device cuda --image path/to/image.jpg --grad-cam
```

3. **Run full test suite**:

```bash
pytest tests/
```

Test outputs will be saved to the timestamped run directory:
`results/run_YYYYMMDD_HHMMSS/`

## 📤 Outputs

After training and evaluation, the following outputs will be generated in `results/run_YYYYMMDD_HHMMSS/`:

- Confusion matrix (`confusion_matrix.png`)
- Training curves:
  - `training_loss.png`
  - `training_accuracy.png`
- Test results:
  - `test_results.txt` - Detailed classification report
  - `examples_correct/` - Correctly classified examples
  - `examples_incorrect/` - Misclassified examples with Grad-CAM
- Model information:
  - `type_detector_info.txt` - Model architecture and parameters

Additional project files:

- `models/checkpoints/type_detector_best.pth` - Best model checkpoint
- `models/exports/` - ONNX exports (if generated)
- `demo_images/` - Example images per class (required for presentation)

Note: All results are organized in timestamped run directories for easy tracking and comparison.

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

## 🚀 Full Pipeline Example

Here's a complete example of running the full classification pipeline:

### 1️⃣ Train model with best tuning config

```bash
PYTHONPATH=. python scripts/train_type_detector.py --device cuda --config results/best_tuning_config.json
```

### 2️⃣ Test entire test set (VG level testing)

```bash
PYTHONPATH=. python scripts/test_dataset_inference.py --device cuda --test_dir data/processed/test --model models/checkpoints/type_detector_best.pth --save-misclassified
```

### 3️⃣ Run single image inference with Grad-CAM

```bash
PYTHONPATH=. python scripts/run_inference.py --device cuda --image "data/processed/test/fork_a/IMG_0941[1].jpg" --grad-cam
```

All outputs will be saved in timestamped directories under `results/run_YYYYMMDD_HHMMSS/`.

```

```
