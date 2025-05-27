# Inference Pipeline Guide

This guide covers how to use the complete inference pipeline for the Cutlery Classifier MVP, including single image inference, batch processing, visualization, and model export.

## Overview

The inference pipeline consists of three main components:

1. **CutleryInferencer** (`src/inference/inferencer.py`) - Core inference engine
2. **CLI Interface** (`scripts/infer_image.py`) - Command-line interface for inference
3. **Model Export** (`scripts/export_model.py`) - Export models for deployment

## Quick Start

### 1. Basic Inference

```bash
# Simple prediction on a single image
python scripts/infer_image.py --model models/checkpoints/type_detector_best.pth --image test.jpg
```

**Output:**

```
============================================================
🔍 CUTLERY CLASSIFICATION RESULTS
============================================================
🏆 Prediction: FORK
📊 Confidence: 87.3%
⚡ Inference Time: 45.2ms
============================================================
```

### 2. Visual Inference

```bash
# Create visualization with prediction overlay
python scripts/infer_image.py --model models/checkpoints/type_detector_best.pth --image test.jpg --visualize --output result.jpg
```

This creates a visualization showing:

- Original image with prediction overlay
- Class name and confidence percentage
- Inference time
- Top-k predictions (if requested)

### 3. Batch Processing

```bash
# Process all images in a directory
python scripts/infer_image.py --model models/checkpoints/type_detector_best.pth --batch data/test_images/ --output results/
```

## Detailed Usage

### Single Image Inference

#### Basic Options

```bash
# Top-3 predictions
python scripts/infer_image.py --model models/checkpoints/type_detector_best.pth --image test.jpg --top-k 3

# Show model information
python scripts/infer_image.py --model models/checkpoints/type_detector_best.pth --image test.jpg --info

# Verbose output with technical details
python scripts/infer_image.py --model models/checkpoints/type_detector_best.pth --image test.jpg --verbose
```

#### Visualization Options

```bash
# Custom font size for overlay text
python scripts/infer_image.py --model models/checkpoints/type_detector_best.pth --image test.jpg --visualize --font-size 32

# Display image after processing (requires GUI)
python scripts/infer_image.py --model models/checkpoints/type_detector_best.pth --image test.jpg --visualize --show

# Custom output filename
python scripts/infer_image.py --model models/checkpoints/type_detector_best.pth --image test.jpg --visualize --output my_result.jpg
```

#### Output Formats

```bash
# Save results as JSON
python scripts/infer_image.py --model models/checkpoints/type_detector_best.pth --image test.jpg --json

# Custom JSON filename
python scripts/infer_image.py --model models/checkpoints/type_detector_best.pth --image test.jpg --json my_results.json
```

### Batch Processing

#### Directory Processing

```bash
# Process all images in directory
python scripts/infer_image.py --model models/checkpoints/type_detector_best.pth --batch data/test_images/

# With output directory for results
python scripts/infer_image.py --model models/checkpoints/type_detector_best.pth --batch data/test_images/ --output results/

# Create visualizations for all images
python scripts/infer_image.py --model models/checkpoints/type_detector_best.pth --batch data/test_images/ --visualize --output results/
```

#### Batch Output

Batch processing creates:

- `batch_results.json` - Complete results for all images
- Individual visualization images (if `--visualize` is used)
- Summary statistics in terminal

**Example batch output:**

```
================================================================================
📊 BATCH PROCESSING RESULTS
================================================================================
✅ Successful: 25/25
  1. fork_001.jpg                → fork           (92.1%)
  2. knife_002.jpg               → knife          (88.7%)
  3. spoon_003.jpg               → spoon          (95.3%)
  ...
================================================================================
```

### Hardware and Performance

#### Device Selection

```bash
# Force CPU usage
python scripts/infer_image.py --model models/checkpoints/type_detector_best.pth --image test.jpg --device cpu

# Force GPU usage (if available)
python scripts/infer_image.py --model models/checkpoints/type_detector_best.pth --image test.jpg --device cuda
```

#### Performance Expectations

| Hardware          | Inference Time | Batch Throughput |
| ----------------- | -------------- | ---------------- |
| RTX 3050          | 15-30ms        | ~50 images/sec   |
| RTX 4060          | 10-20ms        | ~80 images/sec   |
| CPU (Intel i7)    | 100-200ms      | ~8 images/sec    |
| CPU (AMD Ryzen 7) | 80-150ms       | ~12 images/sec   |

## Model Export

### ONNX Export

```bash
# Basic ONNX export
python scripts/export_model.py --model models/checkpoints/type_detector_best.pth --format onnx

# Optimized ONNX export
python scripts/export_model.py --model models/checkpoints/type_detector_best.pth --format onnx --optimize

# Test exported model
python scripts/export_model.py --model models/checkpoints/type_detector_best.pth --format onnx --test
```

### TorchScript Export

```bash
# TorchScript export (tracing method)
python scripts/export_model.py --model models/checkpoints/type_detector_best.pth --format torchscript

# TorchScript export (scripting method)
python scripts/export_model.py --model models/checkpoints/type_detector_best.pth --format torchscript --torchscript-method script
```

### Export All Formats

```bash
# Export to both ONNX and TorchScript
python scripts/export_model.py --model models/checkpoints/type_detector_best.pth --format all --optimize --test
```

## Programming Interface

### Using CutleryInferencer in Code

```python
from src.inference.inferencer import CutleryInferencer

# Initialize inferencer
inferencer = CutleryInferencer(
    model_path="models/checkpoints/type_detector_best.pth",
    device="cuda"  # or "cpu"
)

# Single prediction
results = inferencer.predict("test.jpg", top_k=3)
print(f"Prediction: {results['top_prediction']['class_name']}")
print(f"Confidence: {results['top_prediction']['percentage']:.1f}%")

# Prediction with visualization
results, vis_image = inferencer.predict_with_visualization(
    "test.jpg",
    output_path="result.jpg"
)

# Batch prediction
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
batch_results = inferencer.batch_predict(image_paths)

# Model information
model_info = inferencer.get_model_info()
print(f"Architecture: {model_info['architecture']}")
print(f"Classes: {model_info['class_names']}")
```

### Result Format

```python
# Single prediction result
{
    "predictions": [
        {
            "class_name": "fork",
            "class_index": 0,
            "confidence": 0.873,
            "percentage": 87.3
        },
        # ... additional predictions if top_k > 1
    ],
    "top_prediction": {
        "class_name": "fork",
        "class_index": 0,
        "confidence": 0.873,
        "percentage": 87.3
    },
    "inference_time_ms": 45.2,
    "model_path": "models/checkpoints/type_detector_best.pth",
    "device": "cuda"
}
```

## Troubleshooting

### Common Issues

#### 1. Model Not Found

```
Error: Model file not found: models/checkpoints/type_detector_best.pth
```

**Solution:** Train a model first:

```bash
python scripts/train_type_detector.py --epochs 30
```

#### 2. CUDA Out of Memory

```
Error: CUDA out of memory
```

**Solutions:**

- Use CPU: `--device cpu`
- Reduce batch size in training
- Close other GPU applications

#### 3. Image Not Found

```
Error: Image file not found: test.jpg
```

**Solution:** Check file path and format. Supported formats: JPG, JPEG, PNG, BMP

#### 4. Import Errors

```
ImportError: No module named 'src.inference'
```

**Solution:** Run from project root directory:

```bash
cd cutlery-classifier-mvp
python scripts/infer_image.py ...
```

### Performance Issues

#### Slow Inference

1. **Use GPU:** Add `--device cuda` if available
2. **Check model size:** Larger models are slower
3. **Optimize exports:** Use `--optimize` flag for ONNX export

#### Memory Issues

1. **Reduce batch size:** Process fewer images at once
2. **Use CPU:** Add `--device cpu` to use system RAM instead of VRAM
3. **Close applications:** Free up system memory

### Visualization Issues

#### Font Problems

```
Warning: Could not load font, using default
```

**Solution:** Install system fonts or ignore (default font works fine)

#### Display Issues

```
Warning: Could not display image
```

**Solution:** Remove `--show` flag or ensure GUI environment is available

## Best Practices

### 1. Image Preparation

- **Resolution:** 320x320 pixels optimal, but any size works
- **Format:** JPEG recommended for smaller file sizes
- **Quality:** Clear, well-lit images work best
- **Background:** Varied backgrounds improve generalization

### 2. Batch Processing

- **Organization:** Keep images in organized directories
- **Naming:** Use descriptive filenames for easier result analysis
- **Output:** Always specify output directory for batch results

### 3. Model Selection

- **Type Detector:** Use for general cutlery classification
- **Manufacturer Models:** Use for specific manufacturer identification
- **Export Format:** ONNX for cross-platform, TorchScript for PyTorch ecosystem

### 4. Performance Optimization

- **GPU Usage:** Always use GPU when available for faster inference
- **Batch Size:** Process multiple images together when possible
- **Model Export:** Use optimized exports for production deployment

## Integration Examples

### Web Application

```python
from flask import Flask, request, jsonify
from src.inference.inferencer import CutleryInferencer

app = Flask(__name__)
inferencer = CutleryInferencer("models/checkpoints/type_detector_best.pth")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    results = inferencer.predict(file)
    return jsonify(results)
```

### Real-time Processing

```python
import cv2
from src.inference.inferencer import CutleryInferencer

inferencer = CutleryInferencer("models/checkpoints/type_detector_best.pth")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret:
        results = inferencer.predict(frame)
        # Display results on frame
        cv2.imshow('Cutlery Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### Industrial Integration

```python
# Using ONNX Runtime for production
import onnxruntime as ort
import numpy as np
from PIL import Image

# Load ONNX model
session = ort.InferenceSession("models/exports/type_detector.onnx")

# Preprocess image
image = Image.open("cutlery.jpg").convert("RGB")
image = image.resize((320, 320))
image_array = np.array(image).astype(np.float32) / 255.0
image_array = np.transpose(image_array, (2, 0, 1))  # HWC to CHW
image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

# Run inference
outputs = session.run(None, {"input": image_array})
predictions = outputs[0]
```
