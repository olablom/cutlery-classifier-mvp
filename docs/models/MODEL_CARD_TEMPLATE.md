# Model Card: Cutlery Type Detector

## Model Details

**Model Name:** Cutlery Type Detector v1.0  
**Model Type:** Image Classification (Hierarchical Pipeline Stage 1)  
**Architecture:** ResNet18 with grayscale input adaptation  
**Framework:** PyTorch 2.7.0  
**Created:** [DATE]  
**Version:** 1.0.0

## Intended Use

**Primary Use Case:** Automated cutlery type classification in commercial kitchen environments

**Intended Users:**

- Commercial kitchen automation systems
- Food service equipment manufacturers
- Research institutions studying automated sorting

**Out-of-Scope Uses:**

- General object detection
- Fine-grained cutlery brand identification (use Stage 2 models)
- Real-time video processing without optimization

## Model Architecture

**Base Model:** ResNet18  
**Input:** Grayscale images (1 channel, 320x320 pixels)  
**Output:** 3-class probability distribution (fork, knife, spoon)  
**Parameters:** ~11.2M  
**Model Size:** ~43MB (.pth), ~22MB (.onnx)

**Key Modifications:**

- Adapted first convolutional layer for grayscale input
- Pretrained ImageNet weights averaged across RGB channels
- Custom classifier head for 3-class output

## Training Data

**Dataset Size:** [TO_BE_FILLED] images  
**Classes:** 3 (fork, knife, spoon)  
**Split:** 70% train, 20% validation, 10% test

**Data Collection:**

- Controlled lighting conditions (4000-5000K, 80-100% brightness)
- Standardized background (white surface with black guidelines)
- Multiple angles and positions per cutlery piece
- iPhone 13 camera (4032x3024 resolution)

**Preprocessing:**

- Resize to 320x320 pixels
- Convert to grayscale
- Normalize: mean=0.449, std=0.226

## Performance Metrics

**Test Set Performance:**

- Accuracy: [TO_BE_FILLED]%
- Precision: [TO_BE_FILLED]%
- Recall: [TO_BE_FILLED]%
- F1-Score: [TO_BE_FILLED]%

**Inference Speed:**

- CPU (Intel i7): ~[TO_BE_FILLED]ms per image
- GPU (RTX 3050): ~[TO_BE_FILLED]ms per image

**Confusion Matrix:** [TO_BE_FILLED]

## Training Details

**Training Configuration:**

- Optimizer: Adam (lr=0.001, weight_decay=0.0001)
- Scheduler: StepLR (step_size=10, gamma=0.1)
- Batch Size: 32
- Epochs: 30
- Early Stopping: 8 epochs patience

**Data Augmentation:**

- Horizontal flip (50% probability)
- Rotation (±15 degrees)
- Color jitter (brightness ±20%, contrast ±20%)
- Gaussian blur (10% probability)

**Hardware:** NVIDIA RTX 3050, 16GB RAM  
**Training Time:** ~[TO_BE_FILLED] minutes

## Evaluation

**Validation Strategy:** Stratified train/validation/test split  
**Cross-Validation:** Not performed (sufficient data per class)  
**Interpretability:** Grad-CAM visualizations available

**Known Limitations:**

- Trained on controlled lighting conditions
- Limited to specific cutlery types (standard fork/knife/spoon)
- Performance may degrade with heavily worn or damaged cutlery

## Ethical Considerations

**Bias Assessment:**

- Training data collected from limited manufacturer set
- Potential bias toward specific cutlery designs
- Recommended to validate on target deployment environment

**Environmental Impact:**

- Training carbon footprint: Minimal (short training time, efficient architecture)
- Deployment efficiency: Optimized for edge devices

## Deployment

**Supported Formats:**

- PyTorch (.pth): Full training checkpoint with metadata
- ONNX (.onnx): Cross-platform deployment
- TorchScript (.pt): PyTorch mobile and C++ integration

**Hardware Requirements:**

- Minimum: CPU with 4GB RAM
- Recommended: GPU with 2GB+ VRAM for batch processing
- Edge deployment: ARM processors with 1GB+ RAM

**Integration Example:**

```python
import torch
from PIL import Image
from torchvision import transforms

# Load model
model = torch.jit.load('cutlery_type_detector_v1.pt')
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.449], std=[0.226])
])

# Inference
image = Image.open('cutlery.jpg')
input_tensor = transform(image).unsqueeze(0)
with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.softmax(output, dim=1)
```

## Model Governance

**Version Control:** Semantic versioning (MAJOR.MINOR.PATCH)  
**Update Policy:** New versions for architecture changes or significant performance improvements  
**Monitoring:** Performance tracking on deployment data recommended

**Contact Information:**

- Developer: Ola Blom
- Institution: [INSTITUTION]
- Email: [EMAIL]

## Changelog

### v1.0.0 (Initial Release)

- ResNet18 architecture with grayscale adaptation
- Trained on controlled dataset
- Achieved [TO_BE_FILLED]% accuracy on test set
- Available in PyTorch, ONNX, and TorchScript formats

---

**Note:** This model is part of a hierarchical classification system. For manufacturer-specific classification, use the appropriate Stage 2 models after type detection.
