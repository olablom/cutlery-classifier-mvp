"""Inference module for cutlery classifier."""

import logging
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

from src.models.factory import create_model

logger = logging.getLogger(__name__)


def predict_image(
    model: torch.nn.Module, image_path: str, device: str = "cuda"
) -> tuple[str, float]:
    """Predict the class of an image.

    Args:
        model: The model to use for prediction
        image_path: Path to the image file
        device: Device to run inference on

    Returns:
        tuple[str, float]: Predicted class and confidence score
    """
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")

    # Define transforms
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Preprocess image
    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)

    # Get prediction
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    # Map class index to label
    class_labels = ["fork", "knife", "spoon"]
    predicted_class = class_labels[predicted.item()]
    confidence_score = confidence.item()

    return predicted_class, confidence_score
