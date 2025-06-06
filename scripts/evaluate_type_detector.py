import os
import logging
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from src.models.factory import create_model
from src.data.dataset import CutleryDataset
from src.evaluation.metrics import compute_metrics, plot_confusion_matrix
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Configuration
    model_path = os.path.join("models", "checkpoints", "type_detector_best.pth")
    test_data_dir = os.path.join("data", "processed", "test")
    results_dir = "results"
    batch_size = 32
    input_size = (320, 320)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Create and load model
    model = create_model("resnet18", num_classes=6, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Data transforms
    transform = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229]),
        ]
    )

    # Create test dataset and loader
    test_dataset = CutleryDataset(test_data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Evaluate
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Compute and save metrics
    metrics = compute_metrics(all_labels, all_preds, test_dataset.classes)
    logger.info(f"Test Metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.2f}")

    # Save confusion matrix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cm_path = os.path.join(
        results_dir, "confusion_matrices", f"test_confusion_matrix_{timestamp}.png"
    )
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    plot_confusion_matrix(all_labels, all_preds, test_dataset.classes, cm_path)
    logger.info(f"Confusion matrix saved to: {cm_path}")


if __name__ == "__main__":
    main()
