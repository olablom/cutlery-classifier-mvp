# src/utils/device.py

import torch
import logging

logger = logging.getLogger(__name__)


def get_device(preferred_device: str = "cuda") -> torch.device:
    if preferred_device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        preferred_device = "cpu"
    return torch.device(preferred_device)
