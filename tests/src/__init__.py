"""
Cutlery Classifier MVP

A hierarchical AI classification system for cutlery identification by type and manufacturer.
Optimized for commercial kitchen automation and embedded deployment.
"""

__version__ = "1.0.0"
__author__ = "Ola Blom"
__email__ = "ola.blom@example.com"

# Core modules
from . import data
from . import models
from . import training
from . import evaluation
from . import inference
from . import utils

__all__ = ["data", "models", "training", "evaluation", "inference", "utils"]
