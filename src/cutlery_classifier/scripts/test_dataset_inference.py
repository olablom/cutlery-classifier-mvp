#!/usr/bin/env python3
"""
Console script entry point for test dataset inference.
"""

import sys
from pathlib import Path

# Add project root to path to import the actual script
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.test_dataset_inference import main

if __name__ == "__main__":
    main()
