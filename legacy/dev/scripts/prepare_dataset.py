#!/usr/bin/env python3
"""
[LEGACY] Dataset Preparation Script - NO LONGER MAINTAINED

⚠️ WARNING: This script is no longer part of the active pipeline!
The current project expects a pre-organized dataset structure:

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

Please organize your dataset manually following this structure.
"""

import sys
import logging

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def main():
    logger.error("""
⚠️ This script is no longer maintained or supported!

The current pipeline expects a pre-organized dataset structure:
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

Please:
1. Create these directories manually
2. Place your images in the appropriate folders
3. Proceed with training using:
   python scripts/train_type_detector.py --device cuda

For more information, see the "Dataset Structure" section in README.md
""")
    sys.exit(1)


if __name__ == "__main__":
    main()
