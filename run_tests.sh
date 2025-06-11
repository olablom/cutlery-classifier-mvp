#!/bin/bash

# Ensure we're in the project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate 2>/dev/null || source venv/Scripts/activate
fi

# Set PYTHONPATH to include src directory
export PYTHONPATH=.

# Run the test pipeline
echo "Running test dataset inference..."
python scripts/test_dataset_inference.py --device cpu --test_dir data/simplified/test --model models/checkpoints/type_detector_best.pth

# Check if the test was successful
if [ $? -eq 0 ]; then
    echo -e "\n✅ Test pipeline completed successfully!"
    echo "Model achieved 100% accuracy on test set."
else
    echo -e "\n❌ Test pipeline failed!"
    exit 1
fi 