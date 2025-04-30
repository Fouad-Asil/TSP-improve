#!/bin/bash
# Script to install required dependencies for TSP-Improve

echo "Installing dependencies for TSP-Improve with edge overlap reward..."

# Create a virtual environment (optional)
# python -m venv tsp-env
# source tsp-env/bin/activate

# Install main dependencies
pip install torch numpy tqdm matplotlib

# Install OpenCV for visualization
pip install opencv-python

# Install tensorboard_logger
# The original package might be outdated, so install tensorboardX which is maintained
pip install tensorboardX

# Also install the original tensorboard_logger as fallback
pip install -q tensorboard_logger || echo "Couldn't install tensorboard_logger, will use tensorboardX instead"

# Create required directories
mkdir -p datasets
mkdir -p experiment_logs
mkdir -p outputs

echo "Installing other libraries for visualization and computation..."
pip install -q seaborn pandas scikit-learn || echo "Optional visualization libraries not installed"

echo "Dependencies installed!"
echo "Note: If you see errors with tensorboard_logger, the script will automatically fall back to tensorboardX."
echo "You can now run the experiment with './run_comparison.sh'"

# Make the run script executable
chmod +x run_comparison.sh

echo "Done!" 