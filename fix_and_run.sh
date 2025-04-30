#!/bin/bash
# Script to fix all issues and run the experiment

echo "=== Fixing all issues and running the comparison experiment ==="

# Set working directory
cd "$(dirname "$0")"

# Install TensorBoard alternatives
echo "Installing TensorBoard alternatives..."
pip install -q tensorboardX
pip install -q seaborn pandas matplotlib numpy tqdm

# Create required directories
mkdir -p datasets
mkdir -p experiment_logs
mkdir -p outputs

# Make scripts executable
chmod +x run_comparison.sh
chmod +x install_deps.sh

# Check if the tours file already exists
if [ -f "./datasets/greedy_tours.pkl" ]; then
    echo "Greedy tours file already exists. Do you want to regenerate it? (y/n)"
    read -r REGENERATE
    if [[ $REGENERATE == "y" ]]; then
        rm -f "./datasets/greedy_tours.pkl"
        echo "Deleted existing tours file. Will regenerate."
    else
        echo "Using existing tours file."
    fi
fi

# Ask for experiment parameters
echo "Enter graph size (default: 20):"
read -r SIZE
SIZE=${SIZE:-20}

echo "Enter number of epochs (default: 10):"
read -r EPOCHS
EPOCHS=${EPOCHS:-10}

echo "Enter edge overlap reward weight (beta) between 0.0-1.0 (default: 0.2):"
read -r OVERLAP
OVERLAP=${OVERLAP:-0.2}

echo "Enter breaking penalty weight (gamma) between 0.0-1.0 (default: 0.0):"
read -r PENALTY
PENALTY=${PENALTY:-0.0}

# Create a custom config file to avoid command-line parameter issues
cat > experiment_config.txt << EOF
graph_size=$SIZE
n_epochs=$EPOCHS
overlap_weight=$OVERLAP
break_penalty=$PENALTY
EOF

echo "Configuration saved to experiment_config.txt"

# Run with small batch size first to check for errors in tour generation
echo "Starting tour generation with small batch size to test..."
python generate_greedy_tours.py --graph_size "$SIZE" --num_samples 100 --batch_size 10 --output_path "./datasets/test_tours.pkl"

if [ $? -ne 0 ]; then
    echo "Error in tour generation test. Please check the error message above."
    exit 1
fi

echo "Tour generation test successful. Starting main experiment..."

# Run the actual experiment with configured parameters
./run_comparison.sh \
  --graph_size "$SIZE" \
  --epochs "$EPOCHS" \
  --overlap_weight "$OVERLAP" \
  --break_penalty "$PENALTY"

if [ $? -ne 0 ]; then
    echo "Experiment failed. Please check the error message above."
    exit 1
fi

echo "Experiment completed successfully!"
echo "You can find the results in ./outputs/tsp_${SIZE}/comparison_results.json"
echo "And logs in ./experiment_logs/" 