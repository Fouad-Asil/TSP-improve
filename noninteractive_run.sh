#!/bin/bash
# Non-interactive script to run the comparison experiment

# Default parameters
SIZE=20
EPOCHS=10
OVERLAP=0.2
PENALTY=0.0
REGENERATE=false

# Parse command-line options
while [[ $# -gt 0 ]]; do
  case $1 in
    --graph_size)
      SIZE="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --overlap_weight)
      OVERLAP="$2"
      shift 2
      ;;
    --break_penalty)
      PENALTY="$2"
      shift 2
      ;;
    --regenerate_tours)
      REGENERATE=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--graph_size SIZE] [--epochs EPOCHS] [--overlap_weight OVERLAP] [--break_penalty PENALTY] [--regenerate_tours]"
      exit 1
      ;;
  esac
done

echo "=== Running comparison experiment with non-interactive mode ==="
echo "Graph size: $SIZE"
echo "Epochs: $EPOCHS"
echo "Overlap weight (beta): $OVERLAP"
echo "Break penalty (gamma): $PENALTY"
echo "Regenerate tours: $REGENERATE"

# Set working directory
cd "$(dirname "$0")"

# Install TensorBoard alternatives
echo "Installing TensorBoard alternatives..."
pip install -q tensorboardX
pip install -q seaborn pandas matplotlib numpy tqdm
pip install -q opencv-python  # Add OpenCV for visualization

# Create required directories
mkdir -p datasets
mkdir -p experiment_logs
mkdir -p outputs

# Make scripts executable
chmod +x run_comparison.sh
chmod +x install_deps.sh

# Check if we should regenerate tours
if [ "$REGENERATE" = true ] || [ ! -f "./datasets/greedy_tours.pkl" ]; then
    echo "Generating greedy tours..."
    rm -f "./datasets/greedy_tours.pkl" 2>/dev/null
else
    echo "Using existing tours file."
fi

# Create a custom config file
cat > experiment_config.txt << EOF
graph_size=$SIZE
n_epochs=$EPOCHS
overlap_weight=$OVERLAP
break_penalty=$PENALTY
EOF

echo "Configuration saved to experiment_config.txt"

# Run with small batch size first to check for errors in tour generation
if [ "$REGENERATE" = true ] || [ ! -f "./datasets/greedy_tours.pkl" ]; then
    echo "Starting tour generation with small batch size to test..."
    python generate_greedy_tours.py --graph_size "$SIZE" --num_samples 100 --batch_size 10 --output_path "./datasets/test_tours.pkl"

    if [ $? -ne 0 ]; then
        echo "Error in tour generation test. Please check the error message above."
        exit 1
    fi
    echo "Tour generation test successful."
fi

echo "Starting main experiment..."

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