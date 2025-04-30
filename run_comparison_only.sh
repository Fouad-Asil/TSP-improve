#!/bin/bash
# Script to run only the comparison part of the experiment (without regenerating tours)

# Default parameters
SIZE=20
EPOCHS=10
OVERLAP_WEIGHT=0.2
BREAK_PENALTY=0.0
TOURS_PATH="./datasets/greedy_tours.pkl"

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
      OVERLAP_WEIGHT="$2"
      shift 2
      ;;
    --break_penalty)
      BREAK_PENALTY="$2"
      shift 2
      ;;
    --optimal_tours_path)
      TOURS_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--graph_size SIZE] [--epochs EPOCHS] [--overlap_weight OVERLAP] [--break_penalty PENALTY] [--optimal_tours_path PATH]"
      exit 1
      ;;
  esac
done

echo "=== Running comparison experiment (comparison only) ==="
echo "Graph size: $SIZE"
echo "Epochs: $EPOCHS"
echo "Overlap weight (beta): $OVERLAP_WEIGHT"
echo "Break penalty (gamma): $BREAK_PENALTY"
echo "Tours path: $TOURS_PATH"

# Set working directory
cd "$(dirname "$0")"

# Create output directory for logs
LOG_DIR="./experiment_logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d%H%M%S)
LOG_FILE="$LOG_DIR/comparison_only_$TIMESTAMP.log"

# Install dependencies (including OpenCV)
echo "Installing required packages..." | tee -a "$LOG_FILE"
pip install -q tensorboardX
pip install -q seaborn pandas matplotlib numpy tqdm
pip install -q opencv-python
pip install -q torch

# Create required directories
mkdir -p datasets
mkdir -p experiment_logs
mkdir -p outputs

# Check if tours file exists
if [ ! -f "$TOURS_PATH" ]; then
  echo "Error: Tours file not found at $TOURS_PATH" | tee -a "$LOG_FILE"
  echo "Please run the tour generation first or specify correct path with --optimal_tours_path" | tee -a "$LOG_FILE"
  exit 1
fi

echo "Starting comparison between baseline and edge overlap methods..." | tee -a "$LOG_FILE"

# Run comparison part only
python compare_rewards.py \
  --n_epochs "$EPOCHS" \
  --graph_size "$SIZE" \
  --batch_size 512 \
  --epoch_size 5120 \
  --optimal_tours_path "$TOURS_PATH" \
  --overlap_weight "$OVERLAP_WEIGHT" \
  --break_penalty "$BREAK_PENALTY" \
  --seed 1234 2>&1 | tee -a "$LOG_FILE"

if [ $? -ne 0 ]; then
  echo "Experiment failed. Please check the error message above." | tee -a "$LOG_FILE"
  exit 1
fi

echo "Experiment completed successfully!" | tee -a "$LOG_FILE"
echo "You can find the results in ./outputs/tsp_${SIZE}/comparison_results.json" | tee -a "$LOG_FILE"
echo "And logs in $LOG_FILE" | tee -a "$LOG_FILE"

# Copy the final JSON results to the log directory for easier access
RESULTS_FILE="./outputs/tsp_${SIZE}/comparison_results.json"
if [ -f "$RESULTS_FILE" ]; then
  cp "$RESULTS_FILE" "$LOG_DIR/comparison_results_$TIMESTAMP.json"
  echo "Results also copied to $LOG_DIR/comparison_results_$TIMESTAMP.json" | tee -a "$LOG_FILE"
fi 