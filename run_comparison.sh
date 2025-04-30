#!/bin/bash
# Script to run the complete comparison experiment

# Set defaults
GRAPH_SIZE=20
NUM_SAMPLES=1000
BATCH_SIZE=50
VAL_DATASET="./datasets/tsp_20_10000.pkl"
OPTIMAL_TOURS_PATH="./datasets/greedy_tours.pkl"
EPOCHS=10
TRAINING_BATCH_SIZE=512
EPOCH_SIZE=5120
OVERLAP_WEIGHT=0.2
BREAK_PENALTY=0.0
SEED=1234

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --graph_size)
      GRAPH_SIZE="$2"
      shift 2
      ;;
    --num_samples)
      NUM_SAMPLES="$2"
      shift 2
      ;;
    --val_dataset)
      VAL_DATASET="$2"
      shift 2
      ;;
    --optimal_tours_path)
      OPTIMAL_TOURS_PATH="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch_size)
      TRAINING_BATCH_SIZE="$2"
      shift 2
      ;;
    --epoch_size)
      EPOCH_SIZE="$2"
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
    --seed)
      SEED="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create output directory for logs
LOG_DIR="./experiment_logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d%H%M%S)
LOG_FILE="$LOG_DIR/comparison_experiment_$TIMESTAMP.log"

echo "Starting comparison experiment at $(date)" | tee -a "$LOG_FILE"
echo "Configuration:" | tee -a "$LOG_FILE"
echo "- Graph size: $GRAPH_SIZE" | tee -a "$LOG_FILE"
echo "- Number of samples: $NUM_SAMPLES" | tee -a "$LOG_FILE"
echo "- Validation dataset: $VAL_DATASET" | tee -a "$LOG_FILE"
echo "- Optimal tours path: $OPTIMAL_TOURS_PATH" | tee -a "$LOG_FILE"
echo "- Training epochs: $EPOCHS" | tee -a "$LOG_FILE"
echo "- Training batch size: $TRAINING_BATCH_SIZE" | tee -a "$LOG_FILE"
echo "- Epoch size: $EPOCH_SIZE" | tee -a "$LOG_FILE"
echo "- Overlap weight (β): $OVERLAP_WEIGHT" | tee -a "$LOG_FILE"
echo "- Break penalty (γ): $BREAK_PENALTY" | tee -a "$LOG_FILE"
echo "- Random seed: $SEED" | tee -a "$LOG_FILE"
echo "---------------------------------" | tee -a "$LOG_FILE"

# Ensure all dependencies are installed
echo "Installing required packages..." | tee -a "$LOG_FILE"
pip install -q tensorboardX
pip install -q seaborn pandas matplotlib numpy tqdm
pip install -q opencv-python  # Add OpenCV for visualization

echo "Step 1: Generate tours using the greedy algorithm" | tee -a "$LOG_FILE"
python generate_greedy_tours.py \
  --graph_size "$GRAPH_SIZE" \
  --num_samples "$NUM_SAMPLES" \
  --val_dataset "$VAL_DATASET" \
  --output_path "$OPTIMAL_TOURS_PATH" \
  --batch_size "$BATCH_SIZE" 2>&1 | tee -a "$LOG_FILE"

# Check if tour generation was successful
if [ ! -f "$OPTIMAL_TOURS_PATH" ]; then
  echo "Error: Tour generation failed, optimal tours file not found" | tee -a "$LOG_FILE"
  exit 1
fi

echo "Step 2: Run comparison between baseline and edge overlap methods" | tee -a "$LOG_FILE"
python compare_rewards.py \
  --n_epochs "$EPOCHS" \
  --graph_size "$GRAPH_SIZE" \
  --batch_size "$TRAINING_BATCH_SIZE" \
  --epoch_size "$EPOCH_SIZE" \
  --optimal_tours_path "$OPTIMAL_TOURS_PATH" \
  --overlap_weight "$OVERLAP_WEIGHT" \
  --break_penalty "$BREAK_PENALTY" \
  --seed "$SEED" 2>&1 | tee -a "$LOG_FILE"

echo "Experiment completed at $(date)" | tee -a "$LOG_FILE"
echo "Results saved to outputs directory and logs saved to $LOG_FILE" | tee -a "$LOG_FILE"

# Copy the final JSON results to the log directory for easier access
RESULTS_FILE="./outputs/tsp_${GRAPH_SIZE}/comparison_results.json"
if [ -f "$RESULTS_FILE" ]; then
  cp "$RESULTS_FILE" "$LOG_DIR/comparison_results_$TIMESTAMP.json"
  echo "Results also copied to $LOG_DIR/comparison_results_$TIMESTAMP.json" | tee -a "$LOG_FILE"
fi 