#!/bin/bash
# Script to run the comparison experiment across multiple graph sizes

# --- Configuration ---
GRAPH_SIZES=(20 50 100) # List of graph sizes to test
EPOCHS=10               # Number of training epochs for each size
OVERLAP_WEIGHT=0.2      # Beta value for edge overlap reward
BREAK_PENALTY=0.0       # Gamma value for edge break penalty
SEED=1234               # Random seed
RESULTS_DIR="./experiment_logs" # Directory to store results JSON files
SKIP_IF_TRAINED=false   # Flag to skip sizes if results already exist
# ---------------------

# Parse command-line arguments (optional overrides)
while [[ $# -gt 0 ]]; do
  case $1 in
    --sizes)
      shift
      GRAPH_SIZES=($@)
      break # Assume sizes are the last arguments
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
    --seed)
      SEED="$2"
      shift 2
      ;;
    --results_dir)
      RESULTS_DIR="$2"
      shift 2
      ;;
    --skip_if_trained)
      SKIP_IF_TRAINED=true
      shift 1 # Flag doesn't take an argument
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--sizes S1 S2 ...] [--epochs E] [--overlap_weight W] [--break_penalty P] [--seed S] [--results_dir DIR] [--skip_if_trained]"
      exit 1
      ;;
  esac
done

echo "=== Running Multi-Size Comparison Experiment ==="
echo "Graph Sizes: ${GRAPH_SIZES[@]}"
echo "Epochs per size: $EPOCHS"
echo "Overlap Weight (β): $OVERLAP_WEIGHT"
echo "Break Penalty (γ): $BREAK_PENALTY"
echo "Seed: $SEED"
echo "Results Dir: $RESULTS_DIR"

# Set working directory
cd "$(dirname "$0")"

# Create results directory
mkdir -p "$RESULTS_DIR"

# --- Dependencies ---
echo "Installing dependencies..."
pip install -q tensorboardX opencv-python matplotlib numpy tqdm torch pandas
chmod +x run_comparison.sh install_deps.sh generate_greedy_tours.py compare_rewards.py summarize_results.py
# ------------------

FAILED_SIZES=()
SUCCESS_COUNT=0

# --- Loop through graph sizes ---
for SIZE in "${GRAPH_SIZES[@]}"; do
  echo "\n" + "="*30 + " Starting Size: $SIZE " + "="*30
  
  TIMESTAMP=$(date +%Y%m%d%H%M%S)
  LOG_FILE="$RESULTS_DIR/size_${SIZE}_run_$TIMESTAMP.log"
  echo "Logging output to $LOG_FILE"
  
  # Define file paths for this size
  DATASET_PATH="./datasets/tsp_${SIZE}_val.pkl" # Needs a validation dataset for each size
  TOURS_PATH="./datasets/greedy_tours_${SIZE}.pkl"
  RESULTS_JSON_TARGET="$RESULTS_DIR/comparison_results_${SIZE}.json"
  
  # --- Check if skipping is enabled and results exist ---
  if [ "$SKIP_IF_TRAINED" = true ] && [ -f "$RESULTS_JSON_TARGET" ]; then
      echo "Results file $RESULTS_JSON_TARGET found and --skip_if_trained is set. Skipping size $SIZE." | tee -a "$LOG_FILE"
      SUCCESS_COUNT=$((SUCCESS_COUNT + 1)) # Count skipped as success
      continue # Skip to the next iteration
  fi
  
  # --- Step 1: Generate Validation Dataset (if it doesn't exist) ---
  if [ ! -f "$DATASET_PATH" ]; then
      echo "Validation dataset for size $SIZE not found. Generating..." | tee -a "$LOG_FILE"
      python -c "
import pickle
import numpy as np
import torch

np.random.seed($SEED)
dataset = [torch.FloatTensor($SIZE, 2).uniform_(0, 1) for _ in range(1000)] # Generate 1000 validation instances
with open('$DATASET_PATH', 'wb') as f:
    pickle.dump(dataset, f)
print('Created validation dataset for size $SIZE at $DATASET_PATH')
" 2>&1 | tee -a "$LOG_FILE"
      if [ $? -ne 0 ]; then
          echo "Error generating validation dataset for size $SIZE. Skipping size." | tee -a "$LOG_FILE"
          FAILED_SIZES+=("$SIZE")
          continue
      fi
  else
      echo "Using existing validation dataset: $DATASET_PATH" | tee -a "$LOG_FILE"
  fi

  # --- Step 2: Generate Tours ---
  echo "Generating greedy tours for size $SIZE..." | tee -a "$LOG_FILE"
  python generate_greedy_tours.py \
    --graph_size "$SIZE" \
    --val_dataset "$DATASET_PATH" \
    --num_samples 1000 \
    --output_path "$TOURS_PATH" \
    --batch_size 50 2>&1 | tee -a "$LOG_FILE"
    
  if [ $? -ne 0 ] || [ ! -f "$TOURS_PATH" ]; then
    echo "Error generating tours for size $SIZE. Skipping size." | tee -a "$LOG_FILE"
    FAILED_SIZES+=("$SIZE")
    continue
  fi
  
  # --- Step 3: Run Comparison ---
  echo "Running comparison for size $SIZE..." | tee -a "$LOG_FILE"
  python compare_rewards.py \
    --n_epochs "$EPOCHS" \
    --graph_size "$SIZE" \
    --val_dataset "$DATASET_PATH" \
    --optimal_tours_path "$TOURS_PATH" \
    --overlap_weight "$OVERLAP_WEIGHT" \
    --break_penalty "$BREAK_PENALTY" \
    --seed "$SEED" \
    --no_tb \
    --no_progress_bar 2>&1 | tee -a "$LOG_FILE"
    
  if [ $? -ne 0 ]; then
    echo "Error running comparison for size $SIZE. Skipping size." | tee -a "$LOG_FILE"
    FAILED_SIZES+=("$SIZE")
    continue
  fi
  
  # --- Step 4: Copy Results ---
  RESULTS_JSON="./outputs/tsp_${SIZE}/comparison_results.json"
  if [ -f "$RESULTS_JSON" ]; then
      cp "$RESULTS_JSON" "$RESULTS_DIR/comparison_results_${SIZE}.json"
      echo "Comparison for size $SIZE completed successfully." | tee -a "$LOG_FILE"
      SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
  else
      echo "Error: Results file not found for size $SIZE ($RESULTS_JSON)." | tee -a "$LOG_FILE"
      FAILED_SIZES+=("$SIZE")
  fi

done
# --- End Loop ---

echo "\n" + "="*70
echo "Multi-Size Experiment Finished"
echo "Successful runs: $SUCCESS_COUNT / ${#GRAPH_SIZES[@]}"
if [ ${#FAILED_SIZES[@]} -gt 0 ]; then
  echo "Failed sizes: ${FAILED_SIZES[@]}"
fi
echo "="*70

# --- Step 5: Summarize Results ---
if [ $SUCCESS_COUNT -gt 0 ]; then
    echo "\nGenerating final summary table..."
    python summarize_results.py --results_dir "$RESULTS_DIR"
else
    echo "\nNo successful runs to summarize."
fi

echo "\nExperiment logs and results JSON files are stored in: $RESULTS_DIR" 