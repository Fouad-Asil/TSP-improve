# TSP Reward Comparison Experiment

This document explains how to run the complete experiment comparing the original cost-based reward method with the new edge overlap reward mechanism.

## Quick Start

For the simplest execution, just run the bash script:

```bash
chmod +x run_comparison.sh   # Make the script executable
./run_comparison.sh          # Run with default parameters
```

This will:
1. Generate tours using the greedy algorithm
2. Train models using both reward mechanisms
3. Compare and visualize the results

## Avoiding Segmentation Faults

If you were experiencing segmentation faults with the original implementation, we've created a separate script for tour generation that uses only the greedy nearest neighbor algorithm (no OR-Tools) and processes instances in small batches:

```bash
python generate_greedy_tours.py --graph_size 20 --num_samples 1000 --val_dataset ./datasets/tsp_20_10000.pkl
```

This script:
- Processes instances in small batches (default 10 at a time)
- Saves progress after each batch to avoid losing work
- Uses a simpler algorithm that avoids memory issues
- Has better error handling than the original OR-Tools implementation

## Custom Experiment Configuration

You can customize the experiment by passing parameters to the script:

```bash
./run_comparison.sh --graph_size 50 --epochs 20 --overlap_weight 0.3 --break_penalty 0.1
```

Available parameters:
- `--graph_size`: Size of TSP instances (default: 20)
- `--num_samples`: Number of samples for validation (default: 1000)
- `--val_dataset`: Path to validation dataset (default: ./datasets/tsp_20_10000.pkl)
- `--optimal_tours_path`: Where to save/load tours (default: ./datasets/greedy_tours.pkl)
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Training batch size (default: 512)
- `--epoch_size`: Number of instances per epoch (default: 5120)
- `--overlap_weight`: Weight for edge overlap reward (β) (default: 0.2)
- `--break_penalty`: Weight for penalty when breaking optimal edges (γ) (default: 0.0)
- `--seed`: Random seed (default: 1234)

## Running Components Separately

If you prefer to run each step individually:

### 1. Generate tours with the safer greedy method:

```bash
python generate_greedy_tours.py --output_path ./datasets/greedy_tours.pkl
```

### 2. Run comparison with pre-generated tours:

```bash
python compare_rewards.py --optimal_tours_path ./datasets/greedy_tours.pkl
```

### 3. Evaluate existing models:

If you already have trained models and just want to compare them:

```bash
python compare_rewards.py --eval_only \
  --baseline_model_path ./outputs/tsp_20/comparison_baseline/final.pt \
  --overlap_model_path ./outputs/tsp_20/comparison_overlap/final.pt \
  --optimal_tours_path ./datasets/greedy_tours.pkl
```

## Outputs

The experiment produces the following outputs:

1. **Log Files**: All console output is saved to `experiment_logs/comparison_experiment_TIMESTAMP.log`

2. **Trained Models**: 
   - `outputs/tsp_SIZE/comparison_baseline/final.pt`
   - `outputs/tsp_SIZE/comparison_overlap/final.pt`

3. **Results**:
   - Visual comparison: `outputs/tsp_SIZE/comparison_results.png`
   - Detailed metrics: `outputs/tsp_SIZE/comparison_results.json` (also copied to `experiment_logs/`)

## Interpreting Results

The key metrics to compare:

- **Final Cost**: Lower is better; this is the final tour length
- **Improvement %**: Higher is better; shows how much the model improved from initial solution
- **Time**: Lower is better (as long as solution quality is maintained)

Better performance by the overlap model (compared to baseline) suggests that incorporating edge overlap information helps guide the search toward better solutions. 