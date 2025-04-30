# Comparing Edge Overlap Rewards with Standard Rewards

This document provides instructions for using the `compare_rewards.py` script to compare the performance of the original reward mechanism with the new edge overlap-based approach.

## Overview

The script performs the following tasks:
1. Generates or loads optimal tours for TSP instances
2. Trains two models:
   - **Baseline Model**: Using the original relative improvement reward
   - **Edge Overlap Model**: Using the enhanced reward that combines relative improvement with edge overlap metrics
3. Evaluates both models on the same validation dataset
4. Compares performance metrics including final tour cost, improvement percentage, and computation time
5. Visualizes the comparison with plots

## Prerequisites

In addition to the requirements for the main TSP solver, you'll need:
- matplotlib (for visualization)
- OR-Tools (optional, for generating optimal tours)

To install OR-Tools:
```
pip install ortools
```

## Generating Optimal Tours

Before comparing the reward mechanisms, you need to generate optimal (or high-quality) tours for your dataset. The script can do this automatically using Google's OR-Tools or a greedy heuristic:

```bash
python compare_rewards.py --generate_optimal --optimal_tours_path ./datasets/optimal_tours.pkl
```

This will create a file containing the pre-computed tours.

### Troubleshooting Tour Generation

If you encounter memory errors or segmentation faults during tour generation:

1. **Reduce batch size**: Try generating tours for smaller subsets of the data at a time
2. **Use greedy algorithm**: If OR-Tools is causing issues, use only the greedy algorithm by modifying `generate_optimal_tours` to set `use_ortools=False`
3. **Check available memory**: Tour generation can be memory-intensive for large problem sizes

## Running the Comparison

To run a comparison with default settings:

```bash
python compare_rewards.py --optimal_tours_path ./datasets/optimal_tours.pkl
```

### Evaluation Only Mode

If you already have trained models and just want to evaluate them:

```bash
python compare_rewards.py --eval_only --baseline_model_path ./outputs/tsp_20/comparison_baseline/final.pt --overlap_model_path ./outputs/tsp_20/comparison_overlap/final.pt --optimal_tours_path ./datasets/optimal_tours.pkl
```

This allows you to compare models without retraining them, which is useful for testing different evaluation metrics or parameters.

### Additional Options

- `--n_epochs`: Number of epochs to train each model (default: 10)
- `--graph_size`: Size of TSP instances (default: 20)
- `--batch_size`: Batch size for training (default: 512)
- `--epoch_size`: Number of instances per epoch (default: 5120)
- `--overlap_weight`: Weight for edge overlap reward (β, default: 0.2)
- `--break_penalty`: Weight for penalty when breaking optimal edges (γ, default: 0.0)
- `--seed`: Random seed (default: 1234)
- `--eval_only`: Skip training and evaluate existing models
- `--baseline_model_path`: Path to pre-trained baseline model (when using --eval_only)
- `--overlap_model_path`: Path to pre-trained overlap model (when using --eval_only)

For example, to run a more comprehensive comparison with different edge overlap weight:

```bash
python compare_rewards.py --n_epochs 20 --overlap_weight 0.3 --optimal_tours_path ./datasets/optimal_tours.pkl
```

## Output

The script produces the following outputs:

1. Trained model files saved to:
   - `outputs/tsp_{graph_size}/comparison_baseline/final.pt`
   - `outputs/tsp_{graph_size}/comparison_overlap/final.pt`

2. Comparison visualization saved to:
   - `outputs/tsp_{graph_size}/comparison_results.png`

3. Detailed metrics saved as JSON:
   - `outputs/tsp_{graph_size}/comparison_results.json`

4. Console output showing:
   - Training progress for both models
   - Detailed comparison with metrics
   - Percentage improvement between methods

## Interpreting Results

The script reports several key metrics:

- **Final Cost**: The average length of the final tours (lower is better)
- **Improvement**: The average reduction in tour length from initial to final solution
- **Improvement %**: The percentage improvement from initial to final (higher is better)
- **Time**: The average computation time per instance (in seconds)

The difference column shows how much better (or worse) the edge overlap approach is compared to the baseline.

## Example Visualization

The generated plot will look similar to this, showing a side-by-side comparison of key metrics:

```
Final Tour Cost      Improvement %      Computation Time (s)
|                |  |              |  |                   |
|  +------+     |  |  +------+    |  |  +------+         |
|  |      |     |  |  |      |    |  |  |      |         |
|  | Base |     |  |  | Base |    |  |  | Base |         |
|  |      | +---|  |  |      | +--|  |  |      | +-------|
|  +------+ |   |  |  +------+ |  |  |  +------+ |       |
|           |   |  |           |  |  |           |       |
|       Overlap  |  |       Overlap|  |        Overlap   |
|           |   |  |           |  |  |           |       |
|           +---|  |           +--|  |           +-------|
```

Lower cost, higher improvement percentage, and similar or lower computation time indicate the edge overlap method is performing better.

## Advanced Usage

### Parameter Tuning

To find the optimal balance between improvement reward and edge overlap reward, you can run multiple comparisons with different weights:

```bash
for weight in 0.1 0.2 0.3 0.5; do
    python compare_rewards.py --overlap_weight $weight --optimal_tours_path ./datasets/optimal_tours.pkl
done
```

### Breaking Penalty Experiment

To test the effect of penalizing the breaking of optimal edges, try:

```bash
python compare_rewards.py --overlap_weight 0.2 --break_penalty 0.1 --optimal_tours_path ./datasets/optimal_tours.pkl
```

Gradually increase the penalty value to see if it helps prevent thrashing behavior. 