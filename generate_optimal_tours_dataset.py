#!/usr/bin/env python
"""
Script to generate a fixed TSP dataset and solve every instance with OR-Tools (or a greedy fallback).

Running this script once allows you to re-use the exact same dataset + optimal tours
across many experiments, eliminating the overhead of re-solving the instances inside
individual training runs.

Example usage
-------------
python generate_optimal_tours_dataset.py \
  --graph_size 20 \
  --num_samples 10000 \
  --dataset_path ./datasets/tsp_20_10000.pkl \
  --tours_path   ./datasets/optimal_tours_20_10000.pkl

If the dataset already exists on disk it will be loaded instead of regenerated.
Likewise, if the tours file already exists the solving step will be skipped.
"""

import os
import argparse
import pickle
import numpy as np
from tqdm import tqdm
import torch

# NOTE: The project already defines helper utilities in several modules.  We re-use
# them here where possible to avoid code duplication.
from utils import load_problem    # type: ignore

# Re-use the solver implemented for the compare script.
from compare_rewards import generate_optimal_tours as solve_with_ortools  # type: ignore

# -------------------------------------------------------------------------------------
# Helper utilities
# -------------------------------------------------------------------------------------

def save_dataset(dataset, file_path):
    """Pickle the list of coordinate tensors so it can be reloaded with TSPDataset."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump([item.tolist() for item in dataset.data], f)


def load_or_create_dataset(problem, size, num_samples, file_path):
    """Load an existing dataset or create a brand new one."""
    if file_path is not None and os.path.exists(file_path):
        print(f"Loading existing dataset from {file_path}")
        dataset = problem.make_dataset(filename=file_path)
    else:
        print(f"Creating a new dataset with {num_samples} instances of size {size}")
        dataset = problem.make_dataset(size=size, num_samples=num_samples)
        if file_path is not None:
            save_dataset(dataset, file_path)
            print(f"Saved dataset to {file_path}")
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Generate a fixed TSP dataset and OR-Tools optimal tours")
    parser.add_argument("--graph_size", type=int, default=20, help="Number of nodes per TSP instance")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of instances to generate")
    parser.add_argument("--dataset_path", type=str, default="./datasets/tsp_20_10000.pkl", help="Path to store / load the dataset")
    parser.add_argument("--tours_path", type=str, default="./datasets/optimal_tours_20_10000.pkl", help="Path to store the optimal tours")
    parser.add_argument("--no_ortools", action="store_true", help="Skip OR-Tools and use greedy NN instead (debug)" )
    args = parser.parse_args()

    # Initialise problem & dataset
    problem = load_problem("tsp")(p_size=args.graph_size)
    dataset = load_or_create_dataset(problem, args.graph_size, args.num_samples, args.dataset_path)

    # ------------------------------------------------------------------
    # 2. Solve each instance with OR-Tools (or greedy fallback) only if
    #    we haven't already done so.
    # ------------------------------------------------------------------
    if os.path.exists(args.tours_path):
        print(f"Optimal tours already exist at {args.tours_path} – skipping solving step.")
    else:
        print("Solving every instance – this may take a while the first time…")
        # Re-use the existing implementation so the behaviour stays consistent.
        _ = solve_with_ortools(problem, dataset, args.tours_path, use_ortools=not args.no_ortools)
        print("Finished solving and saved tours.")

    print("All done! You can now pass the dataset/tours paths to your experiments.")


if __name__ == "__main__":
    main() 