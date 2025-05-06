#!/usr/bin/env python
"""
Train the edge–overlap RL model on multiple dataset variants.

For every (graph_size, num_samples) pair, the script will:
  1. Ensure the dataset and its optimal tours exist (creating/solving if necessary).
  2. Train an overlap-reward model using those tours.
  3. Save each trained model in a dedicated sub-folder under outputs/.

Example
-------
python train_overlap_variants.py \
    --graph_sizes 10 20 50 \
    --num_samples 10000 \
    --n_epochs 15 \
    --overlap_weight 0.2

The script relies on the utilities already present in the codebase (compare_rewards,
options, utils, etc.) and re-uses their training logic to avoid duplication.
"""

from __future__ import annotations

import os
import argparse
from typing import List

import torch

# Project imports
from utils import load_problem  # type: ignore
from compare_rewards import generate_optimal_tours, train_model  # type: ignore
from options import get_options  # type: ignore
from tb_logger import TensorboardLogger  # type: ignore (optional)


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def dataset_path(size: int, num_samples: int) -> str:
    return f"./datasets/tsp_{size}_{num_samples}.pkl"


def tours_path(size: int, num_samples: int) -> str:
    return f"./datasets/optimal_tours_{size}_{num_samples}.pkl"


def ensure_dataset_and_tours(problem, size: int, num_samples: int):
    """Create / load dataset and optimal tours for a given size."""
    d_path = dataset_path(size, num_samples)
    t_path = tours_path(size, num_samples)

    # 1. Dataset
    if os.path.exists(d_path):
        print(f"[✓] Dataset exists at {d_path}")
        dataset = problem.make_dataset(filename=d_path)
    else:
        print(f"[+] Generating dataset of {num_samples} instances (size={size}) …")
        dataset = problem.make_dataset(size=size, num_samples=num_samples)
        os.makedirs(os.path.dirname(d_path), exist_ok=True)
        # Save dataset in plain list format (compatible with TSPDataset)
        import pickle
        with open(d_path, "wb") as f:
            pickle.dump([item.tolist() for item in dataset.data], f)
        print(f"[✓] Saved dataset to {d_path}")

    # 2. Optimal tours
    if os.path.exists(t_path):
        print(f"[✓] Optimal tours already exist at {t_path}")
    else:
        print(f"[+] Solving {len(dataset)} instances with OR-Tools … (first time only)")
        _ = generate_optimal_tours(problem, dataset, t_path, use_ortools=True)
        print(f"[✓] Saved optimal tours to {t_path}")

    return dataset, t_path


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------

def train_for_size(size: int, num_samples: int, n_epochs: int, overlap_weight: float, break_penalty: float, seed: int):
    """Load / build everything and train overlap model for a single graph size."""

    # 0. Set up
    problem = load_problem("tsp")(p_size=size)
    val_dataset, opt_tours_path = ensure_dataset_and_tours(problem, size, num_samples)

    # 1. Load optimal tours into problem (needed for reward)
    problem.load_optimal_tours(opt_tours_path)

    # 2. Build opts (similar to compare_rewards but simplified)
    opts_cmd = [
        "--problem", "tsp",
        "--graph_size", str(size),
        "--n_epochs", str(n_epochs),
        "--batch_size", "512",
        "--epoch_size", "5120",
        "--overlap_reward_weight", str(overlap_weight),
        "--break_penalty_weight", str(break_penalty),
        "--seed", str(seed),
        "--no_assert",
        "--no_tb"  # avoid clutter; remove if TensorBoard is desired
    ]
    opts = get_options(opts_cmd)

    # Device
    opts.use_cuda = torch.cuda.is_available()
    opts.device = torch.device("cuda" if opts.use_cuda else "cpu")

    # Logging (optional)
    tb_logger = None
    if not opts.no_tensorboard:
        tb_logger = TensorboardLogger(os.path.join(opts.log_dir, f"tsp_{size}", "overlap"))

    # 3. Train
    print(f"\n=== Training overlap model for size={size} (epochs={n_epochs}) ===")
    _ = train_model(opts, problem, val_dataset, tb_logger, model_name=f"overlap_{size}", use_overlap=True)
    print(f"[✓] Finished training for size={size}\n")


def parse_args():
    p = argparse.ArgumentParser(description="Generate variant datasets and train overlap models on them")
    p.add_argument("--graph_sizes", type=int, nargs="+", default=[20], help="List of TSP sizes to process (e.g. 10 20 50)")
    p.add_argument("--num_samples", type=int, default=10000, help="Number of instances per dataset")
    p.add_argument("--n_epochs", type=int, default=10, help="Training epochs per size")
    p.add_argument("--overlap_weight", type=float, default=0.2, help="Edge overlap reward weight")
    p.add_argument("--break_penalty", type=float, default=0.0, help="Penalty weight for breaking optimal edges")
    p.add_argument("--seed", type=int, default=1234, help="Global random seed")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(args.seed)

    for size in args.graph_sizes:
        train_for_size(size=size,
                       num_samples=args.num_samples,
                       n_epochs=args.n_epochs,
                       overlap_weight=args.overlap_weight,
                       break_penalty=args.break_penalty,
                       seed=args.seed) 