#!/usr/bin/env python
"""
Compare baseline and overlap agents across multiple TSP graph sizes.

The script expects that for every graph size provided there exist two trained
models stored under

    outputs/tsp_<size>/comparison_baseline/final.pt
    outputs/tsp_<size>/comparison_overlap/final.pt

(these are the default folders created by ``compare_rewards.py`` when it is run
in training mode).

For each size we:
  1. Build the problem and a validation dataset (or load an existing .pkl file
     if supplied via ``--val_dataset_pattern``).
  2. Load both trained agents.
  3. Re-use the ``evaluate_and_compare`` routine from ``compare_rewards`` to
     compute the same metrics as in the single-size comparison.
  4. Aggregate the results across sizes and optionally save them as JSON as
     well as a bar-plot.

Example
-------
python compare_agents_variants.py \
    --graph_sizes 10 20 50 \
    --val_size 1000
"""

from __future__ import annotations

import os
import json
import argparse
from typing import List, Dict

import torch
import copy

# Local imports from this project
from compare_rewards import evaluate_and_compare, plot_comparison
from utils import load_problem, get_inner_model, torch_load_cpu
from options import get_options
from nets.attention_model import AttentionModel
from tb_logger import TensorboardLogger  # type: ignore


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def default_model_path(size: int, agent: str) -> str:
    """Return the default checkpoint path for a given agent (baseline/overlap)."""
    return os.path.join(
        "outputs",
        f"tsp_{size}",
        f"comparison_{agent}",
        "final.pt",
    )


def build_model(opts, problem):
    """Instantiate an AttentionModel with parameters from *opts*."""
    return AttentionModel(
        problem=problem,
        embedding_dim=opts.embedding_dim,
        hidden_dim=opts.hidden_dim,
        n_heads=opts.n_heads_encoder,
        n_layers=opts.n_encode_layers,
        normalization=opts.normalization,
        device=opts.device,
    ).to(opts.device)


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Compare baseline and overlap agents over many graph sizes")
    p.add_argument("--graph_sizes", type=int, nargs="+", required=True, help="List of graph sizes (e.g. 10 20 50)")
    p.add_argument("--val_size", type=int, default=1000, help="#instances per validation set if new dataset generated")
    p.add_argument("--val_dataset_pattern", type=str, default=None,
                   help="Optional pattern for existing validation datasets, use {size} placeholder."
                        " Example: ./datasets/tsp_{size}_1000.pkl")
    p.add_argument("--baseline_paths", type=str, nargs="*", default=None,
                   help="Explicit checkpoint paths for baseline agents (same order as graph_sizes)")
    p.add_argument("--overlap_paths", type=str, nargs="*", default=None,
                   help="Explicit checkpoint paths for overlap agents (same order as graph_sizes)")
    p.add_argument("--eval_batch_size", type=int, default=1000, help="Batch size for evaluation")
    p.add_argument("--skip_missing", action="store_true", help="Skip sizes where trained models are missing instead of raising an error")
    p.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    p.add_argument("--output_dir", type=str, default="outputs/variant_comparison", help="Where to store aggregated results")
    return p.parse_args()


def main():
    args = parse_args()

    # Prepare generic opts object – we only need a subset of fields for evaluation
    base_opts = get_options([
        "--problem", "tsp",
        "--graph_size", str(args.graph_sizes[0]),  # dummy, will be overwritten inside loop
        "--eval_only",
        "--batch_size", "512",  # not used
    ])
    base_opts.use_cuda = torch.cuda.is_available() and not args.no_cuda
    base_opts.device = torch.device("cuda" if base_opts.use_cuda else "cpu")
    base_opts.eval_batch_size = args.eval_batch_size
    base_opts.no_tensorboard = True  # disable TB inside evaluate_and_compare

    # Aggregate results per size
    all_results: Dict[int, Dict] = {}

    for idx, size in enumerate(args.graph_sizes):
        print(f"\n=== Evaluating size={size} ===")

        # ------------------------------------------------------------------
        # Resolve checkpoint paths
        # ------------------------------------------------------------------
        if args.baseline_paths is not None and idx < len(args.baseline_paths):
            baseline_ckpt = args.baseline_paths[idx]
        else:
            baseline_ckpt = default_model_path(size, "baseline")

        if args.overlap_paths is not None and idx < len(args.overlap_paths):
            overlap_ckpt = args.overlap_paths[idx]
        else:
            overlap_ckpt = default_model_path(size, "overlap")

        # Skip size if checkpoints are missing and user opted to skip
        if (not os.path.isfile(baseline_ckpt) or not os.path.isfile(overlap_ckpt)) and args.skip_missing:
            print("  [!] One or both checkpoints missing – skipping this size.")
            continue

        # Otherwise assert presence
        if not os.path.isfile(baseline_ckpt):
            raise FileNotFoundError(f"Baseline checkpoint not found: {baseline_ckpt}")
        if not os.path.isfile(overlap_ckpt):
            raise FileNotFoundError(f"Overlap checkpoint not found: {overlap_ckpt}")

        # ------------------------------------------------------------------
        # Prepare opts and problem for this size
        # ------------------------------------------------------------------
        opts = copy.deepcopy(base_opts)
        opts.graph_size = size

        # Problem instance
        problem = load_problem("tsp")(p_size=size)

        # Validation dataset
        if args.val_dataset_pattern is not None:
            ds_path = args.val_dataset_pattern.format(size=size)
            if os.path.exists(ds_path):
                print(f"Loading validation dataset from {ds_path}")
                val_dataset = problem.make_dataset(filename=ds_path)
            else:
                print(f"Dataset pattern given but file does not exist – generating new dataset to {ds_path}")
                val_dataset = problem.make_dataset(size=size, num_samples=args.val_size)
                os.makedirs(os.path.dirname(ds_path), exist_ok=True)
                import pickle
                with open(ds_path, "wb") as f:
                    pickle.dump([item.tolist() for item in val_dataset.data], f)
        else:
            print("Generating new validation dataset on the fly…")
            val_dataset = problem.make_dataset(size=size, num_samples=args.val_size)

        # ------------------------------------------------------------------
        # Load agents
        # ------------------------------------------------------------------
        baseline_data = torch_load_cpu(baseline_ckpt)
        overlap_data = torch_load_cpu(overlap_ckpt)

        baseline_model = build_model(opts, problem)
        overlap_model = build_model(opts, problem)
        get_inner_model(baseline_model).load_state_dict(baseline_data["model"])
        get_inner_model(overlap_model).load_state_dict(overlap_data["model"])

        # ------------------------------------------------------------------
        # Evaluate and compare
        # ------------------------------------------------------------------
        results = evaluate_and_compare(opts, problem, baseline_model, overlap_model, val_dataset)
        all_results[size] = results

        # Store per-size plot
        os.makedirs(args.output_dir, exist_ok=True)
        plot_comparison(results, save_path=os.path.join(args.output_dir, f"comparison_size_{size}.png"))

        # Free memory (especially important on GPU)
        del baseline_model, overlap_model
        torch.cuda.empty_cache()

    # ----------------------------------------------------------------------
    # Save aggregated results
    # ----------------------------------------------------------------------
    agg_path = os.path.join(args.output_dir, "aggregated_results.json")
    with open(agg_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\n[✓] Saved aggregated results to {agg_path}")


if __name__ == "__main__":
    main() 