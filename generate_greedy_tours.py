#!/usr/bin/env python
"""
Simplified script to generate tours using only the greedy nearest neighbor algorithm.
This avoids using OR-Tools which was causing segmentation faults.
"""

import os
import pickle
import argparse
import numpy as np
import torch
from tqdm import tqdm
from utils import load_problem

def greedy_nearest_neighbor(points):
    """
    Simple greedy nearest neighbor algorithm for TSP
    
    Args:
        points: numpy array of shape (n, 2) with coordinates
        
    Returns:
        tour: list of indices representing the tour
    """
    n = len(points)
    tour = [0]  # Start at first node
    unvisited = set(range(1, n))
    
    current = 0
    while unvisited:
        # Find nearest unvisited node
        nearest = min(unvisited, key=lambda j: np.linalg.norm(points[current] - points[j]))
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    return tour

def generate_tours(problem, dataset, save_path, batch_size=10):
    """
    Generate tours for a dataset using the greedy nearest neighbor algorithm
    
    Args:
        problem: Problem instance
        dataset: Dataset of instances
        save_path: Path to save the tours
        batch_size: Number of instances to process at once for progress reporting
    """
    print(f"Generating greedy tours for {len(dataset)} instances...")
    
    optimal_tours = []
    
    # Process in small batches for better progress reporting
    for i in range(0, len(dataset), batch_size):
        batch_end = min(i + batch_size, len(dataset))
        for idx in tqdm(range(i, batch_end), desc=f"Batch {i//batch_size + 1}/{(len(dataset) + batch_size - 1)//batch_size}"):
            try:
                # Get the instance data safely
                instance = dataset[idx]
                if isinstance(instance, torch.Tensor):
                    points = instance.detach().cpu().numpy()
                else:
                    points = np.array(instance)
                
                # Generate tour using greedy algorithm
                tour = greedy_nearest_neighbor(points)
                optimal_tours.append(tour)
                
            except Exception as e:
                print(f"Error processing instance {idx}: {e}")
                # Add a default tour (just sequential nodes) as fallback
                optimal_tours.append(list(range(problem.size)))
        
        # Save progress after each batch
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path + f".partial_{i}", 'wb') as f:
            pickle.dump(optimal_tours, f)
        print(f"Saved progress: {len(optimal_tours)}/{len(dataset)} tours")
    
    # Save final result
    print(f"Saving {len(optimal_tours)} tours to {save_path}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(optimal_tours, f)
    
    # Clean up partial files
    for i in range(0, len(dataset), batch_size):
        partial_path = save_path + f".partial_{i}"
        if os.path.exists(partial_path):
            os.remove(partial_path)
    
    return optimal_tours

def main():
    parser = argparse.ArgumentParser(description="Generate tours using greedy nearest neighbor algorithm")
    parser.add_argument('--graph_size', type=int, default=20, help='Size of TSP instances')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of instances to generate')
    parser.add_argument('--val_dataset', type=str, default=None, help='Path to existing dataset (if not specified, new dataset will be generated)')
    parser.add_argument('--output_path', type=str, default='./datasets/greedy_tours.pkl', help='Path to save the tours')
    parser.add_argument('--batch_size', type=int, default=10, help='Process this many instances at once')
    
    args = parser.parse_args()
    
    # Initialize problem
    problem = load_problem('tsp')(p_size=args.graph_size)
    
    # Create or load dataset
    if args.val_dataset:
        print(f"Loading dataset from {args.val_dataset}")
        dataset = problem.make_dataset(filename=args.val_dataset)
    else:
        print(f"Generating new dataset with {args.num_samples} instances")
        dataset = problem.make_dataset(size=args.graph_size, num_samples=args.num_samples)
    
    # Generate tours
    generate_tours(problem, dataset, args.output_path, args.batch_size)
    
    print("Done!")

if __name__ == "__main__":
    main() 