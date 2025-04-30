#!/usr/bin/env python
"""
Script to compare the original reward mechanism with the new edge overlap-based approach.
Trains and evaluates both methods, then compares their performance.
"""

import os
import time
import json
import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Use our adapter instead of direct import
from tb_logger import TensorboardLogger

from nets.critic_network import CriticNetwork
from train import train_epoch, validate, rollout
from nets.reinforce_baselines import CriticBaseline
from nets.attention_model import AttentionModel
from utils import torch_load_cpu, load_problem, get_inner_model, move_to
from options import get_options


def generate_optimal_tours(problem, dataset, save_path, use_ortools=True):
    """
    Generate optimal (or high-quality) tours for a dataset using OR-Tools or a greedy method.
    
    Args:
        problem: Problem instance
        dataset: Dataset of instances
        save_path: Path to save optimal tours
        use_ortools: Whether to use OR-Tools (if available) or a greedy heuristic
    """
    print(f"Generating optimal tours for {len(dataset)} instances...")
    
    # Helper function for greedy algorithm
    def greedy_nearest_neighbor(points):
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
    
    if use_ortools:
        try:
            from ortools.constraint_solver import routing_enums_pb2
            from ortools.constraint_solver import pywrapcp
            print("Successfully imported OR-Tools")
        except ImportError:
            print("OR-Tools not available, falling back to greedy")
            use_ortools = False
    
    optimal_tours = []
    
    for idx in tqdm(range(len(dataset)), desc="Solving TSPs"):
        # Get the instance data safely
        instance = dataset[idx]
        if isinstance(instance, torch.Tensor):
            points = instance.detach().cpu().numpy()
        else:
            points = np.array(instance)
        
        n = len(points)
        
        if use_ortools:
            try:
                # Create distance matrix
                distances = np.zeros((n, n))
                
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            distances[i][j] = np.linalg.norm(points[i] - points[j])
                
                # Create OR-Tools model
                manager = pywrapcp.RoutingIndexManager(n, 1, 0)  # 1 vehicle, starting at node 0
                routing = pywrapcp.RoutingModel(manager)
                
                def distance_callback(from_index, to_index):
                    from_node = manager.IndexToNode(from_index)
                    to_node = manager.IndexToNode(to_index)
                    return int(distances[from_node][to_node] * 1000)  # Convert to int (required by OR-Tools)
                
                transit_callback_index = routing.RegisterTransitCallback(distance_callback)
                routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
                
                # Set search parameters
                search_parameters = pywrapcp.DefaultRoutingSearchParameters()
                search_parameters.first_solution_strategy = (
                    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
                search_parameters.local_search_metaheuristic = (
                    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
                search_parameters.time_limit.seconds = 5  # Limit per instance
                
                # Solve
                solution = routing.SolveWithParameters(search_parameters)
                
                if solution:
                    tour = []
                    index = routing.Start(0)
                    while not routing.IsEnd(index):
                        tour.append(manager.IndexToNode(index))
                        index = solution.Value(routing.NextVar(index))
                    # Complete the tour
                    tour.append(0)
                    optimal_tours.append(tour[:-1])  # Remove duplicated first node
                else:
                    # Fall back to greedy if OR-Tools fails to find a solution
                    tour = greedy_nearest_neighbor(points)
                    optimal_tours.append(tour)
            except Exception as e:
                print(f"Error using OR-Tools for instance {idx}: {str(e)}")
                # Fall back to greedy on error
                tour = greedy_nearest_neighbor(points)
                optimal_tours.append(tour)
        else:
            # Simple greedy nearest neighbor heuristic
            tour = greedy_nearest_neighbor(points)
            optimal_tours.append(tour)
        
        # Print progress occasionally to see if we're making progress
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(dataset)} instances")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save optimal tours to disk
    print(f"Saving optimal tours to {save_path}")
    with open(save_path, 'wb') as f:
        pickle.dump(optimal_tours, f)
    
    return optimal_tours


def evaluate_and_compare(opts, problem, baseline_model, overlap_model, val_dataset):
    """
    Evaluate both models on the validation dataset and compare performance
    
    Args:
        opts: Options
        problem: Problem instance
        baseline_model: Model trained with original reward
        overlap_model: Model trained with edge overlap reward
        val_dataset: Validation dataset
    
    Returns:
        dict: Performance metrics for both models
    """
    print("\nEvaluating baseline model...")
    baseline_model.eval()
    
    # Stats containers
    baseline_init_value = []
    baseline_best_value = []
    baseline_improvement = []
    baseline_time = []
    
    overlap_init_value = []
    overlap_best_value = []
    overlap_improvement = []
    overlap_time = []
    
    # Evaluate baseline model
    for batch in tqdm(DataLoader(val_dataset, batch_size=opts.eval_batch_size), 
                      desc='evaluate baseline'):
        initial_solution = move_to(
            problem.get_initial_solutions(opts.init_val_met, batch), opts.device)
        
        x_input = move_to(batch, opts.device)
        batch = move_to(batch, opts.device)
        
        initial_value = problem.get_costs(batch, initial_solution)
        baseline_init_value.append(initial_value)
        
        # Run rollout with baseline model
        s_time = time.time()
        bv, improve, _, _ = rollout(problem, 
                                  baseline_model, 
                                  x_input,
                                  batch, 
                                  initial_solution,
                                  initial_value,
                                  opts,
                                  T=opts.T_max,
                                  do_sample=True)
        
        duration = time.time() - s_time
        baseline_time.append(duration)
        baseline_best_value.append(bv.clone())
        baseline_improvement.append(improve.clone())
    
    # Evaluate overlap model
    print("\nEvaluating overlap model...")
    overlap_model.eval()
    
    for batch in tqdm(DataLoader(val_dataset, batch_size=opts.eval_batch_size), 
                      desc='evaluate overlap'):
        initial_solution = move_to(
            problem.get_initial_solutions(opts.init_val_met, batch), opts.device)
        
        x_input = move_to(batch, opts.device)
        batch = move_to(batch, opts.device)
        
        initial_value = problem.get_costs(batch, initial_solution)
        overlap_init_value.append(initial_value)
        
        # Run rollout with overlap model
        s_time = time.time()
        bv, improve, _, _ = rollout(problem, 
                                  overlap_model, 
                                  x_input,
                                  batch, 
                                  initial_solution,
                                  initial_value,
                                  opts,
                                  T=opts.T_max,
                                  do_sample=True)
        
        duration = time.time() - s_time
        overlap_time.append(duration)
        overlap_best_value.append(bv.clone())
        overlap_improvement.append(improve.clone())
    
    # Aggregate results
    baseline_best_value = torch.cat(baseline_best_value, 0)
    baseline_improvement = torch.cat(baseline_improvement, 0)
    baseline_init_value = torch.cat(baseline_init_value, 0).view(-1, 1)
    baseline_time = torch.tensor(baseline_time)
    
    overlap_best_value = torch.cat(overlap_best_value, 0)
    overlap_improvement = torch.cat(overlap_improvement, 0)
    overlap_init_value = torch.cat(overlap_init_value, 0).view(-1, 1)
    overlap_time = torch.tensor(overlap_time)
    
    results = {
        'baseline': {
            'init_cost': baseline_init_value.mean().item(),
            'final_cost': baseline_best_value.mean().item(),
            'improvement': (baseline_init_value.mean() - baseline_best_value.mean()).item(),
            'improvement_percent': (baseline_init_value.mean() - baseline_best_value.mean()).item() / baseline_init_value.mean().item() * 100,
            'time': baseline_time.mean().item()
        },
        'overlap': {
            'init_cost': overlap_init_value.mean().item(),
            'final_cost': overlap_best_value.mean().item(),
            'improvement': (overlap_init_value.mean() - overlap_best_value.mean()).item(),
            'improvement_percent': (overlap_init_value.mean() - overlap_best_value.mean()).item() / overlap_init_value.mean().item() * 100,
            'time': overlap_time.mean().item()
        }
    }
    
    # Print comparison
    print("\n" + "="*50)
    print("COMPARISON RESULTS")
    print("="*50)
    print(f"{'Metric':<20} {'Baseline':<15} {'Edge Overlap':<15} {'Difference':<15}")
    print("-"*65)
    print(f"{'Initial Cost':<20} {results['baseline']['init_cost']:<15.4f} {results['overlap']['init_cost']:<15.4f} {0:<15.4f}")
    print(f"{'Final Cost':<20} {results['baseline']['final_cost']:<15.4f} {results['overlap']['final_cost']:<15.4f} {results['baseline']['final_cost'] - results['overlap']['final_cost']:<15.4f}")
    print(f"{'Improvement':<20} {results['baseline']['improvement']:<15.4f} {results['overlap']['improvement']:<15.4f} {results['overlap']['improvement'] - results['baseline']['improvement']:<15.4f}")
    print(f"{'Improvement %':<20} {results['baseline']['improvement_percent']:<15.2f}% {results['overlap']['improvement_percent']:<15.2f}% {results['overlap']['improvement_percent'] - results['baseline']['improvement_percent']:<15.2f}%")
    print(f"{'Time (s)':<20} {results['baseline']['time']:<15.4f} {results['overlap']['time']:<15.4f} {results['baseline']['time'] - results['overlap']['time']:<15.4f}")
    print("="*65)
    
    return results


def train_model(opts, problem, val_dataset, tb_logger, model_name="baseline", use_overlap=False):
    """
    Train a model with specified reward mechanism
    
    Args:
        opts: Options
        problem: Problem instance
        val_dataset: Validation dataset
        tb_logger: TensorBoard logger
        model_name: Name for this model (for saving)
        use_overlap: Whether to use edge overlap rewards
    
    Returns:
        model: Trained model
    """
    # Ensure device is set
    if not hasattr(opts, 'device'):
        opts.device = torch.device("cuda" if opts.use_cuda else "cpu")
        print(f"Setting device to: {opts.device}")
    
    # Initialize model
    model = AttentionModel(
        problem=problem,
        embedding_dim=opts.embedding_dim,
        hidden_dim=opts.hidden_dim,
        n_heads=opts.n_heads_encoder,
        n_layers=opts.n_encode_layers,
        normalization=opts.normalization,
        device=opts.device
    ).to(opts.device)
    
    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    # Initialize baseline
    baseline = CriticBaseline(
        CriticNetwork(
            problem=problem,
            embedding_dim=opts.embedding_dim,
            hidden_dim=opts.hidden_dim,
            n_heads=opts.n_heads_decoder,
            n_layers=opts.n_encode_layers,
            normalization=opts.normalization,
            device=opts.device
        ).to(opts.device)
    )
    
    # Initialize optimizer
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0 else []
        )
    )
    
    # Initialize learning rate scheduler
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)
    
    # Set overlap reward settings based on argument
    if use_overlap:
        print(f"\nTraining {model_name} with edge overlap rewards...")
        # Make sure the optimal tours are available
        if not hasattr(problem, 'optimal_tours') or problem.optimal_tours is None:
            raise ValueError("Optimal tours must be loaded for edge overlap training")
        # Enable overlap rewards
        opts.use_optimal_tours = True
    else:
        print(f"\nTraining {model_name} with standard rewards...")
        # Disable overlap rewards
        opts.use_optimal_tours = False
    
    # Train for specified number of epochs
    for epoch in range(opts.n_epochs):
        print(f"\nEpoch {epoch+1}/{opts.n_epochs}")
        train_epoch(
            problem,
            model,
            optimizer,
            baseline,
            lr_scheduler,
            epoch,
            val_dataset,
            tb_logger,
            opts
        )
    
    # Save the final model
    save_dir = os.path.join(
        opts.output_dir,
        f"{opts.problem}_{opts.graph_size}",
        f"comparison_{model_name}"
    )
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save(
        {
            'model': get_inner_model(model).state_dict(),
            'optimizer': optimizer.state_dict(),
            'baseline': baseline.state_dict()
        },
        os.path.join(save_dir, f'final.pt')
    )
    
    return model


def plot_comparison(results, save_path=None):
    """Plots the comparison results."""
    
    metrics = list(results['baseline'].keys()) # Assuming both models have the same metrics
    models = list(results.keys())
    num_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 5), sharey=False)
    if num_metrics == 1: # Handle case with only one metric
        axes = [axes] 
        
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in models]
        axes[i].bar(models, values)
        axes[i].set_title(metric.replace('_', ' ').title())
        if metric == 'final_cost':
            axes[i].set_ylabel('Cost (lower is better)')
        elif metric == 'improvement_percent':
            axes[i].set_ylabel('Improvement % (higher is better)')
        else:
            axes[i].set_ylabel('Time in seconds')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) # Ensure directory exists
        plt.savefig(save_path)
        print(f"Saved comparison plot to {save_path}")
    
    # plt.show() # Remove this line


def main():
    parser = argparse.ArgumentParser(description="Compare original and edge overlap reward mechanisms")
    
    # Training settings
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs to train each model')
    parser.add_argument('--graph_size', type=int, default=20, help='Size of TSP instances')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
    parser.add_argument('--epoch_size', type=int, default=5120, help='Number of instances per epoch')
    parser.add_argument('--generate_optimal', action='store_true', help='Generate optimal tours using OR-Tools')
    parser.add_argument('--optimal_tours_path', type=str, default='./datasets/greedy_tours.pkl', 
                        help='Path to save/load optimal tours')
    parser.add_argument('--overlap_weight', type=float, default=0.2, 
                        help='Weight for edge overlap reward')
    parser.add_argument('--break_penalty', type=float, default=0.0, 
                        help='Weight for penalty when breaking optimal edges')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--val_dataset', type=str, default=None, 
                        help='Path to validation dataset .pkl file')
    parser.add_argument('--no_tb', action='store_true', help='Disable Tensorboard logging')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable tqdm progress bars')
    
    # Evaluation settings
    parser.add_argument('--eval_only', action='store_true', 
                        help='Skip training and evaluate existing models')
    parser.add_argument('--baseline_model_path', type=str, default=None,
                        help='Path to pre-trained baseline model (for eval_only)')
    parser.add_argument('--overlap_model_path', type=str, default=None,
                        help='Path to pre-trained overlap model (for eval_only)')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    
    # Parse arguments
    compare_args = parser.parse_args()
    
    # Get standard options and update with comparison-specific ones
    standard_opts_list = [
        '--problem', 'tsp',
        '--graph_size', str(compare_args.graph_size),
        '--n_epochs', str(compare_args.n_epochs),
        '--batch_size', str(compare_args.batch_size),
        '--epoch_size', str(compare_args.epoch_size),
        '--overlap_reward_weight', str(compare_args.overlap_weight),
        '--break_penalty_weight', str(compare_args.break_penalty),
        '--seed', str(compare_args.seed),
        '--no_assert'
    ]
    if compare_args.no_tb:
        standard_opts_list.append('--no_tb')
    if compare_args.no_progress_bar:
        standard_opts_list.append('--no_progress_bar')
        
    opts = get_options(standard_opts_list)
    
    # Set device explicitly
    opts.use_cuda = torch.cuda.is_available() and not compare_args.no_cuda
    opts.device = torch.device("cuda" if opts.use_cuda else "cpu")
    print(f"Using device: {opts.device}")
    
    # Set TensorBoard flag based on compare_args
    opts.no_tensorboard = compare_args.no_tb
    
    # Set progress bar flag based on compare_args
    opts.no_progress_bar = compare_args.no_progress_bar
    
    # Set up TensorBoard logger
    tb_logger = None
    if not opts.no_tensorboard:
        tb_logger = TensorboardLogger(os.path.join(opts.log_dir, f"tsp_{opts.graph_size}", "comparison"))
    
    # Set random seed
    torch.manual_seed(opts.seed)
    
    # Initialize problem
    problem = load_problem(opts.problem)(
        p_size=opts.graph_size,
        with_assert=not opts.no_assert
    )
    
    # Create validation dataset
    if compare_args.val_dataset and os.path.exists(compare_args.val_dataset):
        print(f"Loading validation dataset from {compare_args.val_dataset}")
        val_dataset = problem.make_dataset(filename=compare_args.val_dataset)
    else:
        print("Generating new validation dataset...")
        val_dataset = problem.make_dataset(
            size=opts.graph_size,
            num_samples=opts.val_size
        )
        if compare_args.val_dataset:
            print(f"Saving new validation dataset to {compare_args.val_dataset}")
            os.makedirs(os.path.dirname(compare_args.val_dataset), exist_ok=True)
            with open(compare_args.val_dataset, 'wb') as f:
                # Note: Assuming make_dataset returns the data directly or has a way to get it
                # This might need adjustment based on TSPDataset structure
                pickle.dump([item.tolist() for item in val_dataset.data], f)
    
    # Generate or load optimal tours
    if compare_args.generate_optimal:
        generate_optimal_tours(problem, val_dataset, compare_args.optimal_tours_path)
    
    # Load optimal tours
    problem.load_optimal_tours(compare_args.optimal_tours_path)
    if problem.optimal_tours is None:
        print("Warning: No optimal tours loaded. Will only train baseline model.")
        train_baseline = True
        train_overlap = False
    else:
        train_baseline = True
        train_overlap = True
    
    # Create output directory
    os.makedirs(os.path.join(opts.output_dir, f"tsp_{opts.graph_size}"), exist_ok=True)
    
    # Models dictionary
    models = {}
    
    if compare_args.eval_only:
        print("Evaluation only mode - loading pre-trained models...")
        
        # Load baseline model if path is provided
        if compare_args.baseline_model_path:
            print(f"Loading baseline model from {compare_args.baseline_model_path}")
            baseline_data = torch_load_cpu(compare_args.baseline_model_path)
            baseline_model = AttentionModel(
                problem=problem,
                embedding_dim=opts.embedding_dim,
                hidden_dim=opts.hidden_dim,
                n_heads=opts.n_heads_encoder,
                n_layers=opts.n_encode_layers,
                normalization=opts.normalization,
                device=opts.device
            ).to(opts.device)
            baseline_model_ = get_inner_model(baseline_model)
            baseline_model_.load_state_dict(baseline_data['model'])
            models['baseline'] = baseline_model
        
        # Load overlap model if path is provided
        if compare_args.overlap_model_path:
            print(f"Loading overlap model from {compare_args.overlap_model_path}")
            overlap_data = torch_load_cpu(compare_args.overlap_model_path)
            overlap_model = AttentionModel(
                problem=problem,
                embedding_dim=opts.embedding_dim,
                hidden_dim=opts.hidden_dim,
                n_heads=opts.n_heads_encoder,
                n_layers=opts.n_encode_layers,
                normalization=opts.normalization,
                device=opts.device
            ).to(opts.device)
            overlap_model_ = get_inner_model(overlap_model)
            overlap_model_.load_state_dict(overlap_data['model'])
            models['overlap'] = overlap_model
    else:
        # Train models
        if train_baseline:
            models['baseline'] = train_model(opts, problem, val_dataset, tb_logger, 
                                          model_name="baseline", use_overlap=False)
        
        if train_overlap:
            models['overlap'] = train_model(opts, problem, val_dataset, tb_logger, 
                                         model_name="overlap", use_overlap=True)
    
    # Evaluate and compare models
    if len(models) >= 2 and 'baseline' in models and 'overlap' in models:
        results = evaluate_and_compare(opts, problem, models['baseline'], models['overlap'], val_dataset)
        
        # Plot comparison
        plot_path = os.path.join(opts.output_dir, f"tsp_{opts.graph_size}", "comparison_results.png")
        plot_comparison(results, save_path=plot_path)
        
        # Save results
        with open(os.path.join(opts.output_dir, f"tsp_{opts.graph_size}", "comparison_results.json"), 'w') as f:
            # Convert tensor items to python types
            for model in results:
                for metric in results[model]:
                    if isinstance(results[model][metric], torch.Tensor):
                        results[model][metric] = results[model][metric].item()
            json.dump(results, f, indent=4)
    elif len(models) == 1:
        print("Only one model available. Comparison requires both baseline and overlap models.")
    else:
        print("No models available for evaluation.")
        if compare_args.eval_only:
            print("Please provide paths to pre-trained models using --baseline_model_path and --overlap_model_path.")


if __name__ == "__main__":
    main() 