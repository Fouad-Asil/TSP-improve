import torch
import numpy as np
from argparse import Namespace

# Assuming problem_tsp.py is in the problems directory relative to this script
from problems.problem_tsp import TSP 
from utils import move_to # Assuming utils.py is in the parent directory or accessible

# --- Simulation Parameters ---
graph_size = 50
opts_batch_size = 128  # Corresponds to opts.batch_size
total_optimal_tours = 1000 # Example total number of optimal tours available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# --- End Simulation Parameters ---

# --- Simulate Objects ---
# Dummy opts
opts = Namespace(
    batch_size=opts_batch_size,
    use_optimal_tours=True,
    init_val_met='random', # Method doesn't matter much for this test structure
    device=device,
    overlap_reward_weight=0.1, # Example value, needed for the copied code block
    break_penalty_weight=0.1 # Example value
)

# Instantiate problem
problem = TSP(p_size=graph_size)

# Create dummy optimal tours (List of numpy arrays, as expected by the original code)
# Each tour is a permutation of 0 to graph_size-1
print(f"Generating {total_optimal_tours} dummy optimal tours for graph size {graph_size}...")
problem.optimal_tours = [np.random.permutation(graph_size) for _ in range(total_optimal_tours)]
print("Dummy optimal tours generated.")

# Calculate parameters for the last batch
num_full_batches = total_optimal_tours // opts.batch_size
last_batch_id = num_full_batches
last_batch_actual_size = total_optimal_tours % opts.batch_size
if last_batch_actual_size == 0 and total_optimal_tours > 0:
    # If perfectly divisible, the last batch is a full batch
    last_batch_actual_size = opts.batch_size
    last_batch_id = num_full_batches - 1 # Adjust batch_id if it's the last full batch

print(f"Simulating last batch: batch_id={last_batch_id}, expected size={last_batch_actual_size}")

# Create dummy batch data and solution for the last batch
# batch shape: (batch_size, graph_size, 2) - coordinates don't matter for this test
dummy_batch = torch.rand(last_batch_actual_size, graph_size, 2)
# solution shape: (batch_size, graph_size)
dummy_solution = torch.stack([torch.randperm(graph_size) for _ in range(last_batch_actual_size)]).to(opts.device)

print(f"Dummy batch shape: {dummy_batch.shape}")
print(f"Dummy solution shape: {dummy_solution.shape}")

# --- Start Test ---
# This block is adapted from train_batch in train.py
print("\n--- Testing Optimal Tour Slicing and Initial Overlap ---")
batch_id = last_batch_id
batch = move_to(dummy_batch, opts.device)
solution = dummy_solution # Already on device

# --- Copied/Adapted Code Block ---
optimal_tours = None
if opts.use_optimal_tours and problem.optimal_tours is not None:
    start_idx = batch_id * opts.batch_size
    # Ensure we don't slice beyond the available optimal tours
    end_idx = min(start_idx + opts.batch_size, len(problem.optimal_tours))
    print(f"Calculated slice indices: start={start_idx}, end={end_idx}")
    # Only proceed if there are tours for this slice
    if start_idx < end_idx:
        optimal_tours_slice = problem.optimal_tours[start_idx:end_idx]
        print(f"Sliced optimal tours length: {len(optimal_tours_slice)}")
        # Ensure the slice matches the actual batch size being processed
        current_batch_size = batch.size(0) # Get the real size of this batch
        print(f"Current actual batch size: {current_batch_size}")
        if len(optimal_tours_slice) == current_batch_size:
             print("Slice size matches batch size. Converting to tensor.")
             optimal_tours = move_to(torch.tensor(np.array(optimal_tours_slice)), opts.device) # Ensure conversion works
        else:
             # Handle mismatch - perhaps skip overlap calculation for this batch or log a warning
             print(f"Warning: Batch size ({current_batch_size}) and optimal tours slice size ({len(optimal_tours_slice)}) mismatch for batch_id {batch_id}. Skipping overlap calculation.")
             optimal_tours = None # Prevent using mismatched tours
    else:
        print("start_idx >= end_idx, no optimal tours for this batch_id.")
        optimal_tours = None

#update best_so_far - Dummy calculation for structure
best_so_far = problem.get_costs(batch, solution)
initial_cost = best_so_far.clone()

# Calculate initial edge overlap if using optimal tours
prev_overlap = None
print("\n--- Calculating Initial Overlap ---")
if optimal_tours is not None:
    print(f"Optimal tours tensor shape: {optimal_tours.shape}")
    print(f"Solution tensor shape: {solution.shape}")
    # Ensure the solution batch size matches optimal_tours batch size before calculating
    if solution.size(0) == optimal_tours.size(0):
         print("Solution and optimal_tours batch sizes match. Calculating overlap...")
         try:
             prev_overlap = problem.calculate_edge_overlap(solution, optimal_tours)
             print(f"Successfully calculated prev_overlap. Shape: {prev_overlap.shape}")
             # print(f"Sample overlap values: {prev_overlap[:5]}") # Optional: print some values
         except IndexError as e:
             print(f"!!! IndexError encountered during calculate_edge_overlap: {e}")
         except Exception as e:
             print(f"!!! An unexpected error occurred during calculate_edge_overlap: {e}")
    else:
         print(f"Warning: Mismatch between solution batch size ({solution.size(0)}) and optimal_tours batch size ({optimal_tours.size(0)}). Skipping overlap calculation.")
         optimal_tours = None # Ensure downstream code doesn't use mismatched tours
else:
    print("Optimal tours tensor is None. Skipping overlap calculation.")

print("\n--- Testing Subsequent Overlap/Penalty Calculation (Simulated Step) ---")
# Simulate one step to test calculations inside the loop
exchange = torch.randint(0, graph_size, (solution.size(0), 2), device=opts.device) # Dummy exchange

# Recalculate overlap (like in the loop)
overlap_reward = torch.zeros_like(best_so_far)
if optimal_tours is not None and opts.overlap_reward_weight > 0:
     print("Calculating overlap reward...")
     if solution.size(0) == optimal_tours.size(0):
         curr_overlap = problem.calculate_edge_overlap(solution, optimal_tours)
         if prev_overlap is not None and prev_overlap.size(0) == curr_overlap.size(0):
             overlap_reward = opts.overlap_reward_weight * (curr_overlap - prev_overlap)
             prev_overlap = curr_overlap
             print(f"Overlap reward calculated. Shape: {overlap_reward.shape}")
         else:
             print(f"Warning: Mismatch or missing prev_overlap during overlap reward calculation.")
             overlap_reward = torch.zeros_like(best_so_far)
     else:
         print(f"Warning: Mismatch during overlap reward calculation. Sol size: {solution.size(0)}, Opt size: {optimal_tours.size(0)}")
else:
    print("Optimal tours is None, skipping overlap reward calculation.")


# Calculate penalty
penalty = torch.zeros_like(best_so_far)
if optimal_tours is not None and opts.break_penalty_weight > 0:
     print("Calculating penalty...")
     if solution.size(0) == optimal_tours.size(0):
         try:
            broken_edges = problem.calculate_broken_optimal_edges(solution, exchange, optimal_tours)
            penalty = opts.break_penalty_weight * broken_edges
            print(f"Penalty calculated. Shape: {penalty.shape}")
         except IndexError as e:
             print(f"!!! IndexError encountered during calculate_broken_optimal_edges: {e}")
         except Exception as e:
             print(f"!!! An unexpected error occurred during calculate_broken_optimal_edges: {e}")
     else:
         print(f"Warning: Mismatch during penalty calculation. Sol size: {solution.size(0)}, Opt size: {optimal_tours.size(0)}")
else:
    print("Optimal tours is None, skipping penalty calculation.")


print("\n--- Test Complete ---") 