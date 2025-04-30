#!/bin/bash
# Emergency script with minimal parameters and all visualization disabled

echo "=== EMERGENCY RUN: Minimal parameters, all visualization disabled ==="

# Set working directory
cd "$(dirname "$0")"

# Create required directories
mkdir -p datasets
mkdir -p experiment_logs
mkdir -p outputs

# Install all required packages
echo "Installing all dependencies..."
pip install -q tensorboardX
pip install -q opencv-python
pip install -q matplotlib numpy tqdm torch

# Create a small test dataset and tours if needed
if [ ! -f "./datasets/small_test_tours.pkl" ]; then
    echo "Generating a tiny test dataset and tours..."
    python -c "
import pickle
import numpy as np
import torch

# Create a tiny dataset
n_samples = 5
graph_size = 10
np.random.seed(1234)
dataset = [torch.FloatTensor(graph_size, 2).uniform_(0, 1) for _ in range(n_samples)]

# Save dataset
with open('./datasets/small_test_dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)

# Create simple tours
tours = [list(range(graph_size)) for _ in range(n_samples)]
with open('./datasets/small_test_tours.pkl', 'wb') as f:
    pickle.dump(tours, f)

print('Created tiny test dataset with {} samples of size {}'.format(n_samples, graph_size))
"
fi

# Apply patches to disable plotting
echo "Disabling all plotting and visualization..."

# Create a temporary patch file
cat > temp_patch.py << 'EOF'
def patch_system():
    # Disable matplotlib to avoid GUI errors
    import sys
    import types
    
    # Create mock modules to prevent imports
    class MockModule(types.ModuleType):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            
        def __call__(self, *args, **kwargs):
            return self
            
        def __getattr__(self, name):
            return self
    
    # Replace plotting modules
    sys.modules['matplotlib.pyplot'] = MockModule('matplotlib.pyplot')
    
    # Patch the logger module
    try:
        from utils import logger
        
        # Replace logging functions with dummy versions
        def dummy_log(*args, **kwargs):
            pass
            
        logger.log_to_tb_train = dummy_log
        logger.log_to_tb_val = dummy_log
        logger.log_to_screen = dummy_log
        
        # Replace plots module
        try:
            from utils import plots
            sys.modules['utils.plots'] = MockModule('utils.plots')
        except:
            pass
            
        print("Patched logger and plotting systems")
    except Exception as e:
        print(f"Warning: Could not fully patch logging: {e}")

    return True

# Execute the patch
patch_system()
EOF

# Run with minimal parameters
echo "Running minimal comparison..."
python -c "
import sys
sys.path.append('.')
import temp_patch
temp_patch.patch_system()

# Now run with all visualization disabled
import sys
sys.argv = ['compare_rewards.py', 
    '--graph_size', '10',
    '--n_epochs', '1',
    '--batch_size', '5',
    '--epoch_size', '5',
    '--eval_batch_size', '5',
    '--val_size', '5',
    '--optimal_tours_path', './datasets/small_test_tours.pkl',
    '--overlap_weight', '0.2',
    '--break_penalty', '0.0',
    '--seed', '1234',
    '--no_tb',
    '--no_progress_bar',
    '--no_assert']

# Import the compare_rewards module dynamically
import importlib.util
spec = importlib.util.spec_from_file_location('compare_rewards', 'compare_rewards.py')
compare_rewards = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_rewards)

# Run main function
compare_rewards.main()
"

EXIT_STATUS=$?

# Clean up
rm temp_patch.py

if [ $EXIT_STATUS -eq 0 ]; then
    echo "Success! Emergency run completed."
    echo "If this worked, you can now try with larger parameters by modifying the script."
else
    echo "Emergency run failed with status $EXIT_STATUS."
    echo "Check the output above for specific error messages."
fi 