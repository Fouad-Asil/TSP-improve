"""
Script to patch plotting functions to disable them completely.
This is a more direct approach to fix visualization errors.
"""

import os
import sys
import importlib
import types

def create_dummy_plot_function(*args, **kwargs):
    """Dummy function that returns a blank image"""
    import numpy as np
    # Return a simple 10x10 white image as a dummy
    return np.ones((10, 10, 3), dtype=np.uint8) * 255

def disable_plotting():
    """Patches all plotting functions with dummy versions"""
    try:
        # Try to import the plotting module
        from utils import plots
        
        # Replace the plot functions with dummy versions
        plots.plot_grad_flow = create_dummy_plot_function
        plots.plot_improve_pg = create_dummy_plot_function
        
        print("Successfully disabled plotting functions")
        return True
    except ImportError:
        print("Could not find utils.plots module")
        return False

def patch_logger():
    """Patch the logger to avoid TensorBoard errors"""
    try:
        from utils import logger
        
        # Original log functions
        original_log_to_tb_train = logger.log_to_tb_train
        original_log_to_tb_val = logger.log_to_tb_val
        
        # Create patched versions that skip image logging
        def patched_log_to_tb_train(*args, **kwargs):
            # Remove the last argument which is mini_step
            args_list = list(args)
            mini_step = args_list[-1]
            
            # Get the tb_logger which is the first argument
            tb_logger = args_list[0]
            
            # Skip image logging
            tb_logger.skip_images = True
            
            # Call original function
            return original_log_to_tb_train(*args, **kwargs)
        
        def patched_log_to_tb_val(*args, **kwargs):
            # Get the tb_logger which is the first argument
            tb_logger = args_list[0]
            
            # Skip image logging
            tb_logger.skip_images = True
            
            # Call original function
            return original_log_to_tb_val(*args, **kwargs)
        
        # Replace the functions
        logger.log_to_tb_train = patched_log_to_tb_train
        logger.log_to_tb_val = patched_log_to_tb_val
        
        print("Successfully patched logger functions")
        return True
    except (ImportError, AttributeError) as e:
        print(f"Could not patch logger: {e}")
        return False

def main():
    """Main function to apply all patches"""
    print("Applying patches to disable plotting and fix errors...")
    
    # Disable plotting
    plotting_disabled = disable_plotting()
    
    # Patch logger
    logger_patched = patch_logger()
    
    if plotting_disabled and logger_patched:
        print("All patches applied successfully!")
    else:
        print("Warning: Some patches could not be applied.")
        
    print("\nNow you can run your experiment without visualization errors.")
    print("Use: python -c 'import disable_plots; disable_plots.main()' && ./run_no_log.sh")

if __name__ == "__main__":
    main() 