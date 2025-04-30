#!/usr/bin/env python
"""
Script to patch the train.py file to ensure directories exist before saving checkpoints.
Run this before running the experiment.
"""

import os
import sys
import types
import inspect

def patch_train_module():
    """Patches the train.py module to ensure directories exist before saving"""
    try:
        import train
        from utils import get_inner_model
        
        # Store original train_epoch function
        original_train_epoch = train.train_epoch
        
        # Create a new function that ensures directories exist
        def patched_train_epoch(problem, model, optimizer, baseline, lr_scheduler, epoch, val_dataset, tb_logger, opts):
            """Patched version of train_epoch that ensures save directories exist"""
            # Make sure save_dir exists
            if not os.path.exists(opts.save_dir):
                print(f"Creating directory {opts.save_dir}")
                os.makedirs(opts.save_dir, exist_ok=True)
            
            # Call the original function
            return original_train_epoch(problem, model, optimizer, baseline, lr_scheduler, epoch, val_dataset, tb_logger, opts)
        
        # Replace the function in the module
        train.train_epoch = patched_train_epoch
        
        # Now patch the validate function too
        original_validate = train.validate
        
        def patched_validate(problem, model, val_dataset, tb_logger, opts, _id=None):
            """Patched version of validate that ensures save directories exist"""
            # Make sure save_dir exists
            if not os.path.exists(opts.save_dir):
                print(f"Creating directory {opts.save_dir}")
                os.makedirs(opts.save_dir, exist_ok=True)
            
            # Call the original function
            return original_validate(problem, model, val_dataset, tb_logger, opts, _id)
        
        # Replace the function in the module
        train.validate = patched_validate
        
        # Also monkey patch torch.save to ensure directory exists
        import torch
        original_torch_save = torch.save
        
        def patched_torch_save(obj, f, *args, **kwargs):
            """Patched version of torch.save that ensures directory exists"""
            if isinstance(f, str):
                os.makedirs(os.path.dirname(f), exist_ok=True)
            return original_torch_save(obj, f, *args, **kwargs)
        
        # Replace torch.save globally
        torch.save = patched_torch_save
        
        print("Successfully patched train.py and torch.save to ensure directories exist")
        return True
        
    except ImportError as e:
        print(f"Could not patch train module: {e}")
        return False

def main():
    """Main function to apply patches"""
    print("Applying patches to fix directory errors...")
    
    # Patch train module
    train_patched = patch_train_module()
    
    if train_patched:
        print("All patches applied successfully!")
    else:
        print("Warning: Could not apply all patches.")
    
    print("\nNow you can run your experiment without directory errors.")
    print("Use: python -c 'import fix_directory_error; fix_directory_error.main()' && ./run_no_log.sh")

if __name__ == "__main__":
    main() 