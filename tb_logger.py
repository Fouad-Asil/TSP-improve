"""
Simple adapter for TensorBoard logging that works with either tensorboard_logger or tensorboardX
"""

import os
import importlib.util

# Check if libraries are available but don't import them yet
tensorboardX_available = importlib.util.find_spec("tensorboardX") is not None
tb_logger_available = importlib.util.find_spec("tensorboard_logger") is not None

class TensorboardLogger:
    """
    Simple adapter for TensorBoard logging that works with either tensorboard_logger or tensorboardX
    """
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        
        # Import the appropriate library on demand
        if tensorboardX_available:
            print("Using tensorboardX for logging")
            from tensorboardX import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            self.use_tensorboardx = True
        elif tb_logger_available:
            print("Using tensorboard_logger for logging")
            from tensorboard_logger import Logger
            self.writer = Logger(log_dir)
            self.use_tensorboardx = False
        else:
            print("WARNING: Neither tensorboardX nor tensorboard_logger is available. Logging disabled.")
            self.writer = None
            self.use_tensorboardx = False
    
    def log_value(self, name, value, step):
        """Log a scalar value to tensorboard"""
        if self.writer is None:
            return
            
        if self.use_tensorboardx:
            self.writer.add_scalar(name, value, step)
        else:
            self.writer.log_value(name, value, step)
    
    def log_histogram(self, name, values, step):
        """Log a histogram to tensorboard"""
        if self.writer is None:
            return
            
        if self.use_tensorboardx:
            self.writer.add_histogram(name, values, step)
        else:
            # Basic fallback if using tensorboard_logger which doesn't support histograms
            try:
                self.writer.log_histogram(name, values, step)
            except AttributeError:
                # Just log min/max/mean instead
                import numpy as np
                if isinstance(values, list):
                    values = np.array(values)
                self.log_value(f"{name}/min", values.min(), step)
                self.log_value(f"{name}/max", values.max(), step)
                self.log_value(f"{name}/mean", values.mean(), step)
    
    def log_images(self, name, images, step):
        """Log images to tensorboard"""
        if self.writer is None:
            return
            
        try:
            if self.use_tensorboardx:
                for i, img in enumerate(images):
                    try:
                        self.writer.add_image(f"{name}/{i}", img, step)
                    except TypeError as e:
                        print(f"WARNING: Could not log image {name}/{i}. Error: {e}")
                        # Try to convert to a simple scalar instead
                        try:
                            import numpy as np
                            if hasattr(img, 'mean'):
                                self.log_value(f"{name}/{i}_mean", img.mean(), step)
                            elif isinstance(img, np.ndarray):
                                self.log_value(f"{name}/{i}_mean", np.mean(img), step)
                        except:
                            pass
            else:
                # Basic fallback if using tensorboard_logger
                try:
                    self.writer.log_images(name, images, step)
                except (AttributeError, TypeError) as e:
                    print(f"WARNING: Image logging not supported with current logger: {name}. Error: {e}")
        except Exception as e:
            print(f"WARNING: Failed to log images {name}. Error: {e}")
            # Log won't crash the training even if image logging fails
    
    def close(self):
        """Close the logger"""
        if self.writer is not None and self.use_tensorboardx:
            self.writer.close() 