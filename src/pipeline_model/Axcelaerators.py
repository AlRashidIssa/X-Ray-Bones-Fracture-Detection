import tensorflow as tf
import sys
from abc import ABC, abstractmethod
from tensorflow.python.distribute.distribute_lib import Strategy
from tensorflow.python.framework.ops import device_v2 as device
sys.path.append('/workspaces/DetectionXR/')
from src.utils.reg_log import log_error, log_inference

class IAxcelaerators(ABC):
    """
    Interface for accelerator handling, defining a method for selecting devices (GPU, TPU, or CPU).
    """
    @abstractmethod
    def call(self, AXL: str = 'GPU') -> tf.distribute.Strategy:
        """
        Abstract method to be implemented by subclasses to handle device allocation.
        
        Args:
            AXL (str): The type of accelerator to use ('GPU', 'TPU', or 'CPU').
            
        Returns:
            tf.distribute.Strategy: The distribution strategy for the chosen device.
        """
        pass

class Axcelaerators(IAxcelaerators):
    """
    Concrete class for selecting accelerators (GPU, TPU, or CPU) and applying appropriate
    TensorFlow distribution strategies for multi-device training.
    """
    
    def call(self, AXL: str = 'GPU') -> tf.distribute.Strategy:
        """
        Implements the logic to allocate a GPU, TPU, or CPU for model training.
        
        Args:
            AXL (str): The type of accelerator to use ('GPU', 'TPU', or 'CPU').
            
        Returns:
            tf.distribute.Strategy: The TensorFlow distribution strategy for the chosen device.
        
        Raises:
            RuntimeError: If the specified device is not available or not supported.
        """
        try:
            # Normalize the AXL input to uppercase to avoid case sensitivity issues
            AXL = AXL.upper()

            # Handle GPU allocation
            if AXL == 'GPU':
                print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
                if len(tf.config.list_physical_devices('GPU')) == 0:
                    log_error("No GPU found. Ensure your GPU is installed and recognized by TensorFlow.")
                    raise RuntimeError("No GPU found.")
                
                # Enable memory growth for GPUs
                gpus = tf.config.list_physical_devices('GPU')
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

                # Set GPU for usage
                tf.config.set_visible_devices(gpus[0], 'GPU')
                print(f"Using GPU: {gpus[0]}")
                
                # Define the distributed strategy for GPUs
                strategy = tf.distribute.MirroredStrategy()
                return strategy

            # Handle TPU allocation
            elif AXL == 'TPU':
                try:
                    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
                    tf.config.experimental_connect_to_cluster(resolver)
                    tf.tpu.experimental.initialize_tpu_system(resolver)
                    print("TPU initialized successfully.")

                    # Create TPU strategy
                    strategy = tf.distribute.TPUStrategy(resolver)
                    return strategy

                except ValueError as e:
                    log_error(f"TPU initialization failed: {str(e)}")
                    raise RuntimeError(f"TPU initialization failed: {str(e)}")

            # Handle CPU allocation
            elif AXL == 'CPU':
                # Use default CPU strategy for TensorFlow (no distributed strategy needed)
                print("Using CPU.")
                log_inference("Running on CPU.")
                strategy = tf.distribute.get_strategy()  # Default strategy for single CPU
                return strategy

            # Handle unsupported device types
            else:
                log_error(f"Unsupported accelerator type: {AXL}. Use 'GPU', 'TPU', or 'CPU'.")
                raise ValueError(f"Unsupported accelerator type: {AXL}. Use 'GPU', 'TPU', or 'CPU'.")

        except Exception as e:
            # Log the error and re-raise
            log_error(f"Error during accelerator selection: {str(e)}")
            raise RuntimeError(f"Failed to select the accelerator: {str(e)}")

# Example usage (commented out):
# axc = Axcelaerators()
# strategy = axc.call('GPU')

        