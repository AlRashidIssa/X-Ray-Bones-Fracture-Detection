import os
import tensorflow as tf
import sys

from tensorflow.keras.models import load_model # type: ignore
from abc import ABC, abstractmethod
from typing import Union, Any

# Append the apth to the workspace for importing utility functions
append_path = "/workspaces/X-Ray-Bones-Fracture-Detection"
sys.path.append(append_path)

# Importing custom loaggin utilities
from src.utils.reg_log import log_error, log_inference

class ILoadPreTrainModel(ABC):
    """
    Abstract base class to define the interface for loading per-trained models.
    All classes inheriting this interface must implement the `call`.
    """

    @abstractmethod
    def call(self, model_path: Union[str, None], model_name: str) -> tf.keras.Model:
        """
        Load a pre-trained keras model.

        Args:
            model_path (Union[str, None]): The file path to the model. If None, a default path is used.
            model_name (str): The name of the model file.

        Returns:
            tf.keras.Model: The loaded keras model.
        """
        pass

class LoadPreTrainModel(ILoadPreTrainModel):
    """
    Class for loading pre-trained models. Implements the ILoadPreTrainModel interface.

    If no model path is provided, a default model is loaded from the predefined directory.
    """
    def call(self, model_path: Union[str, None], model_name: str = 'Unknow') -> tf.keras.Model:
        """
        Load a pre-trained model from a specified path. If the path is None, load a default model.

        Args:
            model_path (Union[str, None]): The file path to the model. If None, a default path is used.
            model_name (str): The name of the model file.

        Returns:
            tf.keras.Model: The loaded Keras model.

        Raises:
            FileNotFoundError: If the model file is not found.
            ValueError: If an invalid model path is passed.
            Exception: For any other errors encountered during loading.
        """
        # If model_path is None, set default model path and log the event
        if model_path is None:
            log_error("Model path not provided. Loading default pre-trained model.")
            model_name = "default_model.h5"  # Default model name
            model_path = os.path.join(f"{append_path}/pre-trained_model", model_name)
        
        # Attempt to load the model
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            # Log the inference is starting
            log_inference(f"Loading model from: {model_path}")
            model = load_model(model_path)

            # Log sucessful model loading
            log_inference(f"Model loaded successfully: {model_name}")
            return model
        except FileNotFoundError as fnf_error:
            log_error(f"Fiel not found: {fnf_error}")
            raise

        except ValueError as val_error:
            log_error(f"Invalid model path provided: {val_error}")
            raise
        except Exception as e:
            log_error(f"An unexpected error occurred while loading the model: {str(e)}")
