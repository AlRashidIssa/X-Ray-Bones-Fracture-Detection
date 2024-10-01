import sys
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, Dropout, MaxPooling2D,  # type: ignore
                                     BatchNormalization)
from typing import Tuple
from abc import ABC, abstractmethod

# Local imports
append_path = "/workspaces/X-Ray-Bones-Fracture-Detection"
sys.path.append(append_path)
from src.utils.reg_log import log_inference, log_error

class IBackbone(ABC):
    """
    Interface for the backbone model that defines the structure of the network.
    """

    @abstractmethod
    def call(self, 
             input_tensor: tf.Tensor, 
             filters: list = [64, 128, 256, 512],  # Filter sizes for the backbone
             dropout: bool = True,
             batch_normalization: bool = True,
             activation='relu',
             padding='same') -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Constructs the backbone of the model using convolutional layers.

        Args:
            input_tensor (tf.Tensor): Input tensor to the backbone.
            filters (list): List of filter sizes for each Conv2D layer.
            dropout (bool): Whether to include Dropout layers.
            batch_normalization (bool): Whether to include Batch Normalization layers.
            activation (str): Activation function to use.
            padding (str): Padding method for Conv2D layers.

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: Outputs from the last three convolutional layers.
        """
        pass

class Backbone(IBackbone):
    """
    Implementation of the backbone model that builds a CNN with specified configurations.
    """

    def call(self, 
             input_tensor: tf.Tensor, 
             filters: list = [64, 128, 256, 512], 
             dropout: bool = True, 
             batch_normalization: bool = True, 
             activation='relu', 
             padding='same') -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Constructs the backbone model with convolutional layers, pooling, dropout, and batch normalization.

        Args:
            input_tensor (tf.Tensor): Input tensor to the backbone.
            filters (list): List of filter sizes for each Conv2D layer.
            dropout (bool): Whether to include Dropout layers.
            batch_normalization (bool): Whether to include Batch Normalization layers.
            activation (str): Activation function to use.
            padding (str): Padding method for Conv2D layers.

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: Outputs from the last three convolutional layers (c3, c4, c5).

        Raises:
            ValueError: If the input tensor has an invalid shape.
        """

        # Ensure the input tensor has at least 4 dimensions (batch_size, height, width, channels)
        if len(input_tensor.shape) != 4:
            raise ValueError("Input tensor must have shape (batch_size, height, width, channels).")

        try:
            # Convolutional Layer 1
            c1 = Conv2D(filters[0], (3, 3), activation=activation, padding=padding)(input_tensor)
            if batch_normalization:
                c1 = BatchNormalization()(c1)
            if dropout:
                c1 = Dropout(0.5)(c1)
            p1 = MaxPooling2D(pool_size=(2, 2))(c1)

            # Convolutional Layer 2
            c2 = Conv2D(filters[1], (3, 3), activation=activation, padding=padding)(p1)
            if batch_normalization:
                c2 = BatchNormalization()(c2)
            if dropout:
                c2 = Dropout(0.5)(c2)
            p2 = MaxPooling2D(pool_size=(2, 2))(c2)

            # Convolutional Layer 3 (this will be returned as c3)
            c3 = Conv2D(filters[2], (3, 3), activation=activation, padding=padding)(p2)
            if batch_normalization:
                c3 = BatchNormalization()(c3)
            if dropout:
                c3 = Dropout(0.5)(c3)
            p3 = MaxPooling2D(pool_size=(2, 2))(c3)

            # Convolutional Layer 4 (this will be returned as c4)
            c4 = Conv2D(filters[3], (3, 3), activation=activation, padding=padding)(p3)
            if batch_normalization:
                c4 = BatchNormalization()(c4)
            if dropout:
                c4 = Dropout(0.5)(c4)
            p4 = MaxPooling2D(pool_size=(2, 2))(c4)

            # Convolutional Layer 5 (this will be returned as c5)
            c5 = Conv2D(filters[3], (3, 3), activation=activation, padding=padding)(p4)
            if batch_normalization:
                c5 = BatchNormalization()(c5)
            if dropout:
                c5 = Dropout(0.5)(c5)

            # Return the outputs from the last three layers (c3, c4, c5)
            return c3, c4, c5

        except Exception as e:
            log_error(f"Error in Backbone call: {e}")
            raise RuntimeError(f"Error in Backbone call: {e}")
