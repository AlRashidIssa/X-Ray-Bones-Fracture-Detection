import sys
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, BatchNormalization, UpSampling2D  # type: ignore
import tensorflow as tf
from typing import Tuple
from abc import ABC, abstractmethod

# Local imports
append_path = "/workspaces/X-Ray-Bones-Fracture-Detection"
sys.path.append(append_path)
from src.utils.reg_log import log_inference, log_error

class IBlock(ABC):
    """
    Interface for a generic block in a neural network that applies a series of 
    convolutional layers, optional batch normalization, and dropout to an input tensor.
    """
    
    @abstractmethod
    def call(self, 
             input_tensor: tf.Tensor, 
             dim: int = 512, 
             dropout: bool = True, 
             dropout_val: float = 0.3,
             batchnormalization: bool = True, 
             activation: str = 'relu', 
             padding: str = 'same') -> tf.Tensor:
        """
        Abstract method to process input_tensor with a sequence of convolutional layers,
        batch normalization, and dropout.

        Args:
            input_tensor (tf.Tensor): The input tensor to process.
            dim (int, optional): The base number of filters for Conv2D layers. Default is 512.
            dropout (bool, optional): Whether to apply dropout. Default is True.
            dropout_val (float, optional): Dropout rate if dropout is enabled. Default is 0.3.
            batchnormalization (bool, optional): Whether to apply batch normalization. Default is True.
            activation (str, optional): Activation function to use in Conv2D. Default is 'relu'.
            padding (str, optional): Padding type for Conv2D layers. Default is 'same'.

        Returns:
            tf.Tensor: The processed tensor after applying the block operations.
        """
        pass


class ConvBlock(IBlock):
    """
    A convolutional block consisting of two Conv2D layers, optional batch normalization,
    and optional dropout. It processes the input tensor and applies specified operations.
    """
    
    def call(self, 
             input_tensor: tf.Tensor, 
             dim: int = 512, 
             dropout: bool = True, 
             dropout_val: float = 0.3,
             batchnormalization: bool = True, 
             activation: str = 'relu', 
             padding: str = 'same') -> tf.Tensor:
        """
        Apply two Conv2D layers with optional batch normalization and dropout.

        Args:
            input_tensor (tf.Tensor): The input tensor to process.
            dim (int, optional): Number of filters for the Conv2D layers. Default is 512.
            dropout (bool, optional): Whether to apply dropout. Default is True.
            dropout_val (float, optional): Dropout rate if dropout is enabled. Default is 0.3.
            batchnormalization (bool, optional): Whether to apply batch normalization. Default is True.
            activation (str, optional): Activation function to use. Default is 'relu'.
            padding (str, optional): Padding type for the Conv2D layers. Default is 'same'.

        Returns:
            tf.Tensor: The processed tensor.
        """
        try:
            # First Conv2D layer
            x = Conv2D(dim * 2, (3, 3), activation=activation, padding=padding)(input_tensor)
            if batchnormalization:
                x = BatchNormalization()(x)
            if dropout:
                x = Dropout(dropout_val)(x)

            # Second Conv2D layer
            x = Conv2D(dim, (1, 1), activation=activation, padding=padding)(x)
            if batchnormalization:
                x = BatchNormalization()(x)
            if dropout:
                x = Dropout(dropout_val)(x)

            return x

        except Exception as e:
            raise ValueError(f"Error in ConvBlock: {str(e)}")


class UpSampleBlock(IBlock):
    """
    An up-sampling block consisting of Conv2D and UpSampling2D layers, optional batch normalization,
    and optional dropout. It up-samples the input tensor and applies specified operations.
    """
    
    def call(self, 
             input_tensor: tf.Tensor, 
             dim: int = 512, 
             dropout: bool = True, 
             dropout_val: float = 0.3,
             batchnormalization: bool = True, 
             activation: str = 'relu', 
             padding: str = 'same') -> tf.Tensor:
        """
        Apply an up-sampling operation followed by a Conv2D layer, optional batch normalization, 
        and dropout.

        Args:
            input_tensor (tf.Tensor): The input tensor to process.
            dim (int, optional): Number of filters for the Conv2D layers. Default is 512.
            dropout (bool, optional): Whether to apply dropout. Default is True.
            dropout_val (float, optional): Dropout rate if dropout is enabled. Default is 0.3.
            batchnormalization (bool, optional): Whether to apply batch normalization. Default is True.
            activation (str, optional): Activation function to use. Default is 'relu'.
            padding (str, optional): Padding type for the Conv2D layers. Default is 'same'.

        Returns:
            tf.Tensor: The processed tensor after up-sampling and convolution.
        """
        try:
            # First Conv2D layer (dim // 2 filters)
            x = Conv2D(dim // 2, (1, 1), activation=activation, padding=padding)(input_tensor)
            if batchnormalization:
                x = BatchNormalization()(x)
            if dropout:
                x = Dropout(dropout_val)(x)

            # Up-sampling
            x = UpSampling2D((2, 2))(x)

            return x

        except Exception as e:
            log_error(f"Error in UpSampleBlock: {str(e)}")
            raise ValueError(f"Error in UpSampleBlock: {str(e)}")

if __name__ == "__main__":
    input_tensor = tf.random.normal((1, 128, 128, 64))  # A sample input tensor
    conv_block = ConvBlock()
    output_tensor = conv_block.call(input_tensor, dim=128, dropout=True, dropout_val=0.5)
    print(output_tensor.shape)  # Example output shape
    input_tensor = tf.random.normal((1, 64, 64, 128))  # A sample input tensor
    upsample_block = UpSampleBlock()
    output_tensor = upsample_block.call(input_tensor, dim=128, dropout=False)
    print(output_tensor.shape)  # Example output shape