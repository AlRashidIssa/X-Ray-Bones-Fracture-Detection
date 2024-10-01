import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, Flatten, BatchNormalization, Dense, # type: ignore
                                     MaxPooling2D, Reshape)  # type: ignore
from typing import Tuple
from abc import ABC, abstractmethod

append_path = "/workspaces/X-Ray-Bones-Fracture-Detection"
sys.path.append(append_path)
from src.model.backbone import Backbone
from src.model.convbloc_upsampleblock import ConvBlock, UpSampleBlock
from src.utils.reg_log import log_error


class IHead(ABC):
    """
    Abstract base class for the Head module in the object detection model.
    """

    @abstractmethod
    def call(self, 
             input_tensor: tf.Tensor,
             num_classes: int = 10,
             num_anchors: int = 3,
             activation: str = 'relu',
             activation_class: str = 'softmax',
             activation_objectness: str = 'sigmoid',
             activation_bbox_output: str = 'linear',
             padding: str = 'same'
             ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Abstract method for the call function.

        Args:
            input_tensor (tf.Tensor): Input tensor from the preceding layers.
            num_classes (int): Number of object classes. Default is 10.
            num_anchors (int): Number of anchor boxes. Default is 3.
            activation (str): Activation function for the hidden layers. Default is 'relu'.
            activation_class (str): Activation function for class output. Default is 'softmax'.
            activation_objectness (str): Activation function for objectness output. Default is 'sigmoid'.
            activation_bbox_output (str): Activation function for bounding box output. Default is 'linear'.
            padding (str): Padding type for convolution layers. Default is 'same'.

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: A tuple containing:
                - class_output: Tensor for class predictions.
                - bbox_output: Tensor for bounding box predictions.
                - objectness_output: Tensor for objectness predictions.
        """
        pass


class Head(IHead):
    """
    Implementation of the Head module for the object detection model,
    responsible for generating class, bounding box, and objectness outputs.
    """

    def call(self, 
             input_tensor: tf.Tensor,
             num_classes: int = 10,
             num_anchors: int = 3,
             activation: str = 'relu',
             activation_class: str = 'softmax',
             activation_objectness: str = 'sigmoid',
             activation_bbox_output: str = 'linear',
             padding: str = 'same'
             ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Processes the input tensor to produce class, bounding box, and objectness outputs.

        Args:
            input_tensor (tf.Tensor): Input tensor from the preceding layers.
            num_classes (int): Number of object classes. Default is 10.
            num_anchors (int): Number of anchor boxes. Default is 3.
            activation (str): Activation function for the hidden layers. Default is 'relu'.
            activation_class (str): Activation function for class output. Default is 'softmax'.
            activation_objectness (str): Activation function for objectness output. Default is 'sigmoid'.
            activation_bbox_output (str): Activation function for bounding box output. Default is 'linear'.
            padding (str): Padding type for convolution layers. Default is 'same'.

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: A tuple containing:
                - class_output: Tensor for class predictions.
                - bbox_output: Tensor for bounding box predictions.
                - objectness_output: Tensor for objectness predictions.

        Raises:
            ValueError: If input_tensor is not a tf.Tensor or has unexpected dimensions.
            TypeError: If num_classes or num_anchors is not an integer.
            Exception: For any other unexpected errors during processing.
        """

        if input_tensor.ndim != 4:  # Check if tensor has 4 dimensions (batch, height, width, channels)
            log_error(f"Input tensor does not have 4 dimensions: {input_tensor.shape}.")
            raise ValueError("Expected input_tensor to have 4 dimensions (batch, height, width, channels).")

        # Validate numerical parameters
        if not isinstance(num_classes, int) or not isinstance(num_anchors, int):
            log_error(f"Invalid type for num_classes or num_anchors: {num_classes}, {num_anchors}.")
            raise TypeError("num_classes and num_anchors must be integers.")

        try:
            # Head processing
            x = Conv2D(128, (1, 1), activation=activation, padding=padding)(input_tensor)
            x = BatchNormalization()(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)

            x = Flatten()(x)
            x = Dense(512, activation=activation)(x)
            x = Dense(512, activation=activation)(x)

            # Class output
            class_output = Dense(num_classes, activation=activation_class, name='class_output')(x)

            # Bounding box output
            bbox_output = Dense(num_classes * 4, activation=activation_bbox_output, name='bbox_output')(x)
            bbox_output = Reshape((num_classes, 4), name='reshape_bboxes')(bbox_output)

            # Objectness output
            objectness_output = Dense(num_anchors, activation=activation_objectness, name='objectness_output')(x)

            return class_output, bbox_output, objectness_output

        except Exception as e:
            log_error(f"An error occurred while processing the head: {str(e)}")
            raise  # Re-raise the exception after logging