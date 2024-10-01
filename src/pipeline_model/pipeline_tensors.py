import sys
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Tuple

sys.path.append("/workspaces/DetectionXR/")
from src.model.backbone import Backbone
from src.model.feature_pyramid_network import FeaturePyramidNetwork
from src.model.head import Head
from src.utils.reg_log import log_error, log_train  # Ensure you have a logging utility

class IModelPipeline(ABC):
    """
    Abstract base class for the model pipeline.
    """
    @abstractmethod
    def call(self, 
             inputs_shape: tuple = (512, 512, 3),
             num_classes: int = 10,
             num_anchors: int = 3) -> tf.keras.Model:
        """
        Builds the model pipeline.

        Args:
            inputs_shape (tuple): Shape of the input tensor.
            num_classes (int): Number of output classes.
            num_anchors (int): Number of anchors.

        Returns:
            tf.keras.Model: The constructed model.
        """
        pass

class ModelPipeline(IModelPipeline):
    """
    Implementation of the model pipeline.
    """
    def call(self, 
             inputs_shape: tuple = (512, 512, 3),
             num_classes: int = 10,
             num_anchors: int = 3) -> tf.keras.Model:
        """
        Constructs the model pipeline.

        Args:
            inputs_shape (tuple): Shape of the input tensor.
            num_classes (int): Number of output classes.
            num_anchors (int): Number of anchors.

        Returns:
            tf.keras.Model: The constructed model.
        """
        # Input validation
        if len(inputs_shape) != 3:
            raise ValueError("inputs_shape must be a tuple of three integers representing (height, width, channels).")

        if num_classes <= 0:
            raise ValueError("num_classes must be a positive integer.")

        if num_anchors <= 0:
            raise ValueError("num_anchors must be a positive integer.")

        # Create input tensor
        inputs = tf.keras.Input(shape=inputs_shape)

        # Backbone 
        try:
            backbone = Backbone()  # Ensure you have a proper instance of Backbone
            backbone_output = backbone.call(inputs)  # Pass the input tensor to Backbone
            log_train("Backbone processed successfully.")
        except Exception as e:
            log_error(f"Error in Backbone call: {str(e)}")
            raise

        # Feature Pyramid Network (FPN)
        try:
            fpn = FeaturePyramidNetwork()
            fpn_output = fpn.call(backbone_output)  # Ensure the output from Backbone is correct
            log_train("Feature Pyramid Network processed successfully.")
        except Exception as e:
            log_error(f"Error in Feature Pyramid Network call: {str(e)}")
            raise

        # Detection Head
        try:
            head = Head()
            class_output, bbox_output, objectness_output = head.call(
                fpn_output, 
                num_classes=num_classes,
                num_anchors=num_anchors
            )
            log_train("Detection Head processed successfully.")
        except Exception as e:
            log_error(f"Error in Detection Head call: {str(e)}")
            raise
        
        # Create the final model
        model = tf.keras.Model(inputs=inputs, outputs=[class_output, bbox_output, objectness_output])
        log_train("Model constructed successfully.")

        return model  # Returning the model itself
