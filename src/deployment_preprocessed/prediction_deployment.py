import os
import numpy as np
import tensorflow as tf
import sys
import cv2
from abc import ABC, abstractmethod
from typing import Union, Any, List, Tuple

# Append the path to the workspace for importing utility functions
append_path = "/workspaces/X-Ray-Bones-Fracture-Detection"
sys.path.append(append_path)

# Importing custom logging utilities
from src.utils.reg_log import log_error, log_inference


class IPredictionDeployment(ABC):
    """
    Interface for a prediction deployment system.
    
    Defines a contract for deploying a model to make predictions on input images.
    """
    
    @abstractmethod
    def call(self, model: tf.keras.Model, image: Union[str, np.ndarray, None] = None, 
             confidence_threshold: float = 0.1) -> Tuple[np.ndarray, str]:
        """
        Method to handle predictions on input images.
        
        Args:
            model (tf.keras.Model): The pre-trained object detection model.
            image (Union[str, np.ndarray]): The input image path or image array.
            confidence_threshold (float): Confidence threshold to filter low-confidence predictions.

        Returns:
            Tupe[np.ndarray, str]: Annotated image with bounding boxes and label, lable.
        """
        pass


class PredictionDeployment(IPredictionDeployment):
    """
    Concrete implementation of the IPredictionDeployment interface.
    
    This class handles the deployment of a pre-trained model to process an input image, 
    make predictions, and annotate the image with bounding boxes and class labels.
    """
    
    def call(self, model: tf.keras.Model, image: Union[str, np.ndarray, None] = None, 
             confidence_threshold: float = 0.1) -> Tuple[np.ndarray, str]:
        """
        Processes the image, makes predictions using the model, and returns the annotated image.

        Args:
            model (tf.keras.Model): Pre-trained object detection model.
            image (Union[str, np.ndarray, None]): Path to the image or a numpy array representing the image.
            confidence_threshold (float): The confidence threshold to filter predictions.

        Returns:
            np.ndarray: The image with annotated bounding boxes and class labels.

        Raises:
            ValueError: If image or model is None, or if prediction results do not match expectations.
        """
        # Validate the image input
        if image is None:
            log_error("Image is None. Please provide an image path or numpy array.")
            raise ValueError("Invalid image input: Image cannot be None.")

        # Load the image if a file path is provided
        if isinstance(image, str):
            log_inference(f"Loading image from path: {image}")
            if not os.path.exists(image):
                log_error(f"Image path does not exist: {image}")
                raise FileNotFoundError(f"Image path does not exist: {image}")
            image = cv2.imread(image)

        # Validate the model input
        if model is None:
            log_error("Model is None. A valid model instance must be provided.")
            raise ValueError("Invalid model input: Model cannot be None.")

        # Resize and normalize the image
        try:
            log_inference(f"Resizing image from {image.shape} to (512, 512)")
            image_resized = cv2.resize(image, (512, 512))
            image_normalized = image_resized / 255.0
        except Exception as e:
            log_error(f"Error occurred while resizing/normalizing image: {e}")
            raise

        # Expand dimensions to include batch size
        image_normalized = np.expand_dims(image_normalized, axis=0)

        # Make predictions
        try:
            log_inference("Making predictions with the model.")
            prediction = model.predict(image_normalized)
        except Exception as e:
            log_error(f"Model prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {e}")

        bboxes = prediction[1]
        class_probs = prediction[0] 

        if bboxes.shape[0] == 0:
            log_error("No bounding boxes predicted.")
            raise ValueError("No bounding boxes predicted by the model.")

        # Process predictions
        batch_size = bboxes.shape[0]
        class_ids = np.argmax(class_probs, axis=-1)
        confidence_scores = np.max(class_probs, axis=-1)

        # Filter valid predictions based on the confidence threshold
        valid_indices = np.where(confidence_scores > confidence_threshold)[0]
        if len(valid_indices) == 0:
            log_error(f"No valid predictions above confidence threshold {confidence_threshold}")
            raise ValueError(f"No valid predictions above confidence threshold {confidence_threshold}")

        # Class names for labeling
        class_names = [
            'Elbow Fracture', 'Finger Fracture', 'Forearm Fracture', 'Hand Fracture', 'Shoulder Fracture',
            'Elbow Non-Fracture', 'Finger Non-Fracture', 'Forearm Non-Fracture', 'Hand Non-Fracture', 'Shoulder Non-Fracture'
        ]

        # Annotate image with bounding boxes and labels
        height, width = image.shape[:2]
        log_inference("Annotating image with bounding boxes and labels.")
        
        # Squeeze the boxes to remove extra dimensions
        bboxes = np.squeeze(bboxes)

        for bbox, class_id, conf in zip(bboxes,  class_ids, confidence_scores):
            # Unscale the bounding box
            x_min, y_min, w, h = bbox
            x_min *= width
            y_min *= height
            w *= width * 6
            h *= height * 6

            x_max = x_min + w
            y_max = y_min + h

            # Draw bounding box
            cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

            # Prepare the label with class name and confidence score
            class_name = class_names[class_id]
            bbox_info = f'{class_name}, Conf: {conf:.2f}'
            log_inference(f"Predicted: {bbox_info} at position ({x_min}, {y_min}, {x_max}, {y_max})")

            # Adjust the position of the text
            text_position = (int(x_min), max(int(y_min) - 10, 10))
            cv2.putText(image, bbox_info, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.40, (155, 255, 100), 2)

        return image, class_name
