import os
import numpy as np
import tensorflow as tf
import sys
import cv2
from abc import ABC, abstractmethod
from typing import Union, Any, List

# Append the path to the workspace for importing utility functions
append_path = "/workspaces/X-Ray-Bones-Fracture-Detection"
sys.path.append(append_path)

# Importing custom logging utilities
from src.utils.reg_log import log_error, log_inference
from src.deployment_preprocessed.load__model import LoadPreTrainModel
from src.deployment_preprocessed.prediction_deployment import PredictionDeployment

class IRunRunPipelineDeploymentPrediction(ABC):
    """
    Interface for the deployment prediction pipeline. Defines the structure for the
    deployment process of a model on a given image.

    Methods
    -------
    call(model: tf.keras.Model, image: Union[str, np.ndarray, None], model_name: str) -> np.ndarray
        Abstract method to be implemented for loading the model and predicting on the input image.
    """
    @abstractmethod
    def call(self, model_path: str, image: Union[str, np.ndarray, None] = None, model_name: str = "Unknown") -> np.ndarray:
        """
        Abstract method for the deployment process.

        Parameters
        ----------
        model_path : str
            Path to the pre-trained model.
        image : Union[str, np.ndarray, None], optional
            Input image either as a file path or a NumPy array, by default None.
        model_name : str, optional
            Name of the model for logging purposes, by default "Unknown".

        Returns
        -------
        np.ndarray
            The predicted image with bounding boxes.
        """
        pass

class RunRunPipelineDeploymentPrediction(IRunRunPipelineDeploymentPrediction):
    """
    Concrete implementation of the IRunPipelineDeploymentPrediction interface.
    Handles the process of loading a pre-trained model and predicting the result on a given image.
    """
    def call(self, model_path: str, image: Union[str, np.ndarray, None] = None, model_name: str = "Unknown") -> np.ndarray:
        """
        Executes the model deployment pipeline by loading a pre-trained model and predicting on the given image.

        Parameters
        ----------
        model_path : str
            Path to the pre-trained model.
        image : Union[str, np.ndarray, None], optional
            Input image either as a file path or a NumPy array, by default None.
        model_name : str, optional
            Name of the model for logging purposes, by default "Unknown".

        Returns
        -------
        np.ndarray
            The predicted image with bounding boxes.
        
        Raises
        ------
        ValueError
            If the image is not provided or the model path is incorrect.
        """
        try:
            # Load pre-trained model
            log_inference(f"Loading pre-trained model: {model_name} from path: {model_path}")
            model = LoadPreTrainModel().call(model_name=model_name, model_path=model_path)
            
            if model is None:
                log_error(f"Model could not be loaded. Model path: {model_path}")
                raise ValueError("Model loading failed.")

            # Check and handle image input
            if image is None:
                log_error(f"Image is None. Pass a valid image path or NumPy array.")
                raise ValueError("Image input is required.")

            log_inference(f"Image provided for prediction: {type(image).__name__}")

            # If image is a string, assume it's a file path
            if isinstance(image, str):
                log_inference(f"Loading image from file path: {image}")
                image = cv2.imread(image)

                if image is None:
                    log_error(f"Failed to load image from the path: {image}")
                    raise ValueError("Image could not be loaded from the provided path.")
            else:
                log_inference("Image provided as a NumPy array.")

            # Perform prediction on the image
            img_pred = PredictionDeployment().call(model=model, image=image)
            log_inference("Image prediction completed.")

            return img_pred

        except Exception as e:
            log_error(f"Error occurred during prediction: {str(e)}")
            raise e


import matplotlib.pyplot as plt

if __name__ == "__main__":
    import numpy as np

# Path to the model and the image you want to predict
model_path = "/workspaces/X-Ray-Bones-Fracture-Detection/pre-trained_model/XRayBoneFractureModel.h5"
image_path = "/workspaces/X-Ray-Bones-Fracture-Detection/data/extraction_zip/YOLODataSet/images/train/XR_SHOULDER_positive_1109.png"
model_name = "XRayBoneFractureModel"

# Instantiate the prediction pipeline
pipeline = RunRunPipelineDeploymentPrediction()

try:
    # Call the pipeline to load the model and predict the result on the image
    predicted_image = pipeline.call(model_path=model_path, image=image_path, model_name=model_name)

    # If prediction is successful, perform some post-processing
    # For example, you could display the predicted image using OpenCV or save it to a file
    print("Prediction completed successfully. Displaying the output...")
    
    # Convet BGR to RGB
    img_rgb = cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB)

    # Display the image using matplotlib
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()

except Exception as e:
    # Handle any errors during the process
    print(f"An error occurred: {str(e)}")
