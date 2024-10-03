import unittest
import numpy as np
import cv2
import sys
from pathlib import Path

# Append the apth to the workspace for importing utility functions
append_path = "/workspaces/X-Ray-Bones-Fracture-Detection"
sys.path.append(append_path)
from src.deployment_preprocessed.run_pipeline_prediction_deployment import RunRunPipelineDeploymentPrediction

class TestRunRunPipelineDeploymentPrediction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Set up any state that is shared across all tests.
        """
        # Define a valid model path and image for testing
        cls.model_path = "/workspaces/X-Ray-Bones-Fracture-Detection/pre-trained_model/XRayBoneFractureModel.h5"  # Assuming a valid pre-trained model exists here
        cls.image_path = "/workspaces/X-Ray-Bones-Fracture-Detection/data/extraction_zip/YOLODataSet/images/val/XR_ELBOW_positive_950.png"  # Assuming an image file exists at this path

        # Load the image using OpenCV for testing
        cls.image = cv2.imread(cls.image_path)
        if cls.image is None:
            raise FileNotFoundError(f"Test image could not be loaded from {cls.image_path}")

    def test_prediction_with_image_file(self):
        """
        Test the prediction pipeline with an image file path.
        """
        pipeline = RunRunPipelineDeploymentPrediction()
        result, label = pipeline.call(model_path=self.model_path, image=self.image_path, model_name="TestModel")
        
        # Validate that the result is a NumPy array (as the output should be the image with bounding boxes)
        self.assertIsInstance(result, np.ndarray, "The prediction result should be a NumPy array.")
        self.assertFalse(result is None, "The prediction result should not be None.")

    def test_prediction_with_numpy_array(self):
        """
        Test the prediction pipeline with an image as a NumPy array.
        """
        pipeline = RunRunPipelineDeploymentPrediction()
        result, label = pipeline.call(model_path=self.model_path, image=self.image, model_name="TestModel")
        
        # Validate that the result is a NumPy array
        self.assertIsInstance(result, np.ndarray, "The prediction result should be a NumPy array.")
        self.assertIsInstance(label, str, "The Prediction label it is string.")

        self.assertFalse(result is None, "The prediction result should not be None.")

if __name__ == "__main__":
    unittest.main()
