import unittest
import tensorflow as tf
import sys

sys.path.append("/workspaces/DetectionXR/")
from src.pipeline_model.pipeline_tensors import ModelPipeline  # Adjust the import according to your project structure
from src.utils.reg_log import log_error

class TestModelPipeline(unittest.TestCase):
    """
    Unit tests for the ModelPipeline class.
    """

    def setUp(self):
        """Set up the model pipeline for testing."""
        self.model_pipeline = ModelPipeline()

    def test_model_pipeline_creation(self):
        """Test the model pipeline creation with valid inputs."""
        try:
            model = self.model_pipeline.call(inputs_shape=(512, 512, 3), num_classes=10, num_anchors=3)
            self.assertIsInstance(model, tf.keras.Model, "The model should be a Keras Model instance.")
            self.assertEqual(model.input_shape, (None, 512, 512, 3), "Input shape should match the specified input shape.")
            self.assertEqual(len(model.outputs), 3, "The model should have three outputs (class, bbox, objectness).")
        except ValueError as ve:
            self.fail(f"Model pipeline creation raised a ValueError: {str(ve)}")
        except Exception as e:
            self.fail(f"Model pipeline creation raised an unexpected exception: {str(e)}")

    def test_invalid_input_shape(self):
        """Test model pipeline creation with an invalid input shape."""
        with self.assertRaises(ValueError) as context:
            self.model_pipeline.call(inputs_shape=(512, 512))  # Invalid shape (should be a tuple of 3)
        self.assertIn("inputs_shape must be a tuple of three integers", str(context.exception))

    def test_invalid_number_of_classes(self):
        """Test model pipeline creation with an invalid number of classes."""
        with self.assertRaises(ValueError) as context:
            self.model_pipeline.call(inputs_shape=(512, 512, 3), num_classes=-1, num_anchors=3)  # Invalid number of classes
        self.assertIn("num_classes must be a positive integer", str(context.exception))

    def test_invalid_number_of_anchors(self):
        """Test model pipeline creation with an invalid number of anchors."""
        with self.assertRaises(ValueError) as context:
            self.model_pipeline.call(inputs_shape=(512, 512, 3), num_classes=10, num_anchors=-1)  # Invalid number of anchors
        self.assertIn("num_anchors must be a positive integer", str(context.exception))

if __name__ == '__main__':
    unittest.main()
