import sys
append_path = "/workspaces/X-Ray-Bones-Fracture-Detection"
sys.path.append(append_path)
from src.model.backbone import Backbone

import unittest
import tensorflow as tf
import numpy as np

"""
Explanation of the Test Cases:
    1. setUp Method: This method is called before every test. it creates an instance of the Backbone
    class and initializes a validinput tensor for testing.
    2. test_call_with_valid_input: Test the call method with a valid input tensor and checks that the 
    output shapes match the expected dimensions after applying the convolutional and pooling layers.
    3. test_call_with_invalid_input_shape: Test that a `ValueError` is raised when the input tensor deos
    not have the required shape (missing the channel dimension).
    4. test_call_with_incalid_input_type: Ensures that a `ValueErrir` is raised when a non-tensor input (like a numpy array)
    is passed to the method.
    5. test_call_with_no_dropoutL Checks that disabling droput does not affect the output shapes.
    6. test_call_with_no_batch_norm: Similar to the previous test, but for batch normalization.
"""
class TestBackbone(unittest.TestCase):
    """
    
    """
    def setUp(self):
        """Set up the bacbkone model for testing."""
        self.backbone = Backbone()
        self.input_tensor = tf.random.normal((1, 512, 512, 3)) # Batch size of 1
    
    def test_call_wiht_valid_input(self):
        """Test the Backbone call method with valid input."""
        c3, c4, c5 = self.backbone.call(self.input_tensor)

        # Check the shapes of the outputs 
        self.assertEqual(c3.shape, (1, 128, 128, 256))
        self.assertEqual(c4.shape, (1, 64, 64, 512))
        self.assertEqual(c5.shape, (1, 32, 32, 512))

    
    def test_call_wiht_invalid_input_shape(self):
        """Test the Backbone call method with an invalid input shape."""
        invalid_input_tensor = tf.random.normal((1, 512, 512)) # Missing changel color
        with self.assertRaises(ValueError):
            self.backbone.call(invalid_input_tensor)
    

    def test_call_with_no_dropout(self):
        """Test the Backbone call method with dropout turned off."""
        c3, c4, c5 = self.backbone.call(self.input_tensor, dropout=False)

        # Check that the shapes are the same as with dropout
        self.assertEqual(c3.shape, (1, 128, 128, 256))
        self.assertEqual(c4.shape, (1, 64, 64, 512))
        self.assertEqual(c5.shape, (1, 32, 32, 512))

    def test_call_with_no_batch_norm(self):
        """Test the Backbone call method with batch normalization turned off."""
        c3, c4, c5 = self.backbone.call(self.input_tensor, batch_normalization=False)

        # Check that the shapes are the same as with batch normalization
        self.assertEqual(c3.shape, (1, 128, 128, 256))
        self.assertEqual(c4.shape, (1, 64, 64, 512))
        self.assertEqual(c5.shape, (1, 32, 32, 512))


if __name__ == '__main__':
    unittest.main()
