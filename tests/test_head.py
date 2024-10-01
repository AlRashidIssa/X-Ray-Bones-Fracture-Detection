import sys
import unittest
import tensorflow as tf

sys.path.append("/workspaces/DetectionXR/")
from src.model.head import Head  # Adjust the import based on your project structure

class TestHead(unittest.TestCase):
    
    def setUp(self):
        """Set up a Head instance and a sample input tensor."""
        self.head = Head()
        # Create a sample input tensor with shape (batch_size, height, width, channels)
        self.input_tensor = tf.random.normal((2, 128, 128, 256))  # Example input shape

    def test_output_shapes(self):
        """Test that the output shapes are as expected."""
        num_classes = 10
        num_anchors = 3
        class_output, bbox_output, objectness_output = self.head.call(self.input_tensor, num_classes, num_anchors)
        
        # Check shapes
        self.assertEqual(class_output.shape, (2, num_classes), "Class output shape mismatch.")
        self.assertEqual(bbox_output.shape, (2, num_classes, 4), "Bounding box output shape mismatch.")
        self.assertEqual(objectness_output.shape, (2, num_anchors), "Objectness output shape mismatch.")

    def test_invalid_input_tensor(self):
        """Test that an error is raised for invalid input tensor."""
        invalid_tensor = tf.random.normal((2, 128, 128))  # Invalid shape (not 4D)
        with self.assertRaises(ValueError):
            self.head.call(invalid_tensor)

    def test_invalid_num_classes(self):
        """Test that an error is raised for invalid num_classes."""
        with self.assertRaises(TypeError):
            self.head.call(self.input_tensor, num_classes='invalid')  # Non-integer value

    def test_invalid_num_anchors(self):
        """Test that an error is raised for invalid num_anchors."""
        with self.assertRaises(TypeError):
            self.head.call(self.input_tensor, num_anchors='invalid')  # Non-integer value

    def test_exception_handling(self):
        """Test that any unexpected errors are logged and raised."""
        with self.assertRaises(Exception):
            self.head.call(None)  # Pass None to trigger exception

if __name__ == '__main__':
    unittest.main()
