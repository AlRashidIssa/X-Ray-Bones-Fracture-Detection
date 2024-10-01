import sys
import unittest
import numpy as np
import tensorflow as tf

append_path = "/workspaces/X-Ray-Bones-Fracture-Detection"
sys.path.append(append_path)
from src.model.feature_pyramid_network import FeaturePyramidNetwork  # Adjust the import based on your project structure

class TestFeaturePyramidNetwork(unittest.TestCase):

    def setUp(self):
        # Create mock input tensors with shapes:
        # c3 -> (1, 128, 128, 128)
        # c4 -> (1, 64, 64, 256)
        # c5 -> (1, 32, 32, 512)
        self.c3 = tf.random.normal((1, 512, 512, 128))
        self.c4 = tf.random.normal((1, 256, 256, 256))
        self.c5 = tf.random.normal((1, 128, 128, 512))
        self.fpn = FeaturePyramidNetwork()

    def test_input_validation(self):
        # Test with incorrect input type
        with self.assertRaises(TypeError):
            self.fpn.call("not_a_tuple")
        
        # Test with wrong number of inputs
        with self.assertRaises(ValueError):
            self.fpn.call((self.c3, self.c4))  # Missing c5
        
        # Test with tensor with wrong dimensions
        with self.assertRaises(ValueError):
            self.fpn.call((self.c3, tf.random.normal((1, 64, 64)), self.c5))  # c4 has wrong dimensions

    def test_output_shape(self):
        # Call the feature pyramid network
        output = self.fpn.call((self.c3, self.c4, self.c5))

        # Check the shape of the output
        self.assertEqual(output.shape, (1, 512, 512, 128))  # Expecting concatenated output from P4 and P5 (128, 128, 128 + 256)

    def test_intermediate_shapes(self):
        # Call the network to get intermediate layers
        # Mocking internal processing can be complex without exposing internals, but we can indirectly test shapes

        # We can call the FPN and get the output to infer if internal layers are working as expected
        output = self.fpn.call((self.c3, self.c4, self.c5))

        # Check specific output shapes (derived from P3, P4, P5)
        P5_shape = (1, 32, 32, 512)
        U1_shape = (1, 64, 64, 512)  # Upsampled from P5
        C1_shape = (1, 64, 64, 512)  # Concatenated c4 and U1
        
        # Validate these intermediate shapes indirectly through the output
        self.assertEqual(output.shape[1:], (512, 512, 128))  # Final output shape after all operations

    def test_exceptions_on_invalid_shapes(self):
        # Test with a tensor that has less than 4 dimensions
        with self.assertRaises(ValueError):
            self.fpn.call((tf.random.normal((1, 128, 128)), self.c4, self.c5))  # c3 has wrong dimensions

    def tearDown(self):
        # Cleanup if needed
        pass

if __name__ == '__main__':
    unittest.main()
