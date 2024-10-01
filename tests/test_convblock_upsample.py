"""
Explanation of the Test Cases:
    1.Setup:The `setUp` Funcation creates redusable input tensors (`self.input_tensor_conv` and 
    self.input_tensor_upsample) for the tests, which reduces code duplication.
    2.`test_conv_block_output_shape:Test the output shape of the `ConvBlock` to ensure it produces
    the correct shape based one the input tensor and filters.
    3.`test_upsample_block_output_shape`:Test the output shape of the `UpSampleBlock` after the upsampling operation. The
    upsample block should double the spatial dimensions (height and width).
    4.`test_conv_block_with_invalid_input`:Verifies that the `ConvBlock` raises a `ValuError` when provided with an invalid
    input tensor (less tahn 3D). This ensures proper error handling.
    5.`test_upsample_block_with_invalid_input`:Similarly, it ensures that the `UpSampleblock` raies a `ValueError` when the input tensor 
    is invalid.
    6.`test_dropout_behavior_conv_block:Test the `UpSampleBlock` with droput=False` to ensure that disabling dropout
    doesn't impact the output shape after upsampling.
    7.`test_dropout_behavior_upsample_block`:Test the `UpSampleBlock` with dropout=False` to ensure that disabling
    dropout doesn't impact the output shape after upsampling.
"""
import sys
sys.path.append("/workspaces/DetectionXR/")
from src.model.convbloc_upsampleblock import ConvBlock, UpSampleBlock

import unittest
import tensorflow as tf
import numpy as np

class TestBlock(unittest.TestCase):
    
    def setUp(self):
        """
        Set up function to create reusable input tensors for testing
        """
        self.input_tensor_conv = tf.random.normal((1, 128, 128, 64))
        self.input_tensor_upsample = tf.random.normal((1, 64, 64, 128))
    
    def test_conv_block_output_shape(self):
        """
        Test the output shape of the ConvBlock to ensure correctness.
        """
        conv_block = ConvBlock()
        output_tensor = conv_block.call(self.input_tensor_conv, dim=64, 
                                        dropout=True, dropout_val=0.5)
        
        # Check the output shape: should match the dimensions based on Conv2D
        self.assertEqual(output_tensor.shape, (1, 128, 128, 64))
    
    def test_upsample_block_output_shape(self):
        """
        Test the output shape of the UpSampleBlock to ensure correctness after upsampling.
        """
        upsample_block = UpSampleBlock()
        output_tensor = upsample_block.call(self.input_tensor_upsample, dim=128, dropout=False)

        # Check the output shape: UpSampling2D should double spatial dimensions
        self.assertEqual(output_tensor.shape, (1, 128, 128, 64))
    def test_conv_block_with_invalid_input(self):
        """
        Test ConvBlock with invalid input dimensions and ensure it raises the correct error.
        """
        conv_block = ConvBlock()
        
        # Input tensor with invalid dimensions (less than 3D for Conv2D)
        invalid_input_tensor = tf.random.normal((1, 128))  # Invalid input
        
        with self.assertRaises(ValueError) as context:
            conv_block.call(invalid_input_tensor, dim=64)
        
        self.assertIn("Error in ConvBlock", str(context.exception))
    
    def test_upsample_block_with_invalid_input(self):
        """
        Test UpSampleBlock with invalid input dimensions and ensure it raises the correct error.
        """
        upsample_block = UpSampleBlock()
        
        # Input tensor with invalid dimensions (less than 3D for Conv2D)
        invalid_input_tensor = tf.random.normal((1, 128))  # Invalid input
        
        with self.assertRaises(ValueError) as context:
            upsample_block.call(invalid_input_tensor, dim=128)
        
        self.assertIn("Error in UpSampleBlock", str(context.exception))
    
    def test_dropout_behavior_conv_block(self):
        """
        Test ConvBlock behavior when dropout is disabled.
        """
        conv_block = ConvBlock()
        output_tensor = conv_block.call(self.input_tensor_conv, dim=64, dropout=False)
        
        # Verify that the shape is consistent when dropout is disabled
        self.assertEqual(output_tensor.shape, (1, 128, 128, 64))

    def test_dropout_behavior_upsample_block(self):
        """
        Test UpSampleBlock behavior when dropout is disabled.
        """
        upsample_block = UpSampleBlock()
        output_tensor = upsample_block.call(self.input_tensor_upsample, dim=128, dropout=False)
        
        # Verify that the shape is consistent when dropout is disabled
        self.assertEqual(output_tensor.shape, (1, 128, 128, 64))

if __name__ == '__main__':
    unittest.main()
