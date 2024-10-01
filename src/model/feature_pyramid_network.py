import sys
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Concatenate, MaxPooling2D  # type: ignore
from typing import Tuple
from abc import ABC, abstractmethod

sys.path.append("/workspaces/DetectionXR/")
from src.model.convbloc_upsampleblock import ConvBlock, UpSampleBlock
from src.utils.reg_log import log_error


class IFeaturePyramidNetwork(ABC):
    @abstractmethod
    def call(self, input_tensor: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Processes the input tensors through the feature pyramid network.
        
        Args:
            input_tensor: A tuple containing three tensors (c3, c4, c5).
        
        Returns:
            A tensor representing the output of the feature pyramid network.
        """
        pass


class FeaturePyramidNetwork(IFeaturePyramidNetwork):
    def call(self,
             input_tensor: Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
             dropout: bool = True,
             dropout_val: float = 0.5,
             batchnormalization: bool = True,
             activation: str = 'relu',
             padding: str = 'same') -> tf.Tensor:
        """Constructs the feature pyramid network from the given input tensors.
        
        Args:
            input_tensor: A tuple of tensors (c3, c4, c5).
            dropout: Flag to use dropout in the convolutional blocks.
            dropout_val: Dropout rate.
            batchnormalization: Flag to use batch normalization.
            activation: Activation function to use.
            padding: Padding method to use in convolutional layers.
        
        Returns:
            A tensor representing the final output of the network.
        """
        
        # Validate input tensor
        if not isinstance(input_tensor, tuple):
            log_error("Input must be a tuple.")
            raise TypeError("Expected input_tensor to be a tuple of tensors (c3, c4, c5).")

        if len(input_tensor) != 3:
            log_error(f"Expected input_tensor to contain exactly three tensors, but got {len(input_tensor)}.")
            raise ValueError("Expected input_tensor to contain three tensors (c3, c4, c5).")

        for i, tensor in enumerate(input_tensor):
            if tensor.ndim != 4:
                log_error(f"Tensor {i} in input_tensor does not have 4 dimensions: {tensor.shape}.")
                raise ValueError(f"Expected input_tensor[{i}] to have 4 dimensions (batch, height, width, channels).")

        c3, c4, c5 = input_tensor
        """
        c3=[1, 128, 128,128]  # batch, W, G, Chanle colore
        c4=[1, 64, 64,  256]
        c5=[1, 32, 32,  512]
        """
        try:
            # Process P5
            P5 = Conv2D(512, (1, 1), activation=activation, padding=padding)(c5) # C5=IN->[32, 32, 512], P5=OUT->[32, 32, 512]
            P5 = ConvBlock().call(input_tensor=P5, dim=512, dropout=dropout,dropout_val=dropout_val, 
                                  batchnormalization=batchnormalization,activation=activation, padding=padding) # P5=IN->[32, 32, 512], P5=OUT->[32, 32, 512]
            P5 = ConvBlock().call(input_tensor=P5, dim=512, dropout=dropout,dropout_val=dropout_val, 
                                  batchnormalization=batchnormalization,activation=activation, padding=padding) # P5=IN->[32, 32, 512], P5=OUT->[32, 32, 512]

            # Upsample P5 to match c4's dimensions
            U1 = UpSampleBlock().call(input_tensor=P5, dim=P5.shape[2], dropout=dropout, dropout_val=dropout_val, #P5=IN->[32, 32, 512], P5=OUT->[64, 64, 256]
                                      batchnormalization=batchnormalization, activation=activation, padding=padding)
            C1 = Concatenate()([c4, U1]) # C1=OUT->[64,64,512]

            # Process P4
            P4 = Conv2D(256, (1, 1), activation=activation, padding=padding)(C1) # C1=IN->[64,64,512] P4=OUT->.[64, 64, 256]
            P4 = ConvBlock().call(input_tensor=P4, dim=256, dropout=dropout,dropout_val=dropout_val, #P4=IN->[64, 64, 256] P4=OUT->.[64, 64, 256]
                                  batchnormalization=batchnormalization,activation=activation, padding=padding)
            P4 = ConvBlock().call(input_tensor=P4, dim=256, dropout=dropout,dropout_val=dropout_val,
                                  batchnormalization=batchnormalization, activation=activation, padding=padding) # Like

            # Upsample P4 to match c3's dimensions
            U2 = UpSampleBlock().call(input_tensor=P4, dim=P4.shape[2], dropout=dropout, dropout_val=dropout_val, #P4=IN->[64, 64, 256] P4=OUT->.[128, 128, 128]
                                      batchnormalization=batchnormalization, activation=activation, padding=padding)
            C2 = Concatenate()([c3, U2]) # C2=OUT->[128,128,256]


            # Process P3
            P3 = Conv2D(128, (1, 1), activation=activation, padding=padding)(C2) # C2=IN->[128,128,256] P3=OUT->.[128, 128, 128]
            P3 = ConvBlock().call(input_tensor=P3, dim=128, dropout=dropout, # P3=IN->[128,128,128] P3=OUT->.[128, 128, 128]
                                  dropout_val=dropout_val, batchnormalization=batchnormalization,
                                  activation=activation, padding=padding)
            P3 = ConvBlock().call(input_tensor=P3, dim=128, dropout=dropout,# P3=IN->[128,128,128] P3=OUT->.[128, 128, 128]
                                  dropout_val=dropout_val, batchnormalization=batchnormalization,
                                  activation=activation, padding=padding)
            
            FH = UpSampleBlock().call(input_tensor=P4, dim=P4.shape[2], dropout=dropout, dropout_val=dropout_val, #P4=IN->[64, 64, 256] =OUT->.[128, 128, 128]
                                      batchnormalization=batchnormalization, activation=activation, padding=padding)
            FHH = UpSampleBlock().call(input_tensor=P5, dim=P4.shape[2], dropout=dropout, dropout_val=dropout_val, #P5=IN->[32, 32, 512] =OUT->.[64, 64, 256]
                                      batchnormalization=batchnormalization, activation=activation, padding=padding)
            FF = UpSampleBlock().call(input_tensor=FHH, dim=P4.shape[2], dropout=dropout, dropout_val=dropout_val, #P5=IN->[64, 64, 256] =OUT->.[128, 128, 128]
                                      batchnormalization=batchnormalization, activation=activation, padding=padding)
            
            # Concatenate final outputs
            FLC = Concatenate()([P3, FH, FF])  # Concatenate layers properly
            FLCCONV = Conv2D(128, (1, 1))(FLC)  # Convolution after concatenation

            return FLCCONV

        except Exception as e:
            log_error(f"An error occurred while processing the feature pyramid network: {str(e)}")
            raise