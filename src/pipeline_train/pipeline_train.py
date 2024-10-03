import sys
import tensorflow as tf
from abc import ABC, abstractmethod

append_path = "/workspaces/X-Ray-Bones-Fracture-Detection"
sys.path.append(append_path)

# Importing custom modules
from src.pipeline_model.pipeline_tensors import ModelPipeline
from src.pipeline_model.Axcelaerators import Axcelaerators
from src.utils.reg_log import log_error, log_train

class IPipelineTrain(ABC):
    """
    Interface for the Pipeline model training.

    This interface defines the contract for setting up the model architecture, 
    selecting the accelerator, compiling the model, and managing the training process.

    The implementation must include proper logging (`log_train`) and error handling.
    """
    
    @abstractmethod
    def call(self, 
             AXL: str = "GPU", 
             inputs_shape: tuple = (512, 512, 3), 
             epochs: int = 30, 
             batch_size: int = 256, 
             train_data=None, 
             valid_data=None,
             train_dataset=None,
             valid_dataset=None,
             num_classes: int = 80,
             num_anchors: int = 9,
             parameters: dict = None,
             model_name: str = "trained_model",
             summary: bool = True,
             train: bool = True) -> tf.keras.callbacks.History:
        """
        Abstract method to run the model training pipeline.

        Args:
            AXL (str): The accelerator type ('GPU', 'TPU', or 'CPU').
            inputs_shape (tuple): Input shape for the model.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            train_data: Training data generator.
            valid_data: Validation data generator.
            train_dataset: The training dataset (for step calculations).
            valid_dataset: The validation dataset (for step calculations).
            num_classes (int): Number of classes.
            num_anchors (int): Number of anchors.
            parameters (dict): Parameters for the optimizer.
            model_name (str): Name to save the model with.
            summary (bool): If True, print the model summary.
            train (bool): If True, train the model.

        Returns:
            tf.keras.callbacks.History: The history of the training process.
        """
        pass

class PipelineTrain(IPipelineTrain):
    """
    Concrete class implementing the pipeline for model training.

    Includes logging using `log_train` and error handling with `log_error`.
    The model is built, compiled, and trained based on the selected accelerator (GPU/TPU/CPU).
    """
    
    def call(self, 
             batch_size: int,
             AXL: str = "GPU", 
             inputs_shape: tuple = (512, 512, 3), 
             epochs: int = 30, 
             train_data= None, 
             valid_data= None,
             train_dataset: list = None,
             valid_dataset: list = None,
             num_classes: int = 80,
             num_anchors: int = 9,
             parameters: dict = None,
             model_name: str = "trained_model",
             summary: bool = True,
             train: bool = True) -> tf.keras.callbacks.History:
        """
        Implements the model training pipeline by building the model architecture, 
        selecting the hardware accelerator, compiling the model, and running the training process.
        
        Logs the process and handles errors effectively.

        Args:
            AXL (str): The accelerator type ('GPU', 'TPU', or 'CPU').
            inputs_shape (tuple): Input shape for the model.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            train_data: Training data generator.
            valid_data: Validation data generator.
            train_dataset: The training dataset (for step calculations).
            valid_dataset: The validation dataset (for step calculations).
            num_classes (int): Number of classes.
            num_anchors (int): Number of anchors.
            parameters (dict): Parameters for the optimizer.
            model_name (str): Name to save the model with.
            summary (bool): If True, print the model summary.
            train (bool): If True, train the model.

        Returns:
            tf.keras.callbacks.History: Training history if the model is trained.
        """
        """
        Implements the model training pipeline by building the model architecture, 
        selecting the hardware accelerator, compiling the model, and running the training process.
        """
        try:
            log_train(f"Starting the pipeline with accelerator: {AXL}, input shape: {inputs_shape}, epochs: {epochs}, batch size: {batch_size}")

            # Select the hardware accelerator
            strategy = Axcelaerators().call(AXL=AXL)

            with strategy.scope():
                log_train(f"Building the model with {num_classes} classes and {num_anchors} anchors.")
                
                # Build the model using ModelPipeline
                model = ModelPipeline().call(inputs_shape=inputs_shape,
                                             num_anchors=num_anchors,
                                             num_classes=num_classes)

                # Compile the model with the optimizer and specified parameters
                log_train(f"Compiling the model using Adam.")
                # Instantiate the optimizer using the provided optimization class and parameters
                optimizer = tf.keras.optimizers.Adam(**parameters)

                # Compile the model with the optimizer, losses, and metrics
                model.compile(
                    optimizer=optimizer,
                    loss={
                        'class_output': 'categorical_crossentropy',
                        'reshape_bboxes': 'mean_squared_error',
                        'objectness_output': 'binary_crossentropy'
                    },
                    metrics={
                        'class_output': 'accuracy', 
                        'objectness_output': 'accuracy', 
                        'reshape_bboxes': 'mse'
                    }
                )
            
                # Ensure the model is compiled before training
                if not model:
                    raise ValueError("The model is not compiled. Please ensure the compile step is correctly implemented.")

                # Print model summary if specified
                if summary:
                    log_train("Printing model summary.")
                    model.summary()

                # Train the model if specified
                if train:
                    log_train("Training the model.")
                    history = model.fit(
                        train_data,
                        validation_data=valid_data,
                        epochs=epochs,
                        batch_size=batch_size,
                        steps_per_epoch=len(train_dataset) // batch_size,
                        validation_steps=len(valid_dataset) // batch_size
                    )

                    # Log training info
                    log_train(f"Model training completed successfully for {epochs} epochs.")
                    
                    # Save the trained model
                    model.save(f'/{model_name}.h5')
                    log_train(f"Model saved as '{model_name}.h5'.")

                    return history

        except Exception as e:
            # Log and raise any errors encountered during the process
            log_error(f"Error during the training pipeline: {e}")
            raise RuntimeError(f"Training pipeline failed: {e}")

        return None

# Example usage (commented out):
# pipeline = PipelineTrain()
# history = pipeline.call(
#     AXL='GPU', 
#     inputs_shape=(512, 512, 3), 
#     epochs=50, 
#     batch_size=128, 
#     train_data=train_data_generator, 
#     valid_data=val_data_generator, 
#     train_dataset=train_dataset, 
#     valid_dataset=val_dataset, 
#     num_classes=80, 
#     num_anchors=9, 
#     optimization=tf.keras.optimizers.Adam, 
#     parameters={'learning_rate': 0.001}, 
#     model_name='my_model', 
#     summary=True, 
#     train=True
# )
