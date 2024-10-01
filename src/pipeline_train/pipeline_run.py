import numpy as np
import sys
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import tensorflow as tf

sys.path.append("/workspaces/DetectionXR/")

from src.operation_data.generate_by_batch import DataGenerator
from src.utils.read_yaml import load_config
from config.config import Config
from src.pipeline_train.pipeline_train import PipelineTrain
from src.operation_data.dataInspector import CreateDataset
from src.operation_data.download_drive import Download
from src.utils.unzip import Unzip
from src.utils.reg_log import log_error, log_inference


class IRunPipeline(ABC):
    """
    Interface for Run pipelines handling image and label data processing, and Train model.
    """

    @abstractmethod
    def call(self,
             url: str,
             download: bool,
             output_path: str,
             name_dataset: str,
             zip_dir: str,
             extract_to: str,
             model_config: dict,
             dataset_config: dict) -> Optional[tf.keras.callbacks.History]:
        """
        Abstract method to process image and label data and train the model.
        """
        pass


class RunPipeline(IRunPipeline):
    """
    Run pipeline implementation for downloading, unzipping, and generating data batches, and training the model.
    """

    def call(self,
             url: str,
             download: bool,
             output_path: str,
             name_dataset: str,
             zip_dir: str,
             extract_to: str,
             model_config: dict,
             dataset_config: dict) -> Optional[tf.keras.callbacks.History]:
        """
        Orchestrates the downloading, unzipping, and data generation process and trains the model.
        """
        try:
            # Download and unzip the dataset if necessary
            if download:
                self._download_dataset(url, output_path, name_dataset)
                self._unzip_dataset(zip_dir, extract_to, name_dataset)

            # Proceed to training the model
            history = self.run_train(model_config, dataset_config)

            return history

        except Exception as e:
            log_error(f"Error in processing pipeline: {e}")
            return None  # Return None if an error occurs

    def _download_dataset(self, url: str, output_path: str, name_dataset: str) -> None:
        """
        Downloads the dataset from the given URL.
        """
        Download().call(url=url, output_path=output_path, name_dataset=name_dataset)
        log_inference(f"Downloaded dataset '{name_dataset}' from {url}.")

    def _unzip_dataset(self, zip_dir: str, extract_to: str, name_dataset: str) -> None:
        """
        Unzips the dataset to the specified extraction directory.
        """
        Unzip().call(direcotrys_zip_files=zip_dir, extract_to=extract_to, name_dataset=name_dataset)
        log_inference(f"Extracted dataset '{name_dataset}' to {extract_to}.")

    def run_train(self, model_config: dict, dataset_config: dict) -> tf.keras.callbacks.History:
        """
        Initializes the data generator for the images and labels and trains the model.
        """
        # Extract directories and settings from dataset_config
        image_dir = dataset_config.get('image_dir')
        label_dir = dataset_config.get('label_dir')
        val_image_dir = dataset_config.get('val_image_dir')
        val_label_dir = dataset_config.get('val_label_dir')
        batch_size = dataset_config.get('batch_size')
        image_size = dataset_config.get('image_size')
        num_anchors = dataset_config.get('num_anchors')
        num_classes = dataset_config.get('num_classes')

        # Initialize data generators for training and validation
        train_data_gen = DataGenerator().call(images_dir=image_dir,
                                              labels_dir=label_dir,
                                              batch_size=batch_size,
                                              image_size=image_size,
                                              num_anchors=num_anchors,
                                              num_class=num_classes)

        val_data_gen = DataGenerator().call(images_dir=val_image_dir,
                                            labels_dir=val_label_dir,
                                            batch_size=batch_size,
                                            image_size=image_size,
                                            num_anchors=num_anchors,
                                            num_class=num_classes)

        # Create the training and validation datasets
        train_dataset = CreateDataset().call(image_dir=image_dir,
                                             label_dir=label_dir,
                                             split_name="train")

        val_dataset = CreateDataset().call(image_dir=val_image_dir,
                                           label_dir=val_label_dir,
                                           split_name="val")

        # Run the model training using PipelineTrain
        history = PipelineTrain().call(**model_config,  # Pass model config directly
                                       train_data=train_data_gen,
                                       valid_data=val_data_gen,
                                       train_dataset=train_dataset,
                                       valid_dataset=val_dataset)

        # Optionally, check the history of the training process
        print(history.history)
        return history


# Load configuration from YAML file
config_data = load_config(None)  # Load the config data
config = Config(config_data)  # Initialize the config object with loaded config
configurations = config.to_dict()  # Get model and dataset configurations
model_config = configurations['model']  # Extract model configuration
dataset_config = configurations['dataset']  # Extract dataset configuration

# Extract dataset-specific parameters from dataset_config
url = dataset_config.get('url')
output_path = dataset_config.get('output_path')
name_dataset = dataset_config.get('name_dataset')
extract_to = dataset_config.get('extract_to')
zip_dir = dataset_config.get('zip_dir')

# Instantiate and call the pipeline
history = RunPipeline().call(url=url,
                             download=True,  # Set to True to trigger download and unzip
                             output_path=output_path,
                             name_dataset=name_dataset,
                             zip_dir=zip_dir,
                             extract_to=extract_to,
                             model_config=model_config,
                             dataset_config=dataset_config)
