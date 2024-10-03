import sys
from abc import ABC
from typing import Tuple, Any, Type, Dict

import tensorflow as tf

append_path = "/workspaces/X-Ray-Bones-Fracture-Detection"
sys.path.append(append_path)
from src.utils.reg_log import log_error, log_inference


class IConfig(ABC):
    """
    Abstract base class for configuration. This provides the structure for all configuration classes.
    """

    # Dataset parameters
    url: str
    download: bool
    output_path: str
    dataset_name: str
    image_dir: str
    label_dir: str
    zip_dir: str
    extract_to: str
    batch_size: int
    image_size: Tuple[int, int]
    num_anchors: int
    num_classes: int

    # Deplyment_parameters
    pretrain_model_path: str

    # Model parameters
    AXL: str
    inputs_shape: Tuple[int, int, int]
    epochs: int
    train_data: Any
    valid_data: Any
    num_classes_model: int
    num_anchors_model: int
    optimization: Type
    parameters: Any
    model_name: str
    summary: bool
    train: bool


class Config(IConfig):
    """
    Project configuration class that loads parameters from a YAML file.
    """
    
    def __init__(self, config: dict) -> None:
        log_inference("Obtained Configuration.")
        """Set attributes based on the configuration dictionary."""
        try:
            # Dataset parameters
            self.url = config['dataset']['url']
            self.output_path = config['dataset']['output_path']
            self.name_dataset = config['dataset']['name_dataset']
            self.val_image_dir = config['dataset']['val_image_dir']
            self.val_label_dir = config['dataset']['val_label_dir']
            self.image_dir = config['dataset']['image_dir']
            self.label_dir = config['dataset']['image_dir']
            self.image_dir = config['dataset']['image_dir']
            self.label_dir = config['dataset']['label_dir']
            self.zip_dir = config['dataset']['zip_dir']
            self.extract_to = config['dataset']['extract_to']
            self.batch_size = config['dataset']['batch_size']
            self.image_size = tuple(config['dataset']['image_size'])
            self.num_anchors = config['dataset']['num_anchors']
            self.num_classes = config['dataset']['num_classes']

            # Model parameters
            self.AXL = config['model']['AXL']
            self.inputs_shape = tuple(config['model']['inputs_shape'])
            self.epochs = config['model']['epochs']
            self.batch_size = config['model']['batch_size']  # Redundant with dataset?
            self.num_classes_model = config['model']['num_classes']
            self.num_anchors_model = config['model']['num_anchors']
            self.parameters = config['model']['parameters']
            self.model_name = config['model']['model_name']
            self.summary = config['model']['summary']
            self.train = config['model']['train']

            self.pretrain_model_path = config['deplyment']['pretrain_model_path']
        except KeyError as e:
            log_error(f"Missing configuration key: {e}")
            raise

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns the configuration as a dictionary with separate entries for model and dataset.
        
        Returns:
            dict: A dictionary containing the model and dataset configurations.
        """
        dataset_config = {
            "url": self.url,
            "output_path": self.output_path,
            "name_dataset": self.name_dataset,
            "val_image_dir": self.val_image_dir,
            "val_label_dir": self.val_label_dir,
            "image_dir": self.image_dir,
            "label_dir": self.label_dir,
            "zip_dir": self.zip_dir,
            "extract_to": self.extract_to,
            "batch_size": self.batch_size,
            "image_size": self.image_size,
            "num_anchors": self.num_anchors,
            "num_classes": self.num_classes
        }

        model_config = {
            "AXL": self.AXL,
            "batch_size": self.batch_size,
            "inputs_shape": self.inputs_shape,
            "epochs": self.epochs,
            "num_classes": self.num_classes_model,
            "num_anchors": self.num_anchors_model,
            "parameters": self.parameters,
            "model_name": self.model_name,
            "summary": self.summary,
            "train": self.train
        }

        return {
            "model": model_config,
            "dataset": dataset_config
        }

    def __repr__(self):
        return f"<ProjectConfig {self.to_dict()}>"
