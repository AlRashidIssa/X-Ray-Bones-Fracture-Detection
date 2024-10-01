import sys
import os
import yaml
from typing import Union, Dict, Any


sys.path.append("/workspaces/DetectionXR")
from src.utils.reg_log import log_error, log_inference


def load_config(file_path: Union[str, None]) -> Dict[str, Any]:
    """
    Loads configuration from a YAML file and populates the parameters.

    Args:
        file_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration parameters loaded from the YAML file.    
    """
    if not file_path:
        # Look for a YAML file in the specified directory if no file_path is provided
        config_dir = "/workspaces/DetectionXR/config"
        files = os.listdir(config_dir)
        yaml_files = [file for file in files if file.endswith('.yml') or file.endswith('.yaml')]
        if not yaml_files:
            log_error("No YAML configuration files found in the specified directory.")
            raise FileNotFoundError("No YAML configuration files found.")
    
    # Select the first YAML file found 
    file_path = os.path.join(config_dir, yaml_files[0])

    try:
        with open(file_path, 'r') as file:
            log_inference(f"Reading YAML file from {file_path}")
            config = yaml.safe_load(file)

            if config is None:
                log_error("Loadded configuration is empty.")
                raise ValueError("Configuration is empty.")
            
            return config
    except Exception as e:
        log_error(f"Filed to load configuration file: {e}")
        raise

config = load_config(None)
print(config)