import logging
import os

def setup_logging(log_path: str, logger_name: str, logging_level: int):
    """
    Sets up logging for a specific logger and log file.

    Args:
        log_path (str): Path to the log file.
        logger_name (str): Unique name for the logger.
        logging_level (int): The logging level (e.g., logging.DEBUG, logging.INFO).
    """
    # Ensure the log directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Create a logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging_level)  # Set the desired logging level

    # Check if handlers already exist to avoid duplication
    if not logger.hasHandlers():
        # Create file handler for logger
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

        # Optionally, add console output
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)

    return logger

# Log file paths for different components
api_log_path = "/workspaces/X-Ray-Bones-Fracture-Detection/logs/api_logs/api.log"
error_log_path = "/workspaces/X-Ray-Bones-Fracture-Detection/logs/error_logs/error.log"
inference_log_path = "/workspaces/X-Ray-Bones-Fracture-Detection/logs/inference_logs/inference.log"
train_log_path = "/workspaces/X-Ray-Bones-Fracture-Detection/logs/train_logs/train.log"

log_paths = [api_log_path, error_log_path, inference_log_path, train_log_path]
def ensure_log_exists(log_path):
    """Ensure that log files exists. If the file or its directory does not exists exists use os"""
    # Extract directory from the log file path
    log_dir = os.path.dirname(log_path)

    # Check if the directory exists, if not, create it
    if not os.path.exists(log_dir):
        print(f"Creating directory: {log_dir}")
        os.makedirs(log_dir)
    
    # Check if the log file exists, if not, create an empty file
    if not os.path.exists(log_path):
        print(f"Creatting log file: {log_path}")
        with open(log_path, 'w') as f:
            f.write("") # Create an empty log file

# Ensure all lgo files exist
for log_path in log_paths:
    ensure_log_exists(log_path=log_path)

# Setup separate loggers for each component with appropriate logging levels
api_logger = setup_logging(api_log_path, "API", logging.INFO)
train_logger = setup_logging(train_log_path, "Training", logging.WARNING)
error_logger = setup_logging(error_log_path, "Error", logging.ERROR)
inference_logger = setup_logging(inference_log_path, "Inference", logging.DEBUG)

# Logging functions for each component
def log_api(message: str):
    api_logger.info(message)

def log_train(message: str):
    train_logger.warning(message)

def log_error(message: str):
    error_logger.error(message)

def log_inference(message: str):
    inference_logger.debug(message)

# # Example usage
# log_api("API request received.")
# log_train("Training started.")
# log_error("An error occurred during processing.")
# log_inference("Inference completed successfully.")
