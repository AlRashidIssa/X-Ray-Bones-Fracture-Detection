from abc import ABC, abstractmethod
from pathlib import Path

import os
import numpy as np
import sys
append_path = "/workspaces/X-Ray-Bones-Fracture-Detection"
sys.path.append(append_path)

from src.utils.reg_log import log_inference, log_error

class ISize(ABC):
    """
    Abstract interface for getting the size of a file.
    """
    @abstractmethod
    def call(self, path: str) -> str:
        """
        Abstract method to get the file size.

        :param path: Path to the file.
        :return: File size in GB as a string.
        """
        pass

class Size(ISize):
    def call(self, path: str) -> str:
        """
        Get the size of the file at the given path.

        :param path: Path to the file.
        :return: File size in MB as a string.
        """
        try:
            # Check if the path exists and is a file.
            if not os.path.exists(path):
                log_error(f"The path does not exist or is not a file: {path}")
                raise FileNotFoundError(f"Path does not exist or is not a file: {path}")
            
            # Get file size in GB
            file_size_gb = os.path.getsize(path) / (1024 ** 3)
            log_inference(f"File size: {file_size_gb:.2f} GB")
            return f"File size: {file_size_gb:.2f} GB"

        except FileNotFoundError as fnf_error:
            log_error(f"FileNotFoundError: {fnf_error}")
            raise
        except PermissionError as p_error:
            log_error(f"PermissionError: {p_error}")
            raise
        except OSError as os_error:
            log_error(f"OSError: {os_error}")
            raise
        except Exception as e:
            log_error(f"An unexpected error occurred: {e}")
            raise

# # Example usage:
# size_instance = Size()
# result = size_instance.call(Path('/workspaces/DetectionXR/requirements.txt'))
# print(result)