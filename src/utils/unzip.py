from abc import ABC, abstractmethod
from pathlib import Path
import sys
import zipfile
import os

sys.path.append("/workspaces/DetectionXR/")
from src.utils.reg_log import log_inference, log_error

class IUnzip(ABC):
    """
    Interface for unzipping files.
    """
    @abstractmethod
    def call(self, direcotrys_zip_files: str, extract_to: str) -> None:
        """
        Abstract method to unzip a file.

        :patam direcotrys_zip_files: Path to the ZIP file to be extracted.
        :param extract_to: Directory where the contents should be extracted.
        """
        pass


class Unzip(IUnzip):
    """
    Concrete implementation of IUnzip interface to unzip files.
    """
    def call(self, direcotrys_zip_files: str, extract_to: str, name_dataset: str = "xr_bones") -> None:
        """
        Unzips a file from the given ZIP path to the specified extraction directory.

        :param direcotrys_zip_files: Path to the ZIP file to be extracted.
        :param extract_to: Directory where the contents should be extracted.
        """
        try:
            # Ensure the ZIP file exists
            if not os.path.exists(direcotrys_zip_files):
                log_error(f"The ZIP file does not exist or is not a file: {direcotrys_zip_files}")
                raise FileNotFoundError(f"The ZIP file does not exist or is not a file: {direcotrys_zip_files}")
            
            # Ensure the ouput directory exists
            if not os.path.exists(extract_to):
                log_error(f"Extraction directory does not exist. Creating: {extract_to}")
                raise FileNotFoundError(f"Extraction directoy does not exist., Creating: {extract_to}")
            
            dataset_zip = f"{direcotrys_zip_files}/{name_dataset}.zip"
            # Extract the ZIP file
            log_inference(f"Stargin exraction of {dataset_zip} to {extract_to}")
            with zipfile.ZipFile(dataset_zip, 'r') as  zip_ref:
                zip_ref.extractall(extract_to)
            log_inference(f"Extraction completed successfully to {extract_to}")

        except zipfile.BadZipFile as bad_zip_error:
            log_error(f"BadZipFile error( {bad_zip_error} ): The file is not a ZIP file or it is corrupted: {direcotrys_zip_files}")
            raise

        except FileNotFoundError as fnf_error:
            log_error(f"FileNotFoundError: {fnf_error}")
            raise

        except Exception as e:
            log_error(f"An unexpected error occurred during extraction: {e}")
            raise