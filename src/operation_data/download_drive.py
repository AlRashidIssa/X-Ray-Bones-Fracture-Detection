import gdown
import sys
sys.path.append("/workspaces/DetectionXR/")
from pathlib import Path
from abc import ABC, abstractmethod

from src.utils.reg_log import log_inference, log_error
from src.utils.size import Size

class IDownload(ABC):
    """
    Interface for downloading files.
    """
    @abstractmethod
    def call(self, url: str, output_path: str, name_dataset: str = "XR-Bones") -> None:
        """
        Interface for downloading files.

        :param url: URL of the file to be downloaded.
        :param output_path: The destination where the file should be saved.
        :param name_dataset: Name for the downloaded file.
        """
        pass

class Download(IDownload):
    """
    Concreate implemention of IDownload interface to download files.
    """
    def call(self, url: str, output_path: Path, name_dataset: str = "XR-Bones") -> None:
        """
        Downloads a file from the given URL to the specified output path.

        :param url: URL of the file to be downloaded.
        :param output_path: The destination where the file should be saved.
        :param name_dataset: Name for the downloaded file.
        """
        try:
            # Convert ouput_path ot Path object if it's not already one
            output_path = Path(output_path)

            # Extract the file ID from the Google Drive URL
            file_id = url.split('/d/')[1].split('/')[0]
            download_url = f'https://drive.google.com/uc?id={file_id}'

            # Prepare the output file path
            file_path = output_path / f"{name_dataset}.zip" # type: ignore

            # Download the file using gdown
            log_inference(f"Starting download from {download_url}")
            gdown.download(download_url, str(file_path), quiet=False)

            # Check the size of the downloaded file
            file_size = Size().call(file_path)
            log_inference(f"Download completed. File size: {file_size}")

        except Exception as e:
            log_error(f"An error occurred during the download: {e}")
            raise