# Standard library imports
import os
import glob
import sys

# Third-party library imports
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import cv2
from PIL import Image
import matplotlib.patches as patches
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Tuple

# Local imports
sys.path.append("/workspaces/DetectionXR/")
from src.utils.reg_log import log_inference, log_error


class ICreateDataset(ABC):
    """
    Interface for creating a dataset from images and their corresponding labels.
    """
    
    @abstractmethod
    def call(self, image_dir: str, label_dir: str) -> Union[List[Dict[str, Union[str, List[str]]]], pd.DataFrame]:
        """
        Create a dataset by loading images and their corresponding labels from the specified directories.
        
        Args:
            image_dir (str): Path to the directory containing image files.
            label_dir (str): Path to the directory containing label files.

        Returns:
            Tuple: 
                - List[Dict[str, Union[str, List[str]]]]: A list of dictionaries containing image paths,
                  label paths, and their associated labels.
                - pd.DataFrame: A DataFrame representation of the same dataset.
        """
        pass

class CreateDataset(ICreateDataset):
    """
    Implementation of the dataset creation interface that loads images and labels.
    """
    
    def call(self, image_dir: str, label_dir: str, split_name:str = 'train') -> List[Dict[str, Union[str, List[Any]]]]:
        """
        Loads images and their corresponding labels from the specified directories.
        
        Args:
            image_dir (str): Path to the directory containing image files.
            label_dir (str): Path to the directory containing label files.
            split_name (str): The Data Split (train or validation)

        Returns:
            Tuple: 
                - List[Dict[str, Union[str, List[str]]]]: A list of dictionaries containing image paths,
                  label paths, and their associated labels.
                - pd.DataFrame: A DataFrame representation of the same dataset.
        
        Raises:
            FileNotFoundError: If the provided directories do not exist.
            Exception: If there is an error reading label files.
        """
        
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory '{image_dir}' does not exist.")
        
        if not os.path.exists(label_dir):
            raise FileNotFoundError(f"Label directory '{label_dir}' does not exist.")

        # Retrieve image paths from the specified directory
        image_paths = glob.glob(os.path.join(image_dir, '*.png'))
        dataset = []

        for image_path in image_paths:
            # Construct the corresponding label file path
            label_path = os.path.join(label_dir, os.path.basename(image_path).replace('.png', '.txt'))
            if not os.path.exists(label_path):
                continue  # Skip this image if the label doesn't exist
            
            try:
                with open(label_path, 'r') as file:
                    labels = file.readlines()
            except Exception as e:
                log_error(f"Error reading label file '{label_path}': {e}")
                continue  # Skip this image if there is an error reading the label file

            # Append the data to the dataset
            dataset.append({
                'image_path': image_path,
                'label_path': label_path,
                'labels': [label.strip() for label in labels]  # Strip whitespace from labels
            })
            
        return dataset
    

class ILabelExistence(ABC):
    """
    Interface for checking the existence of labels corresponding to images.
    """

    @abstractmethod
    def call(self, image_dir: str, label_dir: str) -> List[str]:
        """
        Checks for missing label files for the images in the given directory.

        Args:
            image_dir (str): Directory containing image files.
            label_dir (str): Directory containing label files.

        Returns:
            List[str]: List of missing label file paths.
        """
        pass


class LabelExistence(ILabelExistence):
    """
    Implementation of ILabelExistence to find missing labels for images.
    """

    def call(self, image_dir: str, label_dir: str) -> List[str]:
        """
        Identifies missing label files for images in the specified directories.

        Args:
            image_dir (str): Directory containing image files.
            label_dir (str): Directory containing label files.

        Returns:
            List[str]: List of missing label file paths.
        """
        missing_labels = []
        image_paths = glob.glob(os.path.join(image_dir, '*.png'))

        for image_path in image_paths:
            label_path = os.path.join(label_dir, os.path.basename(image_path).replace('.png', '.txt'))
            if not os.path.exists(label_path):
                missing_labels.append(label_path)

        if missing_labels:
            log_error(f"Missing labels found: {missing_labels}")
        else:
            log_inference("All labels found for images.")

        return missing_labels


class IExtractClassIds(ABC):
    """
    Interface for extracting class IDs from a dataset.
    """

    @abstractmethod
    def call(self, dataset: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Extracts class IDs from the provided dataset.

        Args:
            dataset (List[Dict[str, Any]]): Dataset containing image and label information.

        Returns:
            pd.DataFrame: DataFrame containing class distribution counts.
        """
        pass


class ExtractClassIds(IExtractClassIds):
    """
    Implementation of IExtractClassIds to analyze class distribution.
    """

    def call(self, dataset: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Analyzes the dataset to extract class IDs and their counts.

        Args:
            dataset (List[Dict[str, Any]]): Dataset containing image and label information.

        Returns:
            pd.DataFrame: DataFrame containing class distribution counts.
        """
        class_ids = []
        for data in dataset:
            for label in data.get('labels', []):
                try:
                    class_id = int(label.split()[0])
                    class_ids.append(class_id)
                except (ValueError, IndexError) as e:
                    log_error(f"Error extracting class ID from label '{label}': {e}")

        # Count occurrences of each class
        class_distribution = pd.Series(class_ids).value_counts().sort_index()
        class_distribution_df = pd.DataFrame({
            'class_id': class_distribution.index,
            'count': class_distribution.values
        })

        log_inference(f"Class distribution calculated: {class_distribution_df}")
        return class_distribution_df


class IGetImageSizes(ABC):
    """
    Interface for getting image sizes from a dataset.
    """

    @abstractmethod
    def call(self, dataset: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Retrieves the sizes of images in the dataset.

        Args:
            dataset (List[Dict[str, Any]]): Dataset containing image file paths.

        Returns:
            pd.DataFrame: DataFrame containing image widths and heights.
        """
        pass


class GetImageSize(IGetImageSizes):
    """
    Implementation of IGetImageSizes to obtain image dimensions.
    """

    def call(self, dataset: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Retrieves image sizes for each image in the dataset.

        Args:
            dataset (List[Dict[str, Any]]): Dataset containing image file paths.

        Returns:
            pd.DataFrame: DataFrame containing image widths and heights.
        """
        sizes = []
        for data in dataset:
            try:
                image = cv2.imread(data['image_path'])
                height, width, _ = image.shape
                sizes.append((width, height))
            except Exception as e:
                log_error(f"Error reading image '{data['image_path']}': {e}")

        size_df = pd.DataFrame(sizes, columns=['Width', 'Height'])
        log_inference(f"Image sizes retrieved: {size_df.describe()}")
        return size_df


class IBoundingBox(ABC):
    """
    Interface for extracting bounding box information from the dataset.
    """

    @abstractmethod
    def call(self, dataset: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Extracts bounding box coordinates and class IDs from the dataset.

        Args:
            dataset (List[Dict[str, Any]]): Dataset containing labels with bounding box information.

        Returns:
            pd.DataFrame: DataFrame containing bounding box information.
        """
        pass


class BoundingBox(IBoundingBox):
    """
    Implementation of IBoundingBox to extract bounding box data.
    """

    def call(self, dataset: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Extracts bounding box coordinates and class IDs from the dataset.

        Args:
            dataset (List[Dict[str, Any]]): Dataset containing labels with bounding box information.

        Returns:
            pd.DataFrame: DataFrame containing bounding box information.
        """
        all_bboxes = []
        for data in dataset:
            for label in data.get('labels', []):
                parts = label.strip().split()
                if len(parts) > 1:  # Ensure there are bounding box coordinates
                    try:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:])
                        all_bboxes.append((class_id, x_center, y_center, width, height))
                    except (ValueError, IndexError) as e:
                        log_error(f"Error parsing bounding box from label '{label}': {e}")

        bbox_df = pd.DataFrame(all_bboxes, columns=['class_id', 'x_center', 'y_center', 'width', 'height'])
        log_inference(f"Bounding box data extracted: {bbox_df.head()}")
        return bbox_df


class IExtractClassExamplesOne(ABC):
    """
    Interface for extracting one example per class from the dataset.
    """

    @abstractmethod
    def call(self, dataset: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Extracts one example per class from the dataset.

        Args:
            dataset (List[Dict[str, Any]]): Dataset containing image and label information.

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of class examples with image paths and labels.
        """
        pass


class ExtractClassExamplesOne(IExtractClassExamplesOne):
    """
    Implementation of IExtractClassExamplesOne to find one example per class.
    """

    def call(self, dataset: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Extracts one example for each class from the dataset.

        Args:
            dataset (List[Dict[str, Any]]): Dataset containing image and label information.

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of class examples with image paths and labels.
        """
        class_examples = {}
        num_classes = len(set(label.split()[0] for data in dataset for label in data.get('labels', [])))

        for data in dataset:
            image_path = data['image_path']
            labels = data.get('labels', [])
            for label in labels:
                label_info = label.strip().split()
                class_name = label_info[0]
                if class_name not in class_examples:
                    class_examples[class_name] = {
                        'image_path': image_path,
                        'labels': labels
                    }
                    if len(class_examples) >= num_classes:
                        break

            if len(class_examples) >= num_classes:
                break

        log_inference(f"Extracted one example per class: {class_examples}")
        return class_examples


class IVisualization(ABC):
    """
    Interface for visualizing dataset characteristics.
    """

    def __init__(self, dataset: List[Dict[str, Any]], class_names: List[str], num_classes: int = 10) -> None:
        """
        Initializes the visualization interface.

        Args:
            dataset (List[Dict[str, Any]]): Dataset containing image and label information.
            class_names (List[str]): List of class names.
            num_classes (int): Number of classes to visualize.
        """
        pass

    @abstractmethod
    def display_one_image_per_class(self) -> plt.figimage:
        """
        Displays one image for each class with bounding boxes.
        """
        pass

    @abstractmethod
    def plot_class_distribution(self) -> plt.figimage:
        """
        Plots the distribution of classes in the dataset.
        """
        pass

    @abstractmethod
    def plot_distribution_images_sizes(self) -> plt.figimage:
        """
        Plots the distribution of image sizes.
        """
        pass

    @abstractmethod
    def plot_bounding_box_size(self) -> plt.figimage:
        """
        Plots the distribution of bounding box sizes.
        """
        pass

    @abstractmethod
    def plot_bounding_box_width_by_class_id(self) -> plt.figimage:
        """
        Plots bounding box width distribution by class ID.
        """
        pass


class Visualization(IVisualization):
    """
    Implementation of IVisualization to visualize dataset characteristics.
    """

    def __init__(self, dataset: List[Dict[str, Any]], class_names: List[str], num_classes: int = 10) -> None:
        """
        Initializes the visualization with the dataset and class names.

        Args:
            dataset (List[Dict[str, Any]]): Dataset containing image and label information.
            class_names (List[str]): List of class names.
            num_classes (int): Number of classes to visualize.
        """
        self.dataset = dataset
        self.class_names = class_names
        self.class_distribution_df = ExtractClassIds().call(dataset=dataset)
        self.images_size = GetImageSize().call(dataset=dataset)
        self.bbox_df = BoundingBox().call(dataset=dataset)
        self.class_examples = ExtractClassExamplesOne().call(dataset=dataset)

    def display_one_image_per_class(self) -> List[plt.Figure]:
        """
        Displays one image for each class with bounding boxes.

        Returns:
            List[plt.Figure]: List of Figure objects displaying the images.
        """
        figures = []

        for class_name, data in self.class_examples.items():
            fig, ax = plt.subplots(1, figsize=(10, 10))
            image_path = data['image_path']
            labels = data['labels']

            try:
                # Open the image
                image = Image.open(image_path)
                ax.imshow(image)

                # Get image dimensions
                width, height = image.size

                # Draw bounding boxes
                for label in labels:
                    label_info = label.strip().split()
                    class_name_label = label_info[0]
                    x_min, y_min, x_max, y_max = map(float, label_info[1:])

                    # Convert to absolute coordinates if they are normalized
                    x_min_abs = x_min * width
                    y_min_abs = y_min * height
                    x_max_abs = x_max * width
                    y_max_abs = y_max * height

                    # Calculate width and height of the bounding box
                    box_width = x_max_abs - x_min_abs
                    box_height = y_max_abs - y_min_abs

                    # Create a rectangle patch for the bounding box
                    bbox = patches.Rectangle(
                        (x_min_abs, y_min_abs), box_width, box_height,
                        linewidth=2, edgecolor='r', facecolor='none'
                    )

                    # Add the patch to the plot
                    ax.add_patch(bbox)

                    # Add class label text with background
                    ax.text(
                        x_min_abs, y_min_abs, class_name_label, color='white',
                        fontsize=12, bbox=dict(facecolor='red', alpha=0.5)
                    )

                ax.set_title(f"Class: {class_name}", fontsize=14)
                ax.axis('off')  # Hide the axes
                figures.append(fig)

                log_inference(f"Displayed image for class '{class_name}' with bounding boxes.")

            except Exception as e:
                log_error(f"Error displaying image '{image_path}': {e}")

        return figures


    def plot_class_distribution(self) -> plt.Figure:
        """
        Plots the distribution of classes in the dataset.

        Returns:
            plt.figimage: Figure object showing the class distribution.
        """
        fig, ax = plt.subplots(1)
        plt.figure(figsize=(12, 8))
        sns.barplot(x='class_id', y='count', data=self.class_distribution_df)
        plt.title('Class Distribution in Training Dataset')
        plt.xticks(rotation=45)
        plt.ylabel('Count')
        plt.xlabel('Class ID')

        fig.tight_layout()
        log_inference("Plotted class distribution.")
        return fig

    def plot_distribution_images_sizes(self) -> plt.Figure:
        """
        Plots the distribution of image sizes.

        Returns:
            plt.figimage: Figure object showing the distribution of image sizes.
        """
        fig, ax = plt.subplots(1)
        plt.figure(figsize=(12, 6))
        sns.histplot(self.images_size['Width'], bins=30, kde=True, color='blue', label='Width')
        sns.histplot(self.images_size['Height'], bins=30, kde=True, color='red', label='Height')
        plt.title('Image Size Distribution')
        plt.xlabel('Pixel Size')
        plt.ylabel('Frequency')
        plt.legend()

        fig.tight_layout()
        log_inference("Plotted image size distribution.")
        return fig

    def plot_bounding_box_size(self) -> plt.Figure:
        """
        Plots the distribution of bounding box sizes.

        Returns:
            plt.figimage: Figure object showing the distribution of bounding box sizes.
        """
        fig, ax = plt.subplots(1)
        plt.figure(figsize=(12, 6))
        sns.histplot(self.bbox_df['width'], bins=30, kde=True, color='green', label='Width')
        sns.histplot(self.bbox_df['height'], bins=30, kde=True, color='orange', label='Height')
        plt.title('Bounding Box Size Distribution')
        plt.xlabel('Size (normalized)')
        plt.ylabel('Frequency')
        plt.legend()

        fig.tight_layout()
        log_inference("Plotted bounding box size distribution.")
        return fig

    def plot_bounding_box_width_by_class_id(self) -> plt.Figure:
        """
        Plots bounding box width distribution by class ID.

        Returns:
            plt.figimage: Figure object showing bounding box width by class ID.
        """
        fig, ax = plt.subplots(1)
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='class_id', y='width', data=self.bbox_df)
        plt.title('Bounding Box Width by Class ID')
        plt.xlabel('Class ID')
        plt.ylabel('Bounding Box Width (normalized)')
        plt.xticks(ticks=range(len(self.class_names)), labels=self.class_names, rotation=45)

        fig.tight_layout()
        log_inference("Plotted bounding box width by class ID.")
        return fig
    
import matplotlib.pyplot as plt

def main(image_dir: str, label_dir: str,     
         class_names = ['XR_ELBOW_positive', 'XR_FINGER_positive', 'XR_FOREARM_positive','XR_HAND_positive', 'XR_SHOULDER_positive',
                        'XR_ELBOW_negative', 'XR_FINGER_negative', 'XR_FOREARM_negative','XR_HAND_negative', 'XR_SHOULDER_negative'
    ]    ):
    # Sample dataset structure
    dataset = CreateDataset().call(
        image_dir=image_dir,
        label_dir=label_dir
    )

    # Instantiate the Visualization object
    visualization = Visualization(dataset, class_names)

    # Display images with bounding boxes
    figures = visualization.display_one_image_per_class()

    # Show the figures for images with bounding boxes
    for fig in figures:
        plt.show()

    # Collecting all other figures
    all_figures = []

    # Plot class distribution
    class_distribution_fig = visualization.plot_class_distribution()
    all_figures.append(class_distribution_fig)

    # Plot distribution of image sizes
    image_size_distribution_fig = visualization.plot_distribution_images_sizes()
    all_figures.append(image_size_distribution_fig)

    # Plot bounding box size distribution
    bounding_box_size_fig = visualization.plot_bounding_box_size()
    all_figures.append(bounding_box_size_fig)

    # Plot bounding box width by class ID
    bounding_box_width_by_class_fig = visualization.plot_bounding_box_width_by_class_id()
    all_figures.append(bounding_box_width_by_class_fig)

    return all_figures  # Return all plots for further use if needed

if __name__ == "__main__":
    plots = main()  # Call main function and collect plots