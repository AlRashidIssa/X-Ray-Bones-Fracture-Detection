import random
import os
import sys
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod

# Local imports
sys.path.append("/workspaces/DetectionXR/")
from src.utils.reg_log import log_inference, log_error

class IDataGenerator(ABC):
    """
    Interface for data generators in object detection.
    """
    @abstractmethod
    def call(self, 
             images_dir: str, 
             labels_dir: str, 
             batch_size: int = 128, 
             image_size: Tuple[int, int] = (512, 512),
             num_class: int = 10
             ) -> Tuple[np.ndarray, Tuple[list, list, list]]:
        """
        Generates batches of images and their corresponding labels.
        """
        pass

    @abstractmethod
    def parse_label_file(self, label_file: str, num_class: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parses a label file and returns class labels and bounding boxes.
        """
        pass


class DataGenerator(IDataGenerator):
    """
    Data generator for loading images and labels for object detection.
    """
    def call(self, 
             images_dir: str, 
             labels_dir: str,
             batch_size: int = 128,
             image_size: Tuple[int, int] = (512, 512),
             num_anchors: int = 3,
             num_class: int = 10) -> Tuple[np.ndarray, Tuple[list, list, list]]: # type: ignore
        
        """
        Generates batches of images and corresponding labels.

        Args:
            images_dir (str): Directory of images.
            labels_dir (str): Directory of labels.
            batch_size (int): Number of images in each batch.
            image_size (Tuple[int, int]): Target size for images.
            num_anchors (int): Number of anchors for object detection.
            num_class (int): Number of classes for detection.

        Returns:
            Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]: A tuple containing images and labels.
        """

        image_files = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])
        label_files = sorted([os.path.join(labels_dir, os.path.basename(f).replace('.jpg', '.txt').replace('.png', '.txt')) for f in image_files])

        while True:
            data = list(zip(image_files, label_files))
            random.shuffle(data)
            image_files, label_files = zip(*data)

            for start in range(0, len(image_files), batch_size):
                images = []
                # Prepare output arrays
                class_labels_batch = np.zeros((batch_size, num_class), dtype=np.float32)
                bbox_labels_batch = np.zeros((batch_size, num_class, 4), dtype=np.float32)  # (batch_size, num_class, 4)
                objectness_labels_batch = np.zeros((batch_size, num_anchors), dtype=np.float32)  # (batch_size, num_anchors)

                end = min(start + batch_size, len(image_files))
                batch_image_files = image_files[start:end]
                batch_label_files = label_files[start:end]

                for i, (image_file, label_file) in enumerate(zip(batch_image_files, batch_label_files)):
                    try:
                        image = cv2.imread(image_file)  # Corrected from cv2.imshow
                        if image is None:
                            raise FileNotFoundError(f"Image file {image_file} not found or could not be loaded.")
                        # Resize image
                        image = cv2.resize(image, image_size)
                        # Normalize image between [0, 1]
                        image = image / 255.0

                        class_labels, bboxes = self.parse_label_file(label_file, num_class)
                        images.append(image)

                        # Populate class and bounding box labels
                        class_labels_batch[i] = class_labels
                        bbox_labels_batch[i] = bboxes

                        # Assuming one object per anchor for simplicity
                        objectness_labels_batch[i] = np.array([1.0 if np.any(bbox > 0) else 0.0 for bbox in bboxes[:num_anchors]])
                    except Exception as e:
                        log_error(f"Error processing {image_file} with {label_file}: {e}")
                        images.append(np.zeros((*image_size, 3), dtype=np.float32))  # Placeholder for error case

                images = np.array(images)
                yield images, (class_labels_batch, bbox_labels_batch, objectness_labels_batch)

    def parse_label_file(self, label_file: str, num_class: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parses the label file to extract class labels and bounding boxes.

        Args:
            label_file (str): Path to the label file.
            num_class (int): Number of classes.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Class labels and bounding boxes.
        """
        class_labels = np.zeros((num_class,), dtype=np.float32)
        bboxes = np.zeros((num_class, 4), dtype=np.float32)
        try:
            with open(label_file, 'r') as f:
                for line in f.readlines():
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    class_labels[int(class_id)] = 1  # One-hot encoding
                    # Store bbox in (x_min, y_min, x_max, y_max)
                    bboxes[int(class_id)] = [x_center - width / 2, y_center - height / 2, 
                                              x_center + width / 2, y_center + height / 2]
        except Exception as e:
            log_error(f"Error reading {label_file}: {e}")

        return class_labels, bboxes

if __name__ == "__main__":
    # Define directories for images and labels
    images_dir = "/workspaces/DetectionXR/data/extraction_zip/YOLODataSet/images/train"
    labels_dir = "/workspaces/DetectionXR/data/extraction_zip/YOLODataSet/labels/train"
    
    # Create an instance of DataGenerator
    data_generator = DataGenerator()

    # Set parameters
    batch_size = 4
    image_size = (512, 512)
    num_class = 10
    num_anchors = 3

    # Create a generator for the training data
    generator = data_generator.call(images_dir, labels_dir, 
                                          batch_size=batch_size, 
                                          image_size=image_size,
                                          num_class=num_class)
    
    # Get a batch of training data
    train_images, (train_class_labels_batch, train_bbox_labels_batch, train_objectness_labels_batch) = next(generator)


    # Print the shapes
    print("Training data shapes:")
    print("Images:", train_images.shape)  # Should be (batch_size, H, W, C)
    print("Class Labels:", train_class_labels_batch.shape)  # Should be (batch_size, num_classes)
    print("Bounding Box Labels:", train_bbox_labels_batch.shape)  # Should be (batch_size, num_classes, 4)
    print("Objectness Labels:", train_objectness_labels_batch.shape)  # Should be (batch_size, num_anchors)

    # Iterate through the generator
    for images, (class_labels_batch, bbox_labels_batch, objectness_labels_batch) in generator:
        print("Images shape:", images.shape)
        print("Class labels shape:", class_labels_batch.shape)
        print("Bounding boxes shape:", bbox_labels_batch.shape)
        print("Objectness labels shape:", objectness_labels_batch.shape)

        # Break after one batch for demonstration
        break