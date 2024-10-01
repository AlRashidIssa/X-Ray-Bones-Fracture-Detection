# **Project Configuration System Documentation**

## **Overview**

The project configuration system is designed to centralize and manage all configurable parameters for your project in a clear, structured, and flexible way. It allows loading configuration settings from a YAML file, which makes it easy to modify project parameters without altering the codebase. The system also uses a high-level abstraction via an abstract base class (`ABC`) to ensure consistent and extendable configurations.



### Parameters:
- **`AXL` (str)**: A custom parameter.
- **`batch_size` (int)**: The number of samples per gradient update.
- **`directories_zip_files` (str)**: Path to the directory containing ZIP files.
- **`epochs` (int)**: Number of training epochs.
- **`extract_to` (str)**: Directory path where files will be extracted.
- **`image_size` (tuple[int, int])**: The target size of images (width, height).
- **`input_shape` (tuple[int, int, int])**: Shape of the input data (height, width, channels).
- **`model_name` (str)**: The name of the model to use.
- **`num_anchors` (int)**: Number of anchors used in detection models.
- **`num_classes` (int)**: Number of output classes.
- **`optimization` (str)**: Name of the optimization algorithm (e.g., "adam", "sgd").
- **`optimization_parameters` (dict)**: Parameters for the optimization algorithm, such as learning rate.
- **`output_path` (str)**: Directory where outputs will be saved.
- **`name_dataset` (str)**: The name of the dataset being used.
- **`train` (bool)**: Whether to train the model or not.
- **`train_data` (str)**: Path to the training data file.
- **`valid_data` (str)**: Path to the validation data file.
- **`summary` (bool)**: Whether to print a model summary.
- **`url` (str)**: URL to download the dataset from.

---

