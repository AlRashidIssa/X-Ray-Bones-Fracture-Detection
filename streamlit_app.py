import streamlit as st
import numpy as np
from PIL import Image
import base64
import io
import os
import sys

# Append the path to the workspace for importing utility functions
append_path = "/workspaces/X-Ray-Bones-Fracture-Detection"
sys.path.append(append_path)

from src.deployment_preprocessed.run_pipeline_prediction_deployment import RunRunPipelineDeploymentPrediction
from src.utils.reg_log import log_api, log_error
from src.utils.read_yaml import load_config
from config.config import Config

# Configuration and model path
config = Config(config=load_config(None))
model_path = config.pretrain_model_path

def image_prediction(image_np: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Pass image as a numpy array for prediction.
    Args:
        image_np (np.ndarray): The numpy representation of the image.
    Returns:
        Tuple[np.ndarray, str]: Processed image and label.
    """
    try:
        image, label = RunRunPipelineDeploymentPrediction().call(model_path=model_path, image=image_np)
        return image, label
    except Exception as e:
        log_error(f'Error during image prediction: {str(e)}')
        raise  # Re-raise exception to be caught in the caller function

# Streamlit App UI
st.title("X-Ray Bones Fracture Detection")

uploaded_file = st.file_uploader("Upload an X-ray image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded X-ray Image', use_column_width=True)
    
    # Convert the uploaded image to a numpy array
    image_np = np.array(image)

    # Perform prediction
    if st.button('Run Prediction'):
        try:
            processed_image, label = image_prediction(image_np=image_np)

            # Display the result label
            st.write(f"Prediction Label: {label}")

            # Convert numpy array back to image for display
            processed_img = Image.fromarray(processed_image.astype('uint8'))

            # Convert image to a bytes buffer for display
            buffered = io.BytesIO()
            processed_img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # Display processed image
            st.image(processed_img, caption='Processed Image', use_column_width=True)

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

