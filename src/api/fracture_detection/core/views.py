import base64
import numpy as np
from PIL import Image
from typing import Tuple
import io
import os
import sys

# Append the path to the workspace for importing utility functions
append_path = "/workspaces/X-Ray-Bones-Fracture-Detection"
sys.path.append(append_path)

from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.contrib.auth import authenticate, login as auth_login
from django.views.decorators.csrf import csrf_exempt

from .forms import LogIn, SignUp
from config.config import Config
from src.deployment_preprocessed.run_pipeline_prediction_deployment import RunRunPipelineDeploymentPrediction
from src.utils.reg_log import log_api, log_error
from src.utils.read_yaml import load_config

@csrf_exempt
def login_view(request):
    """
    Handle user login.
    """
    if request.method == 'POST':
        form = LogIn(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            auth_login(request, user)
            log_api(f'User {user.username} logged in successfully.')
            return redirect('index')
        else:
            log_error(f'Login failed for user {request.POST.get("username")}: {form.errors}')
    else:
        form = LogIn()
    return render(request, 'login.html', {'form': form})

@csrf_exempt
def signup_view(request):
    """
    Handle user signup (registration).
    """
    if request.method == 'POST':
        form = SignUp(request.POST)
        if form.is_valid():
            user = form.save()
            log_api(f'User {user.username} signed up successfully.')
            return redirect('/login')
        else:
            log_error(f'Signup failed: {form.errors}')
    else:
        form = SignUp()
    return render(request, 'login.html', {'form': form})

@csrf_exempt
def error_view(request):
    """
    Display an error message to the user.
    """
    return render(request, 'error.html', {'message': 'An error occurred. Please try again.'})


@csrf_exempt
def main_view(request):
    """
    Main view to handle both login and signup.
    """
    if request.method == "POST":
        # Check if the user wants to log in
        if 'login' in request.POST:
            login_form = LogIn(request, data=request.POST)
            if login_form.is_valid():
                user = login_form.get_user()
                auth_login(request, user)
                log_api(f'User {user.username} logged in successfully.')
                return redirect('index')
            else:
                log_error(f'Login failed: {login_form.errors}')

        # Check if the user wants to sign up
        elif 'signup' in request.POST:
            signup_form = SignUp(request.POST)
            if signup_form.is_valid():
                user = signup_form.save()
                log_api(f'User {user.username} signed up successfully.')
                return redirect('login')
            else:
                log_error(f'Signup failed: {signup_form.errors}')

    else:
        login_form = LogIn()
        signup_form = SignUp()

    return render(request, 'main.html', {
        'login_form': login_form,
        'signup_form': signup_form,
    })


@csrf_exempt
def index(request):
    """
    Handle image upload and prediction.
    """
    if request.method == "POST":
        image_file = request.FILES.get('image')
        if image_file:
            try:
                # Convert uploaded image to numpy array
                image = Image.open(image_file)
                image_np = np.array(image)
                processed_image, label = image_prediction(image_np=image_np)

                # Store processed image, label in session 
                request.session['processed_image'] = processed_image.tolist()
                request.session['label'] = label

                log_api('Image processed successfully, redirecting to result page.')
                return redirect('result')
            except Exception as e:
                log_error(f'Error processing image: {str(e)}')
                return redirect('error')

    return render(request, "index.html")

@csrf_exempt
def result_view(request):
    """
    Display the processed image and label.
    """
    processed_image = np.array(request.session.get('processed_image', []))
    label = request.session.get('label', 'No label')

    if processed_image.size == 0:
        log_error('No processed image found in session, redirecting to error page.')
        return redirect('error')

    # Convert numpy array to an image for displaying
    img = Image.fromarray(processed_image.astype('uint8'))

    # Convert image to a bytes buffer
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")  # Save as PNG, you can change format if needed
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')  # Convert to base64 string

    # Serve the image with the label
    return render(request, 'result.html', {'label': label, 'image': img_str})


@csrf_exempt  # Use this for testing; remove for production
def api(request):
    """
    API view for image processing and returning a JSON response.
    Handles both GET and POST requests.
    Returns the image path and label.
    """
    if request.method == 'POST':
        image_file = request.FILES.get('image')
        if not image_file:
            log_error('No image provided in API request.')
            return JsonResponse({'error': 'No image provided'}, status=400)

        try:
            # Convert uploaded image to numpy array
            image = Image.open(image_file)
            image_np = np.array(image)

            # Process the image
            processed_image, label = image_prediction(image_np=image_np)

            # Save the processed image to a temporary location
            image_path = save_processed_image(processed_image, label)

            # Log API response
            log_api(f'Image processed successfully, label: {label}')

            # Return JSON response with the image path and label
            return JsonResponse({'label': label, 'status': 'success', 'image_path': image_path})

        except Exception as e:
            log_error(f'Error processing image in API: {str(e)}')
            return JsonResponse({'error': str(e)}, status=500)

    elif request.method == 'GET':
        # For GET requests, return instructions
        return JsonResponse({'message': 'Send a POST request with an image to get predictions.'})

    else:
        log_error('Invalid request method for API.')
        return JsonResponse({'error': 'Invalid request method'}, status=405)

def save_processed_image(image_np, label):
    """
    Save the processed image with a label and return the file path.
    """
    # Create a directory for saving processed images if it doesn't exist
    output_dir = 'media/processed_images'
    os.makedirs(output_dir, exist_ok=True)

    # Generate a filename using the label (or some unique identifier)
    filename = f"{label.replace(' ', '_')}.png"  # Replace spaces with underscores
    file_path = os.path.join(output_dir, filename)

    # Save the image
    image = Image.fromarray(image_np)
    image.save(file_path)

    return file_path
# Configuration and model path
configration = Config(config=load_config(None))
model_path = configration.pretrain_model_path

def image_prediction(image_np: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Passes image as a numpy array for prediction.

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