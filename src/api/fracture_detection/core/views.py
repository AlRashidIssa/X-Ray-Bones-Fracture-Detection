from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
import numpy as np
from PIL import Image
import io
import os
import sys

# Append the path to the workspace for importing utility functions
append_path = "/workspaces/X-Ray-Bones-Fracture-Detection"
sys.path.append(append_path)

from src.deployment_preprocessed.run_pipeline_prediction_deployment import LoadPreTrainModel
from src.utils.reg_log import log_api, log_error
# Create your views here.
