"""
Configuration settings for the Chest X-Ray Disease Detection Application
"""
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model configuration
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'resnet50v2_chest_xray.h5')
XRAY_VALIDATOR_PATH = os.path.join(BASE_DIR, 'models', 'xray_validator.h5')

# Class labels (must match model training order)
CLASS_LABELS = [
    'Bacterial Pneumonia',
    'Covid',
    'Normal',
    'Tuberculosis',
    'Viral Pneumonia'
]

# Image preprocessing
IMG_SIZE = (224, 224)
IMG_CHANNELS = 3

# Upload configuration
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# History configuration
HISTORY_FILE = os.path.join(BASE_DIR, 'data', 'prediction_history.json')
MAX_HISTORY_ITEMS = 20

# Application settings
SECRET_KEY = 'your-secret-key-change-in-production'
DEBUG = True
HOST = '0.0.0.0'
PORT = 5000

# Grad-CAM configuration
GRADCAM_LAYER_NAME = 'conv5_block3_out'  # Last conv layer in ResNet50V2

# X-ray validation threshold
XRAY_CONFIDENCE_THRESHOLD = 0.7

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
