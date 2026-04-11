"""
Configuration for Task 1: LIME Implementation
"""

import os
from pathlib import Path

# Paths
PROJECT_DIR = Path(__file__).parent.parent
IMAGES_DIR = PROJECT_DIR / "Images"
OUTPUT_DIR = PROJECT_DIR / "task1_output"
MODELS_DIR = PROJECT_DIR / "models"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Model Configuration
MODEL_NAME = "inception_v3"
PRETRAINED = True

# Image Configuration
IMAGE_SIZE = 299  # Inception V3 expects 299x299
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Device Configuration
DEVICE = "mps" if __import__("torch").backends.mps.is_available() else "cpu"

# LIME Configuration
NUM_SAMPLES = 1000  # Number of perturbations for LIME
NUM_SUPERPIXELS = 50  # Number of superpixels for segmentation
LIME_NEIGHBORHOOD_DISTANCE = "cosine"
LIME_TOP_PREDICTIONS = 2  # Analyze top 2 predictions

# Visualization
DPI = 100
FIGSIZE = (15, 15)
COLORMAP = "viridis"

# Test Images
TEST_IMAGES = [
    "Schloss-Erlangen02.JPG",
    "Erlangen_Blick_vom_Burgberg_auf_die_Innenstadt_2009_001.JPG",
    "Alte-universitaets-bibliothek_universitaet-erlangen.jpg",
]
