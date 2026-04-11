"""
Configuration for Task 2: SHAP Implementation
"""

from pathlib import Path

# Paths
PROJECT_DIR = Path(__file__).parent.parent
IMAGES_DIR = PROJECT_DIR / "Images"
OUTPUT_DIR = PROJECT_DIR / "task2_output"
MODELS_DIR = PROJECT_DIR / "models"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Model Configuration
MODEL_NAME = "resnet50"  # SHAP works well with ResNet50
PRETRAINED = True

# Image Configuration
IMAGE_SIZE = 224  # ResNet50 expects 224x224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Device Configuration
DEVICE = "mps" if __import__("torch").backends.mps.is_available() else "cpu"

# SHAP Configuration
SHAP_BACKGROUND_SAMPLES = 50  # Background data for SHAP
SHAP_EXPLAIN_SAMPLES = 50  # Number of integration steps for attribution (reduced for speed)
SHAP_TOP_PREDICTIONS = 2  # Analyze top 2 predictions

# Visualization
DPI = 100
FIGSIZE = (15, 15)
COLORMAP = "RdBu"

# Test Images
TEST_IMAGES = [
    "Schloss-Erlangen02.JPG",
    "Erlangen_Blick_vom_Burgberg_auf_die_Innenstadt_2009_001.JPG",
    "Alte-universitaets-bibliothek_universitaet-erlangen.jpg",
]
