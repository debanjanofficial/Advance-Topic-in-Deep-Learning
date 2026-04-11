"""
Task 2: SHAP Implementation
A modularized implementation of SHAP for image classification explainability.
"""

__version__ = "1.0.0"
__author__ = "Deep Learning Course"

from .config import *
from .shap_image_utils import ShapImageProcessor, denormalize_image
from .shap_model_manager import ShapModelManager
from .shap_implementation import ShapExplainer
from .shap_visualizer import ShapVisualizer

__all__ = [
    'ShapImageProcessor',
    'ShapModelManager',
    'ShapExplainer',
    'ShapVisualizer',
    'denormalize_image',
]
