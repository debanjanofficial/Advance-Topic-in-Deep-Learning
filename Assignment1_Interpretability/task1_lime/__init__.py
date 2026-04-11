"""
Task 1: LIME Implementation
A modularized implementation of LIME for image classification explanability.
"""

__version__ = "1.0.0"
__author__ = "Deep Learning Course"

from .config import *
from .image_utils import ImageProcessor, denormalize_image
from .superpixel_utils import SuperpixelSegmenter
from .model_manager import ModelManager
from .lime_implementation import LIMEExplainer
from .visualizer import LIMEVisualizer

__all__ = [
    'ImageProcessor',
    'SuperpixelSegmenter',
    'ModelManager',
    'LIMEExplainer',
    'LIMEVisualizer',
    'denormalize_image',
]
