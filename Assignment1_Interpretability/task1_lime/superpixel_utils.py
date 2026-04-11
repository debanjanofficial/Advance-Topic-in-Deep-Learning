"""
Superpixel segmentation utilities for LIME using scikit-image
"""

import numpy as np
from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage.color import rgb2lab
from typing import Tuple
from PIL import Image
import torch


class SuperpixelSegmenter:
    """Segments images into superpixels for LIME explanations."""
    
    def __init__(self, method: str = "slic", num_segments: int = 50):
        """
        Initialize SuperpixelSegmenter.
        
        Args:
            method: Segmentation method - "slic", "felzenszwalb", or "quickshift"
            num_segments: Approximate number of superpixels to generate
        """
        self.method = method
        self.num_segments = num_segments
    
    def pil_to_numpy(self, image: Image.Image) -> np.ndarray:
        """
        Convert PIL Image to numpy array in [0, 1] range.
        
        Args:
            image: PIL Image
            
        Returns:
            Numpy array of shape (H, W, 3) with values in [0, 1]
        """
        return np.array(image).astype(float) / 255.0
    
    def segment(self, image: Image.Image) -> Tuple[np.ndarray, int]:
        """
        Segment an image into superpixels.
        
        Args:
            image: PIL Image (RGB)
            
        Returns:
            Tuple of (segmentation_map, num_segments)
            - segmentation_map: 2D array where each value is a superpixel label
            - num_segments: Actual number of unique superpixels created
        """
        image_array = self.pil_to_numpy(image)
        
        if self.method == "slic":
            segments = self._segment_slic(image_array)
        elif self.method == "felzenszwalb":
            segments = self._segment_felzenszwalb(image_array)
        elif self.method == "quickshift":
            segments = self._segment_quickshift(image_array)
        else:
            raise ValueError(f"Unknown segmentation method: {self.method}")
        
        num_segments = len(np.unique(segments))
        return segments, num_segments
    
    def _segment_slic(self, image: np.ndarray) -> np.ndarray:
        """
        SLIC (Simple Linear Iterative Clustering) segmentation.
        Fast and produces regular superpixels.
        """
        # Convert to LAB for SLIC
        image_lab = rgb2lab(image)
        segments = slic(
            image_lab,
            n_segments=self.num_segments,
            sigma=1,
            start_label=0,
            compactness=10
        )
        return segments
    
    def _segment_felzenszwalb(self, image: np.ndarray) -> np.ndarray:
        """
        Felzenszwalb's efficient graph-based segmentation.
        More adaptive to image content but slower.
        """
        segments = felzenszwalb(
            image,
            scale=100,
            sigma=0.5,
            min_size=50
        )
        # Relabel to start from 0
        return np.searchsorted(np.unique(segments), segments)
    
    def _segment_quickshift(self, image: np.ndarray) -> np.ndarray:
        """
        Quick Shift segmentation.
        Good balance between speed and quality.
        """
        segments = quickshift(
            image,
            kernel_size=3,
            max_dist=6,
            ratio=0.5
        )
        return segments
    
    def get_superpixel_mask(
        self, 
        segments: np.ndarray, 
        superpixel_id: int
    ) -> np.ndarray:
        """
        Get binary mask for a specific superpixel.
        
        Args:
            segments: Segmentation map from segment()
            superpixel_id: ID of the superpixel
            
        Returns:
            Binary mask of shape (H, W) where 1 = superpixel, 0 = background
        """
        return (segments == superpixel_id).astype(np.uint8)
    
    def get_superpixel_color(
        self, 
        image: np.ndarray, 
        segments: np.ndarray, 
        superpixel_id: int
    ) -> np.ndarray:
        """
        Get mean color of a superpixel.
        
        Args:
            image: Image array of shape (H, W, 3) with values in [0, 1]
            segments: Segmentation map
            superpixel_id: ID of the superpixel
            
        Returns:
            RGB color array of shape (3,)
        """
        mask = segments == superpixel_id
        return image[mask].mean(axis=0)
    
    def visualize_segments(
        self, 
        image: np.ndarray, 
        segments: np.ndarray
    ) -> np.ndarray:
        """
        Visualize superpixel boundaries on image.
        
        Args:
            image: Image array of shape (H, W, 3) with values in [0, 1]
            segments: Segmentation map
            
        Returns:
            Image array with boundaries drawn
        """
        from skimage.segmentation import mark_boundaries
        return mark_boundaries(image, segments, color=(0, 1, 0), mode='thick')
