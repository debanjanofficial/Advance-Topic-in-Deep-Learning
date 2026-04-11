"""
Image preprocessing and utilities for Task 1: LIME Implementation
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Tuple, Union

from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, IMAGES_DIR, DEVICE


class ImageProcessor:
    """Handles image loading, preprocessing, and normalization."""
    
    def __init__(self):
        """Initialize ImageProcessor with Inception V3 preprocessing."""
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD
            )
        ])
        
        self.transform_pil = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
    
    def load_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """
        Load and preprocess an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image tensor of shape (1, 3, 299, 299)
        """
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        return image_tensor.unsqueeze(0).to(DEVICE)
    
    def load_image_pil(self, image_path: Union[str, Path]) -> Tuple[Image.Image, np.ndarray]:
        """
        Load image as PIL Image and normalized numpy array.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (PIL Image, normalized numpy array of shape (3, 299, 299))
        """
        image = Image.open(image_path).convert("RGB")
        image_resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        image_array = self.transform_pil(image_resized).to(DEVICE)
        return image_resized, image_array
    
    def tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
        """
        Convert normalized tensor back to PIL Image.
        
        Args:
            tensor: Normalized image tensor of shape (3, 299, 299) or (1, 3, 299, 299)
            
        Returns:
            PIL Image
        """
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # Denormalize
        tensor = tensor.clone()
        for i, (mean, std) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
            tensor[i] = tensor[i] * std + mean
        
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to PIL
        tensor = tensor.cpu().detach()
        image_array = (tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        return Image.fromarray(image_array)
    
    def get_image_path(self, image_name: str) -> Path:
        """
        Get full path to image in Images directory.
        
        Args:
            image_name: Name of the image file
            
        Returns:
            Full path to the image
        """
        image_path = IMAGES_DIR / image_name
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return image_path


def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalize an image tensor for display.
    
    Args:
        image_tensor: Tensor of shape (3, H, W) or (B, 3, H, W)
        
    Returns:
        Numpy array of shape (H, W, 3) with values in [0, 255]
    """
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)
    
    image = image_tensor.clone().cpu().detach()
    
    for i, (mean, std) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
        image[i] = image[i] * std + mean
    
    image = torch.clamp(image, 0, 1)
    return (image.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
