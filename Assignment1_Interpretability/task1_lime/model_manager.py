"""
Model manager for Inception V3 loading and inference
"""

import torch
import torch.nn as nn
from torchvision.models import inception_v3
import json
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np

from config import DEVICE, PRETRAINED, OUTPUT_DIR


class ModelManager:
    """Manages Inception V3 model loading and inference."""
    
    def __init__(self):
        """Initialize ModelManager and load Inception V3."""
        self.device = DEVICE
        self.model = self._load_model()
        self.imagenet_classes = self._load_imagenet_labels()
    
    def _load_model(self) -> nn.Module:
        """
        Load pretrained Inception V3 model.
        
        Returns:
            Inception V3 model in evaluation mode
        """
        print(f"Loading Inception V3 (pretrained={PRETRAINED}) on device: {self.device}")
        model = inception_v3(pretrained=PRETRAINED, aux_logits=True)
        model = model.to(self.device)
        model.eval()
        return model
    
    def _load_imagenet_labels(self) -> Dict[int, str]:
        """
        Load ImageNet class labels.
        Downloads from online source or uses cached version.
        
        Returns:
            Dictionary mapping class index to class name
        """
        labels_path = OUTPUT_DIR / "imagenet_labels.json"
        
        if labels_path.exists():
            with open(labels_path, 'r') as f:
                return json.load(f)
        
        # Download ImageNet labels
        print("Downloading ImageNet labels...")
        try:
            import urllib.request
            url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
            labels = {}
            with urllib.request.urlopen(url) as response:
                for idx, line in enumerate(response):
                    labels[idx] = line.decode('utf-8').strip()
            
            # Save locally
            with open(labels_path, 'w') as f:
                json.dump(labels, f)
            return labels
        except Exception as e:
            print(f"Warning: Could not download ImageNet labels: {e}")
            print("Using generic labels instead")
            return {i: f"Class {i}" for i in range(1000)}
    
    def predict(self, image_tensor: torch.Tensor, top_k: int = 5) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Get predictions for an image.
        
        Args:
            image_tensor: Normalized image tensor of shape (1, 3, 299, 299)
            top_k: Number of top predictions to return
            
        Returns:
            Tuple of:
            - logits: Raw model output of shape (1, 1000)
            - top_classes: List of top-k class names
            - top_probs: Probabilities of top-k predictions
        """
        with torch.no_grad():
            output = self.model(image_tensor)
            
            # Handle tuple output from Inception V3 with aux_logits
            if isinstance(output, tuple):
                output = output[0]
            
            logits = output.cpu().numpy()
            probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()
        
        # Get top-k predictions
        top_indices = np.argsort(probs[0])[::-1][:top_k]
        top_classes = [self.imagenet_classes.get(int(idx), f"Class {int(idx)}") 
                      for idx in top_indices]
        top_probs = probs[0][top_indices]
        
        return logits, top_classes, top_probs
    
    def get_logits(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Get raw logits from model (for LIME).
        
        Args:
            image_tensor: Image tensor of shape (1, 3, 299, 299)
            
        Returns:
            Logits array of shape (1, 1000)
        """
        with torch.no_grad():
            output = self.model(image_tensor)
            if isinstance(output, tuple):
                output = output[0]
            return output.cpu().numpy()
    
    def get_probabilities(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Get probability predictions from model.
        
        Args:
            image_tensor: Image tensor of shape (N, 3, 299, 299)
            
        Returns:
            Probabilities array of shape (N, 1000)
        """
        with torch.no_grad():
            output = self.model(image_tensor)
            if isinstance(output, tuple):
                output = output[0]
            probs = torch.nn.functional.softmax(output, dim=1)
            return probs.cpu().numpy()
    
    def get_class_index(self, class_name: str) -> int:
        """
        Get class index from class name.
        
        Args:
            class_name: ImageNet class name
            
        Returns:
            Class index (0-999)
        """
        for idx, name in self.imagenet_classes.items():
            if name == class_name:
                return int(idx)
        raise ValueError(f"Class not found: {class_name}")
    
    def get_class_name(self, class_index: int) -> str:
        """
        Get class name from class index.
        
        Args:
            class_index: Class index (0-999)
            
        Returns:
            ImageNet class name
        """
        return self.imagenet_classes.get(class_index, f"Class {class_index}")
