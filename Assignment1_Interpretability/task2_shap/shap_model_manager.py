"""
Model manager for ResNet50 loading and inference (for SHAP explanations)
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50
import json
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np

from config import DEVICE, PRETRAINED, OUTPUT_DIR, IMAGE_SIZE


class ShapModelManager:
    """Manages ResNet50 model loading and inference for SHAP."""
    
    def __init__(self, model_name: str = "resnet50"):
        """
        Initialize ShapModelManager and load ResNet50.
        
        Args:
            model_name: Name of model to load (for future flexibility)
        """
        self.device = DEVICE
        self.model = self._load_model(model_name)
        self.imagenet_classes = self._load_imagenet_labels()
        
        # Hook to capture intermediate features for interpretability
        self.features = None
        self._register_hooks()
    
    def _load_model(self, model_name: str) -> nn.Module:
        """
        Load pretrained ResNet50 model.
        
        Args:
            model_name: Model identifier
            
        Returns:
            ResNet50 model in evaluation mode
        """
        print(f"Loading {model_name.upper()} (pretrained={PRETRAINED}) on device: {self.device}")
        
        if model_name.lower() == "resnet50":
            model = resnet50(pretrained=PRETRAINED)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = model.to(self.device)
        model.eval()
        
        # Remove the final classification layer for intermediate feature extraction
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
        self.classifier = model.fc
        
        return model
    
    def _register_hooks(self):
        """Register forward hooks to capture intermediate activations."""
        def hook_fn(module, input, output):
            self.features = output.detach()
        
        # Register hook on avg pooling layer
        for module in self.model.modules():
            if isinstance(module, nn.AdaptiveAvgPool2d):
                module.register_forward_hook(hook_fn)
                break
    
    def _load_imagenet_labels(self) -> Dict[int, str]:
        """
        Load ImageNet class labels.
        
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
            return {i: f"Class {i}" for i in range(1000)}
    
    def predict(self, image_tensor: torch.Tensor, top_k: int = 5) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Get predictions for an image.
        
        Args:
            image_tensor: Normalized image tensor of shape (1, 3, 224, 224)
            top_k: Number of top predictions to return
            
        Returns:
            Tuple of:
            - logits: Raw model output of shape (1, 1000)
            - top_classes: List of top-k class names
            - top_probs: Probabilities of top-k predictions
        """
        with torch.no_grad():
            output = self.model(image_tensor)
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
        Get raw logits from model.
        
        Args:
            image_tensor: Image tensor of shape (1, 3, 224, 224)
            
        Returns:
            Logits array of shape (1, 1000)
        """
        with torch.no_grad():
            output = self.model(image_tensor)
            return output.cpu().numpy()
    
    def get_probabilities(self, image_tensor) -> np.ndarray:
        """
        Get probability predictions from model.
        
        Args:
            image_tensor: Image tensor/array of shape (N, 3, 224, 224) 
            
        Returns:
            Probabilities array of shape (N, 1000)
        """
        # Handle torch tensor input
        if isinstance(image_tensor, torch.Tensor):
            # Make sure device matches
            if image_tensor.device != torch.device(self.device):
                image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            output = self.model(image_tensor)
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
