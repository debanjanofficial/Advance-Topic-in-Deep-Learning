"""
LIME (Local Interpretable Model-agnostic Explanations) Implementation
Based on: "Why Should I Trust You?": Explaining the Predictions of Any Classifier
(Ribeiro et al., 2016)
"""

import numpy as np
import torch
from typing import Callable, Tuple, List
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from PIL import Image

from config import NUM_SAMPLES, LIME_NEIGHBORHOOD_DISTANCE


class LIMEExplainer:
    """
    LIME explains individual predictions by fitting a local linear model
    around the prediction instance.
    """
    
    def __init__(
        self,
        prediction_fn: Callable,
        num_samples: int = NUM_SAMPLES,
        kernel_width: float = 0.25
    ):
        """
        Initialize LIME explainer.
        
        Args:
            prediction_fn: Function that takes image tensor and returns probabilities
            num_samples: Number of perturbed samples to generate
            kernel_width: Kernel width for exponential kernel (distance weighting)
        """
        self.prediction_fn = prediction_fn
        self.num_samples = num_samples
        self.kernel_width = kernel_width
    
    def explain_instance(
        self,
        image: np.ndarray,
        segments: np.ndarray,
        target_class: int,
        num_samples: int = None,
        device: str = "cpu"
    ) -> Tuple[np.ndarray, List[float], List[int], np.ndarray]:
        """
        Explain a single instance using LIME with superpixels.
        
        Args:
            image: Original image array of shape (H, W, 3) with values in [0, 1]
            segments: Superpixel segmentation map
            target_class: Class index to explain
            num_samples: Override default number of samples
            device: Device to run inference on
            
        Returns:
            Tuple of:
            - explanation: Array of shape (num_superpixels,) with importance scores
            - local_pred: List of predicted probabilities for target_class in perturbed samples
            - sample_indices: List of superpixel IDs used in explanation
            - perturbed_weights: Weights used in the local linear model
        """
        if num_samples is None:
            num_samples = self.num_samples
        
        # Get number of superpixels
        num_superpixels = len(np.unique(segments))
        
        print(f"Generating {num_samples} perturbed samples...")
        
        # Generate perturbed samples by randomly masking superpixels
        perturbed_samples, perturbation_masks = self._generate_perturbed_samples(
            image, segments, num_samples
        )
        
        # Get model predictions for all perturbed samples
        print("Getting model predictions...")
        predictions = self._get_batch_predictions(
            perturbed_samples, device, target_class
        )
        
        # Compute distances between original and perturbed samples
        print("Computing sample weights...")
        distances = self._compute_distances(perturbation_masks)
        
        # Compute kernel weights (exponential kernel)
        weights = np.exp(-(distances ** 2) / (self.kernel_width ** 2))
        
        # Fit weighted linear regression
        print("Fitting local linear model...")
        coefficients = self._fit_weighted_regression(
            perturbation_masks.astype(float),
            predictions,
            weights
        )
        
        return coefficients, predictions.tolist(), list(range(num_superpixels)), weights
    
    def _generate_perturbed_samples(
        self,
        image: np.ndarray,
        segments: np.ndarray,
        num_samples: int,
        background_color: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate perturbed samples by randomly masking superpixels.
        
        Args:
            image: Original image array (H, W, 3)
            segments: Superpixel segmentation map
            num_samples: Number of perturbed samples to generate
            background_color: Color to replace masked superpixels with
            
        Returns:
            Tuple of:
            - perturbed_images: Array of shape (num_samples, H, W, 3)
            - perturbation_masks: Array of shape (num_samples, num_superpixels)
                                 where 1 = superpixel present, 0 = superpixel masked
        """
        num_superpixels = len(np.unique(segments))
        h, w = segments.shape
        
        perturbed_images = []
        perturbation_masks = np.zeros((num_samples, num_superpixels), dtype=np.uint8)
        
        # Always include the original image as first sample
        perturbed_images.append(image.copy())
        perturbation_masks[0] = 1  # All superpixels present
        
        for i in tqdm(range(1, num_samples), desc="Generating perturbations"):
            # Randomly select which superpixels to keep
            mask = np.random.binomial(1, 0.5, num_superpixels)
            perturbation_masks[i] = mask
            
            # Create perturbed image
            perturbed_image = image.copy()
            for sp_id in range(num_superpixels):
                if mask[sp_id] == 0:
                    # Replace this superpixel with background color
                    sp_mask = segments == sp_id
                    for c in range(3):
                        perturbed_image[sp_mask, c] = background_color[c]
            
            perturbed_images.append(perturbed_image)
        
        return np.array(perturbed_images), perturbation_masks
    
    def _get_batch_predictions(
        self,
        images: np.ndarray,
        device: str,
        target_class: int
    ) -> np.ndarray:
        """
        Get model predictions for a batch of perturbed images.
        
        Args:
            images: Array of shape (N, H, W, 3) with values in [0, 1]
            device: Device to run inference on
            target_class: Class index to get predictions for
            
        Returns:
            Array of shape (N,) with predicted probabilities for target_class
        """
        predictions = []
        batch_size = 8  # Process in batches to avoid memory issues
        
        for i in tqdm(range(0, len(images), batch_size), desc="Predicting"):
            batch = images[i:i+batch_size]
            
            # Convert numpy to tensor
            batch_tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).float().to(device)
            
            # Normalize (assuming images are in [0, 1], match ImageNet normalization)
            from torchvision.transforms.functional import normalize
            batch_tensor = normalize(
                batch_tensor,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            
            # Get predictions
            batch_probs = self.prediction_fn(batch_tensor)
            
            # Extract probability for target class
            predictions.append(batch_probs[:, target_class])
        
        return np.concatenate(predictions)
    
    def _compute_distances(self, perturbation_masks: np.ndarray) -> np.ndarray:
        """
        Compute distances between original (all 1s) and perturbed samples.
        Uses Euclidean distance in binary space.
        
        Args:
            perturbation_masks: Array of shape (num_samples, num_superpixels)
            
        Returns:
            Array of shape (num_samples,) with distances
        """
        # Original is all 1s (first sample)
        original = perturbation_masks[0]
        
        # Compute Euclidean distance
        distances = np.linalg.norm(
            perturbation_masks - original,
            axis=1
        )
        
        return distances
    
    def _fit_weighted_regression(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        alpha: float = 1.0
    ) -> np.ndarray:
        """
        Fit weighted Ridge regression to explain the model locally.
        
        Args:
            X: Feature matrix of shape (num_samples, num_features)
                where features are superpixel presence (0/1)
            y: Target values (predicted probabilities) of shape (num_samples,)
            weights: Sample weights of shape (num_samples,)
            alpha: L2 regularization strength
            
        Returns:
            Coefficient array of shape (num_features,) showing importance of each superpixel
        """
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit weighted ridge regression
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X_scaled, y, sample_weight=weights)
        
        # Get coefficients in original feature space
        coefficients = model.coef_ / (scaler.scale_ + 1e-8)
        
        return coefficients
    
    def get_explanation_mask(
        self,
        segments: np.ndarray,
        coefficients: np.ndarray,
        top_k: int = None,
        positive_only: bool = True
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Convert superpixel coefficients to spatial explanation mask.
        
        Args:
            segments: Superpixel segmentation map
            coefficients: Importance scores for each superpixel
            top_k: Show only top-k superpixels (by absolute importance)
            positive_only: If True, only show positive contributions
            
        Returns:
            Tuple of:
            - mask: 2D array same shape as segments with importance values
            - important_superpixels: List of superpixel IDs in order of importance
        """
        mask = np.zeros_like(segments, dtype=float)
        
        # Filter coefficients if needed
        coef_to_use = coefficients.copy()
        if positive_only:
            coef_to_use[coef_to_use < 0] = 0
        
        # Get top-k superpixels
        if top_k is not None:
            top_indices = np.argsort(np.abs(coef_to_use))[-top_k:]
        else:
            top_indices = np.argsort(np.abs(coef_to_use))
        
        # Create mask - iterate over number of superpixels, not image dimensions
        for sp_id in range(len(coefficients)):
            mask[segments == sp_id] = coef_to_use[sp_id]
        
        important_superpixels = sorted(
            top_indices,
            key=lambda x: coef_to_use[x],
            reverse=True
        )
        
        return mask, important_superpixels
