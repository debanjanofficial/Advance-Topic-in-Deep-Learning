"""
Visualization utilities for SHAP explanations
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional

from config import OUTPUT_DIR, DPI


class ShapVisualizer:
    """Handles visualization of SHAP explanations."""
    
    def __init__(self, output_dir: Path = OUTPUT_DIR):
        """
        Initialize ShapVisualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def visualize_shap_explanation(
        self,
        original_image: np.ndarray,
        shap_values: np.ndarray,
        top_predictions: List[Tuple[str, float]],
        target_class_name: str,
        filename: str
    ) -> Path:
        """
        Create comprehensive SHAP explanation visualization.
        
        Args:
            original_image: Original image array (H, W, 3) with values in [0, 1]
            shap_values: Attribution values of shape (H, W) or (3, H, W)
            top_predictions: List of (class_name, probability) tuples
            target_class_name: Name of the explained class
            filename: Output filename
            
        Returns:
            Path to saved figure
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Ensure image is in correct format
        if original_image.max() > 1:
            original_image = original_image / 255.0
        
        # Handle both 2D (H, W) and 3D (3, H, W) attribution shapes
        if len(shap_values.shape) == 3:
            shap_agg = np.mean(np.abs(shap_values), axis=0)  # (H, W)
            positive_shap = np.maximum(np.mean(shap_values, axis=0), 0)  # (H, W)
            negative_shap = np.minimum(np.mean(shap_values, axis=0), 0)  # (H, W)
        else:  # 2D attribution map
            shap_agg = np.abs(shap_values)  # (H, W)
            positive_shap = np.maximum(shap_values, 0)  # (H, W)
            negative_shap = np.minimum(shap_values, 0)  # (H, W)
        
        # 1. Original image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow((original_image * 255).astype(np.uint8))
        ax1.set_title("Original Image", fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # 2. Aggregate SHAP values (mean across channels)
        ax2 = fig.add_subplot(gs[0, 1])
        im = ax2.imshow(shap_agg, cmap='hot')
        ax2.set_title("Attribution Magnitude\n(Pixel Importance)", 
                     fontsize=12, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im, ax=ax2, label='Importance')
        
        # 3. Positive SHAP contributions
        ax3 = fig.add_subplot(gs[0, 2])
        im = ax3.imshow(positive_shap, cmap='Greens')
        ax3.set_title(f"Supporting Evidence\n(For: {target_class_name})", 
                     fontsize=12, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im, ax=ax3, label='Pos. Contribution')
        
        # 4. Negative SHAP contributions
        ax4 = fig.add_subplot(gs[1, 0])
        if len(shap_values.shape) == 3:
            negative_shap = np.abs(np.minimum(np.mean(shap_values, axis=0), 0))
        else:
            negative_shap = np.abs(np.minimum(shap_values, 0))
        im = ax4.imshow(negative_shap, cmap='Reds')
        ax4.set_title("Contradicting Evidence\n(Against this class)", 
                     fontsize=12, fontweight='bold')
        ax4.axis('off')
        plt.colorbar(im, ax=ax4, label='Neg. Contribution')
        
        # 5. SHAP overlay on original image
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.imshow((original_image * 255).astype(np.uint8))
        im = ax5.imshow(shap_agg, cmap='RdBu_r', alpha=0.6)
        ax5.set_title("SHAP Overlay on Image", fontsize=12, fontweight='bold')
        ax5.axis('off')
        plt.colorbar(im, ax=ax5, label='SHAP Value')
        
        # 6. Top predictions
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_predictions(ax6, top_predictions, target_class_name)
        
        # Main title
        fig.suptitle(f"SHAP Explanation for Image Classification", 
                    fontsize=14, fontweight='bold', y=0.98)
        
        # Save figure
        output_path = self.output_dir / filename
        print(f"Saving SHAP explanation to {output_path}")
        fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
        
        return output_path
    
    def visualize_shap_heatmap(
        self,
        original_image: np.ndarray,
        shap_values: np.ndarray,
        filename: str,
        colormap: str = "RdBu_r"
    ) -> Path:
        """
        Create a simple SHAP heatmap visualization.
        
        Args:
            original_image: Original image array (H, W, 3)
            shap_values: SHAP values of shape (3, H, W)
            filename: Output filename
            colormap: Matplotlib colormap to use
            
        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Normalize image
        if original_image.max() > 1:
            original_image = original_image / 255.0
        
        # Aggregate SHAP values
        shap_agg = np.mean(shap_values, axis=0)
        
        # 1. Original
        axes[0].imshow((original_image * 255).astype(np.uint8))
        axes[0].set_title("Original Image", fontweight='bold')
        axes[0].axis('off')
        
        # 2. Heatmap
        im = axes[1].imshow(shap_agg, cmap=colormap)
        axes[1].set_title("SHAP Values Heatmap", fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        # 3. Overlay
        axes[2].imshow((original_image * 255).astype(np.uint8))
        im = axes[2].imshow(shap_agg, cmap=colormap, alpha=0.6)
        axes[2].set_title("SHAP Overlay", fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2])
        
        output_path = self.output_dir / filename
        fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
        
        return output_path
    
    def visualize_shap_decomposition(
        self,
        original_image: np.ndarray,
        shap_values: np.ndarray,
        baseline_value: float,
        prediction: float,
        filename: str
    ) -> Path:
        """
        Visualize SHAP decomposition showing prediction breakdown.
        
        Args:
            original_image: Original image
            shap_values: SHAP values
            baseline_value: Expected model output (baseline)
            prediction: Model's actual prediction
            filename: Output filename
            
        Returns:
            Path to saved figure
        """
        fig = plt.figure(figsize=(15, 8))
        gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
        
        # Normalize
        if original_image.max() > 1:
            original_image = original_image / 255.0
        
        # 1. Original image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow((original_image * 255).astype(np.uint8))
        ax1.set_title("Original Image", fontweight='bold')
        ax1.axis('off')
        
        # 2. Per-channel SHAP or overall magnitude
        ax2 = fig.add_subplot(gs[0, 1])
        if len(shap_values.shape) == 3:
            # 3D case: per-channel analysis
            shap_magnitude = np.sqrt(np.sum(shap_values ** 2, axis=(1, 2)))
            channels = ['Red', 'Green', 'Blue']
            ax2.bar(channels, shap_magnitude)
            ax2.set_ylabel('SHAP Magnitude')
            ax2.set_title('Per-Channel Attribution', fontweight='bold')
        else:
            # 2D case: show distribution of attribution values
            ax2.hist(shap_values.flatten(), bins=50, edgecolor='black', alpha=0.7)
            ax2.set_xlabel('Attribution Value')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Attribution Value Distribution', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. SHAP heatmap
        ax3 = fig.add_subplot(gs[1, 0])
        if len(shap_values.shape) == 3:
            shap_agg = np.mean(np.abs(shap_values), axis=0)
        else:
            shap_agg = np.abs(shap_values)
        im = ax3.imshow(shap_agg, cmap='hot')
        ax3.set_title('Attribution Magnitude Heatmap', fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im, ax=ax3)
        
        # 4. Prediction breakdown
        ax4 = fig.add_subplot(gs[1, 1])
        breakdown = [baseline_value, prediction - baseline_value, prediction]
        labels = [f'Baseline\n{baseline_value:.4f}', 
                 f'SHAP Effect\n{prediction - baseline_value:.4f}',
                 f'Prediction\n{prediction:.4f}']
        colors = ['lightblue', 'orange', 'green']
        ax4.bar(range(len(breakdown)), breakdown, color=colors)
        ax4.set_xticks(range(len(breakdown)))
        ax4.set_xticklabels(labels)
        ax4.set_ylabel('Value')
        ax4.set_title('Prediction Decomposition', fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        
        fig.suptitle('SHAP Decomposition Analysis', fontsize=14, fontweight='bold')
        
        output_path = self.output_dir / filename
        fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
        
        return output_path
    
    def _plot_predictions(
        self,
        ax,
        predictions: List[Tuple[str, float]],
        target_class_name: str
    ):
        """
        Plot bar chart of top predictions.
        
        Args:
            ax: Matplotlib axis
            predictions: List of (class_name, probability) tuples
            target_class_name: Name of explained class
        """
        names = [p[0][:30] for p in predictions]  # Truncate long names
        probs = [p[1] for p in predictions]
        
        # Color target class differently
        colors = ['#FF6B6B' if name == target_class_name else '#4ECDC4' 
                 for name in names]
        
        y_pos = np.arange(len(names))
        ax.barh(y_pos, probs, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel('Probability')
        ax.set_title('Top Predictions', fontweight='bold')
        ax.set_xlim(0, 1)
        
        # Add value labels
        for i, (prob, name) in enumerate(zip(probs, names)):
            ax.text(prob + 0.01, i, f'{prob:.3f}', va='center', fontsize=8)
