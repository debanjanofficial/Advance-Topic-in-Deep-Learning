"""
Visualization utilities for LIME explanations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional

from config import OUTPUT_DIR, FIGSIZE, DPI


class LIMEVisualizer:
    """Handles visualization of LIME explanations."""
    
    def __init__(self, output_dir: Path = OUTPUT_DIR):
        """
        Initialize LIMEVisualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def visualize_explanation(
        self,
        original_image: np.ndarray,
        segments: np.ndarray,
        explanation_mask: np.ndarray,
        top_predictions: List[Tuple[str, float]],
        target_class_idx: int,
        target_class_name: str,
        filename: str,
        important_superpixels: List[int] = None
    ) -> Path:
        """
        Create comprehensive visualization of LIME explanation.
        
        Args:
            original_image: Original image array (H, W, 3) with values in [0, 1]
            segments: Superpixel segmentation map
            explanation_mask: Importance scores for each pixel
            top_predictions: List of (class_name, probability) tuples
            target_class_idx: Index of the explained class
            target_class_name: Name of the explained class
            filename: Output filename
            important_superpixels: List of important superpixel IDs
            
        Returns:
            Path to saved figure
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Original image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow((original_image * 255).astype(np.uint8))
        ax1.set_title("Original Image", fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # 2. Superpixel segmentation
        ax2 = fig.add_subplot(gs[0, 1])
        segmentation_colored = self._colorize_segments(segments)
        ax2.imshow(segmentation_colored)
        ax2.set_title("Superpixel Segmentation", fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # 3. Explanation heatmap
        ax3 = fig.add_subplot(gs[0, 2])
        im = ax3.imshow(explanation_mask, cmap='RdBu_r')
        ax3.set_title(f"LIME Explanation\n(For: {target_class_name})", 
                     fontsize=12, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im, ax=ax3, label='Importance')
        
        # 4. Positive contributions
        ax4 = fig.add_subplot(gs[1, 0])
        positive_mask = explanation_mask.copy()
        positive_mask[positive_mask < 0] = 0
        ax4.imshow((original_image * 255).astype(np.uint8))
        im = ax4.imshow(positive_mask, cmap='Greens', alpha=0.6)
        ax4.set_title("Supporting Evidence\n(Positive Contributions)", 
                     fontsize=12, fontweight='bold')
        ax4.axis('off')
        plt.colorbar(im, ax=ax4, label='Importance')
        
        # 5. Negative contributions
        ax5 = fig.add_subplot(gs[1, 1])
        negative_mask = explanation_mask.copy()
        negative_mask[negative_mask > 0] = 0
        negative_mask = np.abs(negative_mask)
        ax5.imshow((original_image * 255).astype(np.uint8))
        im = ax5.imshow(negative_mask, cmap='Reds', alpha=0.6)
        ax5.set_title("Contradicting Evidence\n(Negative Contributions)", 
                     fontsize=12, fontweight='bold')
        ax5.axis('off')
        plt.colorbar(im, ax=ax5, label='Importance')
        
        # 6. Top predictions
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_predictions(ax6, top_predictions, target_class_idx, target_class_name)
        
        # Main title
        fig.suptitle(f"LIME Explanation for Image Classification", 
                    fontsize=14, fontweight='bold', y=0.98)
        
        # Save figure
        output_path = self.output_dir / filename
        print(f"Saving explanation to {output_path}")
        fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
        
        return output_path
    
    def visualize_top_superpixels(
        self,
        original_image: np.ndarray,
        segments: np.ndarray,
        explanation_coefficients: np.ndarray,
        top_k: int = 5,
        filename: str = "top_superpixels.png"
    ) -> Path:
        """
        Visualize top-k contributing superpixels.
        
        Args:
            original_image: Original image array (H, W, 3)
            segments: Superpixel segmentation map
            explanation_coefficients: Importance scores for each superpixel
            top_k: Number of top superpixels to show
            filename: Output filename
            
        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(2, (top_k + 1) // 2, figsize=(20, 8))
        axes = axes.flatten()
        
        # Show original
        axes[0].imshow((original_image * 255).astype(np.uint8))
        axes[0].set_title("Original Image", fontweight='bold')
        axes[0].axis('off')
        
        # Get top superpixels
        top_indices = np.argsort(np.abs(explanation_coefficients))[-top_k:][::-1]
        
        # Show each superpixel
        for idx, (ax, sp_id) in enumerate(zip(axes[1:len(top_indices)+1], top_indices)):
            # Create image showing only this superpixel
            sp_image = self._highlight_superpixel(original_image, segments, sp_id)
            ax.imshow((sp_image * 255).astype(np.uint8))
            
            coef = explanation_coefficients[sp_id]
            color = 'green' if coef > 0 else 'red'
            ax.set_title(f"SP {sp_id}\n(importance: {coef:.3f})", 
                        fontweight='bold', color=color)
            ax.axis('off')
        
        # Hide unused subplots
        for ax in axes[len(top_indices)+1:]:
            ax.axis('off')
        
        fig.suptitle(f"Top {top_k} Contributing Superpixels", 
                    fontsize=14, fontweight='bold')
        
        output_path = self.output_dir / filename
        fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
        
        return output_path
    
    def _colorize_segments(self, segments: np.ndarray) -> np.ndarray:
        """
        Create colored visualization of superpixel segments.
        
        Args:
            segments: Segmentation map
            
        Returns:
            RGB image with different colors for each superpixel
        """
        num_segments = len(np.unique(segments))
        
        # Create random colors for each segment
        colors = np.random.randint(0, 256, (num_segments, 3), dtype=np.uint8)
        
        # Map segments to colors
        colored = colors[segments]
        return colored.astype(float) / 255.0
    
    def _highlight_superpixel(
        self,
        image: np.ndarray,
        segments: np.ndarray,
        superpixel_id: int,
        background_intensity: float = 0.3
    ) -> np.ndarray:
        """
        Highlight a single superpixel while dimming the rest.
        
        Args:
            image: Original image
            segments: Segmentation map
            superpixel_id: ID of superpixel to highlight
            background_intensity: Brightness of background (0-1)
            
        Returns:
            Image with highlighted superpixel
        """
        highlighted = image.copy()
        
        # Darken background
        mask = segments != superpixel_id
        highlighted[mask] = highlighted[mask] * background_intensity
        
        return highlighted
    
    def _plot_predictions(
        self,
        ax,
        predictions: List[Tuple[str, float]],
        target_class_idx: int,
        target_class_name: str
    ):
        """
        Plot bar chart of top predictions.
        
        Args:
            ax: Matplotlib axis
            predictions: List of (class_name, probability) tuples
            target_class_idx: Index of explained class
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
    
    def create_comparison_figure(
        self,
        original_image: np.ndarray,
        explanations: dict,
        image_name: str
    ) -> Path:
        """
        Create side-by-side comparison of explanations for top 2 predictions.
        
        Args:
            original_image: Original image array
            explanations: Dict with keys being class names and values being explanation masks
            image_name: Name of the image for the title
            
        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(1, 1 + len(explanations), figsize=(18, 6))
        
        # Original image
        axes[0].imshow((original_image * 255).astype(np.uint8))
        axes[0].set_title("Original Image", fontweight='bold')
        axes[0].axis('off')
        
        # Explanations for each class
        for ax, (class_name, mask) in zip(axes[1:], explanations.items()):
            ax.imshow((original_image * 255).astype(np.uint8))
            im = ax.imshow(mask, cmap='RdBu_r', alpha=0.7)
            ax.set_title(f"LIME for\n{class_name[:30]}", fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        fig.suptitle(f"LIME Explanations - {image_name}", 
                    fontsize=14, fontweight='bold')
        
        output_path = self.output_dir / f"comparison_{image_name}.png"
        fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
        
        return output_path
