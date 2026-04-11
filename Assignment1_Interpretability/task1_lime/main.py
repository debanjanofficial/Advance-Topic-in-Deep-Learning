"""
Main script for Task 1: LIME Explanation Analysis

This script:
1. Loads images from the Images directory
2. Gets top predictions from Inception V3
3. Generates LIME explanations for top 2 predictions
4. Visualizes the explanations
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import TEST_IMAGES, IMAGES_DIR, OUTPUT_DIR, DEVICE, NUM_SUPERPIXELS
from image_utils import ImageProcessor
from model_manager import ModelManager
from superpixel_utils import SuperpixelSegmenter
from lime_implementation import LIMEExplainer
from visualizer import LIMEVisualizer


def main(args):
    """
    Main execution function.
    
    Args:
        args: Command line arguments
    """
    print("=" * 80)
    print("TASK 1: LIME EXPLANATION ANALYSIS")
    print("=" * 80)
    
    # Initialize components
    print("\n[1/5] Initializing components...")
    image_processor = ImageProcessor()
    model_manager = ModelManager()
    superpixel_segmenter = SuperpixelSegmenter(method=args.segmentation_method, 
                                              num_segments=args.num_superpixels)
    visualizer = LIMEVisualizer()
    
    # Create LIME explainer
    lime_explainer = LIMEExplainer(
        prediction_fn=model_manager.get_probabilities,
        num_samples=args.num_samples
    )
    
    # Get test images
    test_images = args.test_images if args.test_images else TEST_IMAGES
    print(f"Will analyze {len(test_images)} images")
    
    # Process each image
    for img_idx, image_name in enumerate(test_images):
        print(f"\n{'='*80}")
        print(f"Processing image {img_idx + 1}/{len(test_images)}: {image_name}")
        print(f"{'='*80}")
        
        image_path = image_processor.get_image_path(image_name)
        
        # Load image
        print(f"\n[2/5] Loading image...")
        image_pil, image_tensor = image_processor.load_image_pil(image_path)
        image_array = image_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        
        # Denormalize for processing
        from torchvision.transforms.functional import normalize
        image_denorm = image_array.clone() if isinstance(image_array, torch.Tensor) else image_array.copy()
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        image_denorm = image_denorm * std + mean
        image_denorm = np.clip(image_denorm, 0, 1)
        
        # Get initial predictions
        print(f"[2/5] Getting model predictions...")
        model_image = image_processor.load_image(image_path)
        logits, top_classes, top_probs = model_manager.predict(model_image, top_k=5)
        
        print(f"\nTop 5 Predictions:")
        for cls, prob in zip(top_classes, top_probs):
            print(f"  {cls}: {prob:.4f}")
        
        # Generate superpixel segmentation
        print(f"\n[3/5] Generating superpixel segmentation...")
        segments, num_segments = superpixel_segmenter.segment(image_pil)
        print(f"Generated {num_segments} superpixels using {args.segmentation_method}")
        
        # Explain top 2 predictions
        explanations_data = {}
        fig_data = {
            'original_image': image_denorm,
            'top_predictions': [(c, p) for c, p in zip(top_classes, top_probs)],
            'segments': segments
        }
        
        for pred_idx in range(min(2, len(top_classes))):
            target_class_name = top_classes[pred_idx]
            print(f"\n[4/5] Explaining prediction {pred_idx + 1}: {target_class_name}")
            
            # Get class index
            try:
                target_class_idx = model_manager.get_class_index(target_class_name)
            except ValueError:
                # If exact match not found, use the prediction index
                logits_all = model_manager.get_logits(model_image)
                target_class_idx = np.argsort(logits_all[0])[::-1][pred_idx]
            
            # Generate LIME explanation
            print(f"Generating LIME explanation with {args.num_samples} samples...")
            coefficients, predictions, sp_ids, weights = lime_explainer.explain_instance(
                image=image_denorm,
                segments=segments,
                target_class=target_class_idx,
                num_samples=args.num_samples,
                device=DEVICE
            )
            
            # Get explanation masks
            explanation_mask, important_sps = lime_explainer.get_explanation_mask(
                segments, coefficients, top_k=10
            )
            
            explanations_data[target_class_name] = {
                'coefficients': coefficients,
                'explanation_mask': explanation_mask,
                'class_idx': target_class_idx,
                'important_superpixels': important_sps
            }
            
            # Save explanation details
            print(f"Top contributing superpixels:")
            for sp_id in important_sps[:5]:
                coef = coefficients[sp_id]
                direction = "↑ (supporting)" if coef > 0 else "↓ (contradicting)"
                print(f"  Superpixel {sp_id}: {coef:+.4f} {direction}")
        
        # Visualization
        print(f"\n[5/5] Creating visualizations...")
        
        # Create individual explanations for each top class
        for pred_idx, (target_class_name, expl_data) in enumerate(explanations_data.items()):
            output_filename = f"lime_explanation_{Path(image_name).stem}_class{pred_idx+1}.png"
            
            visualizer.visualize_explanation(
                original_image=image_denorm,
                segments=segments,
                explanation_mask=expl_data['explanation_mask'],
                top_predictions=fig_data['top_predictions'],
                target_class_idx=expl_data['class_idx'],
                target_class_name=target_class_name,
                filename=output_filename,
                important_superpixels=expl_data['important_superpixels']
            )
            
            # Create superpixel visualization
            sp_filename = f"superpixels_{Path(image_name).stem}_class{pred_idx+1}.png"
            visualizer.visualize_top_superpixels(
                original_image=image_denorm,
                segments=segments,
                explanation_coefficients=expl_data['coefficients'],
                top_k=5,
                filename=sp_filename
            )
        
        print(f"\n✓ Completed analysis for {image_name}")
        print(f"  Visualizations saved to: {OUTPUT_DIR}")
    
    print(f"\n{'='*80}")
    print("✓ All analyses complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LIME Explanation Analysis for Image Classification"
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of perturbed samples for LIME (default: 1000)"
    )
    
    parser.add_argument(
        "--num-superpixels",
        type=int,
        default=NUM_SUPERPIXELS,
        help=f"Number of superpixels (default: {NUM_SUPERPIXELS})"
    )
    
    parser.add_argument(
        "--segmentation-method",
        type=str,
        default="slic",
        choices=["slic", "felzenszwalb", "quickshift"],
        help="Superpixel segmentation method (default: slic)"
    )
    
    parser.add_argument(
        "--test-images",
        type=str,
        nargs="+",
        help="List of test image names (default: all images in Images folder)"
    )
    
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
