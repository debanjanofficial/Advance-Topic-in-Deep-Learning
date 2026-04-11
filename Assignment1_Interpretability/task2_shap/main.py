"""
Main script for Task 2: SHAP Explanation Analysis

This script:
1. Loads images from the Images directory
2. Gets predictions from ResNet50
3. Generates SHAP explanations for top 2 predictions
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

from config import TEST_IMAGES, IMAGES_DIR, OUTPUT_DIR, DEVICE, SHAP_BACKGROUND_SAMPLES
from shap_image_utils import ShapImageProcessor, denormalize_image
from shap_model_manager import ShapModelManager
from shap_implementation import ShapExplainer
from shap_visualizer import ShapVisualizer


def create_background_data(
    image_processor: ShapImageProcessor,
    num_samples: int = SHAP_BACKGROUND_SAMPLES
) -> np.ndarray:
    """
    Create background data for SHAP from test images.
    
    Args:
        image_processor: Image processor instance
        num_samples: Number of background samples
        
    Returns:
        Background data array of shape (num_samples, 3, 224, 224)
    """
    print(f"\n[Preparing] Creating {num_samples} background samples...")
    
    background_images = []
    image_files = list(IMAGES_DIR.glob("*.JPG")) + list(IMAGES_DIR.glob("*.jpg"))
    
    for i in range(num_samples):
        img_path = image_files[i % len(image_files)]
        _, image_array = image_processor.load_image_pil(img_path)
        
        # Denormalize for SHAP
        image_denorm = denormalize_image(image_array)
        background_images.append(image_denorm)
    
    background_data = np.array(background_images)
    print(f"Background data shape: {background_data.shape}")
    
    return background_data


def main(args):
    """
    Main execution function for SHAP analysis.
    
    Args:
        args: Command line arguments
    """
    print("=" * 80)
    print("TASK 2: SHAP EXPLANATION ANALYSIS")
    print("=" * 80)
    
    # Initialize components
    print("\n[1/5] Initializing components...")
    image_processor = ShapImageProcessor()
    model_manager = ShapModelManager()
    visualizer = ShapVisualizer()
    
    # Create background data for SHAP
    background_data = create_background_data(image_processor, args.background_samples)
    
    # Create SHAP explainer
    print("[Preparing] Initializing SHAP Partition Explainer...")
    explainer = ShapExplainer(
        prediction_fn=model_manager.get_probabilities,
        background_data=background_data,
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
        image_denorm = denormalize_image(image_tensor)
        
        # Get initial predictions
        print(f"[2/5] Getting model predictions...")
        model_image = image_processor.load_image(image_path)
        logits, top_classes, top_probs = model_manager.predict(model_image, top_k=5)
        
        print(f"\nTop 5 Predictions:")
        for cls, prob in zip(top_classes, top_probs):
            print(f"  {cls}: {prob:.4f}")
        
        # Explain top 2 predictions
        explanations_data = {}
        fig_data = {
            'original_image': image_denorm,
            'top_predictions': [(c, p) for c, p in zip(top_classes, top_probs)],
        }
        
        for pred_idx in range(min(2, len(top_classes))):
            target_class_name = top_classes[pred_idx]
            print(f"\n[3/5] Explaining prediction {pred_idx + 1}: {target_class_name}")
            
            # Get class index
            try:
                target_class_idx = model_manager.get_class_index(target_class_name)
            except ValueError:
                # If exact match not found, use direct index
                logits_all = model_manager.get_logits(model_image)
                target_class_idx = np.argsort(logits_all[0])[::-1][pred_idx]
            
            # Generate SHAP explanation
            print(f"Generating SHAP explanation...")
            shap_values, baseline_value, prediction = explainer.explain_instance(
                image=image_tensor.squeeze(0),
                target_class=target_class_idx
            )
            
            # Handle output format
            if isinstance(shap_values, list):
                # DeepExplainer returns list
                shap_values = shap_values[target_class_idx]
            
            explanations_data[target_class_name] = {
                'shap_values': shap_values,
                'baseline_value': baseline_value if isinstance(baseline_value, (int, float)) else baseline_value[target_class_idx],
                'prediction': prediction[0, target_class_idx] if prediction.ndim > 1 else prediction[target_class_idx],
                'class_idx': target_class_idx
            }
            
            print(f"Prediction: {explanations_data[target_class_name]['prediction']:.4f}")
        
        # Visualization
        print(f"\n[4/5] Creating visualizations...")
        
        # Create individual explanations for each top class
        for pred_idx, (target_class_name, expl_data) in enumerate(explanations_data.items()):
            output_filename = f"shap_explanation_{Path(image_name).stem}_class{pred_idx+1}.png"
            
            visualizer.visualize_shap_explanation(
                original_image=image_denorm,
                shap_values=expl_data['shap_values'],
                top_predictions=fig_data['top_predictions'],
                target_class_name=target_class_name,
                filename=output_filename
            )
            
            # Create decomposition visualization
            decomp_filename = f"shap_decomposition_{Path(image_name).stem}_class{pred_idx+1}.png"
            visualizer.visualize_shap_decomposition(
                original_image=image_denorm,
                shap_values=expl_data['shap_values'],
                baseline_value=expl_data['baseline_value'],
                prediction=expl_data['prediction'],
                filename=decomp_filename
            )
        
        print(f"\n✓ Completed SHAP analysis for {image_name}")
        print(f"  Visualizations saved to: {OUTPUT_DIR}")
    
    print(f"\n{'='*80}")
    print("✓ All SHAP analyses complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SHAP Explanation Analysis for Image Classification"
    )
    
    parser.add_argument(
        "--background-samples",
        type=int,
        default=SHAP_BACKGROUND_SAMPLES,
        help=f"Number of background samples for SHAP (default: {SHAP_BACKGROUND_SAMPLES})"
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2048,
        help="Number of samples for SHAP estimation (default: 2048)"
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
