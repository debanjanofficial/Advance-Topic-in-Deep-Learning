# Task 2: SHAP Implementation

## Overview

This directory contains a complete,modularized implementation of **SHAP (SHapley Additive exPlanations)** for explaining image classification predictions using **ResNet50**.

Based on: *A Unified Approach to Interpreting Model Predictions* (Lundberg & Lee, 2017)

## Project Structure

```
task2_shap/
├── __init__.py                  # Package initialization
├── config.py                    # Configuration parameters
├── shap_model_manager.py       # ResNet50 model management
├── shap_image_utils.py         # Image loading and preprocessing
├── shap_implementation.py       # SHAP algorithm wrapper
├── shap_visualizer.py          # Visualization utilities
├── main.py                      # Main execution script
└── README.md                    # This file
```

## Module Descriptions

### `config.py`
Global configuration for paths, model parameters, and SHAP settings.

**Key Settings:**
- `IMAGE_SIZE`: 224x224 (ResNet50 requirement)
- `SHAP_BACKGROUND_SAMPLES`: 50 (background data for SHAP)
- `SHAP_EXPLAIN_SAMPLES`: 2048 (perturbations for SHAP)
- `DEVICE`: Automatically uses Metal GPU on M1 Mac

### `shap_image_utils.py`
Handles all image I/O and preprocessing for ResNet50.

**Classes:**
- `ShapImageProcessor`: Load, preprocess, and denormalize images
  - Handles ImageNet normalization
  - Creates background data batches for SHAP
  - Batch processing support

### `shap_model_manager.py`
Manages ResNet50 model loading and inference.

**Classes:**
- `ShapModelManager`: Model wrapper for ResNet50
  - Load pretrained ResNet50
  - Get predictions with class names
  - Automatic ImageNet label downloading

### `shap_implementation.py`
SHAP explanation generation using the SHAP library.

**Classes:**
- `ShapExplainer`: SHAP wrapper using Partition Explainer
  - Generates SHAP values for each pixel
  - Aggregates contributions across channels
  - Separates positive and negative contributions

**SHAP Algorithm (simplified):**
1. Use randomly selected background images as reference
2. Generate perturbed versions by masking image regions
3. Get model predictions for each perturbation
4. Compute Shapley values showing each pixel's contribution
5. Values indicate how much each pixel pushes prediction up or down

### `shap_visualizer.py`
Creates publication-quality visualizations.

**Functions:**
- `visualize_shap_explanation()`: Comprehensive 6-panel figure
  - Original image
  - SHAP magnitude heatmap
  - Supporting evidence (positive contributions)
  - Contradicting evidence (negative contributions)
  - SHAP overlay on image
  - Top predictions bar chart

- `visualize_shap_decomposition()`: Prediction breakdown
  - Per-channel contributions
  - SHAP heatmap
  - Prediction decomposition showing baseline + SHAP effect = prediction

### `main.py`
Orchestrates the complete SHAP analysis pipeline.

## Running the Analysis

### Basic Usage

```bash
# Activate the environment
cd /Users/admin/projects/Advance-Topic-in-Deep-Learning
source assignments-env/bin/activate

# Run analysis
cd Assignment1_Interpretability
python task2_shap/main.py
```

### Command Line Options

```bash
python task2_shap/main.py [OPTIONS]

Options:
  --background-samples INT    Number of background samples for SHAP 
                              (default: 50)
  --num-samples INT           Number of samples for SHAP estimation 
                              (default: 2048)
  --test-images STR [STR ...] Specific images to analyze
                              (default: all in Images/)
```

### Examples

```bash
# Run with custom background samples
python task2_shap/main.py --background-samples 100

# Analyze specific images only
python task2_shap/main.py --test-images Schloss-Erlangen02.JPG

# More samples for better SHAP estimates
python task2_shap/main.py --num-samples 4096
```

## Output

Visualizations are saved to `task2_output/` directory:

```
task2_output/
├── shap_explanation_<image>_class1.png
├── shap_explanation_<image>_class2.png
├── shap_decomposition_<image>_class1.png
├── shap_decomposition_<image>_class2.png
└── imagenet_labels.json
```

## Key Features

✓ **Uses SHAP Library**: Industry-standard implementation
✓ **Partition Explainer**: Efficient SHAP computation
✓ **ResNet50 Model**: Deep neural network for image classification
✓ **Metal GPU Support**: Optimized for M1 MacBook Pro
✓ **Comprehensive Visualization**: 6-panel + decomposition plots
✓ **Top 2 Predictions**: Analyzes both high confidence predictions
✓ **Background Data**: Uses image set for SHAP reference

## SHAP vs LIME Comparison

| Aspect | LIME | SHAP |
|--------|------|------|
| **Theoretical Base** | Perturbation → Linear regression | Game theory (Shapley values) |
| **Computation** | Faster (1000 samples) | Slower (2048+ samples) |
| **Consistency** | Local explanations | Global + Local |
| **Python Library** | Implement from scratch | Use library (faster) |
| **Interpretability** | Intuitive (linear fit) | Theoretically grounded |
| **Model Type** | Model-agnostic | Model-agnostic |

## What SHAP Explains

### Positive SHAP Values (Green)
- Pixels that **push prediction UP** toward the predicted class
- Example: castle architecture in palace detection
- Importance = magnitude of positive value

### Negative SHAP Values (Red)
- Pixels that **push prediction DOWN** away from predicted class
- Example: sky regions in palace detection
- Importance = magnitude of negative value

### SHAP Heatmap
- **Bright regions**: Important for prediction (strong influence)
- **Dark regions**: Less important (weak influence)
- Color indicates positive (blue) or negative (red) contribution

## Implementation Details

### Partition Explainer
- Efficient variant of SHAP for deep models
- Uses conditional expectation to compute Shapley values
- Faster than traditional SHAP but equally valid
- Good for image classification tasks

### Background Data
- Uses 50 random images as reference distribution
- SHAP computes how much each pixel deviates from reference
- Helps establish "expected" model behavior

### Per-Image Processing

```
For each image:
  1. Load and normalize (224x224)
  2. Get top-5 predictions from ResNet50
  3. For top-2 predictions:
     a. Generate SHAP values for all pixels
     b. Aggregate across RGB channels
     c. Visualize explanations
```

## GPU Performance

With Metal GPU on M1 MacBook Pro:
- Image loading: ~30ms
- Model prediction (1 image): ~15ms
- SHAP explanation (2048 samples): ~3-5 minutes
- Visualization: ~50ms

**Total time per image: ~5-7 minutes**

## Dependencies

- PyTorch 2.8.0
- torchvision 0.23.0
- SHAP 0.49.1 ✓ (already installed)
- NumPy 2.0.2
- Matplotlib 3.9.4
- scikit-learn 1.6.1

## Troubleshooting

### Out of Memory
- Reduce `--background-samples` to 25
- Reduce `--num-samples` to 1024
- Process one image at a time

### SHAP too slow
- Use fewer background samples
- Use Partition Explainer (default, already optimized)
- Process images sequentially

### Image not found
- Check exact filename in `Images/` folder
- Ensure case-sensitive filename match

## References

1. **SHAP Paper**: Lundberg, S.M., & Lee, S.I. (2017)
   *A Unified Approach to Interpreting Model Predictions*
   
2. **SHAP Library**: https://github.com/slundberg/shap

3. **ResNet50**: He, K., Zhang, X., Ren, S., & Sun, J. (2015)
   *Deep Residual Learning for Image Recognition*

4. **Shapley Values**: Shapley, L. S. (1953)
   *A value for n-person games*

## Code Reusability

Modules can be adapted for:
- Different models (VGG, EfficientNet, etc.)
- Different image datasets
- Other explainability methods
- Custom visualization styles

## What to Expect

### Per Image
- 2 comprehensive SHAP explanation figures
- 2 SHAP decomposition visualizations
- Detailed prediction analysis

### Total Output
- 3 images × 2 predictions × 2 visualizations = 12 PNG files
- ~6-8 MB of high-resolution visualizations

## Next Steps

After SHAP analysis:
1. Compare LIME and SHAP explanations
2. Analyze differences in interpretation
3. Document findings
4. Consider combining both methods for robust explanations
