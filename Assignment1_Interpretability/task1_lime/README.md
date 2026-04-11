# Task 1: LIME Implementation

## Overview

This directory contains a complete, modularized implementation of **LIME (Local Interpretable Model-agnostic Explanations)** for explaining image classification predictions using **Inception V3**.

Based on the paper: *"Why Should I Trust You?": Explaining the Predictions of Any Classifier* (Ribeiro et al., 2016)

## Project Structure

```
task1_lime/
├── __init__.py                     # Package initialization
├── config.py                       # Configuration parameters
├── model_manager.py               # Inception V3 model management
├── image_utils.py                 # Image loading and preprocessing
├── superpixel_utils.py           # Superpixel segmentation (SLIC/Felzenszwalb/Quickshift)
├── lime_implementation.py         # Core LIME algorithm
├── visualizer.py                  # Visualization utilities
├── main.py                        # Main execution script
└── README.md                      # This file
```

## Module Descriptions

### `config.py`
Global configuration for paths, model parameters, and LIME settings.

**Key Settings:**
- `IMAGE_SIZE`: 299x299 (Inception V3 requirement)
- `NUM_SAMPLES`: 1000 (perturbations for LIME)
- `NUM_SUPERPIXELS`: 50 (segmentation level)
- `DEVICE`: Automatically uses Metal GPU on M1 Mac
- `TEST_IMAGES`: List of test images in `Images/` folder

### `image_utils.py`
Handles all image I/O and preprocessing.

**Classes:**
- `ImageProcessor`: Load, preprocess, and denormalize images
  - Handles ImageNet normalization (mean/std)
  - Converts between PIL, numpy, and PyTorch formats
  - Batch processing support

### `superpixel_utils.py`
Generates superpixel segmentations for perturbation.

**Classes:**
- `SuperpixelSegmenter`: Multiple segmentation methods
  - **SLIC** (default): Fast, regular superpixels
  - **Felzenszwalb**: Content-aware segmentation
  - **Quickshift**: Good balance of speed/quality

**Why Superpixels?**
- LIME perturbs individual pixels → too noisy
- Superpixels group neighboring pixels → meaningful features
- Much fewer features (50 vs. millions of pixels)

### `model_manager.py`
Manages Inception V3 model loading and inference.

**Classes:**
- `ModelManager`: Model wrapper
  - Load pretrained Inception V3
  - Get predictions with class names
  - Automatic ImageNet label downloading

### `lime_implementation.py`
Core LIME algorithm implementation.

**Classes:**
- `LIMEExplainer`: Main LIME implementation
  - Generate perturbed samples by masking superpixels
  - Get model predictions on perturbations
  - Fit weighted linear regression
  - Return interpretable coefficients

**LIME Algorithm:**
1. Generate N perturbed samples (random superpixel masking)
2. Get model predictions for all N samples
3. Compute distance between original and perturbed
4. Fit weighted Ridge regression:
   - Features: superpixel presence (0/1)
   - Target: model predictions
   - Weights: kernel distance weights
5. Interpret coefficients as feature importance

### `visualizer.py`
Creates publication-quality visualizations.

**Functions:**
- `visualize_explanation()`: Comprehensive 6-panel figure
  - Original image
  - Superpixel segmentation
  - Explanation heatmap
  - Supporting evidence (positive)
  - Contradicting evidence (negative)
  - Top predictions bar chart

- `visualize_top_superpixels()`: Individual superpixel importance
- `create_comparison_figure()`: Side-by-side top 2 predictions

### `main.py`
Orchestrates the complete analysis pipeline.

## Running the Analysis

### Basic Usage

```bash
# Activate the environment
cd /Users/admin/projects/Advance-Topic-in-Deep-Learning
source assignments-env/bin/activate

# Run analysis with defaults
cd Assignment1_Interpretability
python task1_lime/main.py
```

### Command Line Options

```bash
python task1_lime/main.py [OPTIONS]

Options:
  --num-samples INT              Number of perturbed samples (default: 1000)
  --num-superpixels INT          Number of superpixels (default: 50)
  --segmentation-method STR      Segmentation method: slic, felzenszwalb, 
                                 quickshift (default: slic)
  --test-images STR [STR ...]    Specific images to analyze
                                 (default: all in Images/)
```

### Examples

```bash
# Run with custom number of samples and superpixels
python task1_lime/main.py --num-samples 500 --num-superpixels 30

# Use Felzenszwalb segmentation
python task1_lime/main.py --segmentation-method felzenszwalb

# Analyze specific images only
python task1_lime/main.py --test-images Schloss-Erlangen02.JPG
```

## Output

Visualizations are saved to `task1_output/` directory:

```
task1_output/
├── lime_explanation_<image>_class1.png    # Explanation for top prediction
├── lime_explanation_<image>_class2.png    # Explanation for 2nd prediction
├── superpixels_<image>_class1.png        # Top contributing superpixels
├── superpixels_<image>_class2.png        # Top contributing superpixels
└── imagenet_labels.json                  # Downloaded class labels
```

## Key Features

✓ **Modularized & Clean Code**: Separate concerns across modules
✓ **Metal GPU Support**: Optimized for M1 MacBook Pro
✓ **Superpixel Segmentation**: 3 methods available
✓ **Comprehensive Visualization**: 6-panel + individual superpixel plots
✓ **No LIME Library**: Pure implementation from research paper
✓ **Inception V3**: State-of-the-art image classification model
✓ **Top 2 Predictions**: Analyzes both high confidence predictions

## Implementation Details

### LIME Algorithm Steps

1. **Perturbation Generation**
   ```
   For each of N iterations:
     - Randomly select which superpixels to keep (50% probability)
     - Replace masked superpixels with gray background (0.5, 0.5, 0.5)
     - Store perturbation as binary vector (1 = present, 0 = masked)
   ```

2. **Distance Computation**
   ```
   distance = ||perturbed_mask - original_mask||_2
   ```

3. **Kernel Weighting**
   ```
   weight = exp(-(distance^2) / (2 * kernel_width^2))
   Kernel width = 0.25 (controls locality)
   ```

4. **Weighted Linear Regression**
   ```
   minimize: Σ weight_i * (y_i - (intercept + Σ coef_j * x_ij))^2 + α * ||coef||_2
   
   Where:
   - y_i: model prediction for perturbed sample i
   - x_ij: superpixel j presence in perturbation i
   - weight_i: kernel weight for sample i
   - α: L2 regularization strength (ridge)
   ```

5. **Interpretation**
   ```
   coef_j > 0: Superpixel j supports the prediction (evidence FOR the class)
   coef_j < 0: Superpixel j contradicts the prediction (evidence AGAINST)
   |coef_j|:   Importance of superpixel j
   ```

## GPU Performance

With Metal GPU on M1 MacBook Pro:
- Image loading: ~50ms
- Model prediction (1 image): ~20ms
- LIME explanation (1000 samples): ~2-3 minutes
- Visualization: ~100ms

**Total time per image: ~3-4 minutes**

## Troubleshooting

### Metal GPU not detected
```python
import torch
print(torch.backends.mps.is_available())  # Should be True
print(torch.backends.mps.is_built())      # Should be True
```

### Out of memory (Metal GPU)
- Reduce `--num-samples` to 500
- Reduce `--num-superpixels` to 30
- Process images individually

### Image not found
- Ensure images are in `Images/` folder at parent level
- Check exact filename spelling

## References

1. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). 
   "Why Should I Trust You?": Explaining the Predictions of Any Classifier. 
   *arXiv preprint arXiv:1602.04938*

2. PyTorch Inception V3:
   https://pytorch.org/vision/stable/models.html

3. Superpixel Methods:
   - SLIC: Achanta et al. (2012)
   - Felzenszwalb: Felzenszwalb & Huttenlocher (2004)
   - Quickshift: Vedaldi & Soatto (2008)

## Author Notes

This implementation focuses on:
- Clear, readable code with extensive comments
- Proper separation of concerns (modular design)
- Comprehensive visualization for interpretability
- Efficient GPU utilization for expensive operations
