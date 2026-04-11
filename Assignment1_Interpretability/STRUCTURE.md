# Task 1: LIME Implementation - Project Structure

## Directory Layout

```
Assignment1_Interpretability/
├── Assignment1_Interpretability.ipynb          # Original notebook (reference)
├── Images/                                     # Test images
│   ├── Schloss-Erlangen02.JPG
│   ├── Erlangen_Blick_vom_Burgberg_auf_die_Innenstadt_2009_001.JPG
│   └── Alte-universitaets-bibliothek_universitaet-erlangen.jpg
│
├── task1_lime/                                 # TASK 1: LIME Implementation (MODULARIZED)
│   ├── __init__.py                            # Package initialization
│   ├── config.py                              # Configuration (paths, hyperparameters)
│   │
│   ├── model_manager.py                       # Inception V3 Model Wrapper
│   │   └── ModelManager class for loading/inference
│   │
│   ├── image_utils.py                         # Image Processing Utilities
│   │   └── ImageProcessor: load, preprocess, denormalize
│   │
│   ├── superpixel_utils.py                    # Superpixel Segmentation
│   │   └── SuperpixelSegmenter: SLIC, Felzenszwalb, Quickshift
│   │
│   ├── lime_implementation.py                 # CORE LIME ALGORITHM
│   │   └── LIMEExplainer: full LIME implementation
│   │
│   ├── visualizer.py                          # Visualization & Plotting
│   │   └── LIMEVisualizer: publication-quality figures
│   │
│   ├── main.py                                # MAIN EXECUTION SCRIPT
│   │   └── Orchestrates entire analysis pipeline
│   │
│   └── README.md                              # Detailed documentation
│
├── task1_output/                              # OUTPUT DIRECTORY (auto-created)
│   ├── lime_explanation_Schloss-Erlangen02_class1.png
│   ├── lime_explanation_Schloss-Erlangen02_class2.png
│   ├── superpixels_Schloss-Erlangen02_class1.png
│   ├── superpixels_Schloss-Erlangen02_class2.png
│   └── ... (same for other images)
│
├── run_task1.sh                               # Quick start script
├── STRUCTURE.md                               # This file
└── examples/                                  # (Optional) Usage examples

../
├── Assignment1_Interpretability.ipynb         # Original notebook
├── README.md                                  # Assignment description
├── STRUCTURE.md                               # Project structure
└── SETUP.md                                   # Environment setup guide
```

## Module Relationships

```
main.py (ORCHESTRATOR)
  │
  ├─→ config.py (GLOBAL SETTINGS)
  │
  ├─→ image_utils.py (IMAGES)
  │    └─ ImageProcessor.load_image()
  │       ImageProcessor.load_image_pil()
  │
  ├─→ model_manager.py (MODEL)
  │    └─ ModelManager.predict()
  │       ModelManager.get_probabilities()
  │
  ├─→ superpixel_utils.py (SEGMENTATION)
  │    └─ SuperpixelSegmenter.segment()
  │
  ├─→ lime_implementation.py (EXPLANATION)
  │    └─ LIMEExplainer.explain_instance()
  │
  └─→ visualizer.py (VISUALIZATION)
       └─ LIMEVisualizer.visualize_explanation()
           LIMEVisualizer.visualize_top_superpixels()
```

## Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│ ANALYZE IMAGE                                               │
│ main.py                                                     │
└───────────────────┬─────────────────────────────────────────┘
                    │
    ┌───────────────┴───────────────┐
    │ [1] Load Image                │
    │ ImageProcessor.load_image()   │
    │ Output: normalized tensor     │
    └───────────────┬───────────────┘
                    │
    ┌───────────────┴───────────────────────┐
    │ [2] Get Initial Predictions           │
    │ ModelManager.predict()                │
    │ Output: top 5 classes & probabilities │
    └───────────────┬───────────────────────┘
                    │
    ┌───────────────┴──────────────────────────┐
    │ [3] Generate Superpixels                 │
    │ SuperpixelSegmenter.segment()            │
    │ Output: segmentation map (H, W)          │
    └───────────────┬──────────────────────────┘
                    │
    ┌───────────────┴──────────────────────────────────┐
    │ [4] LIME Explanation (for each top-2 class)      │
    │ LIMEExplainer.explain_instance()                 │
    │                                                   │
    │ a) Generate 1000 perturbed samples               │
    │    └─ Randomly mask superpixels                  │
    │                                                   │
    │ b) Get predictions for all samples               │
    │    └─ ModelManager.get_probabilities()           │
    │                                                   │
    │ c) Compute distances & weights                   │
    │    └─ Exponential kernel weighting               │
    │                                                   │
    │ d) Fit weighted linear regression                │
    │    └─ Ridge regression: features = superpixels   │
    │                         targets = predictions    │
    │                                                   │
    │ Output: importance scores per superpixel         │
    └───────────────┬──────────────────────────────────┘
                    │
    ┌───────────────┴─────────────────────────────────┐
    │ [5] Visualization                               │
    │ LIMEVisualizer                                  │
    │                                                 │
    │ - 6-panel explanation figure                    │
    │ - Top contributing superpixels                  │
    │ - Supporting vs. contradicting evidence         │
    │                                                 │
    │ Output: PNG files in task1_output/              │
    └─────────────────────────────────────────────────┘
```

## File Sizes & Complexity

| File | Lines | Purpose | Complexity |
|------|-------|---------|-----------|
| config.py | ~45 | Global settings | ⭐ |
| image_utils.py | ~120 | Image I/O | ⭐⭐ |
| superpixel_utils.py | ~180 | Segmentation wrapper | ⭐⭐ |
| model_manager.py | ~130 | Model wrapper | ⭐⭐ |
| lime_implementation.py | ~280 | **CORE ALGORITHM** | ⭐⭐⭐⭐ |
| visualizer.py | ~350 | Visualization | ⭐⭐⭐ |
| main.py | ~200 | Orchestration | ⭐⭐ |
| **TOTAL** | **~1300** | | |

## Key Features

### ✓ Modular Design
- Each module has a single responsibility
- Easy to test, debug, and extend
- Reusable components for future assignments

### ✓ Comprehensive Comments
- Docstrings for all classes and methods
- Inline explanations for complex logic
- References to research paper

### ✓ Error Handling
- Input validation
- Graceful error messages
- Debugging information

### ✓ Performance Optimized
- Metal GPU support (M1 Mac)
- Batch processing where possible
- Progress bars for long operations

### ✓ Publication Quality Output
- High-DPI visualizations (100 DPI)
- Color-coded explanations (red/blue)
- Multiple visualization perspectives

## Running the Code

### Quick Start
```bash
./run_task1.sh
```

### Manual Run
```bash
cd /Users/admin/projects/Advance-Topic-in-Deep-Learning
source assignments-env/bin/activate
cd Assignment1_Interpretability
python task1_lime/main.py
```

### Advanced Options
```bash
# Use 500 samples instead of 1000
python task1_lime/main.py --num-samples 500

# Use Felzenszwalb segmentation
python task1_lime/main.py --segmentation-method felzenszwalb

# Analyze only specific images
python task1_lime/main.py --test-images Schloss-Erlangen02.JPG
```

## Extending the Code

### Adding a New Segmentation Method
1. Add method to `SuperpixelSegmenter._segment_xxx()`
2. Update `__init__` choices
3. Add to README documentation

### Adding Custom Visualization
1. Add method to `LIMEVisualizer` class
2. Call from `main.py` after explanation generation

### Using Different Model
1. Create `YourModelManager` class (copy `ModelManager`)
2. Override `_load_model()` and `get_probabilities()`
3. Pass to `LIMEExplainer` in `main.py`

## Output Interpretation

### LIME Explanation Heatmap
- **Blue regions**: Support the prediction (evidence FOR the class)
- **Red regions**: Contradict the prediction (evidence AGAINST)
- **Brighter colors**: Stronger influence on prediction

### Supporting Evidence
- Green highlights show which image regions cause the classifier to predict this class
- Importance score indicates strength of influence

### Contradicting Evidence
- Red highlights show which regions argue AGAINST this class
- The model is "distracted" by these features

### Top Superpixels
- Individual superpixels sorted by importance
- Positive (green) = supports prediction
- Negative (red) = contradicts prediction

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Image not found" | Check exact filename in `Images/` folder |
| Out of memory | Reduce `--num-samples` to 500 |
| Metal GPU not used | Check `torch.backends.mps.is_available()` |
| Slow inference | Use fewer `--num-samples` or fewer superpixels |

## Next Steps

After completing Task 1 (LIME):
- Move to `task2_shap/` for SHAP explanations
- Use same image set for comparison
- Can reuse `image_utils.py` and `model_manager.py`

## References

1. **LIME Paper**: Ribeiro et al. (2016) - "Why Should I Trust You?"
2. **PyTorch Docs**: https://pytorch.org/
3. **Superpixel Algorithms**: https://scikit-image.org/docs/stable/api/skimage.segmentation.html
4. **ImageNet Classes**: https://github.com/pytorch/hub
