# Assignment Environment Setup

## Overview
A global Python virtual environment has been created for all assignments in this course.

## Environment Details
- **Location**: `assignments-env/` (at project root)
- **Python Version**: 3.9.6
- **Framework**: PyTorch 2.8.0 with Metal GPU support (M1 Mac)
- **GPU**: Metal Performance Shaders (MPS) enabled for Apple Silicon

## Activated GPU Support
```
✓ PyTorch Metal Available: True
✓ PyTorch Metal Built: True
```

## Activating the Environment

### For new terminal sessions:
```bash
cd /Users/admin/projects/Advance-Topic-in-Deep-Learning
source assignments-env/bin/activate
```

### Verify activation:
```bash
which python  # Should point to .../assignments-env/bin/python
python --version  # Should show Python 3.9.6
```

## Installed Packages

### Core Deep Learning
- **torch** 2.8.0 - PyTorch with Metal GPU support
- **torchvision** 0.23.0 - Computer vision utilities
- **torchaudio** 2.8.0 - Audio processing

### Model Interpretability
- **shap** 0.49.1 - SHapley Additive exPlanations
- **scikit-image** 0.24.0 - Image processing (superpixels for LIME)
- **scikit-learn** 1.6.1 - Machine learning utilities

### Data & Visualization
- **numpy** 2.0.2
- **matplotlib** 3.9.4
- **scipy** 1.13.1
- **pillow** 11.3.0 - Image handling
- **pandas** 2.3.3

### Jupyter
- **jupyter** 1.1.1
- **jupyterlab** 4.5.6
- **jupyter** 1.1.1 - Notebook support

## Using Metal GPU in PyTorch

In your code, you can use the Metal device:

```python
import torch

# Check if Metal is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Metal GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Move models and tensors to device
model = model.to(device)
inputs = inputs.to(device)
```

## Assignment 1: Interpretability

### Project Structure
```
Assignment1_Interpretability/
├── Assignment1_Interpretability.ipynb  # Main notebook
├── Images/
│   ├── Schloss-Erlangen02.JPG
│   ├── Erlangen_Blick_vom_Burgberg_auf_die_Innenstadt_2009_001.JPG
│   └── Alte-universitaets-bibliothek_universitaet-erlangen.jpg
└── [ your implementation files ]
```

### Tasks
1. **Implement LIME** (Local Interpretable Model-agnostic Explanations)
   - Use Inception V3 pretrained model
   - Analyze top 1 and top 2 predictions
   - Can use superpixels for segmentation
   - Test on 3 images in `Images/` folder

2. **Test SHAP** (SHapley Additive exPlanations)
   - Use existing SHAP library
   - Use ResNet 50 or Inception V3
   - Visual explanations for same images
   - Can use TensorFlow or PyTorch

## Running Assignment 1

```bash
# Activate environment
cd /Users/admin/projects/Advance-Topic-in-Deep-Learning
source assignments-env/bin/activate

# Start Jupyter Lab
jupyter lab

# Or Jupyter Notebook
jupyter notebook
```

## Installing Additional Packages

If you need additional packages for future assignments:

```bash
source assignments-env/bin/activate
pip install package_name

# Update requirements.txt
pip freeze > requirements.txt
```

## Notes for M1 Mac Users
- Metal GPU acceleration is automatically used by PyTorch when available
- CPU offloading happens transparently if needed
- For optimal performance, use `torch.backends.mps.is_available()` to check before using `.to("mps")`
- Some operations may fall back to CPU if not supported on Metal

## Troubleshooting

### If Metal is not available:
1. Ensure you're using the PyTorch arm64 build
2. Check: `python -c "import torch; print(torch.backends.mps.is_built())"`
3. Update pip: `pip install --upgrade pip`

### To deactivate environment:
```bash
deactivate
```
