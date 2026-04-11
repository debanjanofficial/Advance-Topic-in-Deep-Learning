#!/bin/bash
# Quick start script for Task 2: SHAP Analysis
# Run this script from the Assignment1_Interpretability directory

cd "$(dirname "$0")"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║       Task 2: SHAP Explanation Analysis - Quick Start         ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if assignments-env exists
if [ ! -d "../../assignments-env" ]; then
    echo "✗ Error: Python environment not found at ../../assignments-env"
    echo "  Please run setup from project root first"
    exit 1
fi

# Activate environment
echo "[1/4] Activating Python environment..."
source ../../assignments-env/bin/activate

# Check if task2_shap exists
if [ ! -d "task2_shap" ]; then
    echo "✗ Error: task2_shap directory not found"
    exit 1
fi

# Run SHAP analysis
echo "[2/4] Running SHAP analysis..."
echo "      Images: All in Images/ folder"
echo "      Background samples: 50"
echo "      SHAP samples: 2048"
echo ""

python task2_shap/main.py

# Check if analysis was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                  ✓ Analysis Complete!                         ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Results saved to: task2_output/"
    echo ""
    echo "Generated files:"
    echo "  - shap_explanation_<image>_class[1,2].png      (explanation figures)"
    echo "  - shap_decomposition_<image>_class[1,2].png    (decomposition plots)"
    echo "  - imagenet_labels.json                         (class labels cache)"
    echo ""
    echo "Next steps:"
    echo "  1. View the generated PNG files"
    echo "  2. Compare with LIME explanations"
    echo "  3. Analyze similarities and differences"
else
    echo ""
    echo "✗ Error during analysis. Check output above."
    exit 1
fi
