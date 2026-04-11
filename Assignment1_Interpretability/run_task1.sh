#!/bin/bash
# Quick start script for Task 1: LIME Analysis
# Run this script from the Assignment1_Interpretability directory

cd "$(dirname "$0")"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║       Task 1: LIME Explanation Analysis - Quick Start         ║"
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

# Check if task1_lime exists
if [ ! -d "task1_lime" ]; then
    echo "✗ Error: task1_lime directory not found"
    exit 1
fi

# Run LIME analysis
echo "[2/4] Running LIME analysis..."
echo "      Images: All in Images/ folder"
echo "      Samples: 1000 perturbations"
echo "      Superpixels: 50 segments"
echo ""

python task1_lime/main.py

# Check if analysis was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                  ✓ Analysis Complete!                         ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Results saved to: task1_output/"
    echo ""
    echo "Generated files:"
    echo "  - lime_explanation_<image>_class[1,2].png     (explanation figures)"
    echo "  - superpixels_<image>_class[1,2].png          (top superpixels)"
    echo "  - imagenet_labels.json                        (class labels cache)"
    echo ""
    echo "Next steps:"
    echo "  1. View the generated PNG files"
    echo "  2. Check superpixel contribution analysis"
    echo "  3. Compare explained vs. contradicting evidence"
else
    echo ""
    echo "✗ Error during analysis. Check output above."
    exit 1
fi
