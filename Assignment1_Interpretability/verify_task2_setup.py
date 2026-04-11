#!/usr/bin/env python
"""
Quick verification script to check if task2_shap setup is correct.
"""

import sys
from pathlib import Path

def check_imports():
    """Check if required modules can be imported."""
    required = [
        'torch',
        'torchvision',
        'numpy',
        'matplotlib',
        'shap',
        'PIL',
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT FOUND")
            missing.append(package)
    
    return len(missing) == 0

def check_task2_modules():
    """Check if task2_shap modules exist."""
    task2_dir = Path(__file__).parent / "task2_shap"
    
    required_files = [
        "__init__.py",
        "config.py",
        "shap_model_manager.py",
        "shap_image_utils.py",
        "shap_implementation.py",
        "shap_visualizer.py",
        "main.py",
        "README.md",
    ]
    
    all_exist = True
    for filename in required_files:
        filepath = task2_dir / filename
        if filepath.exists():
            print(f"✓ {filename}")
        else:
            print(f"✗ {filename} - NOT FOUND")
            all_exist = False
    
    return all_exist

def main():
    print("\n" + "="*60)
    print("TASK 2: SHAP SETUP VERIFICATION")
    print("="*60)
    
    print("\n[1] Required Packages")
    print("-" * 30)
    check1 = check_imports()
    
    print("\n[2] Task 2 Modules")
    print("-" * 30)
    check2 = check_task2_modules()
    
    print("\n" + "="*60)
    if check1 and check2:
        print("✓ ALL CHECKS PASSED - Ready to run task2_shap/main.py!")
        print("\nRun: python task2_shap/main.py")
        print("="*60 + "\n")
        return 0
    else:
        print("✗ SOME CHECKS FAILED - See above for details")
        print("="*60 + "\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
