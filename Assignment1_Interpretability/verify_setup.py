#!/usr/bin/env python
"""
Quick verification script to check if all dependencies and setup are correct.
Run this before running main.py to catch any issues early.
"""

import sys
from pathlib import Path

def check_python():
    """Check Python version."""
    print("✓ Python version:", sys.version.split()[0])
    if sys.version_info < (3, 8):
        print("✗ Python 3.8+ required")
        return False
    return True

def check_dependencies():
    """Check if required packages are installed."""
    required = [
        'torch',
        'torchvision',
        'numpy',
        'matplotlib',
        'scipy',
        'sklearn',
        'skimage',
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

def check_gpu():
    """Check if Metal GPU is available."""
    try:
        import torch
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            print("✓ Metal GPU Available (M1 MacBook)")
            return True
        else:
            print("⚠ Metal GPU not available (will use CPU)")
            return True  # Still OK to run on CPU
    except Exception as e:
        print(f"⚠ Could not check GPU: {e}")
        return True

def check_directories():
    """Check if required directories exist."""
    dirs_to_check = {
        'task1_lime': 'LIME implementation module',
        'Images': 'Test images directory',
    }
    
    all_exist = True
    for dir_name, description in dirs_to_check.items():
        dir_path = Path(__file__).parent / dir_name
        if dir_path.exists():
            print(f"✓ {dir_name}/ ({description})")
        else:
            print(f"✗ {dir_name}/ - NOT FOUND ({description})")
            all_exist = False
    
    return all_exist

def check_images():
    """Check if test images exist."""
    images_dir = Path(__file__).parent / "Images"
    if not images_dir.exists():
        print("✗ Images directory not found")
        return False
    
    expected_images = [
        "Schloss-Erlangen02.JPG",
        "Erlangen_Blick_vom_Burgberg_auf_die_Innenstadt_2009_001.JPG",
        "Alte-universitaets-bibliothek_universitaet-erlangen.jpg",
    ]
    
    all_found = True
    for img in expected_images:
        if (images_dir / img).exists():
            print(f"✓ {img}")
        else:
            print(f"✗ {img} - NOT FOUND")
            all_found = False
    
    return all_found

def main():
    """Run all checks."""
    print("\n" + "="*60)
    print("TASK 1: LIME SETUP VERIFICATION")
    print("="*60)
    
    print("\n[1] Python Version")
    print("-" * 30)
    check1 = check_python()
    
    print("\n[2] Required Packages")
    print("-" * 30)
    check2 = check_dependencies()
    
    print("\n[3] GPU Support")
    print("-" * 30)
    check3 = check_gpu()
    
    print("\n[4] Directory Structure")
    print("-" * 30)
    check4 = check_directories()
    
    print("\n[5] Test Images")
    print("-" * 30)
    check5 = check_images()
    
    print("\n" + "="*60)
    if all([check1, check2, check3, check4, check5]):
        print("✓ ALL CHECKS PASSED - Ready to run main.py!")
        print("\nRun: python task1_lime/main.py")
        print("="*60 + "\n")
        return 0
    else:
        print("✗ SOME CHECKS FAILED - See above for details")
        print("="*60 + "\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
