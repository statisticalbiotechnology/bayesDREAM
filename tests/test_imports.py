#!/usr/bin/env python
"""
Test script to verify all required dependencies can be imported.
Run this after installing the conda environment to verify everything works.
"""

import sys

def test_imports():
    """Test all required imports."""
    errors = []

    print("Testing core scientific computing packages...")
    try:
        import numpy as np
        print(f"✓ numpy {np.__version__}")
    except ImportError as e:
        errors.append(f"✗ numpy: {e}")

    try:
        import scipy
        print(f"✓ scipy {scipy.__version__}")
    except ImportError as e:
        errors.append(f"✗ scipy: {e}")

    try:
        import pandas as pd
        print(f"✓ pandas {pd.__version__}")
    except ImportError as e:
        errors.append(f"✗ pandas: {e}")

    try:
        import sklearn
        print(f"✓ scikit-learn {sklearn.__version__}")
        from sklearn.preprocessing import SplineTransformer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        print("  ✓ SplineTransformer, Ridge, make_pipeline")
    except ImportError as e:
        errors.append(f"✗ scikit-learn: {e}")

    print("\nTesting deep learning packages...")
    try:
        import torch
        print(f"✓ torch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        errors.append(f"✗ torch: {e}")

    try:
        import pyro
        print(f"✓ pyro-ppl {pyro.__version__}")
    except ImportError as e:
        errors.append(f"✗ pyro-ppl: {e}")

    print("\nTesting visualization packages...")
    try:
        import matplotlib
        print(f"✓ matplotlib {matplotlib.__version__}")
    except ImportError as e:
        errors.append(f"✗ matplotlib: {e}")

    try:
        import seaborn
        print(f"✓ seaborn {seaborn.__version__}")
    except ImportError as e:
        errors.append(f"✗ seaborn: {e}")

    print("\nTesting data I/O packages...")
    try:
        import h5py
        print(f"✓ h5py {h5py.__version__}")
    except ImportError as e:
        errors.append(f"✗ h5py: {e}")

    print("\nTesting Jupyter packages (optional)...")
    try:
        import jupyterlab
        print(f"✓ jupyterlab")
    except ImportError as e:
        print(f"  (optional) jupyterlab: {e}")

    try:
        import ipywidgets
        print(f"✓ ipywidgets {ipywidgets.__version__}")
    except ImportError as e:
        print(f"  (optional) ipywidgets: {e}")

    print("\nTesting bayesDREAM package...")
    try:
        from bayesDREAM import bayesDREAM, Modality
        print("✓ bayesDREAM package")
        print("  ✓ bayesDREAM class")
        print("  ✓ Modality class")
    except ImportError as e:
        errors.append(f"✗ bayesDREAM: {e}")

    # Print summary
    print("\n" + "="*60)
    if errors:
        print("FAILED - The following imports failed:")
        for error in errors:
            print(f"  {error}")
        return 1
    else:
        print("SUCCESS - All required packages can be imported!")
        return 0

if __name__ == "__main__":
    sys.exit(test_imports())
