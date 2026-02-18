#!/usr/bin/env python
"""Check GPU/CUDA availability for training."""
import sys

print("=== GPU/CUDA Check ===")

# Check PyTorch CUDA
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("PyTorch: NOT INSTALLED")
except Exception as e:
    print(f"PyTorch error: {e}")

# Check XGBoost GPU
try:
    import xgboost as xgb
    print(f"\nXGBoost: {xgb.__version__}")
    try:
        # Try to create a GPU classifier
        clf = xgb.XGBClassifier(tree_method='gpu_hist', n_estimators=1, verbosity=0)
        print("XGBoost GPU (gpu_hist): AVAILABLE")
    except Exception as e:
        print(f"XGBoost GPU: NOT AVAILABLE ({e})")
except ImportError:
    print("\nXGBoost: NOT INSTALLED")
except Exception as e:
    print(f"\nXGBoost error: {e}")

# Check CuPy
try:
    import cupy
    print(f"\nCuPy: {cupy.__version__}")
except ImportError:
    print("\nCuPy: NOT INSTALLED")
except Exception as e:
    print(f"\nCuPy error: {e}")

# Check NVIDIA driver
import subprocess
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        print("\n=== NVIDIA GPU Info ===")
        # Print just the first few lines
        lines = result.stdout.strip().split('\n')
        for line in lines[:12]:
            print(line)
    else:
        print("\nnvidia-smi: FAILED")
except FileNotFoundError:
    print("\nnvidia-smi: NOT FOUND (NVIDIA drivers may not be installed)")
except Exception as e:
    print(f"\nnvidia-smi error: {e}")
