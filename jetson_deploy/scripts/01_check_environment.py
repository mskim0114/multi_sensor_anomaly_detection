#!/usr/bin/env python3
"""Sanity check Jetson environment for ONNX inference.

Usage:
    python3 01_check_environment.py
"""

import platform
import sys


def main():
    print("=" * 60)
    print("JETSON ENVIRONMENT CHECK")
    print("=" * 60)

    print(f"\nPython: {sys.version.split()[0]} ({platform.machine()})")
    print(f"Platform: {platform.platform()}")

    try:
        import numpy as np
        print(f"NumPy: {np.__version__}")
    except ImportError:
        print("NumPy: NOT INSTALLED  ->  pip3 install numpy")

    try:
        import onnxruntime as ort
        print(f"onnxruntime: {ort.__version__}")
        providers = ort.get_available_providers()
        print(f"Available providers: {providers}")
        if "CUDAExecutionProvider" in providers:
            print("  [OK] CUDA EP available")
        elif "TensorrtExecutionProvider" in providers:
            print("  [OK] TensorRT EP available")
        else:
            print("  [WARN] Only CPU EP. Install onnxruntime-gpu for GPU acceleration.")
    except ImportError:
        print("onnxruntime: NOT INSTALLED")
        print("  Install (Jetson JetPack 6.x, Python 3.10):")
        print("    pip3 install onnxruntime-gpu")
        print("  or download official Jetson wheel from")
        print("    https://elinux.org/Jetson_Zoo#ONNX_Runtime")

    try:
        import onnx
        print(f"onnx: {onnx.__version__}")
    except ImportError:
        print("onnx: not installed (optional, only needed for checker)")

    # GPU info
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            print(f"\nGPU: {out.stdout.strip()}")
    except Exception:
        pass

    try:
        with open("/etc/nv_tegra_release") as f:
            print(f"L4T: {f.readline().strip()}")
    except FileNotFoundError:
        print("L4T: file not found (not a Jetson?)")

    print("\nDone.")


if __name__ == "__main__":
    main()
