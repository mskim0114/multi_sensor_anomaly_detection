#!/usr/bin/env python3
"""Export the best model (V2: LSTM+TemporalDiff) to ONNX format.

Usage:
    cd /home/keti/factory_safety
    python -m src.deploy.export_onnx
    python -m src.deploy.export_onnx --simplify
"""

import argparse
import sys

import numpy as np
import onnx
import torch

sys.path.insert(0, "/home/keti/factory_safety")

from src.models.ablation_variants import LSTMWithTemporalDiff

CKPT_PATH = "/home/keti/factory_safety/results/ablation_v2/best_model.pt"
OUTPUT_DIR = "/home/keti/factory_safety/results/deploy"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simplify", action="store_true", help="Run onnx-simplifier")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model
    model = LSTMWithTemporalDiff(sensor_dim=8, hidden_dim=128, num_layers=3, num_classes=4)
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from epoch {ckpt['epoch']} (F1={ckpt['val_f1']:.4f})")

    # Dummy inputs (batch=1, seq=30)
    dummy_sensor = torch.randn(1, 30, 8)
    dummy_thermal = torch.randn(1, 30, 120, 160)

    # Verify PyTorch output
    with torch.no_grad():
        pt_output = model(dummy_sensor, dummy_thermal)
    print(f"PyTorch output shape: {pt_output.shape}")
    print(f"PyTorch output: {pt_output[0].tolist()}")

    # Export to ONNX
    onnx_path = os.path.join(OUTPUT_DIR, "model_v2.onnx")
    print(f"\nExporting to ONNX (opset={args.opset})...")

    torch.onnx.export(
        model,
        (dummy_sensor, dummy_thermal),
        onnx_path,
        opset_version=args.opset,
        input_names=["sensor", "thermal"],
        output_names=["logits"],
        dynamic_axes={
            "sensor": {0: "batch"},
            "thermal": {0: "batch"},
            "logits": {0: "batch"},
        },
    )

    # Validate ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model saved: {onnx_path}")
    print(f"ONNX model size: {os.path.getsize(onnx_path) / 1024 / 1024:.1f} MB")

    # Simplify if requested
    if args.simplify:
        try:
            import onnxsim
            print("\nSimplifying ONNX model...")
            simplified, check = onnxsim.simplify(onnx_model)
            if check:
                simplified_path = os.path.join(OUTPUT_DIR, "model_v2_simplified.onnx")
                onnx.save(simplified, simplified_path)
                print(f"Simplified model saved: {simplified_path}")
                print(f"Simplified size: {os.path.getsize(simplified_path) / 1024 / 1024:.1f} MB")
            else:
                print("Simplification check failed, using original")
        except ImportError:
            print("onnxsim not installed. Run: pip install onnxsim")

    # Verify ONNX Runtime inference matches PyTorch
    import onnxruntime as ort
    print("\nVerifying ONNX Runtime inference...")

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_output = sess.run(
        None,
        {
            "sensor": dummy_sensor.numpy(),
            "thermal": dummy_thermal.numpy(),
        },
    )[0]

    diff = np.abs(pt_output.numpy() - ort_output).max()
    print(f"ONNX Runtime output: {ort_output[0].tolist()}")
    print(f"Max difference (PyTorch vs ONNX): {diff:.8f}")

    if diff < 1e-4:
        print("PASS: ONNX export verified")
    else:
        print(f"WARNING: Difference {diff} exceeds threshold 1e-4")


if __name__ == "__main__":
    main()
