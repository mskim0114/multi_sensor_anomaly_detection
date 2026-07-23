#!/usr/bin/env python3
"""Real-time inference pipeline for Jetson Orin Nano.

This is the skeleton for the production deployment.
Sensor collection code will be added when hardware arrives.

Architecture:
  [Sensors] → SensorCollector → DataBuffer (30-step window)
                                     ↓
                              Preprocessor (normalize)
                                     ↓
                              InferenceEngine (ONNX Runtime)
                                     ↓
                              AlertManager (threshold → action)

Usage (simulation mode with validation data):
    cd /home/keti/factory_safety
    python src/deploy/realtime_pipeline.py --simulate
    python src/deploy/realtime_pipeline.py --simulate --interval 1.0
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import onnxruntime as ort

sys.path.insert(0, "/home/keti/factory_safety")

from src.data.config import DataConfig, SensorStats, ThermalStats
from src.data.normalization import SensorNormalizer, ThermalNormalizer

logger = logging.getLogger(__name__)

STATE_NAMES = ["Normal", "Mild", "Moderate", "Severe"]
STATE_COLORS = {0: "\033[92m", 1: "\033[93m", 2: "\033[33m", 3: "\033[91m"}
RESET = "\033[0m"

ONNX_PATH = "/home/keti/factory_safety/results/deploy/model_v2plus.onnx"


@dataclass
class PredictionResult:
    timestamp: float
    predicted_state: int
    state_name: str
    probabilities: list[float]
    confidence: float
    alert_level: str  # "none", "warning", "danger"


class DataBuffer:
    """Circular buffer for maintaining a sliding window of sensor data."""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.sensor_buffer: deque = deque(maxlen=window_size)
        self.thermal_buffer: deque = deque(maxlen=window_size)

    def add(self, sensor: np.ndarray, thermal: np.ndarray):
        """Add one timestep of data.

        Args:
            sensor: (8,) float32 array of raw sensor values.
            thermal: (120, 160) float32 array of thermal image in °C.
        """
        self.sensor_buffer.append(sensor.astype(np.float32))
        self.thermal_buffer.append(thermal.astype(np.float32))

    @property
    def is_ready(self) -> bool:
        return len(self.sensor_buffer) == self.window_size

    def get_window(self) -> tuple[np.ndarray, np.ndarray]:
        """Get current window as numpy arrays.

        Returns:
            sensor: (1, window_size, 8) float32
            thermal: (1, window_size, 120, 160) float32
        """
        sensor = np.stack(list(self.sensor_buffer))[np.newaxis]     # (1, W, 8)
        thermal = np.stack(list(self.thermal_buffer))[np.newaxis]   # (1, W, 120, 160)
        return sensor, thermal


class Preprocessor:
    """Normalize raw sensor and thermal data."""

    def __init__(self, config: DataConfig):
        self.sensor_norm = SensorNormalizer(config.sensor_stats)
        self.thermal_norm = ThermalNormalizer(config.thermal_stats, config.thermal_norm_mode)

    def __call__(self, sensor: np.ndarray, thermal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Normalize a window of data.

        Args:
            sensor: (1, W, 8) raw values
            thermal: (1, W, 120, 160) raw °C values

        Returns:
            Normalized sensor and thermal arrays.
        """
        sensor_norm = self.sensor_norm(sensor[0])[np.newaxis]      # (1, W, 8)
        thermal_norm = self.thermal_norm(thermal[0])[np.newaxis]    # (1, W, 120, 160)
        return sensor_norm, thermal_norm


class InferenceEngine:
    """ONNX Runtime inference engine."""

    def __init__(self, onnx_path: str, use_gpu: bool = True):
        providers = []
        if use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(onnx_path, providers=providers)
        logger.info(f"Loaded ONNX model: {onnx_path}")
        logger.info(f"Providers: {self.session.get_providers()}")

    def predict(self, sensor: np.ndarray, thermal: np.ndarray) -> tuple[int, np.ndarray]:
        """Run inference.

        Args:
            sensor: (1, W, 8) normalized float32
            thermal: (1, W, 120, 160) normalized float32

        Returns:
            (predicted_class, probabilities)
        """
        logits = self.session.run(None, {"sensor": sensor, "thermal": thermal})[0]
        # Softmax
        exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        pred_class = int(np.argmax(probs, axis=-1)[0])
        return pred_class, probs[0]


class AlertManager:
    """Manage alerts based on prediction results."""

    def __init__(self, warning_threshold: float = 0.7, danger_threshold: float = 0.5):
        self.warning_threshold = warning_threshold
        self.danger_threshold = danger_threshold
        self.consecutive_severe = 0
        self.consecutive_moderate = 0

    def evaluate(self, pred_class: int, probs: np.ndarray) -> str:
        """Determine alert level.

        Returns:
            "none", "warning", or "danger"
        """
        # Track consecutive severe/moderate predictions
        if pred_class == 3:  # Severe
            self.consecutive_severe += 1
            self.consecutive_moderate = 0
        elif pred_class == 2:  # Moderate
            self.consecutive_moderate += 1
            self.consecutive_severe = 0
        else:
            self.consecutive_severe = 0
            self.consecutive_moderate = 0

        # DANGER: Severe with high confidence, or 3+ consecutive severe
        if pred_class == 3 and (probs[3] > self.danger_threshold or self.consecutive_severe >= 3):
            return "danger"

        # WARNING: Moderate with high confidence, or 5+ consecutive moderate
        if pred_class == 2 and (probs[2] > self.warning_threshold or self.consecutive_moderate >= 5):
            return "warning"

        # WARNING: Severe detected but low confidence
        if pred_class == 3:
            return "warning"

        return "none"

    @staticmethod
    def display_alert(result: PredictionResult):
        """Print alert to terminal."""
        color = STATE_COLORS.get(result.predicted_state, "")
        bar = "█" * int(result.confidence * 20)

        line = (
            f"[{time.strftime('%H:%M:%S')}] "
            f"{color}{result.state_name:>8s}{RESET} "
            f"{result.confidence:5.1%} {bar}"
        )

        if result.alert_level == "danger":
            line += f"  {STATE_COLORS[3]}⚠ DANGER - 즉시 점검 필요!{RESET}"
        elif result.alert_level == "warning":
            line += f"  {STATE_COLORS[1]}△ WARNING - 주의 관찰{RESET}"

        print(line)


class RealtimePipeline:
    """Main pipeline orchestrator."""

    def __init__(self, config: DataConfig, onnx_path: str = ONNX_PATH, use_gpu: bool = True):
        self.buffer = DataBuffer(window_size=config.window_size)
        self.preprocessor = Preprocessor(config)
        self.engine = InferenceEngine(onnx_path, use_gpu)
        self.alert_mgr = AlertManager()
        self.results_log: list[dict] = []

    def process_step(self, sensor: np.ndarray, thermal: np.ndarray) -> Optional[PredictionResult]:
        """Process one timestep of data.

        Args:
            sensor: (8,) raw sensor values
            thermal: (120, 160) raw thermal image in °C

        Returns:
            PredictionResult if buffer is full, else None.
        """
        self.buffer.add(sensor, thermal)

        if not self.buffer.is_ready:
            return None

        # Get window and preprocess
        sensor_win, thermal_win = self.buffer.get_window()
        sensor_norm, thermal_norm = self.preprocessor(sensor_win, thermal_win)

        # Inference
        t0 = time.perf_counter()
        pred_class, probs = self.engine.predict(sensor_norm, thermal_norm)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Alert evaluation
        alert_level = self.alert_mgr.evaluate(pred_class, probs)

        result = PredictionResult(
            timestamp=time.time(),
            predicted_state=pred_class,
            state_name=STATE_NAMES[pred_class],
            probabilities=probs.tolist(),
            confidence=float(probs[pred_class]),
            alert_level=alert_level,
        )

        # Log
        self.results_log.append({
            "timestamp": result.timestamp,
            "state": pred_class,
            "confidence": result.confidence,
            "alert": alert_level,
            "latency_ms": round(latency_ms, 2),
        })

        return result

    def save_log(self, path: str):
        with open(path, "w") as f:
            json.dump(self.results_log, f, indent=2)


def run_simulation(config: DataConfig, interval: float = 1.0, max_steps: int = 100):
    """Simulate real-time inference using validation dataset."""

    from src.data.datamodule import ManufacturingDataModule

    print("=" * 60)
    print("  Real-time Inference Pipeline (Simulation Mode)")
    print("=" * 60)

    pipeline = RealtimePipeline(config, use_gpu=True)

    # Load validation data
    cfg = DataConfig()
    cfg.batch_size = 1
    cfg.num_workers = 0
    dm = ManufacturingDataModule(cfg)
    dm.setup()

    # Use denormalizers to get raw data
    sensor_norm = SensorNormalizer(cfg.sensor_stats)
    thermal_norm = ThermalNormalizer(cfg.thermal_stats, cfg.thermal_norm_mode)

    # Pick a session that transitions through states
    dataset = dm.val_dataset
    # Get windows from middle of dataset (likely has state transitions)
    start_idx = len(dataset) // 4

    print(f"\nSimulating {max_steps} timesteps (interval={interval}s)...")
    print(f"{'Time':>10s} {'State':>8s} {'Conf':>6s} {'Bar':>20s} {'Alert':>10s}")
    print("-" * 60)

    step = 0
    for i in range(start_idx, min(start_idx + max_steps, len(dataset))):
        sample = dataset[i]
        # Denormalize to simulate raw sensor input
        sensor_raw = sensor_norm.inverse(sample["sensor"].numpy())  # (30, 8)
        thermal_raw = thermal_norm.inverse(sample["thermal"].numpy())  # (30, 120, 160)

        # Feed last timestep of each window as one "real-time" reading
        raw_sensor = sensor_raw[-1]       # (8,)
        raw_thermal = thermal_raw[-1]     # (120, 160)

        result = pipeline.process_step(raw_sensor, raw_thermal)

        if result is not None:
            AlertManager.display_alert(result)
            step += 1

        if interval > 0:
            time.sleep(interval)

    # Save log
    log_path = "/home/keti/factory_safety/results/deploy/simulation_log.json"
    pipeline.save_log(log_path)
    print(f"\n{'='*60}")
    print(f"Simulation complete. {step} predictions made.")
    print(f"Log saved: {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Real-time Inference Pipeline")
    parser.add_argument("--simulate", action="store_true", help="Run simulation with validation data")
    parser.add_argument("--interval", type=float, default=0.5, help="Simulation interval (seconds)")
    parser.add_argument("--max-steps", type=int, default=50, help="Max simulation steps")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    config = DataConfig()

    if args.simulate:
        run_simulation(config, interval=args.interval, max_steps=args.max_steps)
    else:
        print("Real-time mode requires sensor hardware.")
        print("Use --simulate for simulation with validation data.")
        print("\nWhen Jetson + sensors are ready, implement SensorCollector class")
        print("to feed real data into pipeline.process_step(sensor, thermal)")


if __name__ == "__main__":
    main()
