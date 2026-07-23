"""Normalization transforms for sensor and thermal data."""

from __future__ import annotations

import numpy as np

from .config import SensorStats, ThermalStats


class SensorNormalizer:
    """Z-score normalization with pre-computed training set statistics."""

    def __init__(self, stats: SensorStats):
        self.mean = np.array(stats.mean, dtype=np.float32)
        self.std = np.array(stats.std, dtype=np.float32)

    def __call__(self, sensor: np.ndarray) -> np.ndarray:
        """Normalize sensor data.

        Args:
            sensor: (seq_len, 8) float32 array.

        Returns:
            Z-score normalized array, same shape.
        """
        return (sensor - self.mean) / self.std

    def inverse(self, sensor: np.ndarray) -> np.ndarray:
        """Denormalize back to original scale."""
        return sensor * self.std + self.mean


class ThermalNormalizer:
    """Min-max normalization for thermal images."""

    def __init__(self, stats: ThermalStats, mode: str = "global"):
        self.mode = mode
        self.global_min = np.float32(stats.global_min)
        self.global_range = np.float32(stats.global_max - stats.global_min)

    def __call__(self, thermal: np.ndarray) -> np.ndarray:
        """Normalize thermal images to [0, 1].

        Args:
            thermal: (seq_len, 120, 160) float32 array of temperatures in °C.

        Returns:
            Normalized array in [0, 1], same shape.
        """
        if self.mode == "global":
            return (thermal - self.global_min) / self.global_range
        else:  # per_image
            mins = thermal.min(axis=(1, 2), keepdims=True)
            maxs = thermal.max(axis=(1, 2), keepdims=True)
            return (thermal - mins) / (maxs - mins + 1e-8)

    def inverse(self, thermal: np.ndarray) -> np.ndarray:
        """Denormalize back to °C (global mode only)."""
        return thermal * self.global_range + self.global_min
