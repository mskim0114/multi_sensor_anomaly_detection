"""Optional data augmentation for training."""

from __future__ import annotations

import numpy as np

from .config import DataConfig


class Augmentation:
    """Augmentation applied after normalization, training only."""

    def __init__(self, config: DataConfig):
        self.sensor_noise_std = config.sensor_noise_std
        self.thermal_flip_prob = config.thermal_flip_prob

    def __call__(
        self, sensor: np.ndarray, thermal: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # Sensor: additive Gaussian noise in normalized space
        if self.sensor_noise_std > 0:
            noise = np.random.randn(*sensor.shape).astype(np.float32) * self.sensor_noise_std
            sensor = sensor + noise

        # Thermal: horizontal flip (consistent across all frames in window)
        if np.random.random() < self.thermal_flip_prob:
            thermal = np.ascontiguousarray(thermal[:, :, ::-1])

        return sensor, thermal
