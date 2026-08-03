"""PyTorch Dataset for multimodal manufacturing safety data."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import NamedTuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentation import Augmentation
from .config import DataConfig
from .normalization import SensorNormalizer, ThermalNormalizer
from .session_index import DatasetIndex


class WindowSpec(NamedTuple):
    session_idx: int
    start: int


class ManufacturingDataset(Dataset):
    """Sliding-window dataset with lazy loading.

    Windows never cross session boundaries. Each __getitem__ returns
    a dict with sensor, thermal, label, and metadata.
    """

    def __init__(
        self,
        index: DatasetIndex,
        source_dir: str,
        label_dir: str,
        config: DataConfig,
        sensor_normalizer: SensorNormalizer,
        thermal_normalizer: ThermalNormalizer,
        augmentation: Optional[Augmentation] = None,
    ):
        self.index = index
        self.source_dir = source_dir
        self.label_dir = label_dir
        self.config = config
        self.sensor_normalizer = sensor_normalizer
        self.thermal_normalizer = thermal_normalizer
        self.augmentation = augmentation

        self.windows = self._build_windows()
        # Pre-load all labels into memory (fast lookup)
        self._all_labels = self.get_all_labels()

    @staticmethod
    @lru_cache(maxsize=128)
    def _load_single_sensor(source_dir: str, basename: str) -> np.ndarray:
        """Load single sensor CSV with LRU caching."""
        path = os.path.join(source_dir, basename + ".csv")
        return np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)

    @staticmethod
    @lru_cache(maxsize=128)
    def _load_single_thermal(source_dir: str, basename: str) -> np.ndarray:
        """Load single thermal BIN with LRU caching."""
        path = os.path.join(source_dir, basename + ".bin")
        return np.load(path, mmap_mode="r").astype(np.float32)

    def _build_windows(self) -> list[WindowSpec]:
        """Pre-compute all valid (session_idx, start_idx) pairs."""
        windows = []
        for sess_idx, session in enumerate(self.index.sessions):
            n = len(session)
            if n < self.config.window_size:
                continue
            for start in range(0, n - self.config.window_size + 1, self.config.step_size):
                windows.append(WindowSpec(session_idx=sess_idx, start=start))
        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        spec = self.windows[idx]
        session = self.index.sessions[spec.session_idx]
        end = spec.start + self.config.window_size
        basenames = session.basenames[spec.start:end]
        window_labels = session.labels[spec.start:end]

        # Load sensor data: 30 CSV files
        sensor_seq = self._load_sensor_window(basenames)

        # Load thermal data: 30 BIN files
        thermal_seq = self._load_thermal_window(basenames)

        # Determine label
        label = self._compute_label(window_labels)

        # Normalize
        sensor_seq = self.sensor_normalizer(sensor_seq)
        thermal_seq = self.thermal_normalizer(thermal_seq)

        # Augment (training only)
        if self.augmentation is not None:
            sensor_seq, thermal_seq = self.augmentation(sensor_seq, thermal_seq)

        return {
            "sensor": torch.from_numpy(sensor_seq),
            "thermal": torch.from_numpy(thermal_seq),
            "label": torch.tensor(label, dtype=torch.long),
            "metadata": {
                "session_id": session.session_id,
                "device_id": session.device_id,
                "window_start": spec.start,
            },
        }

    def _load_sensor_window(self, basenames: list[str]) -> np.ndarray:
        """Read CSV files for sensor data with LRU caching."""
        rows = [self._load_single_sensor(self.source_dir, bn) for bn in basenames]
        return np.stack(rows)  # (window_size, 8)

    def _load_thermal_window(self, basenames: list[str]) -> np.ndarray:
        """Load thermal images with LRU caching."""
        frames = [self._load_single_thermal(self.source_dir, bn) for bn in basenames]
        return np.stack(frames)  # (window_size, 120, 160)

    def _compute_label(self, labels: list[int]) -> int:
        if self.config.label_strategy == "last":
            return labels[-1]
        # Default: majority vote
        return int(np.bincount(labels).argmax())

    def get_all_labels(self) -> np.ndarray:
        """Compute labels for all windows without loading data. For sampler."""
        labels = []
        for spec in self.windows:
            session = self.index.sessions[spec.session_idx]
            window_labels = session.labels[spec.start:spec.start + self.config.window_size]
            labels.append(self._compute_label(window_labels))
        return np.array(labels, dtype=np.int64)
