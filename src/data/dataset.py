"""PyTorch Dataset for multimodal manufacturing safety data."""

from __future__ import annotations

import os
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

    Note on caching (2026-08-06): a previous version used
    @lru_cache(maxsize=128) on the per-file loaders. This was measured
    to be ineffective because (a) the dataset has ~90k files against a
    128-entry cache, and (b) DataLoader num_workers>1 gives each worker
    its own process-local cache. Reverted to the plain inline loader.
    Labels are still pre-loaded into memory in __init__ (see
    self._all_labels), which is where the real speedup lives.
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
        # Pre-load all labels into memory (fast lookup for WeightedRandomSampler)
        self._all_labels = self.get_all_labels()

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

        # Load thermal data: 30 BIN (npy) files
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
        """Read CSV files for sensor data. Each CSV is ~80 bytes."""
        rows = []
        for bn in basenames:
            path = os.path.join(self.source_dir, bn + ".csv")
            row = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
            rows.append(row)
        return np.stack(rows)  # (window_size, 8)

    def _load_thermal_window(self, basenames: list[str]) -> np.ndarray:
        """Memory-map BIN (npy-format) files for thermal images.

        The .bin extension is a misnomer: files are actually numpy .npy
        format (magic bytes \\x93NUMPY), so np.load with mmap_mode='r'
        is the correct loader.
        """
        frames = []
        for bn in basenames:
            path = os.path.join(self.source_dir, bn + ".bin")
            frame = np.load(path, mmap_mode="r").astype(np.float32)
            frames.append(frame)
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
