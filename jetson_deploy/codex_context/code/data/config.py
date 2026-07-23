"""Data pipeline configuration."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

import yaml


SENSOR_CHANNELS = ["NTC", "PM1.0", "PM2.5", "PM10", "CT1", "CT2", "CT3", "CT4"]

_BASE = "/home/keti/factory_safety/data/aihub/datasets/extracted"


@dataclasses.dataclass
class SensorStats:
    """Pre-computed per-channel mean/std for Z-score normalization."""

    mean: list[float] = dataclasses.field(
        default_factory=lambda: [32.49, 15.70, 20.79, 36.32, 6.23, 42.73, 24.49, 11.21]
    )
    std: list[float] = dataclasses.field(
        default_factory=lambda: [9.27, 10.39, 12.15, 22.27, 18.61, 46.43, 29.74, 18.99]
    )


@dataclasses.dataclass
class ThermalStats:
    """Training-set global min/max for thermal normalization."""

    global_min: float = 30.98
    global_max: float = 146.10


@dataclasses.dataclass
class DataConfig:
    # --- Paths ---
    train_source_dir: str = f"{_BASE}/Training/원천데이터"
    train_label_dir: str = f"{_BASE}/Training/라벨링데이터"
    val_source_dir: str = f"{_BASE}/Validation/원천데이터"
    val_label_dir: str = f"{_BASE}/Validation/라벨링데이터"
    cache_dir: str = "/home/keti/factory_safety/cache"

    # --- Sliding window ---
    window_size: int = 30
    step_size: int = 10

    # --- Normalization ---
    sensor_stats: SensorStats = dataclasses.field(default_factory=SensorStats)
    thermal_stats: ThermalStats = dataclasses.field(default_factory=ThermalStats)
    thermal_norm_mode: str = "global"  # "global" | "per_image"

    # --- Label strategy ---
    label_strategy: str = "majority"  # "majority" | "last"

    # --- Augmentation ---
    augmentation_enabled: bool = False
    sensor_noise_std: float = 0.01
    thermal_flip_prob: float = 0.5

    # --- DataLoader ---
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2

    # --- Sampling ---
    use_weighted_sampler: bool = True

    # --- Session detection ---
    session_gap_threshold: int = 120  # seconds

    # --- Reproducibility ---
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> DataConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)

        sensor_stats = SensorStats(**raw.pop("sensor_stats", {}))
        thermal_stats = ThermalStats(**raw.pop("thermal_stats", {}))
        return cls(sensor_stats=sensor_stats, thermal_stats=thermal_stats, **raw)

    def to_yaml(self, path: str | Path) -> None:
        d = dataclasses.asdict(self)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False, allow_unicode=True)
