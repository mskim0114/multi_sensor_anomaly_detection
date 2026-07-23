"""Top-level DataModule that wires everything together."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .augmentation import Augmentation
from .config import DataConfig
from .dataset import ManufacturingDataset
from .normalization import SensorNormalizer, ThermalNormalizer
from .sampler import build_weighted_sampler, compute_class_weights
from .session_index import load_or_build_index

logger = logging.getLogger(__name__)


def _worker_init_fn(worker_id: int) -> None:
    """Ensure different random seeds per worker for augmentation."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)


def _collate_fn(batch: list[dict]) -> dict:
    """Custom collate that handles the metadata dict."""
    return {
        "sensor": torch.stack([b["sensor"] for b in batch]),
        "thermal": torch.stack([b["thermal"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "metadata": [b["metadata"] for b in batch],
    }


class ManufacturingDataModule:
    """Factory that builds train/val DataLoaders from config.

    Usage:
        dm = ManufacturingDataModule(config)
        dm.setup()
        for batch in dm.train_dataloader():
            sensor = batch["sensor"]    # (B, 30, 8)
            thermal = batch["thermal"]  # (B, 30, 120, 160)
            label = batch["label"]      # (B,)
    """

    def __init__(self, config: DataConfig):
        self.config = config
        self.train_dataset: Optional[ManufacturingDataset] = None
        self.val_dataset: Optional[ManufacturingDataset] = None
        self._class_weights: Optional[torch.Tensor] = None

    def setup(self) -> None:
        """Load or build session indices, create datasets."""
        cfg = self.config

        # Build session indices
        train_index = load_or_build_index(
            cfg.train_source_dir, cfg.train_label_dir,
            cfg.cache_dir, split="train",
            gap_threshold=cfg.session_gap_threshold,
        )
        val_index = load_or_build_index(
            cfg.val_source_dir, cfg.val_label_dir,
            cfg.cache_dir, split="val",
            gap_threshold=cfg.session_gap_threshold,
        )

        # Normalizers (from training stats only)
        sensor_norm = SensorNormalizer(cfg.sensor_stats)
        thermal_norm = ThermalNormalizer(cfg.thermal_stats, cfg.thermal_norm_mode)

        # Augmentation (training only)
        aug = Augmentation(cfg) if cfg.augmentation_enabled else None

        # Datasets
        self.train_dataset = ManufacturingDataset(
            index=train_index,
            source_dir=cfg.train_source_dir,
            label_dir=cfg.train_label_dir,
            config=cfg,
            sensor_normalizer=sensor_norm,
            thermal_normalizer=thermal_norm,
            augmentation=aug,
        )
        self.val_dataset = ManufacturingDataset(
            index=val_index,
            source_dir=cfg.val_source_dir,
            label_dir=cfg.val_label_dir,
            config=cfg,
            sensor_normalizer=sensor_norm,
            thermal_normalizer=thermal_norm,
            augmentation=None,
        )

        # Pre-compute class weights
        self._class_weights = compute_class_weights(self.train_dataset)

        logger.info(
            f"Train: {len(self.train_dataset)} windows from "
            f"{len(train_index.sessions)} sessions"
        )
        logger.info(
            f"Val: {len(self.val_dataset)} windows from "
            f"{len(val_index.sessions)} sessions"
        )
        logger.info(f"Class weights: {self._class_weights.tolist()}")

    @property
    def class_weights(self) -> torch.Tensor:
        """Class weights for CrossEntropyLoss. Call setup() first."""
        if self._class_weights is None:
            raise RuntimeError("Call setup() before accessing class_weights")
        return self._class_weights

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None, "Call setup() first"

        sampler = None
        shuffle = True
        if self.config.use_weighted_sampler:
            sampler = build_weighted_sampler(self.train_dataset)
            shuffle = False  # mutually exclusive with sampler

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            prefetch_factor=self.config.prefetch_factor,
            persistent_workers=self.config.num_workers > 0,
            worker_init_fn=_worker_init_fn,
            generator=torch.Generator().manual_seed(self.config.seed),
            collate_fn=_collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None, "Call setup() first"

        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            prefetch_factor=self.config.prefetch_factor,
            persistent_workers=self.config.num_workers > 0,
            worker_init_fn=_worker_init_fn,
            collate_fn=_collate_fn,
        )
