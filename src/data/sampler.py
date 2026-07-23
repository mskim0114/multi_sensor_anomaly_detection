"""Class-balanced sampling utilities."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from .dataset import ManufacturingDataset


def build_weighted_sampler(dataset: ManufacturingDataset) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler that balances class frequencies."""
    labels = dataset.get_all_labels()
    class_counts = np.bincount(labels, minlength=4).astype(np.float64)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]

    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(labels),
        replacement=True,
    )


def compute_class_weights(dataset: ManufacturingDataset) -> torch.Tensor:
    """Compute class weights for CrossEntropyLoss(weight=...).

    Returns:
        Tensor of shape (4,) with weights inversely proportional to class frequency,
        normalized so that they sum to num_classes.
    """
    labels = dataset.get_all_labels()
    class_counts = np.bincount(labels, minlength=4).astype(np.float64)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * len(class_counts)
    return torch.tensor(weights, dtype=torch.float32)
