"""V2+ Model: LSTM + Multi-Scale Temporal Diff + Channel Attention + SupCon Loss.

Improvements over V2 (LSTM+TemporalDiff):
1. Multi-scale temporal difference (lag=1,5,10) instead of lag=1 only
2. SE-style channel attention on sensor encoder output
3. Compatible with Supervised Contrastive Loss during training

Inference cost: negligible increase over V2 (~0.1ms on Jetson Orin Nano)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C)
        w = self.fc(x)  # (B, C) channel weights
        return x * w


class V2Plus(nn.Module):
    """LSTM + Multi-Scale Temporal Diff + Channel Attention.

    Args:
        sensor_dim: Number of sensor channels (default: 8).
        hidden_dim: LSTM hidden dimension (default: 128).
        num_layers: LSTM layers (default: 3).
        num_classes: Output classes (default: 4).
        lags: Temporal difference lags (default: [1, 5, 10]).
        dropout: Dropout rate (default: 0.1).
    """

    def __init__(self, sensor_dim: int = 8, hidden_dim: int = 128,
                 num_layers: int = 3, num_classes: int = 4,
                 lags: list[int] = None, dropout: float = 0.1):
        super().__init__()

        self.lags = lags or [1, 5, 10]
        # Input: original + N diff scales = sensor_dim * (1 + len(lags))
        input_dim = sensor_dim * (1 + len(self.lags))

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout,
        )
        self.fc_sensor = nn.Linear(hidden_dim, hidden_dim)
        self.se = SEBlock(hidden_dim)

        # Thermal encoder (same as baseline)
        self.thermal_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc_thermal = nn.Linear(64 * 15 * 20, hidden_dim)

        self.classifier = nn.Linear(2 * hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def encode(self, sensor_data: torch.Tensor,
               thermal_data: torch.Tensor) -> torch.Tensor:
        """Encode inputs to embedding space (for contrastive loss).

        Returns:
            (B, 2*hidden_dim) embedding vector.
        """
        B, T, C = sensor_data.shape

        # Multi-scale temporal difference
        diffs = []
        for lag in self.lags:
            d = torch.zeros_like(sensor_data)
            d[:, lag:, :] = sensor_data[:, lag:, :] - sensor_data[:, :-lag, :]
            diffs.append(d)
        sensor_input = torch.cat([sensor_data] + diffs, dim=-1)  # (B, T, C*(1+N))

        # Sensor branch
        lstm_out, _ = self.lstm(sensor_input)
        sensor_out = self.fc_sensor(lstm_out[:, -1, :])  # (B, hidden)
        sensor_out = self.se(sensor_out)                  # channel attention

        # Thermal branch
        B, T, H, W = thermal_data.shape
        x = thermal_data.view(B * T, 1, H, W)
        x = self.thermal_conv(x)
        x = x.view(B * T, -1)
        x = self.fc_thermal(x)
        thermal_out = x.view(B, T, -1)  # (B, T, hidden)

        # Fusion
        sensor_out = sensor_out.unsqueeze(1).expand_as(thermal_out)
        combined = torch.cat([sensor_out, thermal_out], dim=-1)
        embedding = combined.mean(dim=1)  # (B, 2*hidden)

        return embedding

    def forward(self, sensor_data: torch.Tensor,
                thermal_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sensor_data: (B, T, 8) normalized sensor time series.
            thermal_data: (B, T, 120, 160) normalized thermal images.

        Returns:
            (B, num_classes) classification logits.
        """
        embedding = self.encode(sensor_data, thermal_data)
        return self.classifier(self.dropout(embedding))


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss.

    Reference: Khosla et al., "Supervised Contrastive Learning" (NeurIPS 2020)
    Adapted from https://github.com/HobbitLong/SupContrast
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D) L2-normalized feature vectors.
            labels: (B,) integer class labels.

        Returns:
            Scalar loss.
        """
        device = features.device
        B = features.shape[0]

        if B <= 1:
            return torch.tensor(0.0, device=device)

        # Normalize features
        features = F.normalize(features, dim=1)

        # Similarity matrix
        sim = torch.matmul(features, features.T) / self.temperature  # (B, B)

        # Mask: same class = 1, different class = 0, self = 0
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)  # (B, B)
        self_mask = torch.eye(B, device=device)
        mask = mask - self_mask  # remove self-pairs

        # Number of positives per anchor
        pos_count = mask.sum(dim=1)

        # For numerical stability
        logits_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        # Mask out self-similarity
        logits = logits - self_mask * 1e9

        # Log-sum-exp of all negatives + positives
        exp_logits = torch.exp(logits)
        log_sum_exp = torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # Mean log-prob of positives
        log_prob = logits - log_sum_exp
        mean_log_prob = (mask * log_prob).sum(dim=1) / (pos_count + 1e-8)

        # Only compute loss for anchors with at least 1 positive
        valid = pos_count > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        loss = -mean_log_prob[valid].mean()

        return loss
