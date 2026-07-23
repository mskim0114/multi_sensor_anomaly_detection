"""External SOTA baselines for comparison.

Implements simplified versions of TimesNet and PatchTST adapted for
our multimodal classification task (sensor + thermal → 4-class).

These are faithful reimplementations of the core mechanisms from:
- TimesNet (Wu et al., ICLR 2023): 2D temporal variation modeling
- PatchTST (Nie et al., ICLR 2023): Patch-based Transformer for time series
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# TimesNet for Classification
# ============================================================

class TimesBlock(nn.Module):
    """Core TimesNet block: reshape 1D→2D by discovered period, apply 2D conv."""

    def __init__(self, seq_len, d_model, d_ff, top_k=2):
        super().__init__()
        self.seq_len = seq_len
        self.top_k = top_k
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_ff, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_ff, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape

        # FFT to find dominant periods
        xf = torch.fft.rfft(x, dim=1)
        freq_magnitudes = torch.mean(torch.abs(xf), dim=-1)  # (B, T//2+1)
        freq_magnitudes[:, 0] = 0  # remove DC component

        _, top_indices = torch.topk(freq_magnitudes, self.top_k, dim=1)  # (B, top_k)
        periods = T // (top_indices + 1)  # convert freq index to period
        periods = periods.clamp(min=2, max=T)

        # Use the most common period across the batch
        period = int(periods[:, 0].float().median().item())
        period = max(2, min(period, T))

        # Reshape to 2D: (B, D, period, T//period)
        n_periods = T // period
        x_trimmed = x[:, :n_periods * period, :]  # trim to fit
        x_2d = x_trimmed.reshape(B, n_periods, period, D)
        x_2d = x_2d.permute(0, 3, 2, 1)  # (B, D, period, n_periods)

        # 2D convolution
        out_2d = self.conv(x_2d)  # (B, D, period, n_periods)

        # Reshape back to 1D
        out = out_2d.permute(0, 3, 2, 1).reshape(B, n_periods * period, D)

        # Pad if needed
        if out.shape[1] < T:
            out = F.pad(out, (0, 0, 0, T - out.shape[1]))

        return out + x  # residual


class TimesNetClassifier(nn.Module):
    """TimesNet adapted for multimodal sensor classification.

    Reference: Wu et al., "TimesNet: Temporal 2D-Variation Modeling for
    General Time Series Analysis" (ICLR 2023)
    """

    def __init__(self, sensor_dim=8, d_model=64, d_ff=32,
                 num_layers=2, seq_len=30, num_classes=4, dropout=0.1):
        super().__init__()
        # Sensor branch: TimesNet
        self.sensor_proj = nn.Linear(sensor_dim, d_model)
        self.times_blocks = nn.ModuleList([
            TimesBlock(seq_len, d_model, d_ff) for _ in range(num_layers)
        ])
        self.sensor_norm = nn.LayerNorm(d_model)

        # Thermal branch: simple CNN (same as baseline)
        self.thermal_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc_thermal = nn.Linear(64 * 15 * 20, d_model)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_classes),
        )

    def forward(self, sensor_data, thermal_data):
        # Sensor: TimesNet
        x = self.sensor_proj(sensor_data)  # (B, T, d_model)
        for block in self.times_blocks:
            x = block(x)
        x = self.sensor_norm(x)
        sensor_out = x.mean(dim=1)  # (B, d_model)

        # Thermal: CNN
        B, T, H, W = thermal_data.shape
        t = thermal_data.view(B * T, 1, H, W)
        t = self.thermal_conv(t)
        t = self.fc_thermal(t.view(B * T, -1))
        thermal_out = t.view(B, T, -1).mean(dim=1)  # (B, d_model)

        combined = torch.cat([sensor_out, thermal_out], dim=-1)
        return self.classifier(combined)


# ============================================================
# PatchTST for Classification
# ============================================================

class PatchTSTClassifier(nn.Module):
    """PatchTST adapted for multimodal sensor classification.

    Reference: Nie et al., "A Time Series is Worth 64 Words: Long-term
    Forecasting with Transformers" (ICLR 2023)
    """

    def __init__(self, sensor_dim=8, d_model=64, nhead=4,
                 num_layers=2, seq_len=30, patch_len=5, stride=5,
                 num_classes=4, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.sensor_dim = sensor_dim
        self.d_model = d_model

        # Number of patches
        self.n_patches = (seq_len - patch_len) // stride + 1

        # Channel-independent: project each channel's patches separately
        self.patch_proj = nn.Linear(patch_len, d_model)

        # Positional encoding for patches
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # Thermal branch: same CNN
        self.thermal_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc_thermal = nn.Linear(64 * 15 * 20, d_model)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model * (sensor_dim + 1), num_classes),
        )

    def forward(self, sensor_data, thermal_data):
        B, T, C = sensor_data.shape

        # Channel-independent patching
        # For each of 8 channels, create patches independently
        channel_outs = []
        for ch in range(C):
            ch_data = sensor_data[:, :, ch]  # (B, T)
            # Create patches: unfold along time dimension
            patches = ch_data.unfold(1, self.patch_len, self.stride)  # (B, n_patches, patch_len)
            patches = self.patch_proj(patches)  # (B, n_patches, d_model)
            patches = patches + self.pos_embed
            patches = self.transformer(patches)
            patches = self.norm(patches)
            channel_outs.append(patches.mean(dim=1))  # (B, d_model)

        sensor_out = torch.stack(channel_outs, dim=1)  # (B, C, d_model)
        sensor_out = sensor_out.view(B, -1)  # (B, C * d_model)

        # Thermal branch
        B, T, H, W = thermal_data.shape
        t = thermal_data.view(B * T, 1, H, W)
        t = self.thermal_conv(t)
        t = self.fc_thermal(t.view(B * T, -1))
        thermal_out = t.view(B, T, -1).mean(dim=1)  # (B, d_model)

        combined = torch.cat([sensor_out, thermal_out], dim=-1)  # (B, (C+1)*d_model)
        return self.classifier(combined)
