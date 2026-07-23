"""Ablation study model variants.

Variant 2: LSTM + Temporal Diff (sensor input에 변화율 추가)
Variant 3: LSTM + EfficientNet (열화상 인코더만 교체)
Variant 4: CATFT without Cross-Attention (독립 브랜치 + concat)
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models

from .catft import PositionalEncoding, SensorEncoder, ThermalEncoder


# ============================================================
# Variant 2: Baseline LSTM + Temporal Difference features
# ============================================================

class LSTMWithTemporalDiff(nn.Module):
    """Baseline LSTM with temporal difference features added to sensor input."""

    def __init__(self, sensor_dim: int = 8, hidden_dim: int = 128,
                 num_layers: int = 3, num_classes: int = 4):
        super().__init__()
        # Input: 8 original + 8 diff = 16
        self.lstm = nn.LSTM(
            sensor_dim * 2, hidden_dim, num_layers,
            batch_first=True, dropout=0.1,
        )
        self.fc_sensor = nn.Linear(hidden_dim, hidden_dim)

        # Same thermal encoder as baseline
        self.thermal_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc_thermal = nn.Linear(64 * 15 * 20, hidden_dim)
        self.classifier = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, sensor_data: torch.Tensor,
                thermal_data: torch.Tensor) -> torch.Tensor:
        B, T, C = sensor_data.shape

        # Temporal difference
        diff = torch.zeros_like(sensor_data)
        diff[:, 1:, :] = sensor_data[:, 1:, :] - sensor_data[:, :-1, :]
        sensor_input = torch.cat([sensor_data, diff], dim=-1)  # (B, T, 16)

        # Sensor branch
        lstm_out, _ = self.lstm(sensor_input)
        sensor_out = self.fc_sensor(lstm_out[:, -1, :])  # (B, hidden)

        # Thermal branch (same as baseline)
        B, T, H, W = thermal_data.shape
        x = thermal_data.view(B * T, 1, H, W)
        x = self.thermal_conv(x)
        x = x.view(B * T, -1)
        x = self.fc_thermal(x)
        thermal_out = x.view(B, T, -1)  # (B, T, hidden)

        # Fusion (same as baseline: broadcast + mean)
        sensor_out = sensor_out.unsqueeze(1).expand_as(thermal_out)
        combined = torch.cat([sensor_out, thermal_out], dim=-1)
        combined = combined.mean(dim=1)
        return self.classifier(combined)


# ============================================================
# Variant 3: LSTM + EfficientNet thermal encoder
# ============================================================

class LSTMWithEfficientNet(nn.Module):
    """Baseline LSTM sensor encoder + EfficientNet-B0 thermal encoder."""

    def __init__(self, sensor_dim: int = 8, hidden_dim: int = 128,
                 num_layers: int = 3, num_classes: int = 4):
        super().__init__()

        # Same LSTM sensor encoder as baseline
        self.lstm = nn.LSTM(
            sensor_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0.1,
        )
        self.fc_sensor = nn.Linear(hidden_dim, hidden_dim)

        # EfficientNet-B0 thermal encoder
        efficientnet = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.DEFAULT)
        self.backbone = efficientnet.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        for i, layer in enumerate(self.backbone):
            if i < 5:
                for param in layer.parameters():
                    param.requires_grad = False
        self.input_conv = nn.Conv2d(1, 3, kernel_size=1, bias=False)
        nn.init.constant_(self.input_conv.weight, 1.0 / 3.0)
        self.fc_thermal = nn.Linear(1280, hidden_dim)

        self.classifier = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, sensor_data: torch.Tensor,
                thermal_data: torch.Tensor) -> torch.Tensor:
        # Sensor branch (baseline LSTM)
        lstm_out, _ = self.lstm(sensor_data)
        sensor_out = self.fc_sensor(lstm_out[:, -1, :])  # (B, hidden)

        # Thermal branch (EfficientNet)
        B, T, H, W = thermal_data.shape
        x = thermal_data.view(B * T, 1, H, W)
        x = self.input_conv(x)
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        x = self.fc_thermal(x)
        thermal_out = x.view(B, T, -1)  # (B, T, hidden)

        # Fusion (same as baseline: broadcast + mean)
        sensor_out = sensor_out.unsqueeze(1).expand_as(thermal_out)
        combined = torch.cat([sensor_out, thermal_out], dim=-1)
        combined = combined.mean(dim=1)
        return self.classifier(combined)


# ============================================================
# Variant 4: CATFT without Cross-Attention (independent branches)
# ============================================================

class CATFTNoCrossAttention(nn.Module):
    """CATFT architecture but without cross-attention fusion.
    Both branches encode independently, then concat + classify.
    """

    def __init__(self, sensor_dim: int = 8, d_model: int = 256,
                 nhead: int = 4, num_encoder_layers: int = 2,
                 num_classes: int = 4, dropout: float = 0.1):
        super().__init__()

        self.sensor_encoder = SensorEncoder(
            sensor_dim, d_model, nhead, num_encoder_layers, dropout,
        )
        self.thermal_encoder = ThermalEncoder(
            d_model, nhead, num_encoder_layers, dropout,
        )

        # No cross-attention - directly classify
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, sensor_data: torch.Tensor,
                thermal_data: torch.Tensor) -> torch.Tensor:
        sensor_feat = self.sensor_encoder(sensor_data)    # (B, T, D)
        thermal_feat = self.thermal_encoder(thermal_data)  # (B, T, D)

        # No cross-attention, just pool and concat
        sensor_pooled = sensor_feat.mean(dim=1)
        thermal_pooled = thermal_feat.mean(dim=1)
        combined = torch.cat([sensor_pooled, thermal_pooled], dim=-1)
        return self.classifier(combined)
