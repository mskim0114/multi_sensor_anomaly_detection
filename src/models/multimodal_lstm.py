"""Multimodal LSTM baseline model (adapted from AI Hub code).

Architecture:
  - Sensor branch: LSTM encoder (8-dim → 128-dim)
  - Thermal branch: 3-layer CNN encoder (120×160 → 128-dim per frame)
  - Fusion: broadcast + concat + temporal mean → Linear(256, num_classes)
"""

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 128,
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout,
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 8)
        lstm_out, _ = self.lstm(x)   # (B, T, hidden)
        out = lstm_out[:, -1, :]      # (B, hidden) - last timestep
        return self.fc(out)


class ThermalImageEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),      # 120×160 → 60×80
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),      # → 30×40
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),      # → 15×20
        )
        self.fc = nn.Linear(64 * 15 * 20, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 120, 160)
        B, T, H, W = x.shape
        x = x.view(B * T, 1, H, W)           # add channel dim
        x = self.conv_layers(x)                # (B*T, 64, 15, 20)
        x = x.view(B * T, -1)                 # (B*T, 19200)
        x = self.fc(x)                        # (B*T, hidden)
        return x.view(B, T, -1)               # (B, T, hidden)


class MultimodalLSTM(nn.Module):
    """Multimodal LSTM with late fusion (AI Hub baseline)."""

    def __init__(self, sensor_dim: int = 8, hidden_dim: int = 128,
                 num_layers: int = 3, num_classes: int = 4):
        super().__init__()
        self.sensor_encoder = LSTMEncoder(sensor_dim, hidden_dim, num_layers)
        self.thermal_encoder = ThermalImageEncoder(hidden_dim)
        self.fc = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, sensor_data: torch.Tensor,
                thermal_data: torch.Tensor) -> torch.Tensor:
        # sensor_data: (B, T, 8)
        # thermal_data: (B, T, 120, 160)
        sensor_out = self.sensor_encoder(sensor_data)       # (B, hidden)
        thermal_out = self.thermal_encoder(thermal_data)     # (B, T, hidden)

        # Broadcast sensor summary to match temporal dim
        sensor_out = sensor_out.unsqueeze(1).expand_as(thermal_out)
        combined = torch.cat([sensor_out, thermal_out], dim=-1)  # (B, T, 2*hidden)
        combined = combined.mean(dim=1)                          # (B, 2*hidden)
        return self.fc(combined)                                 # (B, num_classes)
