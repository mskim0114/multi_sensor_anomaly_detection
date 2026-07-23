"""Cross-Attention Temporal Fusion Transformer (CATFT).

Addresses two key weaknesses of the Multimodal LSTM baseline:
1. Normal↔Mild confusion → temporal difference features + cross-attention
2. No temporal alignment → bidirectional cross-attention per timestep

Architecture:
  - Sensor branch: temporal diff + projection + Transformer encoder
  - Thermal branch: EfficientNet-B0 per-frame + Transformer encoder
  - Fusion: bidirectional cross-attention (2 layers)
  - Head: temporal mean pooling + MLP classifier
"""

import math

import torch
import torch.nn as nn
import torchvision.models as tv_models


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class SensorEncoder(nn.Module):
    """Sensor time-series encoder with temporal difference features."""

    def __init__(self, sensor_dim: int = 8, d_model: int = 256,
                 nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        # Input: original 8 channels + 8 temporal diff channels = 16
        self.input_proj = nn.Linear(sensor_dim * 2, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, sensor: torch.Tensor) -> torch.Tensor:
        # sensor: (B, T, 8)
        B, T, C = sensor.shape

        # Compute temporal difference (rate of change)
        diff = torch.zeros_like(sensor)
        diff[:, 1:, :] = sensor[:, 1:, :] - sensor[:, :-1, :]

        # Concatenate original + diff
        x = torch.cat([sensor, diff], dim=-1)  # (B, T, 16)

        x = self.input_proj(x)                  # (B, T, D)
        x = self.pos_enc(x)
        x = self.transformer(x)                 # (B, T, D)
        return self.norm(x)


class ThermalEncoder(nn.Module):
    """Thermal image sequence encoder using EfficientNet-B0."""

    def __init__(self, d_model: int = 256, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.1,
                 freeze_backbone_layers: int = 5):
        super().__init__()

        # EfficientNet-B0 backbone (pretrained)
        efficientnet = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.DEFAULT)
        # Extract features (remove classifier)
        self.backbone = efficientnet.features  # output: (B, 1280, H', W')
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Freeze early layers
        for i, layer in enumerate(self.backbone):
            if i < freeze_backbone_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        # Project 1280 → d_model
        self.proj = nn.Linear(1280, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # For converting single-channel thermal to 3-channel for EfficientNet
        self.input_conv = nn.Conv2d(1, 3, kernel_size=1, bias=False)
        nn.init.constant_(self.input_conv.weight, 1.0 / 3.0)

    def forward(self, thermal: torch.Tensor) -> torch.Tensor:
        # thermal: (B, T, 120, 160), values in [0, 1]
        B, T, H, W = thermal.shape

        x = thermal.view(B * T, 1, H, W)          # (B*T, 1, 120, 160)
        x = self.input_conv(x)                      # (B*T, 3, 120, 160)
        x = self.backbone(x)                        # (B*T, 1280, H', W')
        x = self.pool(x).flatten(1)                 # (B*T, 1280)
        x = self.proj(x)                            # (B*T, D)
        x = x.view(B, T, -1)                        # (B, T, D)

        x = self.pos_enc(x)
        x = self.transformer(x)                     # (B, T, D)
        return self.norm(x)


class CrossAttentionFusionLayer(nn.Module):
    """Bidirectional cross-attention between sensor and thermal features."""

    def __init__(self, d_model: int = 256, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        # Sensor attends to thermal
        self.cross_attn_s2t = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
        )
        self.norm_s1 = nn.LayerNorm(d_model)
        self.ffn_s = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm_s2 = nn.LayerNorm(d_model)

        # Thermal attends to sensor
        self.cross_attn_t2s = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
        )
        self.norm_t1 = nn.LayerNorm(d_model)
        self.ffn_t = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm_t2 = nn.LayerNorm(d_model)

    def forward(
        self, sensor_feat: torch.Tensor, thermal_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # sensor_feat, thermal_feat: (B, T, D)

        # Sensor queries, thermal K/V (pre-norm)
        s_normed = self.norm_s1(sensor_feat)
        t_normed = self.norm_t1(thermal_feat)

        s_cross, _ = self.cross_attn_s2t(s_normed, t_normed, t_normed)
        sensor_feat = sensor_feat + s_cross
        sensor_feat = sensor_feat + self.ffn_s(self.norm_s2(sensor_feat))

        # Thermal queries, sensor K/V
        t_cross, _ = self.cross_attn_t2s(t_normed, s_normed, s_normed)
        thermal_feat = thermal_feat + t_cross
        thermal_feat = thermal_feat + self.ffn_t(self.norm_t2(thermal_feat))

        return sensor_feat, thermal_feat


class CATFT(nn.Module):
    """Cross-Attention Temporal Fusion Transformer.

    Args:
        sensor_dim: Number of sensor channels (default: 8).
        d_model: Hidden dimension (default: 256).
        nhead: Number of attention heads (default: 4).
        num_encoder_layers: Transformer encoder layers per branch (default: 2).
        num_fusion_layers: Cross-attention fusion layers (default: 2).
        num_classes: Output classes (default: 4).
        dropout: Dropout rate (default: 0.1).
    """

    def __init__(
        self,
        sensor_dim: int = 8,
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_fusion_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.sensor_encoder = SensorEncoder(
            sensor_dim, d_model, nhead, num_encoder_layers, dropout,
        )
        self.thermal_encoder = ThermalEncoder(
            d_model, nhead, num_encoder_layers, dropout,
        )

        self.fusion_layers = nn.ModuleList([
            CrossAttentionFusionLayer(d_model, nhead, dropout)
            for _ in range(num_fusion_layers)
        ])

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(
        self, sensor_data: torch.Tensor, thermal_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            sensor_data: (B, T, 8) normalized sensor time series.
            thermal_data: (B, T, 120, 160) normalized thermal images.

        Returns:
            (B, num_classes) classification logits.
        """
        # Encode each modality independently
        sensor_feat = self.sensor_encoder(sensor_data)     # (B, T, D)
        thermal_feat = self.thermal_encoder(thermal_data)   # (B, T, D)

        # Bidirectional cross-attention fusion
        for fusion_layer in self.fusion_layers:
            sensor_feat, thermal_feat = fusion_layer(sensor_feat, thermal_feat)

        # Temporal mean pooling
        sensor_pooled = sensor_feat.mean(dim=1)   # (B, D)
        thermal_pooled = thermal_feat.mean(dim=1)  # (B, D)

        # Concatenate and classify
        combined = torch.cat([sensor_pooled, thermal_pooled], dim=-1)  # (B, 2D)
        return self.classifier(combined)  # (B, num_classes)
