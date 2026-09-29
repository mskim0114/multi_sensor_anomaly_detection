"""Normal-only anomaly detector arms S/T/F for EXP-004 §3.3. Contains no training.

Every structure here is a PROPOSED pre-implementation choice recorded in
``docs/연구노트/연구노트_25_합성이상_모달리티_비교설계.md`` §3.3. Constructing a model
never runs an optimiser, never reads calibration or held-out data, and never loads a
pretrained checkpoint. The sensor branch and the thermal frame CNN reproduce
``v2_plus.V2Plus``'s structures; the thermal temporal aggregation (a shared 3-layer
LSTM instead of V2+'s time mean) and all decoders are new unvalidated candidates.
Such a model is not V2+ and must not be reported as V2+.

Nothing here is protocol-frozen: these structures are implementation candidates awaiting
independent review and server verification. Only the protocol freeze makes them final.

No SupCon objective and no 4-class head appear here: these arms are trained on Normal
windows only (AE) or on a 1:1 Normal/synthetic batch (BIN).
"""
from __future__ import annotations

import hashlib
import json

import torch
from torch import nn

__all__ = [
    "WINDOW_TICKS", "SENSOR_DIM", "THERMAL_SHAPE", "HIDDEN_DIM", "NUM_LAYERS", "LAGS",
    "AI_HUB_CHANNELS", "FIELD_SENSOR_CHANNELS", "ARMS", "OBJECTIVES", "ARM_MODALITIES",
    "TRAINING_CONTRACT", "field_channel_indices", "SEBlock", "SensorEncoder",
    "ThermalEncoder", "SensorDecoder", "ThermalDecoder", "AnomalyDetector",
    "multi_scale_diff", "reconstruction_error", "autoencoder_loss", "residual_score",
    "build_arm_set", "structure_manifest", "state_tensor_sha256",
]

WINDOW_TICKS = 30
SENSOR_DIM = 5
THERMAL_SHAPE = (120, 160)
HIDDEN_DIM = 128
NUM_LAYERS = 3
LAGS = (1, 5, 10)
DROPOUT = 0.1
SE_REDUCTION = 8
# Three stride-2 poolings on 120x160 leave 15x20 at 64 channels.
FRAME_FEATURE_SHAPE = (64, 15, 20)

# The synthetic bank stores AI Hub's eight channels; these arms consume the five-channel
# field view. D-028: CT2..CT4 are never reconstructed, one CT channel is selected.
AI_HUB_CHANNELS = ("NTC", "PM1.0", "PM2.5", "PM10", "CT1", "CT2", "CT3", "CT4")
FIELD_SENSOR_CHANNELS = ("NTC", "PM1.0", "PM2.5", "PM10", "CT")

ARMS = ("S", "T", "F")
OBJECTIVES = ("AE", "BIN")
ARM_MODALITIES = {"S": ("sensor",), "T": ("thermal",), "F": ("sensor", "thermal")}

# §3.3(7) initial budget. Recorded as data; this module does not run it.
TRAINING_CONTRACT = {
    "epochs": 30, "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 0.01,
    "lr_schedule": "fixed_cosine", "precision": "FP32", "effective_batch": 16,
    "seeds": (42, 123, 456), "deterministic": True,
    "bin_normal_to_synthetic_ratio": "1:1",
    "no_extra_class_weight_or_sampler": True,
    "checkpoint_selection": {"AE": "min_dev_Normal_loss", "BIN": "min_dev_BCE",
                             "tie_rule": "earlier_epoch"},
    "budget_status": "initial_proposal_may_change_only_on_dev_and_only_for_all_arms",
    "heldout_used_for_any_selection": False,
    "pretrained_or_supcon_weights_used": False,
}


def field_channel_indices(ct_channel="CT1"):
    """Five-channel field view of an eight-channel bank window."""
    if ct_channel not in {"CT1", "CT2"}:
        raise ValueError("only the preregistered CT1/CT2 comparison is defined")
    return (0, 1, 2, 3, AI_HUB_CHANNELS.index(ct_channel))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention; structurally identical to v2_plus.SEBlock.

    Duplicated rather than imported so this module stays leaf-loadable. Kept honest by
    a test that compares the layer sequence against v2_plus.SEBlock.
    """

    def __init__(self, channels: int, reduction: int = SE_REDUCTION):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


def multi_scale_diff(sensor: torch.Tensor, lags=LAGS) -> torch.Tensor:
    """Concatenate the window with its lagged differences, zero-padded at the front."""
    diffs = []
    for lag in lags:
        d = torch.zeros_like(sensor)
        d[:, lag:, :] = sensor[:, lag:, :] - sensor[:, :-lag, :]
        diffs.append(d)
    return torch.cat([sensor] + diffs, dim=-1)


class SensorEncoder(nn.Module):
    """V2+ sensor branch: multi-scale diff -> 3-layer LSTM -> projection -> SE."""

    def __init__(self, sensor_dim: int = SENSOR_DIM, hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = NUM_LAYERS, lags=LAGS, dropout: float = DROPOUT):
        super().__init__()
        self.sensor_dim, self.lags = sensor_dim, tuple(lags)
        self.lstm = nn.LSTM(sensor_dim * (1 + len(self.lags)), hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc_sensor = nn.Linear(hidden_dim, hidden_dim)
        self.se = SEBlock(hidden_dim)

    def forward(self, sensor: torch.Tensor) -> torch.Tensor:
        if sensor.dim() != 3 or sensor.shape[1:] != (WINDOW_TICKS, self.sensor_dim):
            raise ValueError(f"sensor window must be (B, {WINDOW_TICKS}, {self.sensor_dim})")
        out, _ = self.lstm(multi_scale_diff(sensor, self.lags))
        return self.se(self.fc_sensor(out[:, -1, :]))


class ThermalEncoder(nn.Module):
    """V2+ frame CNN and frame projection, then a shared 3-layer LSTM over time.

    V2+ averaged frames and lost temporal order. §3.3(2) replaces that with an LSTM;
    this is the unvalidated part of the structure.
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM, num_layers: int = NUM_LAYERS,
                 dropout: float = DROPOUT):
        super().__init__()
        self.thermal_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        c, h, w = FRAME_FEATURE_SHAPE
        self.fc_thermal = nn.Linear(c * h * w, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

    def forward(self, thermal: torch.Tensor) -> torch.Tensor:
        if thermal.dim() != 4 or thermal.shape[1:] != (WINDOW_TICKS, *THERMAL_SHAPE):
            raise ValueError(f"thermal window must be (B, {WINDOW_TICKS}, {THERMAL_SHAPE[0]}, {THERMAL_SHAPE[1]})")
        b, t = thermal.shape[:2]
        x = self.thermal_conv(thermal.reshape(b * t, 1, *THERMAL_SHAPE))
        x = self.fc_thermal(x.reshape(b * t, -1)).reshape(b, t, -1)
        out, _ = self.lstm(x)
        return out[:, -1, :]


class _PositionalSequenceDecoder(nn.Module):
    """Expand a bottleneck to a 30-step sequence using only z and the tick position.

    Proposed implementation choice, kept compact: the per-step input is
    ``concat(z, position[t])`` fed to a single-layer LSTM. There is no path from the
    original window into this module (no skip, no teacher forcing), which is what makes
    the AE residual a genuine reconstruction error.

    ``position`` is a plain parameter rather than ``nn.Embedding`` because every tick is
    used on every step, so no lookup is needed, and ``embedding_dense_backward`` has no
    deterministic CUDA implementation — an embedding would make strict
    ``use_deterministic_algorithms(True)`` raise during training.
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM, window: int = WINDOW_TICKS):
        super().__init__()
        self.window, self.hidden_dim = window, hidden_dim
        self.position = nn.Parameter(torch.randn(window, hidden_dim))
        self.lstm = nn.LSTM(2 * hidden_dim, hidden_dim, 1, batch_first=True, dropout=0.0)

    def sequence(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 2 or z.shape[1] != self.hidden_dim:
            raise ValueError(f"bottleneck must be (B, {self.hidden_dim})")
        pos = self.position.unsqueeze(0).expand(z.shape[0], -1, -1)
        out, _ = self.lstm(torch.cat([z.unsqueeze(1).expand(-1, self.window, -1), pos], dim=-1))
        return out


class SensorDecoder(_PositionalSequenceDecoder):
    """Reconstruct (B, 30, sensor_dim) from the bottleneck alone."""

    def __init__(self, sensor_dim: int = SENSOR_DIM, hidden_dim: int = HIDDEN_DIM,
                 window: int = WINDOW_TICKS):
        super().__init__(hidden_dim=hidden_dim, window=window)
        self.head = nn.Linear(hidden_dim, sensor_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(self.sequence(z))


class ThermalDecoder(_PositionalSequenceDecoder):
    """Reconstruct (B, 30, 120, 160) from the bottleneck alone.

    Mirrors the encoder: a frame feature map at 64x15x20 is upsampled by three
    transposed convolutions back to 120x160. Identical structure in T and F (§3.3(4)).
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM, window: int = WINDOW_TICKS):
        super().__init__(hidden_dim=hidden_dim, window=window)
        c, h, w = FRAME_FEATURE_SHAPE
        self.to_frame = nn.Linear(hidden_dim, c * h * w)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2), nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]
        x = self.to_frame(self.sequence(z)).reshape(b * self.window, *FRAME_FEATURE_SHAPE)
        return self.deconv(x).reshape(b, self.window, *THERMAL_SHAPE)


class AnomalyDetector(nn.Module):
    """One (arm, objective) configuration. Modules are built in a declared order.

    The order is fixed for readability only: branch initial weights are made identical
    across arms by explicit copying in :func:`build_arm_set`, never by RNG call order.
    """

    def __init__(self, arm: str, objective: str, sensor_dim: int = SENSOR_DIM,
                 hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        if arm not in ARMS or objective not in OBJECTIVES:
            raise ValueError("arm must be S/T/F and objective AE/BIN")
        self.arm, self.objective = arm, objective
        self.sensor_dim, self.hidden_dim = sensor_dim, hidden_dim
        self.modalities = ARM_MODALITIES[arm]
        if "sensor" in self.modalities:
            self.sensor_encoder = SensorEncoder(sensor_dim=sensor_dim, hidden_dim=hidden_dim)
        if "thermal" in self.modalities:
            self.thermal_encoder = ThermalEncoder(hidden_dim=hidden_dim)
        if arm == "F":
            self.fusion = nn.Linear(2 * hidden_dim, hidden_dim)
        if objective == "AE":
            if "sensor" in self.modalities:
                self.sensor_decoder = SensorDecoder(sensor_dim=sensor_dim, hidden_dim=hidden_dim)
            if "thermal" in self.modalities:
                self.thermal_decoder = ThermalDecoder(hidden_dim=hidden_dim)
        else:
            self.head = nn.Linear(hidden_dim, 1)

    def bottleneck(self, sensor=None, thermal=None) -> torch.Tensor:
        """128-d bottleneck. A modality outside this arm is ignored, not consumed."""
        for name, value in (("sensor", sensor), ("thermal", thermal)):
            if name in self.modalities and value is None:
                raise ValueError(f"arm {self.arm} requires the {name} modality")
        if self.arm == "S":
            return self.sensor_encoder(sensor)
        if self.arm == "T":
            return self.thermal_encoder(thermal)
        return self.fusion(torch.cat([self.sensor_encoder(sensor),
                                      self.thermal_encoder(thermal)], dim=-1))

    def forward(self, sensor=None, thermal=None) -> dict:
        z = self.bottleneck(sensor=sensor, thermal=thermal)
        out = {"bottleneck": z, "sensor_reconstruction": None,
               "thermal_reconstruction": None, "logit": None}
        if self.objective == "AE":
            if "sensor" in self.modalities:
                out["sensor_reconstruction"] = self.sensor_decoder(z)
            if "thermal" in self.modalities:
                out["thermal_reconstruction"] = self.thermal_decoder(z)
        else:
            out["logit"] = self.head(z).squeeze(-1)
        return out


def reconstruction_error(prediction: torch.Tensor, target: torch.Tensor, *,
                         per_sample: bool = False) -> torch.Tensor:
    """Element-mean squared error, never a sum (§3.3(5)).

    A summed MSE would let 576,000 thermal elements dominate 150 sensor elements.
    """
    if prediction.shape != target.shape:
        raise ValueError("reconstruction and target shapes differ")
    squared = (prediction - target) ** 2
    return squared.flatten(1).mean(dim=1) if per_sample else squared.mean()


def autoencoder_loss(outputs: dict, sensor=None, thermal=None, *, arm: str,
                     per_sample: bool = False):
    """Return (total, per-modality) element-mean MSE; F weights each modality 0.5."""
    if arm not in ARMS:
        raise ValueError("unknown arm")
    parts = {}
    if "sensor" in ARM_MODALITIES[arm]:
        parts["sensor"] = reconstruction_error(outputs["sensor_reconstruction"], sensor,
                                               per_sample=per_sample)
    if "thermal" in ARM_MODALITIES[arm]:
        parts["thermal"] = reconstruction_error(outputs["thermal_reconstruction"], thermal,
                                                per_sample=per_sample)
    if arm == "F":
        total = 0.5 * parts["sensor"] + 0.5 * parts["thermal"]
    else:
        total = parts[ARM_MODALITIES[arm][0]]
    return total, parts


def residual_score(model: AnomalyDetector, sensor=None, thermal=None) -> torch.Tensor:
    """Per-window AE residual, using the identical formula as the training loss.

    This is the evaluation API, so it requires ``eval()`` and runs under ``no_grad``: the
    encoder LSTMs carry dropout, and a score taken in train mode would not be
    reproducible and could not be compared against a calibrated threshold. A trainer must
    call :func:`autoencoder_loss` directly instead of relaxing this guard.
    """
    if model.objective != "AE":
        raise ValueError("residual score is defined for the AE objective only")
    if model.training:
        raise ValueError("residual_score requires model.eval(); dropout makes train-mode "
                         "scores non-reproducible against a calibrated threshold")
    with torch.no_grad():
        outputs = model(sensor=sensor, thermal=thermal)
        total, _ = autoencoder_loss(outputs, sensor=sensor, thermal=thermal,
                                    arm=model.arm, per_sample=True)
    return total


def state_tensor_sha256(module_or_state) -> str:
    """Hash actual tensor bytes, not just parameter names. Accepts a Module or state dict.

    A manifest built from names and counts alone would be identical for two differently
    initialised models, so branch-copy equality and seed reproducibility must be checked
    against this digest. Checkpoints use this same schema (key, shape, dtype, bytes) so a
    saved state and a live model are directly comparable.
    """
    state = module_or_state.state_dict() if isinstance(module_or_state, nn.Module) else module_or_state
    h = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        t = tensor.detach().cpu().contiguous()
        h.update(key.encode())
        h.update(f"{tuple(t.shape)}|{t.dtype}".encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def _copy_branch(source: nn.Module, target: nn.Module, prefix: str) -> list[str]:
    target.load_state_dict(source.state_dict(), strict=True)
    return [f"{prefix}.{k}" for k in sorted(source.state_dict())]


def build_arm_set(objective: str, *, seed: int, sensor_dim: int = SENSOR_DIM):
    """Build S/T/F with branch initial weights made identical by explicit copying.

    Seeding alone is not enough: F builds its thermal encoder after its sensor encoder,
    so F's thermal RNG draws differ from T's. §3.3(3) forbids relying on call order, so
    every shared branch is copied and the copied parameter names are returned.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an int")
    models = {}
    for arm in ARMS:
        torch.manual_seed(seed)
        models[arm] = AnomalyDetector(arm, objective, sensor_dim=sensor_dim)
    copied = _copy_branch(models["S"].sensor_encoder, models["F"].sensor_encoder, "sensor_encoder")
    copied += _copy_branch(models["T"].thermal_encoder, models["F"].thermal_encoder, "thermal_encoder")
    if objective == "AE":
        copied += _copy_branch(models["S"].sensor_decoder, models["F"].sensor_decoder, "sensor_decoder")
        copied += _copy_branch(models["T"].thermal_decoder, models["F"].thermal_decoder, "thermal_decoder")
    # F's fusion has no counterpart; BIN heads exist in all three arms but copying one
    # arm's head into F would be an arbitrary choice, so heads stay independently drawn.
    uncopied = sorted(n for n, _ in models["F"].named_parameters()
                      if not any(n.startswith(p.split(".")[0]) for p in copied))
    shared = {"sensor_encoder": ("S", "F"), "thermal_encoder": ("T", "F")}
    if objective == "AE":
        shared.update(sensor_decoder=("S", "F"), thermal_decoder=("T", "F"))
    manifest = {"objective": objective, "seed": seed, "sensor_dim": sensor_dim,
                "copied_parameters": copied, "uncopied_F_parameters": uncopied,
                "copy_is_explicit_not_rng_order": True,
                "initial_state_sha256": {a: state_tensor_sha256(m) for a, m in models.items()},
                "shared_branch_sha256": {
                    branch: {arm: state_tensor_sha256(getattr(models[arm], branch)) for arm in arms}
                    for branch, arms in shared.items()},
                "parameter_counts": {a: sum(p.numel() for p in m.parameters())
                                     for a, m in models.items()},
                "trainable_parameter_counts": {a: sum(p.numel() for p in m.parameters() if p.requires_grad)
                                               for a, m in models.items()},
                "gradient_training_runs": 0}
    manifest["sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return models, manifest


def structure_manifest(sensor_dim: int = SENSOR_DIM) -> dict:
    """Structure record for the review document; builds models but never trains."""
    record = {"window_ticks": WINDOW_TICKS, "sensor_dim": sensor_dim,
              "thermal_shape": list(THERMAL_SHAPE), "hidden_dim": HIDDEN_DIM,
              "encoder_lstm_layers": NUM_LAYERS, "lags": list(LAGS),
              "frame_feature_shape": list(FRAME_FEATURE_SHAPE),
              "decoder": {"input": "bottleneck_and_tick_position_only",
                          "position": "nn.Parameter_not_nn.Embedding_for_deterministic_backward",
                          "lstm_layers": 1, "skip_connections": False,
                          "teacher_forcing": False,
                          "thermal_upsampling": "3x_ConvTranspose2d_stride2_then_Conv2d_1ch"},
              "fusion": "concat_256_then_Linear_128",
              "ae_loss": "element_mean_MSE_per_modality_F_is_0.5_each",
              "bin_head": "Linear(128,1)_BCEWithLogits",
              "supcon_or_4class_head": False,
              "training_contract": {k: list(v) if isinstance(v, tuple) else v
                                    for k, v in TRAINING_CONTRACT.items()},
              "status": "implementation_candidate_pending_review_and_server_verification",
              "protocol_frozen": False,
              "full_P0_gate": "NOT_EVALUATED"}
    for objective in OBJECTIVES:
        _, manifest = build_arm_set(objective, seed=42, sensor_dim=sensor_dim)
        record[f"{objective}_arm_set"] = {k: manifest[k] for k in
                                         ("parameter_counts", "copied_parameters",
                                          "uncopied_F_parameters", "sha256")}
    record["sha256"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return record
