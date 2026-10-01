"""EXP-004 BIN metadata schedule and dev weights; no tensors, readers or torch.

Build separate fit/dev indexes from selected pure-Normal windows and all nine
nominal bank records per base. The caller must first verify the original manifest
file SHA and split/normalizer provenance. This leaf checks the selected metadata
and hashes its canonical content; that digest is not the original JSONL file SHA.

Fit: one Normal and one accepted synthetic per base/epoch, grouped in pairs.
Event cycles and base ordering depend on epoch, fold and training seed only.
Dev: half the weight per class, then equal equipment, equal bases within equipment,
and equal accepted events within a base. Invisible positives remain positives.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re

import numpy as np


PROTOCOL = "synthetic-modality-v3-proposed"
SCHEMA = "exp004-base-paired-bin-v1"
GRID = frozenset((f, s) for f in ("sensor-only", "thermal-only", "coupled")
                 for s in (1, 2, 4))
CHANGES = {"sensor-only": (True, False), "thermal-only": (False, True),
           "coupled": (True, True)}


def _digest(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative JSON integer (not bool)")
    return value


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _sha(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


@dataclass(frozen=True)
class EventRef:
    event_id: str
    event_instance_id: str
    family: str
    strength: int
    sensor_changed: bool
    thermal_changed: bool


@dataclass(frozen=True)
class BaseEvents:
    base_window_id: str
    device: str
    window_index: int
    events: tuple[EventRef, ...]


@dataclass(frozen=True)
class EventIndex:
    """Immutable selected metadata, constructed by ``build_event_index``."""

    role: str
    fold: int
    generator_seed: int
    ct_channel: str
    normalizer_sha256: str
    fit_ids_sha256: str
    bases: tuple[BaseEvents, ...]
    selected_window_ids: tuple[str, ...]
    sha256: str

    def record(self):
        return {"schema": SCHEMA, "protocol_id": PROTOCOL, "role": self.role,
                "fold": self.fold, "generator_seed": self.generator_seed,
                "ct_channel": self.ct_channel,
                "normalizer_sha256": self.normalizer_sha256,
                "fit_ids_sha256": self.fit_ids_sha256,
                "selected_metadata_sha256": self.sha256,
                "selected_window_order_sha256": _digest(self.selected_window_ids),
                "base_count": len(self.bases),
                "accepted_events": sum(len(b.events) for b in self.bases),
                "devices": sorted({b.device for b in self.bases})}


def build_event_index(windows, roles, events, *, role, fold, generator_seed,
                      ct_channel, normalizer_sha256, fit_ids_sha256):
    """Validate one selected role's complete nominal bank before any scheduling.

    ``windows`` and ``events`` must already be filtered to the requested role.
    Injection metadata lacks role/device/base_window_id; the bank writer must
    add those caller-owned fields before passing its complete nominal records.
    Calibration/heldout entries, missing bases/equipment, duplicate or truncated
    grids, foreign namespaces and accepted no-ops fail rather than being ignored.
    Old frozen banks may omit event_instance_id: it is explicitly derived from
    [namespace, ct_channel], and a supplied value must match that definition.
    """
    if role not in ("fit", "dev"):
        raise ValueError("BIN indexes permit fit/dev only")
    if _integer(fold, "fold") not in range(4):
        raise ValueError("fold must be in 0..3")
    _integer(generator_seed, "generator_seed")
    if ct_channel not in ("CT1", "CT2"):
        raise ValueError("unknown CT channel")
    _sha(normalizer_sha256, "normalizer_sha256")
    _sha(fit_ids_sha256, "fit_ids_sha256")
    if not isinstance(roles, dict):
        raise ValueError("roles must be the equipment role mapping")
    expected_devices = {d for d, r in roles.items() if r == role}
    selected, positions = {}, {}
    for position, window in enumerate(windows):
        identifier = _identifier(window.get("base_window_id"), "base_window_id")
        device = _identifier(window.get("device"), "device")
        if roles.get(device) != role:
            raise ValueError(f"window {identifier} is outside selected role {role}")
        if window.get("hard_valid") is not True or window.get("pure_Normal") is not True:
            raise ValueError(f"window {identifier} must be hard-valid pure Normal")
        if identifier in selected:
            raise ValueError(f"duplicate base {identifier}")
        selected[identifier] = dict(window)
        positions[identifier] = position
    if not selected or {w["device"] for w in selected.values()} != expected_devices:
        raise ValueError(f"selected {role} windows do not cover all assigned equipment")

    nominal, accepted, seen, source = {}, {}, set(), []
    for event in events:
        identifier = event.get("base_window_id")
        if identifier not in selected:
            raise ValueError("event base is not a selected Normal window")
        if event.get("role") != role or event.get("device") != selected[identifier]["device"]:
            raise ValueError("event role/device disagrees with its base")
        family, strength = event.get("family"), event.get("strength")
        if type(strength) is not int or (family, strength) not in GRID:
            raise ValueError("event is outside the nominal 3-by-3 grid")
        expected = [PROTOCOL, fold, role, identifier, family, strength, generator_seed]
        namespace = event.get("namespace")
        if (not isinstance(namespace, list) or len(namespace) != 7
                or type(namespace[1]) is not int or type(namespace[5]) is not int
                or type(namespace[6]) is not int or namespace != expected):
            raise ValueError("event namespace disagrees with the selected bank")
        if event.get("ct_channel") != ct_channel:
            raise ValueError("event CT channel disagrees with the selected bank")
        if (event.get("normalizer_sha256") != normalizer_sha256
                or event.get("fit_ids_sha256") != fit_ids_sha256):
            raise ValueError("event fit/normalizer provenance mismatch")
        event_id = _digest(namespace)
        instance_id = _digest([namespace, ct_channel])
        if event.get("event_id") != event_id:
            raise ValueError("event ID does not hash its namespace")
        if "event_instance_id" in event and event["event_instance_id"] != instance_id:
            raise ValueError("event instance ID does not hash namespace and CT")
        cell = (family, strength)
        cells = nominal.setdefault(identifier, set())
        if event_id in seen or cell in cells:
            raise ValueError("duplicate nominal event")
        seen.add(event_id)
        cells.add(cell)
        if type(event.get("accepted")) is not bool:
            raise ValueError("accepted must be a JSON boolean")
        changed = event.get("sensor_changed"), event.get("thermal_changed")
        if any(type(c) is not bool for c in changed):
            raise ValueError("modality change flags must be JSON booleans")
        reasons = event.get("rejection_reasons")
        if (not isinstance(reasons, list) or any(not isinstance(r, str) for r in reasons)
                or bool(reasons) == event["accepted"]):
            raise ValueError("rejection reasons disagree with acceptance")
        if event["accepted"]:
            if changed != CHANGES[family]:
                raise ValueError("accepted event violates the modality-change contract")
            accepted.setdefault(identifier, []).append(
                EventRef(event_id, instance_id, family, strength, *changed))
        source.append(dict(event))
    if set(nominal) != set(selected) or any(cells != GRID for cells in nominal.values()):
        raise ValueError("every base must have the complete nine nominal events")
    if set(accepted) != set(selected):
        raise ValueError("every base must have at least one accepted event")
    bases = tuple(BaseEvents(i, selected[i]["device"], positions[i],
                            tuple(sorted(accepted[i], key=lambda e: e.event_id)))
                  for i in sorted(selected))
    window_ids = tuple(selected)
    binding = {"schema": SCHEMA, "protocol_id": PROTOCOL, "role": role, "fold": fold,
               "generator_seed": generator_seed, "ct_channel": ct_channel,
               "normalizer_sha256": normalizer_sha256, "fit_ids_sha256": fit_ids_sha256,
               "selected_window_ids": window_ids,
               "windows": [selected[i] for i in sorted(selected)],
               "nominal_events": sorted(source, key=lambda e: e["event_id"])}
    return EventIndex(role, fold, generator_seed, ct_channel, normalizer_sha256,
                      fit_ids_sha256, bases, window_ids, _digest(binding))


@dataclass(frozen=True)
class BinPair:
    base_window_id: str
    device: str
    window_index: int
    event_id: str
    event_instance_id: str
    family: str
    strength: int
    ct_channel: str
    sensor_changed: bool
    thermal_changed: bool


@dataclass(frozen=True)
class BasePairedSampler:
    index: EventIndex
    training_seed: int
    paired_bases_per_batch: int = 8
    planned_epochs: int = 30

    def __post_init__(self):
        if not isinstance(self.index, EventIndex) or self.index.role != "fit":
            raise ValueError("fit sampler requires a validated fit EventIndex")
        _integer(self.training_seed, "training_seed")
        if _integer(self.paired_bases_per_batch, "paired_bases_per_batch") == 0:
            raise ValueError("a batch must contain at least one pair")
        if _integer(self.planned_epochs, "planned_epochs") == 0:
            raise ValueError("planned_epochs must be positive")

    def _permutation(self, count, *suffix):
        # Same explicit digest-to-PCG64 conversion as the generator, but a
        # separate training namespace. No modality, model, CT or generator seed.
        key = [SCHEMA, PROTOCOL, self.index.fold, self.training_seed, "fit", *suffix]
        rng = np.random.Generator(np.random.PCG64(int(_digest(key)[:32], 16)))
        return rng.permutation(count)

    def epoch_pairs(self, epoch):
        """One-based epoch, restricted to 1..planned_epochs, like the P1a runner.

        Internally e=epoch-1: event permutation cycle=floor(e/k), offset=e%k.
        window_index refers to the original role-filtered windows order, not the
        combined fit+dev cache index. The runner
        must assert that window's base ID before using its full cache identity,
        inject via the supplied family/strength/CT, then verify accepted=True,
        IDs and change flags against this pair before normalizing/projection.
        """
        _integer(epoch, "epoch")
        if not 1 <= epoch <= self.planned_epochs:
            raise ValueError("epoch must be in 1..planned_epochs")
        e = epoch - 1
        pairs = []
        for i in self._permutation(len(self.index.bases), "base-order", e):
            base = self.index.bases[int(i)]
            cycle, offset = divmod(e, len(base.events))
            order = self._permutation(len(base.events), "event-cycle", base.base_window_id, cycle)
            event = base.events[int(order[offset])]
            pairs.append(BinPair(base.base_window_id, base.device, base.window_index,
                                 event.event_id, event.event_instance_id,
                                 event.family, event.strength, self.index.ct_channel,
                                 event.sensor_changed, event.thermal_changed))
        return tuple(pairs)

    def epoch_batches(self, epoch):
        """Each pair expands to [Normal(label=0), synthetic(label=1)] in the runner.

        Microbatching may split these examples but must accumulate loss over the
        actual effective batch and apply one optimizer update after all examples.
        """
        pairs = self.epoch_pairs(epoch)
        n = self.paired_bases_per_batch
        return tuple(pairs[i:i + n] for i in range(0, len(pairs), n))

    def epoch_manifest(self, epoch):
        pairs = self.epoch_pairs(epoch)
        record = {**self.index.record(), "training_seed": self.training_seed,
                  "epoch": epoch, "epoch_zero_based": epoch - 1,
                  "planned_epochs": self.planned_epochs,
                  "rng": "PCG64(int(SHA256(canonical_JSON(namespace))[:32],16))",
                  "rng_namespace": "[schema,protocol,fold,training_seed,fit,kind,...]",
                  "policy": "one_Normal_and_one_accepted_per_base_per_epoch",
                  "paired_bases_per_batch": self.paired_bases_per_batch,
                  "example_order_within_pair": ["Normal", "synthetic"],
                  "normal_examples": len(pairs), "synthetic_examples": len(pairs),
                  "optimizer_steps": math.ceil(len(pairs) / self.paired_bases_per_batch),
                  "pairs": [asdict(p) for p in pairs]}
        record["sha256"] = _digest(record)
        return record


@dataclass(frozen=True)
class WeightedExample:
    base_window_id: str
    device: str
    window_index: int
    event_id: str | None
    event_instance_id: str | None
    family: str | None
    strength: int | None
    ct_channel: str
    sensor_changed: bool
    thermal_changed: bool
    label: int
    weight: float


def dev_examples(index):
    """All dev examples with weights summing to 1 (1/2 for each class).

    Use sum(weight * unreduced_BCE), not a second mean or a batch-size division.
    Normal event_id is None; every accepted synthetic has label 1, irrespective
    of what the evaluated modality can observe. Calibration weights are separate.
    """
    if not isinstance(index, EventIndex) or index.role != "dev":
        raise ValueError("dev weights require a validated dev EventIndex")
    counts = Counter(b.device for b in index.bases)
    examples = []
    for base in sorted(index.bases, key=lambda b: (b.device, b.base_window_id)):
        weight = 0.5 / (len(counts) * counts[base.device])
        examples.append(WeightedExample(base.base_window_id, base.device, base.window_index,
                                        None, None, None, None, index.ct_channel,
                                        False, False, 0, weight))
        for event in base.events:
            examples.append(WeightedExample(base.base_window_id, base.device, base.window_index,
                                            event.event_id, event.event_instance_id,
                                            event.family, event.strength, index.ct_channel,
                                            event.sensor_changed, event.thermal_changed,
                                            1, weight / len(base.events)))
    return tuple(examples)


def dev_weight_manifest(index):
    examples = dev_examples(index)
    record = {**index.record(), "policy": "class_then_equipment_then_base_then_event",
              "checkpoint": "minimum_weighted_BCE_earlier_epoch_on_exact_tie",
              "class_weight_sum": {str(label): math.fsum(e.weight for e in examples
                                                         if e.label == label)
                                   for label in (0, 1)},
              "examples": [asdict(e) for e in examples]}
    record["sha256"] = _digest(record)
    return record
