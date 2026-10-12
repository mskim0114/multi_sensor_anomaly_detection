"""Intervention event log and human state annotations (labels) for received units.

Two layers, kept apart on purpose (docs/LABELING_GUIDE.md):

1. **Intervention events** are recorded live by the operator ([개입 시작] / [개입 종료] on the
   web while lighting a lighter, opening the ethanol dish, shaking dust). They are facts about
   what was done and when, never state labels. Times are hub clock; the device clock offset
   measured from heartbeats is stored with each event so it can be mapped to device ticks.
2. **Annotations** are made afterwards by a person, per unit, as tick intervals with a label,
   a confidence and optional per-modality response onsets. Unlabelled ticks stay unlabelled.
   Every save is a new immutable revision bound to the unit's manifest SHA-256; nothing in the
   raw unit is ever changed. Nothing here derives a label from data.

The baseline reference (mean ± k·std of the experiment's baseline phase) is a visual aid only.
"""
from __future__ import annotations

import bisect
import csv
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import secrets
import time

from .store import HubError, Store, _atomic_write, _json_bytes, _read_json, check_device, check_unit, utc_now

ANNOTATION_SCHEMA = "anno-v1"
CLASSES = ("normal", "combustion", "vapor_release", "dust")
LABELS = CLASSES + ("transition", "invalid")
CONFIDENCE = ("high", "medium", "low")
MODALITIES = ("flir", "ntc", "pm", "gas", "co2", "ct")
EVENT_KINDS = ("intervention_start", "intervention_end", "note")
EVENT_SCENARIOS = ("combustion", "vapor_release", "dust", "other")
MAX_INTERVALS = 500
BASELINE_K = 3.0


def _epoch(text: str | None) -> float | None:
    if not isinstance(text, str):
        return None
    try:
        value = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.timestamp()


class Labels:
    def __init__(self, store: Store):
        self.store = store
        self.root = store.root / "annotations" / ANNOTATION_SCHEMA
        self.root.mkdir(parents=True, exist_ok=True)
        with store._db_lock:
            store.db.executescript("""
                CREATE TABLE IF NOT EXISTS events(
                    id TEXT PRIMARY KEY, device TEXT, kind TEXT, scenario TEXT, note TEXT,
                    hub_epoch REAL, hub_utc TEXT, device_clock_offset_s REAL, user TEXT,
                    voided INTEGER DEFAULT 0, voided_by TEXT, voided_utc TEXT);
                CREATE INDEX IF NOT EXISTS events_device_epoch ON events(device, hub_epoch);
            """)
            store.db.commit()

    # ------------------------------------------------------------- events
    def add_event(self, device: str, kind: str, user: str, scenario: str | None = None, note: str = "") -> dict:
        check_device(device)
        if kind not in EVENT_KINDS:
            raise HubError(400, "kind must be intervention_start, intervention_end or note")
        if scenario is not None and scenario not in EVENT_SCENARIOS:
            raise HubError(400, "unknown scenario")
        if device not in self.store.devices():
            raise HubError(404, "no such device")
        status = (self.store.live.get(device) or {}).get("status") or {}
        offset = status.get("clock_offset_s")
        record = {"id": secrets.token_hex(12), "device": device, "kind": kind, "scenario": scenario,
                  "note": (note or "")[:500], "hub_epoch": time.time(), "hub_utc": utc_now(),
                  "device_clock_offset_s": offset if isinstance(offset, (int, float)) else None, "user": user}
        with self.store._db_lock:
            self.store.db.execute("INSERT INTO events(id, device, kind, scenario, note, hub_epoch, hub_utc, "
                                  "device_clock_offset_s, user) VALUES(?,?,?,?,?,?,?,?,?)",
                                  tuple(record[k] for k in ("id", "device", "kind", "scenario", "note", "hub_epoch",
                                                            "hub_utc", "device_clock_offset_s", "user")))
            self.store.db.commit()
        return record

    def void_event(self, event_id: str, user: str) -> dict:
        """Mistaken clicks are voided (kept, marked), never deleted."""
        if not isinstance(event_id, str) or not re.fullmatch(r"[0-9a-f]{24}", event_id):
            raise HubError(400, "invalid event id")
        with self.store._db_lock:
            cursor = self.store.db.execute("UPDATE events SET voided=1, voided_by=?, voided_utc=? WHERE id=? AND voided=0",
                                           (user, utc_now(), event_id))
            self.store.db.commit()
        if cursor.rowcount != 1:
            raise HubError(404, "no such active event")
        return {"id": event_id, "voided": True}

    def events(self, device: str, since_epoch: float = 0.0, until_epoch: float | None = None,
               include_voided: bool = True) -> list[dict]:
        check_device(device)
        query = ("SELECT id, kind, scenario, note, hub_epoch, hub_utc, device_clock_offset_s, user, voided, voided_by "
                 "FROM events WHERE device=? AND hub_epoch>=?")
        args = [device, since_epoch]
        if until_epoch is not None:
            query += " AND hub_epoch<=?"
            args.append(until_epoch)
        if not include_voided:
            query += " AND voided=0"
        with self.store._db_lock:
            rows = self.store.db.execute(query + " ORDER BY hub_epoch", args).fetchall()
        keys = ("id", "kind", "scenario", "note", "hub_epoch", "hub_utc", "device_clock_offset_s", "user", "voided",
                "voided_by")
        return [dict(zip(keys, r), voided=bool(r[8])) for r in rows]

    def open_intervention(self, device: str) -> dict | None:
        """The latest non-voided start without a later end (for the live panel)."""
        last = None
        for event in self.events(device, time.time() - 3600, include_voided=False):
            if event["kind"] == "intervention_start":
                last = event
            elif event["kind"] == "intervention_end":
                last = None
        return last

    # --------------------------------------------------------- unit timeline
    def _timeline(self, device: str, unit: str) -> tuple[list[float], dict]:
        base = self.store.raw_dir(device, unit)
        if not self.store._committed(device, unit) or not (base / "scalars.csv").is_file():
            raise HubError(404, "unit not found or has no scalars.csv")
        stamps = []
        with open(base / "scalars.csv", newline="") as stream:
            for row in csv.DictReader(stream):
                stamps.append(_epoch(row.get("timestamp_utc")))
        experiment = _read_json(base / "experiment.json") or {}
        return stamps, experiment

    def unit_events(self, device: str, unit: str) -> dict:
        """Events overlapping the unit, mapped to the nearest device tick."""
        stamps, _ = self._timeline(device, unit)
        known = [(t, i) for i, t in enumerate(stamps) if t is not None]
        if not known:
            return {"ticks": len(stamps), "events": []}
        times = [t for t, _ in known]
        first, last = times[0], times[-1]
        out = []
        for event in self.events(device, first - 3600, last + 3600):
            offset = event["device_clock_offset_s"] or 0.0
            device_time = event["hub_epoch"] - offset          # device clock ~= hub clock - offset
            if not first - 5 <= device_time <= last + 5:
                continue
            j = bisect.bisect_left(times, device_time)
            candidates = [k for k in (j - 1, j) if 0 <= k < len(times)]
            k = min(candidates, key=lambda c: abs(times[c] - device_time))
            out.append(dict(event, tick=known[k][1], tick_error_s=round(abs(times[k] - device_time), 3),
                            clock_offset_known=event["device_clock_offset_s"] is not None))
        return {"ticks": len(stamps), "events": out}

    def baseline(self, device: str, unit: str, columns: list[str]) -> dict:
        """mean ± k·std per column over the experiment's baseline phase (visual reference only)."""
        base = self.store.raw_dir(device, unit)
        _, experiment = self._timeline(device, unit)
        phase = next((p for p in experiment.get("phase_plan") or [] if p.get("phase") == "baseline"), None)
        if not phase:
            return {"available": False, "reason": "no baseline phase in this unit"}
        lo, hi = int(phase["start_tick"]), int(phase["end_tick"])
        values: dict[str, list[float]] = {c: [] for c in columns[:8]}
        with open(base / "scalars.csv", newline="") as stream:
            for i, row in enumerate(csv.DictReader(stream)):
                if lo <= i <= hi:
                    for c in values:
                        try:
                            x = float(row.get(c, ""))
                        except ValueError:
                            continue
                        if math.isfinite(x):
                            values[c].append(x)
        stats = {}
        for c, xs in values.items():
            if len(xs) >= 10:
                mean = sum(xs) / len(xs)
                std = math.sqrt(sum((x - mean) ** 2 for x in xs) / len(xs))
                stats[c] = {"mean": mean, "std": std, "low": mean - BASELINE_K * std, "high": mean + BASELINE_K * std,
                            "n": len(xs)}
        return {"available": True, "phase": {"start_tick": lo, "end_tick": hi}, "k": BASELINE_K, "stats": stats,
                "note": "visual reference only; not a threshold and not a label"}

    # ---------------------------------------------------------- annotations
    def _dir(self, device: str, unit: str) -> Path:
        check_device(device)
        return self.root / device / Path(*check_unit(unit))

    def revisions(self, device: str, unit: str) -> list[Path]:
        directory = self._dir(device, unit)
        return sorted(directory.glob("rev-*.json")) if directory.is_dir() else []

    def annotation(self, device: str, unit: str) -> dict:
        revisions = self.revisions(device, unit)
        current = _read_json(revisions[-1]) if revisions else None
        return {"schema": ANNOTATION_SCHEMA, "labels": list(LABELS), "classes": list(CLASSES),
                "confidence": list(CONFIDENCE), "modalities": list(MODALITIES),
                "revision": len(revisions), "current": current,
                "history": [{"revision": i + 1, "annotator": (_read_json(p) or {}).get("annotator"),
                             "saved_utc": (_read_json(p) or {}).get("saved_utc")} for i, p in enumerate(revisions)]}

    def save(self, device: str, unit: str, user: str, body: dict) -> dict:
        receipt = self.store._committed(device, unit)
        if not receipt:
            raise HubError(404, "unit not found")
        stamps, experiment = self._timeline(device, unit)
        n = len(stamps)
        base_revision = body.get("base_revision")
        if isinstance(base_revision, bool) or not isinstance(base_revision, int):
            raise HubError(400, "base_revision (int) is required")
        intervals = body.get("intervals")
        if not isinstance(intervals, list) or len(intervals) > MAX_INTERVALS:
            raise HubError(400, "intervals must be a list")
        clean = []
        for item in intervals:
            if not isinstance(item, dict):
                raise HubError(400, "intervals must be objects")
            start, end, label = item.get("start_tick"), item.get("end_tick"), item.get("label")
            if any(isinstance(v, bool) or not isinstance(v, int) for v in (start, end)) or not 0 <= start <= end < n:
                raise HubError(400, f"interval ticks must satisfy 0 <= start <= end < {n}")
            if label not in LABELS:
                raise HubError(400, f"label must be one of {', '.join(LABELS)}")
            confidence = item.get("confidence", "high")
            if confidence not in CONFIDENCE:
                raise HubError(400, "confidence must be high, medium or low")
            clean.append({"start_tick": start, "end_tick": end, "label": label, "confidence": confidence,
                          "note": str(item.get("note") or "")[:300]})
        clean.sort(key=lambda x: x["start_tick"])
        for a, b in zip(clean, clean[1:]):
            if b["start_tick"] <= a["end_tick"]:
                raise HubError(400, f"intervals overlap at tick {b['start_tick']}")
        onsets = body.get("onsets") or {}
        if not isinstance(onsets, dict):
            raise HubError(400, "onsets must be an object")
        clean_onsets = {}
        for modality, tick in onsets.items():
            if modality not in MODALITIES:
                raise HubError(400, f"unknown modality {modality!r}")
            if tick is not None and (isinstance(tick, bool) or not isinstance(tick, int) or not 0 <= tick < n):
                raise HubError(400, f"onset for {modality} out of range")
            clean_onsets[modality] = tick
        directory = self._dir(device, unit)
        revisions = self.revisions(device, unit)
        if base_revision != len(revisions):
            raise HubError(409, "annotation changed since you loaded it; reload first", revision=len(revisions))
        record = {"schema": ANNOTATION_SCHEMA, "device": device, "unit": unit,
                  "manifest_sha256": receipt["manifest_sha256"], "ticks": n,
                  "scenario_id": experiment.get("scenario_id"), "revision": len(revisions) + 1,
                  "annotator": user, "saved_utc": utc_now(), "guide_version": GUIDE_VERSION,
                  "intervals": clean, "onsets": clean_onsets, "note": str(body.get("note") or "")[:1000],
                  "events_snapshot": self.unit_events(device, unit)["events"]}
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"rev-{record['revision']:04d}.json"
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)   # a revision is never rewritten
        except FileExistsError:
            raise HubError(409, "another save won the race; reload first", revision=len(self.revisions(device, unit)))
        with os.fdopen(fd, "wb") as stream:
            stream.write(_json_bytes(record))
            stream.flush()
            os.fsync(stream.fileno())
        return {"revision": record["revision"], "saved_utc": record["saved_utc"]}

    def export(self) -> list[dict]:
        """Current revision of every annotated unit (input of the dataset builder, P2)."""
        out = []
        for directory in sorted({p.parent for p in self.root.rglob("rev-*.json")}):
            revisions = sorted(directory.glob("rev-*.json"))
            current = _read_json(revisions[-1])
            if current:
                receipt = self.store._committed(current["device"], current["unit"])
                current["raw_matches"] = bool(receipt and receipt["manifest_sha256"] == current["manifest_sha256"])
                out.append(current)
        return out


GUIDE_VERSION = "labeling-guide-v1"
