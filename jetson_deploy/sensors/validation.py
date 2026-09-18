"""Verify the actual saved acquisition, including thermal chunk references."""
from __future__ import annotations

import json
import csv
from pathlib import Path


def validate_run(run_dir: Path, *, save_thermal: bool,
                 expected_snapshots: int | None = None) -> dict:
    import numpy as np

    errors = []
    count = 0
    frames_checked = 0
    csv_count = 0
    previous_sequence = None
    cached_name = None
    cached_frames = cached_sequences = None
    try:
        with (run_dir / "snapshots.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                snap = json.loads(line)
                sequence = snap["sequence"]
                if previous_sequence is None and sequence != 0:
                    raise ValueError(f"first sequence must be 0, got {sequence}")
                if previous_sequence is not None and sequence != previous_sequence + 1:
                    raise ValueError(f"non-consecutive sequence: {previous_sequence} -> {sequence}")
                previous_sequence = sequence
                count += 1
                thermal = snap["sensors"]["flir"]
                name = thermal.get("thermal_chunk")
                if not name:
                    if save_thermal and thermal["status"] == "ok":
                        raise ValueError(f"tick {sequence}: thermal frame has no saved reference")
                    continue
                if Path(name).name != name or not name.startswith("thermal_"):
                    raise ValueError(f"invalid thermal chunk path: {name}")
                if name != cached_name:
                    with np.load(run_dir / name, allow_pickle=False) as chunk:
                        cached_frames = chunk["frames"]
                        cached_sequences = chunk["sequences"]
                    cached_name = name
                    if (cached_frames.dtype != np.uint16 or cached_frames.shape[1:] != (120, 160)
                            or len(cached_frames) != len(cached_sequences)):
                        raise ValueError(f"invalid thermal arrays: {name}")
                index = thermal["thermal_index"]
                if not isinstance(index, int) or not 0 <= index < len(cached_frames):
                    raise ValueError(f"tick {sequence}: invalid index in {name}")
                stored_sequence = int(cached_sequences[index])
                if stored_sequence != sequence:
                    raise ValueError(f"tick {sequence}: wrong saved sequence {stored_sequence}")
                frames_checked += 1
        with (run_dir / "scalars.csv").open(newline="", encoding="utf-8") as stream:
            for csv_count, row in enumerate(csv.DictReader(stream), 1):
                if int(row["sequence"]) != csv_count - 1:
                    raise ValueError(f"CSV sequence mismatch on row {csv_count}")
        if csv_count != count:
            raise ValueError(f"CSV rows {csv_count} != JSONL snapshots {count}")
        if expected_snapshots is not None and count != expected_snapshots:
            raise ValueError(f"saved snapshots {count} != collector count {expected_snapshots}")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    if count == 0:
        errors.append("no snapshots saved")
    return {"passed": not errors, "snapshots_checked": count,
            "csv_rows_checked": csv_count,
            "thermal_references_checked": frames_checked, "errors": errors}
