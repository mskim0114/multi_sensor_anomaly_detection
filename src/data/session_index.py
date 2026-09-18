"""Session discovery and index building.

Scans flat directories of CSV/BIN/JSON files, groups them into temporal
sessions, and caches the result as a pickle for fast reuse.
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
import pickle
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Bump when session grouping or label extraction semantics change.
INDEX_CACHE_VERSION = 1


@dataclass
class SessionInfo:
    session_id: str          # e.g. "agv02_0902_1306"
    device_id: str           # e.g. "agv02"
    device_type: str         # "agv" | "oht"
    basenames: list[str]     # sorted chronologically
    labels: list[int]        # parallel to basenames

    def __len__(self) -> int:
        return len(self.basenames)


@dataclass
class DatasetIndex:
    sessions: list[SessionInfo] = field(default_factory=list)
    split: str = ""

    @property
    def total_samples(self) -> int:
        return sum(len(s) for s in self.sessions)

    def summary(self) -> str:
        n_agv = sum(1 for s in self.sessions if s.device_type == "agv")
        n_oht = sum(1 for s in self.sessions if s.device_type == "oht")
        return (
            f"DatasetIndex(split={self.split}, sessions={len(self.sessions)}, "
            f"agv={n_agv}, oht={n_oht}, total_samples={self.total_samples})"
        )


# Filename pattern: {device}_{MMDD}_{HHMMSS}
_FNAME_RE = re.compile(r"^((?:agv|oht)\d+)_(\d{4})_(\d{6})$")


def _parse_basename(basename: str) -> tuple[str, str, int] | None:
    """Parse basename into (device_id, date_str, seconds_since_midnight)."""
    m = _FNAME_RE.match(basename)
    if m is None:
        return None
    device_id = m.group(1)
    date_str = m.group(2)  # MMDD
    time_str = m.group(3)  # HHMMSS
    h, mi, s = int(time_str[:2]), int(time_str[2:4]), int(time_str[4:6])
    seconds = h * 3600 + mi * 60 + s
    return device_id, date_str, seconds


def _read_label(label_dir: str, basename: str) -> int:
    """Read state label from JSON file."""
    path = os.path.join(label_dir, basename + ".json")
    with open(path) as f:
        data = json.load(f)
    return int(data["annotations"][0]["tagging"][0]["state"])


def _read_labels_batch(label_dir: str, basenames: list[str]) -> list[int]:
    """Read multiple labels in parallel using thread pool.

    Raises RuntimeError if any label fails to load. Silently defaulting to
    label 0 (as an earlier version did) risks training on corrupted labels
    without the operator noticing.
    """
    results = {}
    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_bn = {
            executor.submit(_read_label, label_dir, bn): bn
            for bn in basenames
        }
        for future in as_completed(future_to_bn):
            bn = future_to_bn[future]
            try:
                results[bn] = future.result()
            except Exception as e:
                failures.append((bn, str(e)))

    if failures:
        summary = "; ".join(f"{bn}: {err}" for bn, err in failures[:5])
        raise RuntimeError(
            f"Failed to read {len(failures)} label file(s). "
            f"First few: {summary}"
        )

    return [results[bn] for bn in basenames]


def build_session_index(
    source_dir: str,
    label_dir: str,
    split: str = "",
    gap_threshold: int = 120,
) -> DatasetIndex:
    """Scan source directory and build session index.

    Args:
        source_dir: Directory containing CSV + BIN files.
        label_dir: Directory containing JSON label files.
        split: Label for this split ("train" or "val").
        gap_threshold: Max gap in seconds between consecutive samples
            within the same session. Gaps larger than this start a new session.
    """
    # Collect basenames from CSV files
    csv_files = [f[:-4] for f in os.listdir(source_dir) if f.endswith(".csv")]
    logger.info(f"Found {len(csv_files)} CSV files in {source_dir}")

    # Parse and group by (device_id, date)
    parsed: dict[tuple[str, str], list[tuple[int, str]]] = {}
    skipped = 0
    for bn in csv_files:
        result = _parse_basename(bn)
        if result is None:
            skipped += 1
            continue
        device_id, date_str, seconds = result
        key = (device_id, date_str)
        parsed.setdefault(key, []).append((seconds, bn))

    if skipped:
        logger.warning(f"Skipped {skipped} files with unrecognized names")

    # Collect all basenames first for batch label loading
    all_basenames = []
    session_boundaries: list[tuple[str, str, int, list[str]]] = []
    
    for (device_id, date_str), entries in sorted(parsed.items()):
        entries.sort()  # sort by seconds

        # Split by temporal gaps
        current_basenames: list[str] = [entries[0][1]]
        current_start_time = entries[0][0]
        prev_time = entries[0][0]

        for seconds, bn in entries[1:]:
            if seconds - prev_time > gap_threshold:
                # Record session boundary
                session_boundaries.append((device_id, date_str, current_start_time, current_basenames.copy()))
                all_basenames.extend(current_basenames)
                # Start new session
                current_basenames = [bn]
                current_start_time = seconds
            else:
                current_basenames.append(bn)
            prev_time = seconds

        # Flush last session
        if current_basenames:
            session_boundaries.append((device_id, date_str, current_start_time, current_basenames.copy()))
            all_basenames.extend(current_basenames)

    # Load all labels in parallel (major speedup!)
    logger.info(f"Loading {len(all_basenames)} labels in parallel...")
    all_labels = _read_labels_batch(label_dir, all_basenames)
    
    # Build session index from pre-loaded labels
    sessions: list[SessionInfo] = []
    label_idx = 0
    for device_id, date_str, start_time, basenames in session_boundaries:
        n = len(basenames)
        labels = all_labels[label_idx:label_idx + n]
        label_idx += n
        session_id = f"{device_id}_{date_str}_{basenames[0].split('_')[-1][:4]}"
        device_type = "agv" if "agv" in device_id else "oht"
        sessions.append(SessionInfo(
            session_id=session_id,
            device_id=device_id,
            device_type=device_type,
            basenames=basenames,
            labels=labels,
        ))

    sessions.sort(key=lambda s: s.session_id)
    index = DatasetIndex(sessions=sessions, split=split)
    logger.info(f"Built index: {index.summary()}")

    # Warn about short sessions
    short = [s for s in sessions if len(s) < 30]
    if short:
        logger.warning(
            f"{len(short)} sessions have fewer than 30 samples: "
            f"{[s.session_id for s in short[:5]]}"
        )

    return index


def _index_provenance(
    source_dir: str, label_dir: str, split: str, gap_threshold: int
) -> dict:
    """Fingerprint index inputs without opening sensor/thermal data.

    Index membership depends on recognized CSV names; labels depend on JSON
    contents. Size, mtime and ctime detect ordinary label edits/restores without
    parsing every JSON on a cache hit. This is cache invalidation metadata, NOT
    a cryptographic proof of dataset integrity; raw manifest checks remain required.
    """
    source = Path(source_dir).expanduser().resolve()
    labels = Path(label_dir).expanduser().resolve()
    with os.scandir(source) as entries:
        basenames = sorted(
            entry.name[:-4] for entry in entries
            if entry.name.endswith(".csv") and entry.is_file()
            and _parse_basename(entry.name[:-4]) is not None
        )
    wanted = {bn + ".json" for bn in basenames}
    label_stats = {}
    with os.scandir(labels) as entries:
        for entry in entries:
            if entry.name in wanted and entry.is_file():
                stat = entry.stat()
                label_stats[entry.name] = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    signature = hashlib.sha256()
    for bn in basenames:
        name = bn + ".json"
        if name not in label_stats:
            raise FileNotFoundError(f"Missing label for indexed sample: {labels / name}")
        signature.update(json.dumps([bn, *label_stats[name]], separators=(",", ":")).encode())
        signature.update(b"\n")
    return {
        "version": INDEX_CACHE_VERSION,
        "source_dir": str(source),
        "label_dir": str(labels),
        "split": split,
        "gap_threshold": gap_threshold,
        "input_signature": signature.hexdigest(),
    }


def load_or_build_index(
    source_dir: str,
    label_dir: str,
    cache_dir: str,
    split: str = "",
    gap_threshold: int = 120,
    force_rebuild: bool = False,
) -> DatasetIndex:
    """Reuse a local, trusted cache only when its index inputs still match.

    Legacy bare DatasetIndex pickles and invalid caches are rebuilt. Never
    accept cache pickles from an untrusted source. This cache does not validate
    CSV/BIN contents or replace the dataset integrity gate.
    """
    provenance = _index_provenance(source_dir, label_dir, split, gap_threshold)
    cache_path = Path(cache_dir) / f"session_index_{split}.pkl"

    if cache_path.exists() and not force_rebuild:
        try:
            with cache_path.open("rb") as f:
                cached = pickle.load(f)
            if (isinstance(cached, dict)
                    and cached.get("provenance") == provenance
                    and isinstance(cached.get("index"), DatasetIndex)
                    and cached["index"].split == split):
                index = cached["index"]
                logger.info(f"Loaded matching cache: {index.summary()}")
                return index
            logger.warning(f"Rebuilding legacy or stale index cache: {cache_path}")
        except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ImportError,
                ValueError, TypeError) as exc:
            logger.warning(f"Rebuilding unreadable index cache {cache_path}: {exc}")

    index = build_session_index(source_dir, label_dir, split, gap_threshold)
    if _index_provenance(source_dir, label_dir, split, gap_threshold) != provenance:
        raise RuntimeError("Index inputs changed during build; no cache was written")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=cache_path.parent,
                                         prefix=cache_path.name + ".", delete=False) as f:
            temporary = Path(f.name)
            pickle.dump({"provenance": provenance, "index": index}, f)
        os.replace(temporary, cache_path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    logger.info(f"Saved index to {cache_path}")

    return index
