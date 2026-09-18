"""Disposable, bounded live previews; acquisition files remain authoritative."""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import stat
import sys
import tempfile
import threading
import time

from .runtime import process_identity

MAX_PREVIEW_BYTES = 512 * 1024


def preview_dir() -> Path:
    """Use user-owned volatile storage, never add a per-tick flash fsync."""
    uid = os.getuid()
    user_runtime = Path("/run/user") / str(uid)
    if user_runtime.is_dir() and not user_runtime.is_symlink() and user_runtime.stat().st_uid == uid:
        root = user_runtime / "factory-safety-preview"
    else:
        root = Path("/dev/shm") / f"factory-safety-preview-{uid}"
    root.mkdir(mode=0o700, exist_ok=True)
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != uid or info.st_mode & 0o077:
        raise PermissionError(f"Preview directory must be private and user-owned: {root}")
    return root


def preview_path(run_dir: Path, root: Path | None = None) -> Path:
    key = hashlib.sha256(str(Path(run_dir).resolve()).encode()).hexdigest()[:24]
    return (root if root is not None else preview_dir()) / f"run-{key}.json"


def clean_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    return value


def volatile_json(path: Path, value: dict) -> None:
    """Atomic visibility only: this cache is neither durable nor raw data."""
    data = json.dumps(clean_json(value), ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")
    if len(data) > MAX_PREVIEW_BYTES:
        raise ValueError("Live preview exceeds its bounded size")
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def thermal_payload(snapshot: dict, selected_frame) -> dict | None:
    """Encode exactly the frame selected for this snapshot, not camera latest."""
    if selected_frame is None:
        return None
    sequence, frame = selected_frame
    if sequence != snapshot["sequence"] or frame is None:
        return None
    if frame.shape != (120, 160) or frame.dtype.kind != "u" or frame.dtype.itemsize != 2:
        raise ValueError("Preview frame must be 120x160 uint16")
    flir = snapshot["sensors"].get("flir") or {}
    return {"encoding": "base64", "dtype": "<u2", "shape": [120, 160],
            "data": base64.b64encode(frame.astype("<u2", copy=False).tobytes()).decode("ascii"),
            "snapshot_sequence": sequence, "frame_sequence": flir.get("frame_sequence"),
            "conversion": "raw / 100 - 273.15"}


class PreviewPublisher:
    """A latest-only worker: queue pressure or failure never waits on hardware."""

    def __init__(self, run_dir: Path, *, root: Path | None = None):
        self.run_dir = Path(run_dir).resolve()
        self.root = root if root is not None else preview_dir()
        self.path = preview_path(self.run_dir, self.root)
        self.identity = process_identity(os.getpid())
        self.queue = queue.Queue(maxsize=1)
        self.closed = threading.Event()
        self.submitted = 0
        self.published = 0
        self.dropped = 0
        self.error = None
        self.thread = threading.Thread(target=self._work, name="sensor-preview", daemon=True)

    def start(self):
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        volatile_json(self.root / "active.json", {
            "run_dir": str(self.run_dir), "identity": self.identity,
            "started_monotonic_ns": time.monotonic_ns()})
        # Cache growth stays bounded across repeated collection runs.
        previous = sorted(self.root.glob("run-*.json"), key=lambda p: p.stat().st_mtime_ns,
                          reverse=True)
        for path in previous[7:]:
            if path != self.path:
                path.unlink(missing_ok=True)
        self.thread.start()

    def offer(self, snapshot: dict, selected_frame=None) -> None:
        if self.closed.is_set() or self.error is not None:
            return
        self.submitted += 1
        item = (snapshot, selected_frame)
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(item)
            except queue.Full:
                self.dropped += 1

    def stats(self) -> dict:
        return {"submitted": self.submitted, "published": self.published,
                "dropped_updates": self.dropped, "last_error": self.error,
                "worker_alive": self.thread.is_alive(), "cache_path": str(self.path)}

    def _work(self):
        try:
            while not self.closed.is_set() or not self.queue.empty():
                try:
                    snapshot, frame = self.queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                payload = {"schema": 1, "run_id": self.run_dir.name,
                           "run_dir": str(self.run_dir), "identity": self.identity,
                           "published_monotonic_ns": time.monotonic_ns(),
                           "snapshot": snapshot, "thermal": thermal_payload(snapshot, frame),
                           "publisher": self.stats()}
                volatile_json(self.path, payload)
                self.published += 1
        except Exception as exc:
            # Latch once, no retries. This is an optional display failure, not
            # a sensor failure: the raw collector continues and reports it.
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"Live preview disabled: {self.error}", file=sys.stderr, flush=True)

    def close(self):
        self.closed.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=0.25)

