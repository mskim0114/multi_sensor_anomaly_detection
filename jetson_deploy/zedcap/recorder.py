"""ZED session writer: one saved frame per collector tick, stored in segment files under <run>/zed/.

Layout (schema ``zed-v1``)::

    <run>/zed/zed.json                         camera, calibration, config, tick clock, sessions, status
    <run>/zed/frames.csv                       one row per saved frame (written when its segment is final)
    <run>/zed/seg_SS_NNNNN_left.npz            left_jpeg uint8 (all JPEGs back to back), left_offsets int64 (N+1)
    <run>/zed/seg_SS_NNNNN_depth.npz           depth_mm uint16 (N, H, W): 0 = no data, 65535 = beyond range
                                               (only when depth_mode is not NONE; the default is left RGB only)

Tick alignment: the collector schedules tick n at ``t0 + n * 1 s`` on CLOCK_MONOTONIC, and t0 is the
``target_monotonic_ns`` of its first snapshot. The first frame captured within half a frame period
(+5 ms) of a tick target is saved as that tick, so it is the closest frame in practice; ticks
without such a frame are counted as missed, never filled.

Segments are written by one background thread (compressing a minute of depth takes about a second
and must not stall grabbing), each with tmp + rename and only then appended to frames.csv, so every
index row points at a complete file. A crash loses at most the segments not yet written.
"""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
import queue
import threading
import time

import numpy as np

from . import SCHEMA

NO_DATA, BEYOND_RANGE = 0, 65535
FRAME_FIELDS = ("session", "segment", "index", "tick", "frame_monotonic_ns", "image_timestamp_ns",
                "offset_ms", "imu_ax", "imu_ay", "imu_az", "imu_gx", "imu_gy", "imu_gz",
                "imu_qx", "imu_qy", "imu_qz", "imu_qw", "imu_temperature_c")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def depth_to_mm(depth_m: np.ndarray) -> np.ndarray:
    """float32 metres (NaN = occluded/unknown, -inf = too close, +inf = too far) -> uint16 mm."""
    depth_m = np.asarray(depth_m, dtype=np.float32)
    out = np.zeros(depth_m.shape, dtype=np.uint16)
    finite = np.isfinite(depth_m) & (depth_m > 0)
    out[finite] = np.clip(np.rint(depth_m[finite] * 1000.0), 1, BEYOND_RANGE - 1).astype(np.uint16)
    out[np.isposinf(depth_m)] = BEYOND_RANGE
    return out


def read_tick_clock(run_dir: Path) -> int | None:
    """t0 of the collector: target_monotonic_ns of snapshot 0, read-only, None until written."""
    try:
        with open(run_dir / "snapshots.jsonl", "rb") as stream:
            first = stream.readline(1 << 20)
        if not first.endswith(b"\n"):
            return None                     # first line still being written
        snapshot = json.loads(first)
    except (OSError, ValueError):
        return None
    t0 = snapshot.get("target_monotonic_ns") if isinstance(snapshot, dict) else None
    sequence = snapshot.get("sequence") if isinstance(snapshot, dict) else None
    if isinstance(t0, int) and isinstance(sequence, int) and sequence >= 0:
        return t0 - sequence * 1_000_000_000
    return None


class Session:
    """One continuous ZED recording inside one run. Feed frames with ``offer``; ``close`` ends it."""

    def __init__(self, run_dir: Path, *, camera_info: dict, config: dict, encode_jpeg, t0_ns: int,
                 tick_clock: str, recorder: dict, period_ns: int = 1_000_000_000,
                 tolerance_ns: int | None = None):
        self.dir = Path(run_dir) / "zed"
        self.dir.mkdir(exist_ok=True)
        self.meta_path = self.dir / "zed.json"
        self.index_path = self.dir / "frames.csv"
        self.encode_jpeg = encode_jpeg
        self.t0, self.period = int(t0_ns), int(period_ns)
        fps = max(1, int(config.get("fps", 15)))
        self.tolerance = int(tolerance_ns if tolerance_ns is not None else 0.5e9 / fps + 5e6)
        self.segment_frames = int(config.get("segment_frames", 60))
        self.with_depth = config.get("depth_mode", "NONE") != "NONE"
        self._lock = threading.Lock()                 # record/meta are shared with the writer
        meta = self._load_meta()
        for earlier in meta.get("sessions", []):
            if isinstance(earlier, dict) and earlier.get("stopped_utc") is None:
                earlier["stop_reason"] = "interrupted"   # its recorder died; the next one says so
        previous = [s.get("session", 0) for s in meta.get("sessions", []) if isinstance(s, dict)]
        self.number = max(previous, default=0) + 1
        self.record = {"session": self.number, "started_utc": utc_now(), "stopped_utc": None,
                       "stop_reason": None, "frames": 0, "first_tick": None, "last_tick": None,
                       "missed_ticks": 0, "segments": 0, "error": None}
        self.meta = meta
        self.meta.update(schema=SCHEMA, run_id=Path(run_dir).name, status="recording", recorder=recorder,
                         camera=camera_info, config={k: v for k, v in config.items() if k != "roots"},
                         tick_clock={"source": tick_clock, "t0_monotonic_ns": self.t0, "period_ns": self.period,
                                     "tolerance_ns": self.tolerance,
                                     "rule": "first frame within tolerance of a tick target; missing ticks stay missing"},
                         depth_encoding={"dtype": "uint16", "unit": "mm", "no_data": NO_DATA,
                                         "beyond_range": BEYOND_RANGE,
                                         "size": list(config.get("depth_size", [640, 360]))}
                         if self.with_depth else None,
                         files={"index": "frames.csv", "left": "seg_SS_NNNNN_left.npz (left_jpeg, left_offsets)",
                                "depth": "seg_SS_NNNNN_depth.npz (depth_mm)" if self.with_depth else None})
        self.meta.setdefault("sessions", []).append(self.record)
        self._write_meta()
        self.segment = 0                               # next segment number handed to the writer
        self.buffer: list[dict] = []
        self.last_tick: int | None = None
        self.bytes_written = 0
        self.write_error: str | None = None
        self._queue: queue.Queue = queue.Queue(maxsize=3)
        self._writer = threading.Thread(target=self._write_loop, name="zed-writer", daemon=True)
        self._writer.start()

    def _load_meta(self) -> dict:
        try:
            value = json.loads(self.meta_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write_meta(self) -> None:
        with self._lock:
            self.meta["updated_utc"] = utc_now()
            text = json.dumps(self.meta, indent=1, ensure_ascii=False)
        temp = self.meta_path.with_name(self.meta_path.name + ".partial")
        with open(temp, "w", encoding="utf-8") as stream:
            stream.write(text + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, self.meta_path)

    def tick_for(self, frame_monotonic_ns: int) -> tuple[int, float] | None:
        """(tick, offset ms) when this frame is the one to save, else None."""
        delta = frame_monotonic_ns - self.t0
        tick = int(round(delta / self.period))
        offset = delta - tick * self.period
        if tick < 0 or abs(offset) > self.tolerance:
            return None
        if self.last_tick is not None and tick <= self.last_tick:
            return None
        return tick, offset / 1e6

    def wants(self, frame_monotonic_ns: int) -> bool:
        return self.tick_for(frame_monotonic_ns) is not None

    def offer(self, frame_monotonic_ns: int, image_timestamp_ns: int, left_rgb, depth_m, imu: dict | None = None):
        """Store this frame if it belongs to the next tick. Returns the tick or None."""
        if self.write_error:
            raise RuntimeError(f"segment writer failed: {self.write_error}")
        chosen = self.tick_for(frame_monotonic_ns)
        if chosen is None:
            return None
        tick, offset_ms = chosen
        imu = imu or {}
        frame = {"tick": tick, "frame_monotonic_ns": int(frame_monotonic_ns),
                 "image_timestamp_ns": int(image_timestamp_ns), "offset_ms": round(offset_ms, 3),
                 "jpeg": self.encode_jpeg(left_rgb),
                 "depth": depth_to_mm(depth_m) if self.with_depth else None,
                 **{f: imu.get(f) for f in FRAME_FIELDS if f.startswith("imu_")}}
        with self._lock:
            if self.last_tick is not None:
                self.record["missed_ticks"] += tick - self.last_tick - 1
            else:
                self.record["first_tick"] = tick
            self.record["frames"] += 1
            self.record["last_tick"] = tick
        self.last_tick = tick
        self.buffer.append(frame)
        if len(self.buffer) >= self.segment_frames:
            self.flush()
        return tick

    def flush(self) -> None:
        """Hand the buffered frames to the writer (blocks only if three segments are queued)."""
        if not self.buffer:
            return
        frames, self.buffer = self.buffer, []
        self._queue.put((self.segment, frames))
        self.segment += 1

    def _write_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            if self.write_error:
                continue                             # drain; the session is ending with an error
            try:
                self._write_segment(*item)
            except Exception as exc:
                self.write_error = f"{type(exc).__name__}: {exc}"[:300]

    def _write_segment(self, segment: int, frames: list[dict]) -> None:
        stem = f"seg_{self.number:02d}_{segment:05d}"
        jpegs = [f["jpeg"] for f in frames]
        offsets = np.zeros(len(jpegs) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(j) for j in jpegs])
        left = np.frombuffer(b"".join(jpegs), dtype=np.uint8)
        # JPEG is already compressed: stored (seekable in place by the hub); depth: deflated.
        written = self._write_npz(self.dir / f"{stem}_left.npz", np.savez, left_jpeg=left, left_offsets=offsets)
        if self.with_depth:
            depth = np.stack([f["depth"] for f in frames])
            written += self._write_npz(self.dir / f"{stem}_depth.npz", np.savez_compressed, depth_mm=depth)
        new_index = not self.index_path.exists()
        with open(self.index_path, "a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FRAME_FIELDS)
            if new_index:
                writer.writeheader()
            for index, frame in enumerate(frames):
                writer.writerow({**{k: frame.get(k) for k in FRAME_FIELDS},
                                 "session": self.number, "segment": segment, "index": index})
            stream.flush()
            os.fsync(stream.fileno())
        self.bytes_written += written
        with self._lock:
            self.record["segments"] = segment + 1
        self._write_meta()

    @staticmethod
    def _write_npz(path: Path, save, **arrays) -> int:
        temp = path.with_name(path.name + ".partial")
        with open(temp, "wb") as stream:
            save(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        return path.stat().st_size

    def close(self, reason: str, error: str | None = None) -> dict:
        """Write what is buffered, wait for the writer, record why the session ended, mark complete."""
        try:
            self.flush()
        finally:
            self._queue.put(None)
            self._writer.join()
            with self._lock:
                self.record.update(stopped_utc=utc_now(), stop_reason=reason,
                                   error=error or (f"segment writer failed: {self.write_error}"
                                                   if self.write_error else None))
                self.meta["status"] = "complete"
            self._write_meta()
        return self.record


def encode_jpeg_pillow(quality: int, size: tuple[int, int] | None = None):
    """JPEG encoder for RGB uint8 arrays (Pillow lives in the ZED venv only)."""
    from PIL import Image

    def encode(rgb) -> bytes:
        image = Image.fromarray(np.ascontiguousarray(rgb))
        if size is not None and image.size != tuple(size):
            image = image.resize(tuple(size))
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=int(quality))
        return buffer.getvalue()
    return encode
