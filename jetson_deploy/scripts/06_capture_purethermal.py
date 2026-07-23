#!/usr/bin/env python3
"""Capture a PureThermal UVC frame and save raw + converted outputs.

Outputs per capture:
  - *_gray16.raw: original GRAY16_LE frame from PureThermal
  - *_preview.png: contrast-stretched 8-bit PNG for visual inspection
  - *_celsius.csv: temperature matrix converted from centi-Kelvin
  - *_metadata.json: capture settings and basic statistics

Usage:
    python3 scripts/06_capture_purethermal.py
    python3 scripts/06_capture_purethermal.py --device /dev/video0 --count 5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
import zlib


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DEFAULT_OUT_DIR = ROOT / "results" / "camera"
WIDTH = 160
HEIGHT = 120
FORMAT = "GRAY16_LE"
FRAMERATE = "9/1"


def find_default_device() -> str:
    by_id = Path("/dev/v4l/by-id")
    if by_id.exists():
        matches = sorted(by_id.glob("*PureThermal*video-index0"))
        if matches:
            return str(matches[0].resolve())

    for dev in ("/dev/video0", "/dev/video1"):
        if Path(dev).exists():
            return dev

    return "/dev/video0"


def run_gst_capture(device: str, raw_path: Path, width: int, height: int) -> None:
    gst = shutil.which("gst-launch-1.0")
    if gst is None:
        raise SystemExit("gst-launch-1.0 not found. Install GStreamer tools first.")

    caps = f"video/x-raw,format={FORMAT},width={width},height={height},framerate={FRAMERATE}"
    cmd = [
        gst,
        "-q",
        "v4l2src",
        f"device={device}",
        "num-buffers=1",
        "!",
        caps,
        "!",
        "filesink",
        f"location={raw_path}",
    ]
    subprocess.run(cmd, check=True)


def read_gray16(raw_path: Path, width: int, height: int) -> list[int]:
    data = raw_path.read_bytes()
    expected = width * height * 2
    if len(data) != expected:
        raise ValueError(f"Unexpected raw size: got {len(data)} bytes, expected {expected}")
    return list(struct.unpack("<" + "H" * (width * height), data))


def raw_to_celsius(values: list[int]) -> list[float]:
    # PureThermal radiometric GRAY16 values are commonly centi-Kelvin.
    return [(v / 100.0) - 273.15 for v in values]


def png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def write_preview_png(values: list[int], png_path: Path, width: int, height: int) -> None:
    mn = min(values)
    mx = max(values)
    if mx > mn:
        pixels = bytes(int((v - mn) * 255 / (mx - mn)) for v in values)
    else:
        pixels = bytes([0] * len(values))

    scanlines = b"".join(
        b"\x00" + pixels[row * width : (row + 1) * width] for row in range(height)
    )
    png = b"\x89PNG\r\n\x1a\n"
    png += png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
    png += png_chunk(b"IDAT", zlib.compress(scanlines, 9))
    png += png_chunk(b"IEND", b"")
    png_path.write_bytes(png)


def write_celsius_csv(celsius: list[float], csv_path: Path, width: int, height: int) -> None:
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        for row in range(height):
            start = row * width
            writer.writerow(f"{v:.2f}" for v in celsius[start : start + width])


def stats(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def capture_one(device: str, out_dir: Path, prefix: str, width: int, height: int) -> dict[str, object]:
    raw_path = out_dir / f"{prefix}_gray16.raw"
    png_path = out_dir / f"{prefix}_preview.png"
    csv_path = out_dir / f"{prefix}_celsius.csv"
    json_path = out_dir / f"{prefix}_metadata.json"

    run_gst_capture(device, raw_path, width, height)
    raw_values = read_gray16(raw_path, width, height)
    celsius = raw_to_celsius(raw_values)

    write_preview_png(raw_values, png_path, width, height)
    write_celsius_csv(celsius, csv_path, width, height)

    center_idx = (height // 2) * width + (width // 2)
    metadata = {
        "device": device,
        "width": width,
        "height": height,
        "format": FORMAT,
        "framerate": FRAMERATE,
        "raw_path": str(raw_path),
        "preview_png_path": str(png_path),
        "celsius_csv_path": str(csv_path),
        "raw_stats": stats([float(v) for v in raw_values]),
        "celsius_stats": stats(celsius),
        "center_raw": raw_values[center_idx],
        "center_celsius": celsius[center_idx],
    }
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metadata["metadata_path"] = str(json_path)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=find_default_device(), help="V4L2 device path")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Directory for captures")
    parser.add_argument("--count", type=int, default=1, help="Number of frames to capture")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between captures")
    parser.add_argument("--prefix", default="purethermal", help="Output filename prefix")
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {args.device}")
    print(f"Output directory: {out_dir}")

    for i in range(args.count):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        suffix = f"{stamp}_{i:03d}" if args.count > 1 else stamp
        prefix = f"{args.prefix}_{suffix}"
        meta = capture_one(args.device, out_dir, prefix, args.width, args.height)

        c = meta["celsius_stats"]
        print(
            f"[{i + 1}/{args.count}] saved {prefix}: "
            f"{c['min']:.2f}C..{c['max']:.2f}C mean={c['mean']:.2f}C"
        )
        print(f"  raw:      {meta['raw_path']}")
        print(f"  preview:  {meta['preview_png_path']}")
        print(f"  celsius:  {meta['celsius_csv_path']}")
        print(f"  metadata: {meta['metadata_path']}")

        if i + 1 < args.count:
            time.sleep(args.interval)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
