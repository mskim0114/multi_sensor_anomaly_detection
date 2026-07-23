#!/usr/bin/env python3
"""Read Sensirion SPS30 particulate measurements over Jetson I2C.

Usage:
    python3 scripts/07_read_sps30.py
    python3 scripts/07_read_sps30.py --i2c-port /dev/i2c-1 --samples 10

The current Jetson Orin Nano Super DevKit maps 40-pin header pin 27/28 I2C to
/dev/i2c-1 at 100 kHz, which matches the SPS30 I2C speed limit. If permission is
denied, run with sudo or add the user to the i2c group and log in again.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

from sensirion_driver_adapters.i2c_adapter.i2c_channel import I2cChannel
from sensirion_i2c_driver import CrcCalculator, I2cConnection, LinuxI2cTransceiver
from sensirion_i2c_sps30.commands import OutputFormat
from sensirion_i2c_sps30.device import Sps30Device


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DEFAULT_OUT_DIR = ROOT / "results" / "sps30"
DEFAULT_I2C_PORT = "/dev/i2c-1"
SPS30_I2C_ADDRESS = 0x69

FIELDS = [
    "timestamp",
    "data_ready",
    "pm1_0_ug_m3",
    "pm2_5_ug_m3",
    "pm4_0_ug_m3",
    "pm10_ug_m3",
    "nc0_5_per_cm3",
    "nc1_0_per_cm3",
    "nc2_5_per_cm3",
    "nc4_0_per_cm3",
    "nc10_per_cm3",
    "typical_particle_size_um",
]


def as_plain(value: object) -> object:
    """Convert Sensirion value wrappers/enums to JSON-friendly values."""
    if hasattr(value, "value"):
        return as_plain(getattr(value, "value"))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)


def measurement_to_row(values: tuple[object, ...], data_ready: object) -> dict[str, object]:
    plain = [as_plain(v) for v in values]
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "data_ready": as_plain(data_ready),
        "pm1_0_ug_m3": plain[0],
        "pm2_5_ug_m3": plain[1],
        "pm4_0_ug_m3": plain[2],
        "pm10_ug_m3": plain[3],
        "nc0_5_per_cm3": plain[4],
        "nc1_0_per_cm3": plain[5],
        "nc2_5_per_cm3": plain[6],
        "nc4_0_per_cm3": plain[7],
        "nc10_per_cm3": plain[8],
        "typical_particle_size_um": plain[9],
    }


def make_sensor(i2c_port: str) -> Sps30Device:
    transceiver = LinuxI2cTransceiver(i2c_port)
    channel = I2cChannel(
        I2cConnection(transceiver),
        slave_address=SPS30_I2C_ADDRESS,
        crc=CrcCalculator(8, 0x31, 0xFF, 0x00),
    )
    return Sps30Device(channel)


def print_identity(sensor: Sps30Device) -> dict[str, object]:
    identity: dict[str, object] = {}
    for label, func in (
        ("serial_number", sensor.read_serial_number),
        ("product_type", sensor.read_product_type),
        ("firmware_version", sensor.read_firmware_version),
        ("device_status", sensor.read_device_status_register),
    ):
        try:
            identity[label] = as_plain(func())
        except Exception as exc:  # Keep sampling even if an info command fails.
            identity[label] = f"unavailable: {exc}"

    print("SPS30 identity:")
    for key, value in identity.items():
        print(f"  {key}: {value}")
    return identity


def write_outputs(out_dir: Path, prefix: str, metadata: dict[str, object], rows: list[dict[str, object]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{prefix}.csv"
    json_path = out_dir / f"{prefix}.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    payload = dict(metadata)
    payload["samples"] = rows
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--i2c-port", default=DEFAULT_I2C_PORT)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--format", choices=("float", "uint16"), default="float")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    prefix = f"sps30_{time.strftime('%Y%m%d_%H%M%S')}"

    print(f"I2C port: {args.i2c_port}")
    print(f"I2C address: 0x{SPS30_I2C_ADDRESS:02x}")
    print(f"Output format: {args.format}")

    with LinuxI2cTransceiver(args.i2c_port) as transceiver:
        channel = I2cChannel(
            I2cConnection(transceiver),
            slave_address=SPS30_I2C_ADDRESS,
            crc=CrcCalculator(8, 0x31, 0xFF, 0x00),
        )
        sensor = Sps30Device(channel)

        try:
            sensor.wake_up_sequence()
        except Exception:
            pass

        try:
            sensor.stop_measurement()
        except Exception:
            pass

        identity = print_identity(sensor)

        output_format = (
            OutputFormat.OUTPUT_FORMAT_FLOAT
            if args.format == "float"
            else OutputFormat.OUTPUT_FORMAT_UINT16
        )
        sensor.start_measurement(output_format)

        rows: list[dict[str, object]] = []
        try:
            for idx in range(args.samples):
                time.sleep(args.interval)
                data_ready = sensor.read_data_ready_flag()
                values = (
                    sensor.read_measurement_values_float()
                    if args.format == "float"
                    else sensor.read_measurement_values_uint16()
                )
                row = measurement_to_row(values, data_ready)
                rows.append(row)

                print(
                    f"[{idx + 1}/{args.samples}] "
                    f"ready={row['data_ready']} "
                    f"PM1.0={row['pm1_0_ug_m3']} "
                    f"PM2.5={row['pm2_5_ug_m3']} "
                    f"PM10={row['pm10_ug_m3']} ug/m3"
                )
        finally:
            try:
                sensor.stop_measurement()
            except Exception:
                pass

        metadata = {
            "i2c_port": args.i2c_port,
            "i2c_address": f"0x{SPS30_I2C_ADDRESS:02x}",
            "output_format": args.format,
            "identity": identity,
            "field_order_for_model": ["NTC", "PM1.0", "PM2.5", "PM10", "CT1", "CT2", "CT3", "CT4"],
        }
        write_outputs(out_dir, prefix, metadata, rows)

    return 0


def print_troubleshooting(exc: Exception) -> None:
    print(f"SPS30 read failed: {exc}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Checks:", file=sys.stderr)
    print("  1. Confirm `i2cdetect -r -y 1` shows address 0x69.", file=sys.stderr)
    print("  2. Power-cycle the SPS30 while SEL is already tied to GND.", file=sys.stderr)
    print("  3. Confirm VDD=5V and Jetson/SPS30 grounds are common.", file=sys.stderr)
    print("  4. Confirm SDA=pin 27 and SCL=pin 28 are not swapped.", file=sys.stderr)
    print("  5. Add/verify 10k pull-ups from SDA/SCL to 3.3V, not 5V.", file=sys.stderr)
    print("  6. Keep wires short while testing.", file=sys.stderr)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PermissionError as exc:
        print(f"Permission denied: {exc}", file=sys.stderr)
        print("Try: sudo -E python3 scripts/07_read_sps30.py", file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print_troubleshooting(exc)
        raise SystemExit(2)
