#!/usr/bin/env python3
"""Read a Bosch BME680 over Jetson I2C.

Usage:
    python3 scripts/10_read_bme680_i2c.py
    python3 scripts/10_read_bme680_i2c.py --i2c-port /dev/i2c-7 --address 0x77 --samples 30

Verified wiring on this board (breakout silk MISO/SCLK/CS/MOSI/GND/VCC):

    SCLK -> pin 5  (SCL)        MOSI -> pin 3  (SDA)
    CS   -> 3.3V                MISO -> 3.3V for 0x77, GND for 0x76
    GND  -> GND                 VCC  -> 3.3V

CS must sit at VDDIO or the part stays in SPI mode and never answers on I2C.
Once CSB has been driven low even once, I2C stays disabled until a true
power-on reset; a warm `reboot` is not enough. See
docs/JETSON_SPI_BME680_SETUP.md.

The Adafruit driver issues a soft reset (register 0xE0 <- 0xB6) in its
constructor. That resets volatile registers only; the factory calibration NVM
is read-only and is not touched.

Note: the BME680 gas heater warms the die, so its temperature reads above
ambient. The model's temperature channel is the NTC, not this sensor.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time

from fcntl import ioctl


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DEFAULT_OUT_DIR = ROOT / "results" / "bme680"
DEFAULT_I2C_PORT = "/dev/i2c-7"
DEFAULT_ADDRESS = 0x77

I2C_SLAVE = 0x0703
REG_CHIP_ID = 0xD0
REG_VARIANT_ID = 0xF0
EXPECTED_CHIP_ID = 0x61
VARIANT_NAMES = {0x00: "BME680", 0x01: "BME680", 0x02: "BME688"}

FIELDS = [
    "timestamp",
    "temperature_c",
    "relative_humidity_pct",
    "pressure_hpa",
    "gas_ohm",
]


def parse_int(value: str) -> int:
    return int(value, 0)


def bus_number(i2c_port: str) -> int:
    """Extract the Linux bus number ExtendedI2C needs from /dev/i2c-N."""
    name = Path(i2c_port).name
    if not name.startswith("i2c-") or not name[4:].isdigit():
        raise ValueError(f"cannot derive a bus number from {i2c_port!r}; expected /dev/i2c-N")
    return int(name[4:])


def read_identity(i2c_port: str, address: int) -> dict[str, object]:
    """Read the chip and variant id registers directly, before the driver runs."""
    fd = os.open(i2c_port, os.O_RDWR)
    try:
        ioctl(fd, I2C_SLAVE, address)

        def reg(register: int) -> int:
            os.write(fd, bytes([register]))
            data = os.read(fd, 1)
            if len(data) != 1:
                raise OSError(f"short read from BME680 register 0x{register:02x}")
            return data[0]

        chip_id = reg(REG_CHIP_ID)
        try:
            variant_id: int | None = reg(REG_VARIANT_ID)
        except OSError:
            variant_id = None
    finally:
        os.close(fd)

    return {
        "chip_id": chip_id,
        "variant_id": variant_id,
        "variant_name": VARIANT_NAMES.get(variant_id, "unknown") if variant_id is not None else "unavailable",
    }


def physically_plausible(rows: list[dict[str, object]]) -> tuple[bool, list[str]]:
    """Broad sanity only. Deliberately not an environmental threshold check."""
    problems: list[str] = []
    for idx, row in enumerate(rows, start=1):
        t = float(row["temperature_c"])          # type: ignore[arg-type]
        h = float(row["relative_humidity_pct"])  # type: ignore[arg-type]
        p = float(row["pressure_hpa"])           # type: ignore[arg-type]
        g = float(row["gas_ohm"])                # type: ignore[arg-type]
        if not all(math.isfinite(v) for v in (t, h, p, g)):
            problems.append(f"sample {idx}: non-finite value")
        if not 0.0 <= h <= 100.0:
            problems.append(f"sample {idx}: relative humidity {h} outside 0..100 %")
        if not 300.0 <= p <= 1100.0:
            problems.append(f"sample {idx}: pressure {p} hPa outside 300..1100 hPa")
        if g <= 0.0:
            problems.append(f"sample {idx}: gas resistance {g} ohm is not positive")
    return (not problems), problems


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
    parser.add_argument("--address", type=parse_int, default=DEFAULT_ADDRESS)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    if args.samples < 1:
        print("--samples must be at least 1", file=sys.stderr)
        return 2

    print(f"I2C port: {args.i2c_port}")
    print(f"I2C address: 0x{args.address:02x}")

    identity = read_identity(args.i2c_port, args.address)
    chip_id = int(identity["chip_id"])  # type: ignore[arg-type]
    variant_id = identity["variant_id"]
    print(f"chip id: 0x{chip_id:02x} (expected 0x{EXPECTED_CHIP_ID:02x})")
    if variant_id is None:
        print("variant id: unavailable")
    else:
        print(f"variant id: 0x{int(variant_id):02x} ({identity['variant_name']})")

    if chip_id != EXPECTED_CHIP_ID:
        print("", file=sys.stderr)
        print(
            f"BME680 identity check FAILED: chip id 0x{chip_id:02x} at "
            f"{args.i2c_port} 0x{args.address:02x}, expected 0x{EXPECTED_CHIP_ID:02x}.",
            file=sys.stderr,
        )
        print("Not sampling. See the checks below.", file=sys.stderr)
        print_troubleshooting_i2c()
        return 2

    print(f"Samples: {args.samples}, interval: {args.interval:.3f} s")

    # Imported after the identity check so a wiring fault reports as a wiring
    # fault rather than as a driver constructor error.
    from adafruit_extended_bus import ExtendedI2C
    import adafruit_bme680

    i2c = ExtendedI2C(bus_number(args.i2c_port))
    sensor = adafruit_bme680.Adafruit_BME680_I2C(i2c, address=args.address)

    rows: list[dict[str, object]] = []
    print(f"{'#':>4} {'temperature':>13} {'humidity':>11} {'pressure':>13} {'gas':>13}")
    for idx in range(args.samples):
        row = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "temperature_c": sensor.temperature,
            "relative_humidity_pct": sensor.relative_humidity,
            "pressure_hpa": sensor.pressure,
            "gas_ohm": sensor.gas,
        }
        rows.append(row)
        print(
            f"{idx + 1:>4}"
            f" {row['temperature_c']:>10.2f} C"
            f" {row['relative_humidity_pct']:>9.2f} %"
            f" {row['pressure_hpa']:>9.2f} hPa"
            f" {row['gas_ohm']:>9d} ohm"
        )
        if idx + 1 < args.samples:
            time.sleep(args.interval)

    ok, problems = physically_plausible(rows)
    print("")
    print(f"Reads: {len(rows)}/{args.samples} successful")
    print(f"Physical sanity: {'PASS' if ok else 'FAIL'}")
    for problem in problems:
        print(f"  {problem}", file=sys.stderr)
    print("Note: gas resistance drifts for minutes while the heater settles;")
    print("      early samples are expected to differ. Temperature reads above")
    print("      ambient because of that heater - the model uses the NTC.")

    metadata = {
        "i2c_port": args.i2c_port,
        "i2c_address": f"0x{args.address:02x}",
        "interface": "i2c",
        "chip_id": f"0x{chip_id:02x}",
        "variant_id": None if variant_id is None else f"0x{int(variant_id):02x}",
        "variant_name": identity["variant_name"],
        "driver": "adafruit-circuitpython-bme680",
        "samples_requested": args.samples,
        "interval_s": args.interval,
        "physical_sanity_pass": ok,
        "field_order_for_model": ["NTC", "PM1.0", "PM2.5", "PM10", "CT1", "CT2", "CT3", "CT4"],
        "note": "BME680 is a collection/context sensor. It is NOT part of the model input vector.",
    }
    write_outputs(Path(args.out_dir), f"bme680_{time.strftime('%Y%m%d_%H%M%S')}", metadata, rows)
    return 0 if ok else 2


def print_troubleshooting_i2c() -> None:
    print("", file=sys.stderr)
    print("Checks:", file=sys.stderr)
    print("  1. CS must be tied to 3.3V. Left floating or low the part stays in", file=sys.stderr)
    print("     SPI mode and never answers on I2C.", file=sys.stderr)
    print("  2. MISO/SDO selects the address: 3.3V -> 0x77, GND -> 0x76.", file=sys.stderr)
    print("  3. If CSB was ever driven low, power-cycle the board properly.", file=sys.stderr)
    print("     Unplug the adapter for 10 s; a warm reboot is not enough.", file=sys.stderr)
    print("  4. I2C needs pull-ups. The Jetson header provides none, so another", file=sys.stderr)
    print("     I2C module must share the bus.", file=sys.stderr)
    print("  5. Confirm SCLK=pin 5 (SCL) and MOSI=pin 3 (SDA) are not swapped.", file=sys.stderr)
    print("  6. Confirm the module is not inserted backwards. Reverse insertion", file=sys.stderr)
    print("     destroyed three modules during bring-up.", file=sys.stderr)
    print("  7. Verify a control device on the same bus still answers, e.g.", file=sys.stderr)
    print("     ADS1115 at 0x48 on /dev/i2c-7.", file=sys.stderr)


def print_troubleshooting(exc: Exception) -> None:
    print(f"BME680 read failed: {exc}", file=sys.stderr)
    print_troubleshooting_i2c()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PermissionError as exc:
        print(f"Permission denied: {exc}", file=sys.stderr)
        print("Add the user to the 'i2c' group and log in again.", file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("", file=sys.stderr)
        print("Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print_troubleshooting(exc)
        raise SystemExit(2)
