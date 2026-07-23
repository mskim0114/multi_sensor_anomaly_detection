#!/usr/bin/env python3
"""Read a 10K B3950 NTC thermistor through an ADS1115 on Jetson I2C.

Default wiring:
    3.3V -> 10k fixed resistor -> ADS1115 A0 -> NTC probe -> GND

Usage:
    python3 scripts/08_read_ntc_ads1115.py
    python3 scripts/08_read_ntc_ads1115.py --samples 30 --channel 0
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
DEFAULT_OUT_DIR = ROOT / "results" / "ntc"
DEFAULT_I2C_PORT = "/dev/i2c-1"
DEFAULT_ADDRESS = 0x48

I2C_SLAVE = 0x0703
REG_CONVERSION = 0x00
REG_CONFIG = 0x01

PGA_FSR = {
    6.144: (0x0000, 6.144),
    4.096: (0x0200, 4.096),
    2.048: (0x0400, 2.048),
    1.024: (0x0600, 1.024),
    0.512: (0x0800, 0.512),
    0.256: (0x0A00, 0.256),
}

DATA_RATE_BITS = {
    8: 0x0000,
    16: 0x0020,
    32: 0x0040,
    64: 0x0060,
    128: 0x0080,
    250: 0x00A0,
    475: 0x00C0,
    860: 0x00E0,
}

FIELDS = [
    "timestamp",
    "channel",
    "raw",
    "voltage_v",
    "ntc_resistance_ohm",
    "temperature_c",
]


def parse_int(value: str) -> int:
    return int(value, 0)


def open_i2c_device(i2c_port: str, address: int) -> int:
    fd = os.open(i2c_port, os.O_RDWR)
    try:
        ioctl(fd, I2C_SLAVE, address)
    except Exception:
        os.close(fd)
        raise
    return fd


def write_register(fd: int, register: int, value: int) -> None:
    os.write(fd, bytes([register, (value >> 8) & 0xFF, value & 0xFF]))


def read_register(fd: int, register: int) -> int:
    os.write(fd, bytes([register]))
    data = os.read(fd, 2)
    if len(data) != 2:
        raise OSError(f"short read from ADS1115: {len(data)} bytes")
    return (data[0] << 8) | data[1]


def twos_complement_16(value: int) -> int:
    return value - 0x10000 if value & 0x8000 else value


def read_ads1115_single_ended(
    fd: int,
    channel: int,
    pga: float,
    data_rate: int,
) -> tuple[int, float]:
    if channel < 0 or channel > 3:
        raise ValueError("ADS1115 channel must be 0..3")
    if pga not in PGA_FSR:
        raise ValueError(f"unsupported PGA FSR: {pga}")
    if data_rate not in DATA_RATE_BITS:
        raise ValueError(f"unsupported ADS1115 data rate: {data_rate}")

    pga_bits, fsr = PGA_FSR[pga]
    mux_bits = (0x04 + channel) << 12
    config = (
        0x8000  # start single conversion
        | mux_bits
        | pga_bits
        | 0x0100  # single-shot mode
        | DATA_RATE_BITS[data_rate]
        | 0x0003  # disable comparator
    )

    write_register(fd, REG_CONFIG, config)
    time.sleep((1.0 / data_rate) + 0.01)

    raw = twos_complement_16(read_register(fd, REG_CONVERSION))
    voltage = raw * fsr / 32768.0
    return raw, voltage


def ntc_resistance_from_divider(
    voltage: float,
    vcc: float,
    fixed_resistor: float,
    divider: str,
) -> float:
    if voltage <= 0.0 or voltage >= vcc:
        raise ValueError(f"divider voltage out of range: {voltage:.6f} V for Vcc={vcc:.6f} V")

    if divider == "fixed-top":
        return fixed_resistor * voltage / (vcc - voltage)
    if divider == "ntc-top":
        return fixed_resistor * (vcc - voltage) / voltage
    raise ValueError(f"unknown divider orientation: {divider}")


def temperature_c_from_beta(
    resistance: float,
    nominal_resistance: float,
    beta: float,
    nominal_temp_c: float,
) -> float:
    t0 = nominal_temp_c + 273.15
    inv_t = (1.0 / t0) + (math.log(resistance / nominal_resistance) / beta)
    return (1.0 / inv_t) - 273.15


def read_one(args: argparse.Namespace, fd: int) -> dict[str, object]:
    raw, voltage = read_ads1115_single_ended(fd, args.channel, args.pga, args.data_rate)
    resistance = ntc_resistance_from_divider(
        voltage=voltage,
        vcc=args.vcc,
        fixed_resistor=args.fixed_resistor,
        divider=args.divider,
    )
    temperature_c = temperature_c_from_beta(
        resistance=resistance,
        nominal_resistance=args.nominal_resistance,
        beta=args.beta,
        nominal_temp_c=args.nominal_temp_c,
    )

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "channel": args.channel,
        "raw": raw,
        "voltage_v": voltage,
        "ntc_resistance_ohm": resistance,
        "temperature_c": temperature_c,
    }


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
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--vcc", type=float, default=3.3)
    parser.add_argument("--fixed-resistor", type=float, default=10000.0)
    parser.add_argument("--nominal-resistance", type=float, default=10000.0)
    parser.add_argument("--nominal-temp-c", type=float, default=25.0)
    parser.add_argument("--beta", type=float, default=3950.0)
    parser.add_argument("--pga", type=float, choices=sorted(PGA_FSR), default=4.096)
    parser.add_argument("--data-rate", type=int, choices=sorted(DATA_RATE_BITS), default=128)
    parser.add_argument(
        "--divider",
        choices=("fixed-top", "ntc-top"),
        default="fixed-top",
        help="fixed-top: 3.3V -> fixed resistor -> node -> NTC -> GND",
    )
    args = parser.parse_args()

    print(f"I2C port: {args.i2c_port}")
    print(f"ADS1115 address: 0x{args.address:02x}")
    print(f"ADS1115 channel: A{args.channel}")
    print(f"Divider: {args.divider}, Vcc={args.vcc:.3f} V, R_fixed={args.fixed_resistor:.1f} ohm")

    fd = open_i2c_device(args.i2c_port, args.address)
    rows: list[dict[str, object]] = []
    try:
        for idx in range(args.samples):
            row = read_one(args, fd)
            rows.append(row)
            print(
                f"[{idx + 1}/{args.samples}] "
                f"A{row['channel']}={row['voltage_v']:.4f} V "
                f"Rntc={row['ntc_resistance_ohm']:.1f} ohm "
                f"T={row['temperature_c']:.2f} C"
            )
            if idx + 1 < args.samples:
                time.sleep(args.interval)
    finally:
        os.close(fd)

    metadata = {
        "i2c_port": args.i2c_port,
        "address": f"0x{args.address:02x}",
        "channel": args.channel,
        "divider": args.divider,
        "vcc": args.vcc,
        "fixed_resistor": args.fixed_resistor,
        "nominal_resistance": args.nominal_resistance,
        "nominal_temp_c": args.nominal_temp_c,
        "beta": args.beta,
        "pga": args.pga,
        "data_rate": args.data_rate,
        "field_order_for_model": ["NTC", "PM1.0", "PM2.5", "PM10", "CT1", "CT2", "CT3", "CT4"],
    }
    write_outputs(Path(args.out_dir), f"ntc_{time.strftime('%Y%m%d_%H%M%S')}", metadata, rows)
    return 0


def print_troubleshooting(exc: Exception) -> None:
    print(f"NTC/ADS1115 read failed: {exc}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Checks:", file=sys.stderr)
    print("  1. Confirm `sudo i2cdetect -r -y 1` shows ADS1115 at 0x48.", file=sys.stderr)
    print("  2. Power ADS1115 from Jetson 3.3V, not 5V.", file=sys.stderr)
    print("  3. Confirm ADS1115 GND and Jetson GND are common.", file=sys.stderr)
    print("  4. Confirm ADS1115 SDA=pin 27 and SCL=pin 28.", file=sys.stderr)
    print("  5. Confirm ADS1115 ADDR is tied to GND for address 0x48.", file=sys.stderr)
    print("  6. Confirm A0 is connected to the divider midpoint.", file=sys.stderr)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PermissionError as exc:
        print(f"Permission denied: {exc}", file=sys.stderr)
        print("Try: sudo -E python3 scripts/08_read_ntc_ads1115.py", file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print_troubleshooting(exc)
        raise SystemExit(2)
