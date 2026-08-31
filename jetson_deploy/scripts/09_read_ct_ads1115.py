#!/usr/bin/env python3
"""CT diagnostic read via ADS1115 hardware differential AIN0-AIN1.

Sensor: YHDC SCT024TS, 400 A / 1 A current-output CT.

Wiring assumed:
    CT k -> ADS1115 A0
    CT l -> ADS1115 A1 + VBIAS (10k/10k divider from 3.3V)
    0.68 ohm / 5W burden resistor across A0-A1
    ADS1115 on MAIN I2C bus: Jetson pin 3 (SDA) / pin 5 (SCL) -> /dev/i2c-7, 0x48

Validated design range of this system: 0 ~ 400 A RMS (CT nameplate rating).
The +/-2.048 V PGA is selected to keep ADC headroom at the 400 A rating. It does
NOT extend the measurement range beyond the CT rating, and the ADS1115 absolute
input rail limits apply independently of the differential FSR.

This script reports measured values only. It does NOT classify the result as
no-current / noise / valid-current. A zero-current noise floor must be
established separately from repeated baseline measurements on real hardware.

NTC (A2) is intentionally not measured here: one ADS1115 is a single MUXed ADC,
so CT continuous conversion and NTC conversion cannot run at the same time.
Do not run this script concurrently with 08_read_ntc_ads1115.py.

Usage:
    python3 scripts/09_read_ct_ads1115.py
    python3 scripts/09_read_ct_ads1115.py --duration 5.0
    python3 scripts/09_read_ct_ads1115.py --i2c-port /dev/i2c-7 --address 0x48
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

from fcntl import ioctl


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DEFAULT_OUT_DIR = ROOT / "results" / "ct"

# MAIN I2C bus: Jetson 40-pin pin 3 (SDA) / pin 5 (SCL), 400 kHz.
DEFAULT_I2C_PORT = "/dev/i2c-7"
DEFAULT_ADDRESS = 0x48

# ---------------------------------------------------------------------------
# Low-level ADS1115 access.
#
# Copied from scripts/08_read_ntc_ads1115.py (verified working on this board).
# That module name starts with a digit and therefore cannot be imported, and 08
# must not be modified, so the verified primitives are duplicated here. A shared
# jetson_deploy/sensors/ads1115.py driver will replace both copies when the
# integrated SensorCollector is implemented.
# ---------------------------------------------------------------------------
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

# Config register field values used by this script.
OS_NO_EFFECT = 0x0000       # bit 15, ignored while MODE=continuous
OS_START_SINGLE = 0x8000    # bit 15, begins one conversion in single-shot mode
MUX_DIFF_AIN0_AIN1 = 0x0000  # bits 14:12 = 000
MODE_CONTINUOUS = 0x0000    # bit 8 = 0
MODE_SINGLE_SHOT = 0x0100   # bit 8 = 1
COMP_QUE_DISABLE = 0x0003   # bits 1:0 = 11

# A0/A1 absolute-voltage sanity gate, measured single-ended before entering
# continuous differential mode. +/-4.096 V FSR is used so that the whole
# 0.1..3.2 V window is representable without clipping.
SANITY_PGA = 4.096
SANITY_DATA_RATE = 128
SANITY_MIN_V = 0.1
SANITY_MAX_V = 3.2

# Config register masks. OS (bit 15) is both a status bit on read and a
# "start single conversion" command on write, so it is masked out when restoring
# a config whose MODE was single-shot. Everything else (MUX/PGA/MODE/DR/comparator)
# lives in bits 14:0 and must be restored bit-for-bit.
CONFIG_OS_MASK = 0x8000
CONFIG_SETTINGS_MASK = 0x7FFF

ADC_CODE_MAX = 32767
ADC_CODE_MIN = -32768

SAMPLE_FIELDS = ["index", "monotonic_ns", "t_rel_ms", "raw_code", "differential_v", "ac_v"]


def parse_int(value: str) -> int:
    return int(value, 0)


MUX_NAMES = {
    0b000: "AIN0-AIN1", 0b001: "AIN0-AIN3", 0b010: "AIN1-AIN3", 0b011: "AIN2-AIN3",
    0b100: "AIN0-GND", 0b101: "AIN1-GND", 0b110: "AIN2-GND", 0b111: "AIN3-GND",
}
PGA_NAMES = {
    0b000: 6.144, 0b001: 4.096, 0b010: 2.048,
    0b011: 1.024, 0b100: 0.512, 0b101: 0.256, 0b110: 0.256, 0b111: 0.256,
}
DR_NAMES = {0b000: 8, 0b001: 16, 0b010: 32, 0b011: 64,
            0b100: 128, 0b101: 250, 0b110: 475, 0b111: 860}


def decode_config(value: int) -> dict[str, object]:
    """Decode a Config register value into its fields (OS excluded on purpose)."""
    return {
        "mux": MUX_NAMES[(value >> 12) & 0b111],
        "pga_fsr_v": PGA_NAMES[(value >> 9) & 0b111],
        "mode": "single-shot" if (value >> 8) & 0b1 else "continuous",
        "data_rate_sps": DR_NAMES[(value >> 5) & 0b111],
        "comp_que": (value & 0b11),
    }


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


# ---------------------------------------------------------------------------
# Config register construction
# ---------------------------------------------------------------------------
def build_single_ended_config(channel: int, pga: float, data_rate: int) -> tuple[int, float]:
    """Single-shot single-ended config, used only for the A0/A1 sanity gate."""
    if channel < 0 or channel > 3:
        raise ValueError("ADS1115 channel must be 0..3")
    pga_bits, fsr = PGA_FSR[pga]
    config = (
        OS_START_SINGLE
        | ((0x04 + channel) << 12)
        | pga_bits
        | MODE_SINGLE_SHOT
        | DATA_RATE_BITS[data_rate]
        | COMP_QUE_DISABLE
    )
    return config, fsr


def build_continuous_diff_config(pga: float, data_rate: int) -> tuple[int, float]:
    """Continuous-conversion differential AIN0-AIN1 config.

    For pga=2.048 and data_rate=860 this yields 0x04E3:
        bit 15    OS        = 0    (no effect in continuous mode)
        bits 14:12 MUX      = 000  (AIN_P = AIN0, AIN_N = AIN1)
        bits 11:9  PGA      = 010  (FSR = +/-2.048 V)
        bit 8     MODE      = 0    (continuous conversion)
        bits 7:5   DR       = 111  (860 SPS)
        bit 4     COMP_MODE = 0
        bit 3     COMP_POL  = 0
        bit 2     COMP_LAT  = 0
        bits 1:0   COMP_QUE = 11   (comparator disabled, ALERT/RDY high-Z)
    """
    if pga not in PGA_FSR:
        raise ValueError(f"unsupported PGA FSR: {pga}")
    if data_rate not in DATA_RATE_BITS:
        raise ValueError(f"unsupported ADS1115 data rate: {data_rate}")
    pga_bits, fsr = PGA_FSR[pga]
    config = (
        OS_NO_EFFECT
        | MUX_DIFF_AIN0_AIN1
        | pga_bits
        | MODE_CONTINUOUS
        | DATA_RATE_BITS[data_rate]
        | COMP_QUE_DISABLE
    )
    return config, fsr


def read_single_ended(fd: int, channel: int, pga: float, data_rate: int) -> tuple[int, float]:
    config, fsr = build_single_ended_config(channel, pga, data_rate)
    write_register(fd, REG_CONFIG, config)
    time.sleep((1.0 / data_rate) + 0.01)
    raw = twos_complement_16(read_register(fd, REG_CONVERSION))
    return raw, raw * fsr / 32768.0


# ---------------------------------------------------------------------------
# Shared-resource warning: one ADS1115 cannot serve CT and NTC simultaneously.
# ---------------------------------------------------------------------------
def self_and_ancestor_pids() -> set[int]:
    """Return this pid plus every ancestor pid, walking /proc/<pid>/stat PPIDs."""
    pids: set[int] = set()
    pid = os.getpid()
    while pid > 0 and pid not in pids:
        pids.add(pid)
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8") as handle:
                # "pid (comm) state ppid ..." - comm may contain spaces/parens.
                fields = handle.read().rsplit(") ", 1)[1].split()
            pid = int(fields[1])
        except (OSError, IndexError, ValueError):
            break
    return pids


def find_conflicting_processes(i2c_port: str) -> list[tuple[int, str]]:
    """Report other processes that actually hold the same I2C device open.

    Detection is strictly fd-based: a process is reported only when one of its
    /proc/<pid>/fd entries resolves to the same device path. A process command
    line is never used as evidence, because the shell that launched this script
    carries the script name in its own cmdline and would be a false positive.

    This process and all of its ancestors are excluded. Processes owned by other
    users cannot be inspected and are silently skipped, so this check can miss a
    real conflict - it is a diagnostic warning only. Nothing is ever signalled,
    killed, or terminated here.
    """
    target = os.path.realpath(i2c_port)
    excluded = self_and_ancestor_pids()
    conflicts: list[tuple[int, str]] = []

    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid in excluded:
            continue

        fd_dir = f"/proc/{pid}/fd"
        try:
            fd_names = os.listdir(fd_dir)
        except OSError:
            continue  # process gone, or not inspectable by this user

        holds_device = False
        for fd_name in fd_names:
            try:
                if os.path.realpath(os.path.join(fd_dir, fd_name)) == target:
                    holds_device = True
                    break
            except OSError:
                continue
        if not holds_device:
            continue

        try:
            with open(f"/proc/{pid}/cmdline", "rb") as handle:
                cmdline = handle.read().replace(b"\x00", b" ").decode("utf-8", "replace").strip()
        except OSError:
            cmdline = "(cmdline unavailable)"
        conflicts.append((pid, cmdline or "(empty cmdline)"))

    return conflicts


# ---------------------------------------------------------------------------
# A0/A1 absolute-voltage sanity gate
# ---------------------------------------------------------------------------
def sanity_check_inputs(fd: int, samples_per_channel: int) -> tuple[bool, dict[str, object]]:
    results: dict[str, object] = {}
    all_ok = True

    for channel in (0, 1):
        readings: list[float] = []
        for _ in range(samples_per_channel):
            _, voltage = read_single_ended(fd, channel, SANITY_PGA, SANITY_DATA_RATE)
            readings.append(voltage)
        in_range = all(SANITY_MIN_V <= value <= SANITY_MAX_V for value in readings)
        all_ok = all_ok and in_range
        results[f"a{channel}"] = {
            "readings_v": readings,
            "mean_v": sum(readings) / len(readings),
            "min_v": min(readings),
            "max_v": max(readings),
            "in_range": in_range,
        }

    return all_ok, results


# ---------------------------------------------------------------------------
# Continuous acquisition
# ---------------------------------------------------------------------------
def acquire(
    fd: int,
    duration_s: float,
    data_rate: int,
    timestamps: list[int],
    codes: list[int],
) -> int:
    """Read the conversion register continuously for duration_s.

    The address pointer is written once; afterwards each sample is a bare 2-byte
    read. Reads are paced toward the nominal conversion period with a monotonic
    deadline, and every sample carries its own time.monotonic_ns() timestamp so
    the real acquisition rate can be measured instead of assumed.

    The very first conversion read after the configuration change is discarded as
    a warm-up sample: it sits on the config-switch boundary. It is returned for
    the record but never enters the RMS statistics or the CSV samples.
    """
    period_ns = int(1_000_000_000 / data_rate)
    duration_ns = int(duration_s * 1_000_000_000)

    os.write(fd, bytes([REG_CONVERSION]))  # set address pointer once

    # Warm-up: discard the first conversion read after the configuration change.
    # One conversion period is then allowed to elapse so that the first recorded
    # sample comes from a fresh conversion rather than the boundary one.
    warmup = os.read(fd, 2)
    if len(warmup) != 2:
        raise OSError(f"short read from ADS1115: {len(warmup)} bytes")
    discarded_code = twos_complement_16((warmup[0] << 8) | warmup[1])
    time.sleep(period_ns / 1_000_000_000)

    t_start = time.monotonic_ns()
    deadline = t_start
    while True:
        data = os.read(fd, 2)
        now = time.monotonic_ns()
        if len(data) != 2:
            raise OSError(f"short read from ADS1115: {len(data)} bytes")
        timestamps.append(now)
        codes.append(twos_complement_16((data[0] << 8) | data[1]))

        if now - t_start >= duration_ns:
            break

        deadline += period_ns
        sleep_ns = deadline - time.monotonic_ns()
        if sleep_ns > 0:
            time.sleep(sleep_ns / 1_000_000_000)

    return discarded_code


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def safe_ratio(numerator: float, denominator: float):
    """Return numerator/denominator, or None when the result is not meaningful."""
    if not math.isfinite(denominator) or denominator <= 0.0:
        return None
    if not math.isfinite(numerator):
        return None
    value = numerator / denominator
    return value if math.isfinite(value) else None


def summarize(
    timestamps: list[int],
    codes: list[int],
    fsr: float,
    burden_ohm: float,
    ct_primary_a: float,
    ct_secondary_a: float,
    headroom_v: float,
) -> tuple[dict[str, object], list[float], list[float]]:
    count = len(codes)
    lsb_v = fsr / 32768.0
    turns_ratio = ct_primary_a / ct_secondary_a
    scale_a_per_v = turns_ratio / burden_ohm

    summary: dict[str, object] = {
        "total_sample_count": count,
        "lsb_v": lsb_v,
        "turns_ratio": turns_ratio,
        "burden_ohm": burden_ohm,
        "scale_a_per_v": scale_a_per_v,
    }
    if count == 0:
        return summary, [], []

    differential_v = [code * lsb_v for code in codes]
    offset_v = sum(differential_v) / count
    ac_v = [value - offset_v for value in differential_v]

    elapsed_s = (timestamps[-1] - timestamps[0]) / 1e9
    intervals_ms = [
        (timestamps[index] - timestamps[index - 1]) / 1e6 for index in range(1, count)
    ]

    ac_rms_v = math.sqrt(sum(value * value for value in ac_v) / count)
    secondary_rms_a = ac_rms_v / burden_ohm
    primary_rms_a = secondary_rms_a * turns_ratio

    clipping_count = sum(1 for code in codes if code >= ADC_CODE_MAX or code <= ADC_CODE_MIN)
    headroom_exceed_count = sum(1 for value in differential_v if abs(value) >= headroom_v)
    consecutive_equal_code_count = sum(
        1 for index in range(1, count) if codes[index] == codes[index - 1]
    )

    summary.update(
        {
            "acquisition_duration_s": elapsed_s,
            "effective_sample_rate_sps": safe_ratio(float(count - 1), elapsed_s),
            "inter_sample_interval_mean_ms": (
                statistics.fmean(intervals_ms) if intervals_ms else None
            ),
            "inter_sample_interval_std_ms": (
                statistics.stdev(intervals_ms) if len(intervals_ms) >= 2 else None
            ),
            "inter_sample_interval_min_ms": min(intervals_ms) if intervals_ms else None,
            "inter_sample_interval_max_ms": max(intervals_ms) if intervals_ms else None,
            "raw_differential_mean_v": offset_v,
            "min_differential_v": min(differential_v),
            "max_differential_v": max(differential_v),
            "peak_to_peak_v": max(differential_v) - min(differential_v),
            "ac_rms_v": ac_rms_v,
            "secondary_rms_a": secondary_rms_a,
            "primary_rms_a": primary_rms_a,
            "clipping_count": clipping_count,
            "headroom_threshold_v": headroom_v,
            "headroom_exceed_count": headroom_exceed_count,
            "clipping_warning": bool(clipping_count or headroom_exceed_count),
            # Diagnostic reference only. Two consecutive conversions can legitimately
            # produce the same ADC code, so this is NOT evidence of duplicated
            # conversions and is never used to drop or invalidate samples.
            "consecutive_equal_code_count": consecutive_equal_code_count,
            "crest_factor": safe_ratio(max(abs(value) for value in ac_v), ac_rms_v),
            "vpp_to_rms_ratio": safe_ratio(max(ac_v) - min(ac_v), ac_rms_v),
        }
    )
    return summary, differential_v, ac_v


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_outputs(
    out_dir: Path,
    prefix: str,
    metadata: dict[str, object],
    timestamps: list[int],
    codes: list[int],
    differential_v: list[float],
    ac_v: list[float],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{prefix}.csv"
    json_path = out_dir / f"{prefix}.json"

    t0 = timestamps[0] if timestamps else 0
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        for index, timestamp in enumerate(timestamps):
            writer.writerow(
                {
                    "index": index,
                    "monotonic_ns": timestamp,
                    "t_rel_ms": (timestamp - t0) / 1e6,
                    "raw_code": codes[index],
                    "differential_v": differential_v[index],
                    "ac_v": ac_v[index],
                }
            )

    # Per-sample rows live in the CSV; the JSON keeps metadata plus the summary.
    json_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")


def format_value(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def print_summary(summary: dict[str, object]) -> None:
    order = [
        ("total_sample_count", ""),
        ("acquisition_duration_s", "s"),
        ("effective_sample_rate_sps", "SPS"),
        ("inter_sample_interval_mean_ms", "ms"),
        ("inter_sample_interval_std_ms", "ms"),
        ("inter_sample_interval_min_ms", "ms"),
        ("inter_sample_interval_max_ms", "ms"),
        ("raw_differential_mean_v", "V"),
        ("min_differential_v", "V"),
        ("max_differential_v", "V"),
        ("peak_to_peak_v", "V"),
        ("ac_rms_v", "V"),
        ("secondary_rms_a", "A"),
        ("primary_rms_a", "A"),
        ("clipping_count", ""),
        ("headroom_exceed_count", ""),
        ("clipping_warning", ""),
        ("consecutive_equal_code_count", ""),
        ("crest_factor", ""),
        ("vpp_to_rms_ratio", ""),
    ]
    print("")
    print("Diagnostics:")
    for key, unit in order:
        if key not in summary:
            continue
        suffix = f" {unit}" if unit else ""
        print(f"  {key:32s} {format_value(summary[key])}{suffix}")


def print_sanity_failure(sanity: dict[str, object]) -> None:
    print("", file=sys.stderr)
    print("A0/A1 절대 전압 sanity check 실패 -> CT 연속 측정을 시작하지 않았습니다.", file=sys.stderr)
    print(f"허용 범위: {SANITY_MIN_V} V <= V <= {SANITY_MAX_V} V", file=sys.stderr)
    for name in ("a0", "a1"):
        entry = sanity.get(name)
        if not isinstance(entry, dict):
            continue
        state = "정상" if entry.get("in_range") else "범위 이탈"
        print(
            f"  {name.upper()}: mean={entry['mean_v']:.4f} V "
            f"min={entry['min_v']:.4f} V max={entry['max_v']:.4f} V -> {state}",
            file=sys.stderr,
        )
    print("", file=sys.stderr)
    print("확인할 지점:", file=sys.stderr)
    print("  1. 0 V 근처면 해당 채널 배선이 끊어졌거나 ADS1115 입력 핀 접촉 불량입니다.", file=sys.stderr)
    print("  2. 3.2 V 이상이면 VBIAS 분압 저항(10k/10k) 한쪽이 단선 또는 오결선입니다.", file=sys.stderr)
    print("  3. VBIAS 중간 노드 -> ADS1115 A1 연결을 확인하세요.", file=sys.stderr)
    print("  4. 0.68 ohm burden 저항이 A0-A1 사이에 실제로 붙어 있는지 확인하세요.", file=sys.stderr)
    print("  5. ADS1115 VDD가 3.3 V이고 GND가 Jetson GND와 공통인지 확인하세요.", file=sys.stderr)
    print("  배선은 변경하지 말고 위 지점만 점검한 뒤 다시 실행하세요.", file=sys.stderr)


def print_troubleshooting(exc: Exception) -> None:
    print(f"CT/ADS1115 read failed: {exc}", file=sys.stderr)
    print("", file=sys.stderr)
    print("확인할 지점:", file=sys.stderr)
    print("  1. ADS1115가 MAIN 버스(Jetson pin 3/5 -> /dev/i2c-7)에 있는지 확인하세요.", file=sys.stderr)
    print("  2. ADS1115 ADDR 핀이 GND에 묶여 주소가 0x48인지 확인하세요.", file=sys.stderr)
    print("  3. ADS1115 VDD는 3.3 V여야 합니다. 5 V를 쓰면 Jetson SDA/SCL이 손상됩니다.", file=sys.stderr)
    print("  4. ADS1115 GND와 Jetson GND가 공통인지 확인하세요.", file=sys.stderr)
    print("  5. 08_read_ntc_ads1115.py가 동시에 실행 중이 아닌지 확인하세요.", file=sys.stderr)
    print("     하나의 ADS1115는 단일 MUX ADC이므로 동시 접근이 충돌합니다.", file=sys.stderr)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="CT diagnostic read via ADS1115 differential AIN0-AIN1 (0-400 A RMS design range)."
    )
    parser.add_argument("--i2c-port", default=DEFAULT_I2C_PORT)
    parser.add_argument("--address", type=parse_int, default=DEFAULT_ADDRESS)
    parser.add_argument("--duration", type=float, default=3.0, help="acquisition seconds")
    parser.add_argument("--pga", type=float, choices=sorted(PGA_FSR), default=2.048)
    parser.add_argument("--data-rate", type=int, choices=sorted(DATA_RATE_BITS), default=860)
    parser.add_argument("--burden-ohm", type=float, default=0.68)
    parser.add_argument("--ct-primary", type=float, default=400.0)
    parser.add_argument("--ct-secondary", type=float, default=1.0)
    parser.add_argument("--headroom-v", type=float, default=1.9)
    parser.add_argument("--sanity-samples", type=int, default=3)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    if args.duration <= 0:
        parser.error("--duration must be > 0")
    if args.burden_ohm <= 0:
        parser.error("--burden-ohm must be > 0")
    if args.ct_secondary <= 0:
        parser.error("--ct-secondary must be > 0")
    if args.sanity_samples < 1:
        parser.error("--sanity-samples must be >= 1")

    config_value, fsr = build_continuous_diff_config(args.pga, args.data_rate)
    turns_ratio = args.ct_primary / args.ct_secondary
    scale_a_per_v = turns_ratio / args.burden_ohm

    print("CT diagnostic (YHDC SCT024TS via ADS1115 differential AIN0-AIN1)")
    print(f"I2C port: {args.i2c_port}")
    print(f"ADS1115 address: 0x{args.address:02x}")
    print(f"MUX: AIN0-AIN1 (hardware differential)")
    print(f"PGA: +/-{fsr} V   MODE: continuous   DR: {args.data_rate} SPS")
    print(f"Config register write value: 0x{config_value:04X}")
    print(f"Burden: {args.burden_ohm} ohm   CT: {args.ct_primary}/{args.ct_secondary} A")
    print(f"Scale: {scale_a_per_v:.10f} A per volt RMS")
    print(f"Acquisition duration: {args.duration} s")
    print("Validated design range: 0 ~ 400 A RMS (CT nameplate rating)")
    print("No current/noise classification is performed. Measured values only.")
    print("First conversion after the config switch is discarded as warm-up.")

    conflicts = find_conflicting_processes(args.i2c_port)
    if conflicts:
        print("")
        print("WARNING: 동일 ADS1115 또는 동일 I2C 포트를 사용 중인 프로세스가 있습니다.", file=sys.stderr)
        print("하나의 ADS1115는 단일 MUX ADC이므로 동시 접근 시 값이 오염될 수 있습니다.", file=sys.stderr)
        for pid, cmdline in conflicts:
            print(f"  pid {pid}: {cmdline}", file=sys.stderr)
        print("계속 진행하지만 결과 해석에 주의하세요.", file=sys.stderr)

    fd = open_i2c_device(args.i2c_port, args.address)
    original_config: int | None = None
    sanity: dict[str, object] = {}
    restore_info: dict[str, object] = {}
    timestamps: list[int] = []
    codes: list[int] = []
    discarded_code: int | None = None
    interrupted = False

    try:
        original_config = read_register(fd, REG_CONFIG)
        print("")
        print(f"Original config register: 0x{original_config:04X} (will be restored on exit)")

        print("")
        print("Step 1: A0/A1 absolute-voltage sanity check (single-ended, +/-4.096 V)")
        sanity_ok, sanity = sanity_check_inputs(fd, args.sanity_samples)
        for name in ("a0", "a1"):
            entry = sanity[name]
            readings = " ".join(f"{value:.4f}" for value in entry["readings_v"])
            print(
                f"  {name.upper()}: readings=[{readings}] V "
                f"mean={entry['mean_v']:.4f} V "
                f"min={entry['min_v']:.4f} V max={entry['max_v']:.4f} V "
                f"in_range={entry['in_range']}"
            )
        if not sanity_ok:
            print_sanity_failure(sanity)
            return 3

        print("")
        print("Step 2: continuous differential acquisition")
        write_register(fd, REG_CONFIG, config_value)
        time.sleep(2.0 / args.data_rate)  # let the first conversions settle
        try:
            discarded_code = acquire(fd, args.duration, args.data_rate, timestamps, codes)
        except KeyboardInterrupt:
            interrupted = True
            print("")
            print("Interrupted - reporting the samples collected so far.", file=sys.stderr)
    finally:
        if original_config is not None:
            try:
                original_is_single_shot = bool(original_config & MODE_SINGLE_SHOT)
                if original_is_single_shot:
                    # Writing OS=1 in single-shot mode would start a stray conversion.
                    restore_config = original_config & CONFIG_SETTINGS_MASK
                else:
                    restore_config = original_config
                write_register(fd, REG_CONFIG, restore_config)

                readback = read_register(fd, REG_CONFIG)
                fields_match = (
                    readback & CONFIG_SETTINGS_MASK
                ) == (original_config & CONFIG_SETTINGS_MASK)
                restore_info = {
                    "original": f"0x{original_config:04X}",
                    "original_fields": decode_config(original_config),
                    "os_bit_masked": original_is_single_shot,
                    "written": f"0x{restore_config:04X}",
                    "readback": f"0x{readback:04X}",
                    "readback_fields": decode_config(readback),
                    # OS (bit 15) is a status bit on read and is deliberately excluded.
                    "settings_fields_match": fields_match,
                    "restore_ok": True,
                }
                print(
                    f"Restored config: wrote 0x{restore_config:04X} "
                    f"(original 0x{original_config:04X}"
                    f"{', OS bit masked' if original_is_single_shot else ''}), "
                    f"readback 0x{readback:04X}, "
                    f"MUX/PGA/MODE/DR match: {fields_match}"
                )
                if not fields_match:
                    print(
                        "WARNING: config readback does not match the original settings fields.",
                        file=sys.stderr,
                    )
            except Exception as restore_exc:  # noqa: BLE001
                restore_info = {
                    "original": f"0x{original_config:04X}",
                    "restore_ok": False,
                    "error": str(restore_exc),
                }
                print(f"WARNING: config restore failed: {restore_exc}", file=sys.stderr)
        os.close(fd)

    if not codes:
        print("No samples acquired.", file=sys.stderr)
        return 2

    summary, differential_v, ac_v = summarize(
        timestamps=timestamps,
        codes=codes,
        fsr=fsr,
        burden_ohm=args.burden_ohm,
        ct_primary_a=args.ct_primary,
        ct_secondary_a=args.ct_secondary,
        headroom_v=args.headroom_v,
    )
    print_summary(summary)

    if summary.get("clipping_warning"):
        print("")
        print("WARNING: ADC headroom warning.", file=sys.stderr)
        print(
            f"  clipping_count={summary['clipping_count']} "
            f"headroom_exceed_count={summary['headroom_exceed_count']} "
            f"(threshold {args.headroom_v} V of +/-{fsr} V FSR)",
            file=sys.stderr,
        )
        print("  PGA는 자동으로 변경하지 않습니다. 설정 변경은 사용자 판단으로 하세요.", file=sys.stderr)

    metadata = {
        "sensor": "YHDC SCT024TS",
        "ct_primary_a": args.ct_primary,
        "ct_secondary_a": args.ct_secondary,
        "burden_ohm": args.burden_ohm,
        "validated_design_range_a_rms": [0, 400],
        "i2c_port": args.i2c_port,
        "address": f"0x{args.address:02x}",
        "mux": "differential AIN0-AIN1",
        "pga_fsr_v": fsr,
        "mode": "continuous",
        "data_rate_sps": args.data_rate,
        "config_register_written": f"0x{config_value:04X}",
        "config_register_original": f"0x{original_config:04X}" if original_config is not None else None,
        "config_restore": restore_info,
        "sanity_check": sanity,
        "requested_duration_s": args.duration,
        "warmup_discarded_code": discarded_code,
        "warmup_discarded_v": (
            discarded_code * (fsr / 32768.0) if discarded_code is not None else None
        ),
        "warmup_sample_excluded_from_statistics": True,
        "interrupted": interrupted,
        "classification_performed": False,
        "ntc_measured": False,
        "summary": summary,
    }
    write_outputs(
        Path(args.out_dir),
        f"ct_{time.strftime('%Y%m%d_%H%M%S')}",
        metadata,
        timestamps,
        codes,
        differential_v,
        ac_v,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PermissionError as exc:
        print(f"Permission denied: {exc}", file=sys.stderr)
        print("사용자가 i2c 그룹에 속해 있는지 확인하세요: id | grep i2c", file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        raise
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print_troubleshooting(exc)
        raise SystemExit(2)
