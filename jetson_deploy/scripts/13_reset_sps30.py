"""Operator maintenance: soft-reset the SPS30 (Sensirion "device reset") and check it answers.

Usage:
    ./jetson_deploy/run_python.sh jetson_deploy/scripts/13_reset_sps30.py [--i2c-port /dev/i2c-1] [--no-measure]

Run only between acquisitions. It takes the same hardware lock as the collector and trial
runner, so it refuses while either holds the bus. It never runs automatically: the collector
still stops on the first SPS30 error and keeps that error (no in-run retry); this tool is the
explicit, logged step an operator takes before starting a new run.

Steps: probe (read product type) -> device_reset -> wait -> read identity and status ->
optional short measurement (start, wait for data-ready, read once, stop). Nothing is written to
raw data; a JSON record goes to ~/.cache/factory_safety/maintenance/.

A soft reset needs a sensor that still answers on I2C. If the probe and the reset both time
out, the sensor (or bus) is stuck harder than a reset can reach; only a power cycle helps.

Exit codes: 0 sensor answers and measures after reset; 1 reset attempted but sensor not OK;
2 could not open the bus; 3 acquisition active (nothing was sent to the sensor).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sensors.runtime import AcquisitionBusyError, AcquisitionLock  # noqa: E402

DEFAULT_I2C_PORT = "/dev/i2c-1"
SPS30_I2C_ADDRESS = 0x69
RESET_WAIT_S = 1.0          # datasheet: reset completes well within this; the sensor returns to idle
DATA_READY_TIMEOUT_S = 5.0  # first measurement after start is available after about 1 s
LOG_DIR = Path("~/.cache/factory_safety/maintenance").expanduser()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def plain(value):
    if hasattr(value, "value"):
        return plain(value.value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return str(value)


def _step(record: dict, name: str, call):
    started = time.monotonic()
    try:
        value = call()
        record[name] = {"ok": True, "value": plain(value), "ms": round((time.monotonic() - started) * 1000, 1)}
        return True, value
    except Exception as error:  # report every failure as it happened; never retry
        record[name] = {"ok": False, "error": f"{type(error).__name__}: {error}",
                        "ms": round((time.monotonic() - started) * 1000, 1)}
        return False, None


def reset_sps30(device, *, measure: bool = True, sleep=time.sleep, monotonic=time.monotonic) -> dict:
    """Run the reset sequence on an opened Sps30Device-like object. Returns a JSON-able record."""
    record = {"started_utc": utc_now(), "steps": {}}
    steps = record["steps"]
    _step(steps, "probe_before", device.read_product_type)
    reset_ok, _ = _step(steps, "device_reset", device.device_reset)
    sleep(RESET_WAIT_S)
    identity_ok, _ = _step(steps, "product_type", device.read_product_type)
    _step(steps, "serial", device.read_serial_number)
    _step(steps, "firmware", device.read_firmware_version)
    _step(steps, "status_register", device.read_device_status_register)
    measured = None
    if measure and identity_ok:
        from_start, _ = _step(steps, "start_measurement", lambda: device.start_measurement(_float_format()))
        if from_start:
            deadline = monotonic() + DATA_READY_TIMEOUT_S
            ready = False
            while monotonic() < deadline:
                ok, flag = _step(steps, "data_ready", device.read_data_ready_flag)
                if not ok:
                    break
                if plain(flag):   # same truthiness test as the collector
                    ready = True
                    break
                sleep(0.2)
            if ready:
                ok, values = _step(steps, "read_values", device.read_measurement_values_float)
                if ok:
                    names = ("pm1_0", "pm2_5", "pm4_0", "pm10")
                    measured = {n: plain(v) for n, v in zip(names, list(values)[:4])}
            _step(steps, "stop_measurement", device.stop_measurement)
    record["measurement"] = measured
    record["ok"] = bool(identity_ok and (measured is not None or not measure))
    record["reset_command_acknowledged"] = reset_ok
    if not record["ok"]:
        record["hint"] = ("sensor did not answer after the soft reset; a power cycle (unplug/replug, or a "
                          "switched supply) is needed" if not identity_ok else
                          "sensor answers but did not produce a measurement")
    record["finished_utc"] = utc_now()
    return record


def _float_format():
    from sensirion_i2c_sps30.commands import OutputFormat
    return OutputFormat.OUTPUT_FORMAT_FLOAT


def open_device(port: str):
    from sensirion_driver_adapters.i2c_adapter.i2c_channel import I2cChannel
    from sensirion_i2c_driver import CrcCalculator, I2cConnection, LinuxI2cTransceiver
    from sensirion_i2c_sps30.device import Sps30Device
    transceiver = LinuxI2cTransceiver(port)
    transceiver.open()
    channel = I2cChannel(I2cConnection(transceiver), slave_address=SPS30_I2C_ADDRESS,
                         crc=CrcCalculator(8, 0x31, 0xFF, 0x00))
    return transceiver, Sps30Device(channel)


def save(record: dict) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"sps30-reset-{record['started_utc'].replace(':', '').replace('+', 'p')}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--i2c-port", default=DEFAULT_I2C_PORT)
    parser.add_argument("--no-measure", action="store_true", help="skip the short measurement check")
    args = parser.parse_args(argv)
    record = {"tool": "13_reset_sps30", "i2c_port": args.i2c_port, "address": hex(SPS30_I2C_ADDRESS)}
    try:
        lock = AcquisitionLock().acquire()
    except AcquisitionBusyError as error:
        record.update(ok=False, error=f"acquisition active; stop the collector first ({error})")
        print(json.dumps(record, ensure_ascii=False))
        return 3
    try:
        try:
            transceiver, device = open_device(args.i2c_port)
        except Exception as error:
            record.update(ok=False, error=f"cannot open {args.i2c_port}: {type(error).__name__}: {error}")
            record["log"] = str(save(dict(record, started_utc=utc_now())))
            print(json.dumps(record, ensure_ascii=False))
            return 2
        try:
            record.update(reset_sps30(device, measure=not args.no_measure))
        finally:
            try:
                transceiver.close()
            except Exception:
                pass
    finally:
        lock.release()
    record["log"] = str(save(record))
    print(json.dumps(record, ensure_ascii=False))
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
