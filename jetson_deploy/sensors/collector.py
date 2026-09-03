"""Synchronized multi-sensor collector for the Jetson runtime.

Master tick
-----------
The trained model consumes 30 ticks at 1 Hz per window, so the master tick is
exactly 1.0 s and is scheduled on absolute monotonic deadlines
(``start + n * period``). Repeated ``time.sleep(1.0)`` would accumulate drift
and is deliberately not used.

Bus arbitration
---------------
Two I2C buses, each accessed serially by exactly one owner:

  /dev/i2c-7   SGP30 0x58, BME680 0x77, ADS1115 0x48 (CT + NTC)
               -> all read from the master tick thread, in that order.
  /dev/i2c-1   SPS30 0x69, SCD30 0x61
               -> one background thread polling data-ready.

The ADS1115 has a single MUX, so CT (AIN0-AIN1 differential) and NTC (A2
single-ended) are time-multiplexed by one owner. Running the standalone 08/09
diagnostics at the same time as this collector would corrupt both.

The FLIR Lepton is USB/UVC and independent of the I2C schedule; a background
thread keeps the latest frame.

This module is a raw acquisition layer. It does not touch the model input
vector or the inference pipeline.
"""

from __future__ import annotations

import math
import os
import queue
import threading
import time
from dataclasses import dataclass
from fcntl import ioctl
from typing import Any

from .snapshot import (
    SCHEMA_VERSION,
    WINDOW_TICKS,
    STATUS_DISABLED,
    STATUS_ERROR,
    STATUS_OK,
    STATUS_STALE,
    STATUS_WARMING_UP,
    observation,
    tick_quality,
    window_quality,
)

# --------------------------------------------------------------------------
# Timing constants
# --------------------------------------------------------------------------
MASTER_PERIOD_S = 1.0
MASTER_PERIOD_NS = 1_000_000_000

# SGP30 initialisation phase. Sensirion: after a successful iaq_init the first
# 15 s are the initialisation phase, during which measure_iaq returns the fixed
# 400 ppm CO2eq / 0 ppb TVOC. The phase is defined by ELAPSED TIME since that
# iaq_init, never by the returned values.
SGP30_WARMUP_S = 15.0

SGP30_INTERVAL_WARN_LOW_MS = 900.0
SGP30_INTERVAL_WARN_HIGH_MS = 1100.0
SGP30_INTERVAL_FAIL_MS = 1500.0

SCD30_STALE_AGE_MS = 4500.0
FLIR_AGE_WARN_MS = 500.0

# --------------------------------------------------------------------------
# Bus / device map (matches AGENTS.md section 4)
# --------------------------------------------------------------------------
BUS7 = "/dev/i2c-7"
BUS1 = "/dev/i2c-1"
ADS1115_ADDRESS = 0x48
SGP30_ADDRESS = 0x58
BME680_ADDRESS = 0x77
SCD30_ADDRESS = 0x61
SPS30_ADDRESS = 0x69

# --------------------------------------------------------------------------
# ADS1115 primitives.
#
# Same register semantics as scripts/08 and scripts/09, which are the verified
# implementations on this board. Those scripts start with a digit and cannot be
# imported; this module is the shared home their headers anticipated. They are
# intentionally left untouched so the standalone diagnostics keep working.
# --------------------------------------------------------------------------
I2C_SLAVE = 0x0703
REG_CONVERSION = 0x00
REG_CONFIG = 0x01

PGA_FSR = {6.144: 0x0000, 4.096: 0x0200, 2.048: 0x0400,
           1.024: 0x0600, 0.512: 0x0800, 0.256: 0x0A00}
DATA_RATE_BITS = {8: 0x0000, 16: 0x0020, 32: 0x0040, 64: 0x0060,
                  128: 0x0080, 250: 0x00A0, 475: 0x00C0, 860: 0x00E0}

OS_NO_EFFECT = 0x0000
OS_START_SINGLE = 0x8000
MUX_DIFF_AIN0_AIN1 = 0x0000
MODE_CONTINUOUS = 0x0000
MODE_SINGLE_SHOT = 0x0100
COMP_QUE_DISABLE = 0x0003

ADC_CODE_MAX = 32767
ADC_CODE_MIN = -32768

# CT front-end, unchanged from the verified 09 diagnostic.
CT_PGA = 2.048
CT_DATA_RATE = 860
CT_BURDEN_OHM = 0.68
CT_PRIMARY_A = 400.0

# NTC divider, unchanged from the verified 08 diagnostic.
NTC_CHANNEL = 2
NTC_PGA = 4.096
NTC_DATA_RATE = 128
NTC_VCC = 3.3
NTC_FIXED_R = 10000.0
NTC_R0 = 10000.0
NTC_BETA = 3950.0
NTC_T0_K = 298.15


def _twos_complement_16(value: int) -> int:
    return value - 0x10000 if value & 0x8000 else value


class Ads1115Owner:
    """Sole owner of the ADS1115. Time-multiplexes CT (AIN0-AIN1) and NTC (A2)."""

    def __init__(self, i2c_port: str = BUS7, address: int = ADS1115_ADDRESS,
                 ct_burst_s: float = 0.5) -> None:
        self.i2c_port = i2c_port
        self.address = address
        self.ct_burst_s = ct_burst_s
        self.ct_fsr = CT_PGA
        self.ct_config = (OS_NO_EFFECT | MUX_DIFF_AIN0_AIN1 | PGA_FSR[CT_PGA]
                          | MODE_CONTINUOUS | DATA_RATE_BITS[CT_DATA_RATE]
                          | COMP_QUE_DISABLE)
        self.fd = os.open(i2c_port, os.O_RDWR)
        try:
            ioctl(self.fd, I2C_SLAVE, address)
        except Exception:
            os.close(self.fd)
            raise

    # -- low level ---------------------------------------------------------
    def _write_reg(self, register: int, value: int) -> None:
        os.write(self.fd, bytes([register, (value >> 8) & 0xFF, value & 0xFF]))

    def _read_reg(self, register: int) -> int:
        os.write(self.fd, bytes([register]))
        data = os.read(self.fd, 2)
        if len(data) != 2:
            raise OSError(f"short read from ADS1115: {len(data)} bytes")
        return (data[0] << 8) | data[1]

    def probe(self) -> int:
        """Read the config register. Raises if the device is not reachable."""
        return self._read_reg(REG_CONFIG)

    def enter_ct_mode(self) -> None:
        self._write_reg(REG_CONFIG, self.ct_config)

    # -- CT ----------------------------------------------------------------
    def ct_burst(self) -> dict[str, Any]:
        """Continuous AIN0-AIN1 burst of ``ct_burst_s`` at 860 SPS.

        The address pointer is written once and every sample is then a bare
        2-byte read paced by a monotonic deadline. The first conversion after
        the config change sits on the switch boundary and is discarded.
        """
        import numpy as np

        self.enter_ct_mode()
        period_ns = int(1_000_000_000 / CT_DATA_RATE)
        duration_ns = int(self.ct_burst_s * 1_000_000_000)

        os.write(self.fd, bytes([REG_CONVERSION]))
        warmup = os.read(self.fd, 2)
        if len(warmup) != 2:
            raise OSError(f"short read from ADS1115: {len(warmup)} bytes")
        time.sleep(period_ns / 1_000_000_000)

        codes: list[int] = []
        t_start = time.monotonic_ns()
        deadline = t_start
        while True:
            data = os.read(self.fd, 2)
            now = time.monotonic_ns()
            if len(data) != 2:
                raise OSError(f"short read from ADS1115: {len(data)} bytes")
            codes.append(_twos_complement_16((data[0] << 8) | data[1]))
            if now - t_start >= duration_ns:
                break
            deadline += period_ns
            sleep_ns = deadline - time.monotonic_ns()
            if sleep_ns > 0:
                time.sleep(sleep_ns / 1_000_000_000)
        t_end = time.monotonic_ns()

        arr = np.asarray(codes, dtype=np.int32)
        volts = arr.astype(np.float64) * self.ct_fsr / 32768.0
        offset = float(volts.mean())
        ac = volts - offset
        vrms = float(math.sqrt(float((ac ** 2).mean())))
        capture_ms = (t_end - t_start) / 1e6
        actual_sps = (len(codes) / (capture_ms / 1000.0)) if capture_ms > 0 else None
        clipping = int(((arr >= ADC_CODE_MAX) | (arr <= ADC_CODE_MIN)).sum())

        return {
            "sample_count": len(codes),
            "capture_duration_ms": round(capture_ms, 3),
            "actual_sample_rate": None if actual_sps is None else round(actual_sps, 2),
            "vdiff_mean": offset,
            "vdiff_min": float(volts.min()),
            "vdiff_max": float(volts.max()),
            "vrms": vrms,
            "current_a_nominal": vrms / CT_BURDEN_OHM * CT_PRIMARY_A,
            "clipping": clipping,
            "_codes": arr,
        }

    # -- NTC ---------------------------------------------------------------
    def read_ntc(self) -> dict[str, Any]:
        """One single-shot A2 conversion, then back to the CT configuration."""
        config = (OS_START_SINGLE | ((0x04 + NTC_CHANNEL) << 12) | PGA_FSR[NTC_PGA]
                  | MODE_SINGLE_SHOT | DATA_RATE_BITS[NTC_DATA_RATE] | COMP_QUE_DISABLE)
        settle = (1.0 / NTC_DATA_RATE) + 0.01
        try:
            # First conversion after the MUX/PGA change is discarded.
            self._write_reg(REG_CONFIG, config)
            time.sleep(settle)
            self._read_reg(REG_CONVERSION)

            self._write_reg(REG_CONFIG, config)
            time.sleep(settle)
            raw = _twos_complement_16(self._read_reg(REG_CONVERSION))
        finally:
            # Restore CT configuration for the next tick regardless of outcome.
            try:
                self.enter_ct_mode()
            except Exception:
                pass

        voltage = raw * NTC_PGA / 32768.0
        if voltage <= 0.0 or voltage >= NTC_VCC:
            raise ValueError(f"NTC divider voltage out of range: {voltage:.6f} V")
        resistance = NTC_FIXED_R * voltage / (NTC_VCC - voltage)
        inv_t = (1.0 / NTC_T0_K) + (math.log(resistance / NTC_R0) / NTC_BETA)
        return {
            "raw_code": raw,
            "voltage_v": voltage,
            "resistance_ohm": resistance,
            "temperature_c": (1.0 / inv_t) - 273.15,
        }

    def close(self) -> None:
        try:
            os.close(self.fd)
        except Exception:
            pass


# --------------------------------------------------------------------------
# Per-sensor state
# --------------------------------------------------------------------------
@dataclass
class SensorState:
    name: str
    consecutive_errors: int = 0
    total_errors: int = 0
    total_ok: int = 0
    last_error: str | None = None
    handle: Any = None
    _retry_at_seq: int = 0

    def ok(self) -> None:
        self.consecutive_errors = 0
        self.total_ok += 1
        self.last_error = None

    def fail(self, exc: BaseException) -> None:
        self.consecutive_errors += 1
        self.total_errors += 1
        self.last_error = f"{type(exc).__name__}: {exc}"


@dataclass
class Reading:
    """A value published by a background worker, with its acquisition time."""
    values: dict
    monotonic_ns: int
    sequence: int = 0


# --------------------------------------------------------------------------
# Bus 1 worker: SPS30 + SCD30
# --------------------------------------------------------------------------
class Bus1Worker(threading.Thread):
    """Owns /dev/i2c-1. Polls data-ready and publishes the latest readings.

    Polling at 4 Hz rather than reading at 1 Hz keeps `age_ms` honest: a new
    SCD30 sample (0.5 Hz native) is picked up within ~250 ms of becoming
    available instead of up to a full second later.
    """

    POLL_S = 0.25

    def __init__(self, i2c_port: str = BUS1) -> None:
        super().__init__(name="bus1", daemon=True)
        self.i2c_port = i2c_port
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._sps30: Reading | None = None
        self._scd30: Reading | None = None
        self.sps30_state = SensorState("sps30")
        self.scd30_state = SensorState("scd30")
        self._transceiver = None
        self._sps = None
        self._scd = None
        self.startup_error: str | None = None
        # Physical identity, read once at setup. Only values actually returned
        # by the device are stored; nothing is synthesised.
        self.identity: dict[str, dict] = {}

    # -- setup -------------------------------------------------------------
    def _setup(self) -> None:
        from sensirion_driver_adapters.i2c_adapter.i2c_channel import I2cChannel
        from sensirion_i2c_driver import CrcCalculator, I2cConnection, LinuxI2cTransceiver
        from sensirion_i2c_sps30.commands import OutputFormat
        from sensirion_i2c_sps30.device import Sps30Device
        from sensirion_i2c_scd30.device import Scd30Device

        self._transceiver = LinuxI2cTransceiver(self.i2c_port)
        self._transceiver.open()
        conn = I2cConnection(self._transceiver)
        crc = CrcCalculator(8, 0x31, 0xFF, 0x00)

        try:
            self._sps = Sps30Device(I2cChannel(conn, slave_address=SPS30_ADDRESS, crc=crc))
            try:
                self._sps.wake_up_sequence()
            except Exception:
                pass
            try:
                self._sps.stop_measurement()
            except Exception:
                pass
            self.identity["sps30"] = self._read_identity(self._sps, (
                ("serial", "read_serial_number"),
                ("product_type", "read_product_type"),
                ("firmware", "read_firmware_version"),
            ))
            # Started once and left running; not restarted per tick.
            self._sps.start_measurement(OutputFormat.OUTPUT_FORMAT_FLOAT)
        except Exception as exc:
            self.sps30_state.fail(exc)
            self._sps = None

        try:
            self._scd = Scd30Device(I2cChannel(conn, slave_address=SCD30_ADDRESS, crc=crc))
            self.identity["scd30"] = self._read_identity(self._scd, (
                ("serial", "read_serial_number"),
                ("firmware", "read_firmware_version"),
            ))
            self._scd.start_periodic_measurement(0)
        except Exception as exc:
            self.scd30_state.fail(exc)
            self._scd = None

    # -- helpers -----------------------------------------------------------
    @classmethod
    def _read_identity(cls, device, fields) -> dict:
        """Read identity fields. A field the device does not answer is omitted.

        No value is ever invented: if the device provides no unique serial, the
        key simply does not appear.
        """
        out: dict = {}
        for key, method in fields:
            try:
                value = cls._plain(getattr(device, method)())
            except Exception:
                continue
            if value is not None:
                out[key] = str(value)
        return out

    @staticmethod
    def _plain(value):
        if hasattr(value, "value"):
            return Bus1Worker._plain(getattr(value, "value"))
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value)

    def _publish(self, attr: str, values: dict) -> None:
        with self._lock:
            prev = getattr(self, attr)
            seq = (prev.sequence + 1) if prev else 1
            setattr(self, attr, Reading(values, time.monotonic_ns(), seq))

    def latest(self) -> tuple[Reading | None, Reading | None]:
        with self._lock:
            return self._sps30, self._scd30

    # -- loop --------------------------------------------------------------
    def run(self) -> None:
        try:
            self._setup()
        except Exception as exc:
            self.startup_error = f"{type(exc).__name__}: {exc}"
            return

        while not self._stop_event.is_set():
            if self._scd is not None:
                try:
                    if self._scd.get_data_ready():
                        co2, temp, rh = self._scd.read_measurement_data()
                        self._publish("_scd30", {
                            "co2_ppm": float(co2),
                            "temperature_c": float(temp),
                            "humidity_pct": float(rh),
                        })
                        self.scd30_state.ok()
                except Exception as exc:
                    self.scd30_state.fail(exc)

            if self._sps is not None:
                try:
                    if self._plain(self._sps.read_data_ready_flag()):
                        v = [self._plain(x) for x in self._sps.read_measurement_values_float()]
                        self._publish("_sps30", {
                            "pm1_0_ug_m3": v[0], "pm2_5_ug_m3": v[1],
                            "pm4_0_ug_m3": v[2], "pm10_ug_m3": v[3],
                            "nc0_5_per_cm3": v[4], "nc1_0_per_cm3": v[5],
                            "nc2_5_per_cm3": v[6], "nc4_0_per_cm3": v[7],
                            "nc10_per_cm3": v[8], "typical_particle_size_um": v[9],
                        })
                        self.sps30_state.ok()
                except Exception as exc:
                    self.sps30_state.fail(exc)

            self._stop_event.wait(self.POLL_S)

    def shutdown(self) -> None:
        self._stop_event.set()
        self.join(timeout=3.0)
        for dev, stop in ((self._sps, "stop_measurement"),
                          (self._scd, "stop_periodic_measurement")):
            if dev is not None:
                try:
                    getattr(dev, stop)()
                except Exception:
                    pass
        if self._transceiver is not None:
            try:
                self._transceiver.close()
            except Exception:
                pass


# --------------------------------------------------------------------------
# FLIR Lepton worker (USB UVC, independent of the I2C schedule)
# --------------------------------------------------------------------------
class ThermalWorker(threading.Thread):
    """Keeps the latest PureThermal frame. The camera is opened once.

    The PureThermal stream is 160x122 GRAY16: rows 0..119 are the image and
    rows 120..121 are telemetry (verified on this board - the telemetry rows
    contain values such as 0 and 58744 that are not temperatures). Only the
    first 120 rows are kept, which is the shape the model expects.
    """

    MODEL_HEIGHT = 120
    MODEL_WIDTH = 160

    def __init__(self, device: str = "/dev/video0") -> None:
        super().__init__(name="thermal", daemon=True)
        self.device = device
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._frame = None
        self._frame_ns = 0
        self._sequence = 0
        self.state = SensorState("flir")
        self.shape_failures = 0
        self.startup_error: str | None = None
        self._cap = None

    def _open(self) -> None:
        import cv2
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {self.device}")
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("Y", "1", "6", " "))
        self._cap = cap

    def run(self) -> None:
        try:
            self._open()
        except Exception as exc:
            self.startup_error = f"{type(exc).__name__}: {exc}"
            return
        while not self._stop_event.is_set():
            try:
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    self.state.fail(RuntimeError("frame read failed"))
                    time.sleep(0.02)
                    continue
                if frame.ndim != 2 or frame.shape[1] != self.MODEL_WIDTH \
                        or frame.shape[0] < self.MODEL_HEIGHT:
                    self.shape_failures += 1
                    self.state.fail(RuntimeError(f"unexpected frame shape {frame.shape}"))
                    continue
                image = frame[: self.MODEL_HEIGHT, : self.MODEL_WIDTH].copy()
                with self._lock:
                    self._frame = image
                    self._frame_ns = time.monotonic_ns()
                    self._sequence += 1
                self.state.ok()
            except Exception as exc:
                self.state.fail(exc)
                time.sleep(0.05)

    def read_identity(self) -> dict:
        """USB descriptors from sysfs. Only fields actually present are kept."""
        import glob
        out: dict = {"device": self.device}
        name = os.path.basename(os.path.realpath(self.device))
        link = f"/sys/class/video4linux/{name}/device"
        try:
            usb_iface = os.path.realpath(link)
            usb_dev = os.path.dirname(usb_iface)
        except Exception:
            return out
        for key, fname in (("serial", "serial"), ("usb_vendor_id", "idVendor"),
                           ("usb_product_id", "idProduct"),
                           ("manufacturer", "manufacturer"), ("product", "product")):
            path = os.path.join(usb_dev, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    value = f.read().strip()
                if value:
                    out[key] = value
            except Exception:
                continue
        try:
            v4l = sorted(glob.glob(f"/sys/class/video4linux/{name}/name"))
            if v4l:
                out["v4l2_name"] = open(v4l[0], encoding="utf-8").read().strip()
        except Exception:
            pass
        return out

    def latest(self):
        with self._lock:
            if self._frame is None:
                return None, 0, 0
            return self._frame, self._frame_ns, self._sequence

    def shutdown(self) -> None:
        self._stop_event.set()
        self.join(timeout=3.0)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass


# --------------------------------------------------------------------------
# Chunk writer: keeps NPZ compression off the acquisition critical path
# --------------------------------------------------------------------------
class ChunkWriter(threading.Thread):
    """Single background writer behind a BOUNDED queue.

    np.savez_compressed of a 30-frame thermal chunk costs ~190 ms. Measured on
    this board over a 1800-tick soak, doing it inline made every 30th tick
    955.8 ms against a 1000 ms budget (18 ms headroom) while ordinary ticks sat
    at 779.6 ms. Moving it here removes that periodic spike.

    The queue is bounded on purpose. If the writer cannot keep up, the backlog
    is NOT silently discarded: the submit is recorded as a storage degradation
    with the affected chunk name, and `dropped` is counted so the run report
    shows it. Shutdown drains the queue, flushes and joins the thread.
    """

    QUEUE_MAXSIZE = 8            # 8 thermal chunks = 240 s of buffer
    SUBMIT_TIMEOUT_S = 0.05      # absorb transient slowness, never a long stall

    def __init__(self) -> None:
        super().__init__(name="chunkwriter", daemon=False)
        self.q: queue.Queue = queue.Queue(maxsize=self.QUEUE_MAXSIZE)
        self.written = 0
        self.errors = 0
        self.dropped = 0
        self.degraded_events: list[dict] = []
        self.max_queue_depth = 0
        self.last_error: str | None = None
        self._sentinel = object()

    def submit(self, path, arrays: dict) -> bool:
        """Queue one chunk. Returns False and records degradation if refused."""
        try:
            self.q.put((path, arrays), timeout=self.SUBMIT_TIMEOUT_S)
        except queue.Full:
            self.dropped += 1
            self.degraded_events.append({
                "chunk": str(getattr(path, "name", path)),
                "reason": "writer_queue_full",
                "queue_maxsize": self.QUEUE_MAXSIZE,
            })
            return False
        depth = self.q.qsize()
        if depth > self.max_queue_depth:
            self.max_queue_depth = depth
        return True

    def run(self) -> None:
        import numpy as np
        while True:
            item = self.q.get()
            try:
                if item is self._sentinel:
                    return
                path, arrays = item
                try:
                    np.savez_compressed(path, **arrays)
                    self.written += 1
                except Exception as exc:
                    self.errors += 1
                    self.last_error = f"{type(exc).__name__}: {exc}"
            finally:
                self.q.task_done()

    def shutdown(self, timeout_s: float = 60.0) -> None:
        """Drain, flush and join. Anything still queued is reported, not hidden."""
        self.q.put(self._sentinel)
        self.join(timeout=timeout_s)
        if self.is_alive() or not self.q.empty():
            remaining = self.q.qsize()
            self.degraded_events.append({
                "reason": "writer_did_not_drain_before_shutdown",
                "remaining": remaining,
            })

    def report(self) -> dict:
        return {
            "chunks_written": self.written,
            "queue_maxsize": self.QUEUE_MAXSIZE,
            "queue_max_depth": self.max_queue_depth,
            "dropped_chunks": self.dropped,
            "write_errors": self.errors,
            "last_error": self.last_error,
            "degraded_events": self.degraded_events,
        }


# --------------------------------------------------------------------------
# Bus 7 readers driven from the master tick thread
# --------------------------------------------------------------------------
class Sgp30Reader:
    """SGP30 on /dev/i2c-7 @0x58. Strict 1 Hz: iaq_measure once per tick.

    The device is initialised once (iaq_init) and the instance is kept for the
    whole run; re-initialising per tick would restart the IAQ algorithm.

    Warm-up is an INITIALISATION SESSION, not a property of the values. Each
    successful iaq_init opens a session with its own monotonic start time; for
    the first SGP30_WARMUP_S of that session the status is warming_up, after
    which it is ok. The returned values never affect the status: 400/0 is the
    documented output during initialisation, but a live reading that happens to
    land on 400/0 later must not push the status back to warming_up.

    warming_up -> ok therefore happens at most once per session. A genuine
    reinitialisation after a communication failure opens a NEW session and
    legitimately restarts the 15 s phase; the session id and init timestamp are
    recorded so that is distinguishable from oscillation.
    """

    RETRY_BACKOFF_TICKS = (10, 20, 40, 80, 160, 300)

    def __init__(self, bus: int = 7, address: int = SGP30_ADDRESS) -> None:
        self.bus = bus
        self.address = address
        self.state = SensorState("sgp30")
        self._sensor = None
        self._last_measure_ns: int | None = None
        self.intervals_ms: list[float] = []
        self.measure_count = 0
        self.serial: str | None = None
        self._retry_step = 0
        self._ticks_since_init = 0
        # initialisation session
        self._session_id = 0
        self._session_init_ns: int | None = None
        self._session_init_sequence: int | None = None
        self.init_count = 0
        self.session_log: list[dict] = []

    def _schedule_retry(self, sequence: int) -> None:
        """Bounded exponential backoff.

        Re-initialising through Blinka issues an I2CDevice probe whose
        zero-length write is not supported by the Tegra adapter (Errno 95).
        Hammering an absent or marginal device with that every 10 ticks adds
        useless traffic to a bus shared with the ADS1115, so the interval
        backs off and is capped.
        """
        step = min(self._retry_step, len(self.RETRY_BACKOFF_TICKS) - 1)
        self.state._retry_at_seq = sequence + self.RETRY_BACKOFF_TICKS[step]
        self._retry_step = min(self._retry_step + 1, len(self.RETRY_BACKOFF_TICKS) - 1)

    def _init(self) -> None:
        from adafruit_extended_bus import ExtendedI2C
        import adafruit_sgp30
        i2c = ExtendedI2C(self.bus)
        sensor = adafruit_sgp30.Adafruit_SGP30(i2c, address=self.address)
        self.serial = "".join("%04X" % w for w in sensor.serial)
        sensor.iaq_init()
        # A new initialisation session starts here, not at first read.
        self._sensor = sensor
        self._retry_step = 0
        self._ticks_since_init = 0
        self._last_measure_ns = None
        self._session_id += 1
        self.init_count += 1
        self._session_init_ns = time.monotonic_ns()
        self._session_init_sequence = None   # filled by the caller's sequence below

    def prime(self) -> None:
        """Initialise before the first tick.

        Two reasons this cannot wait for tick 0. The run manifest is written
        right after start() and must be able to record the serial, which is only
        known once iaq_init has succeeded. And the 15 s initialisation phase
        should be measured from the actual iaq_init, not from the first read.

        Failure is not fatal: the reader falls back to its normal backoff and
        the manifest simply carries no serial for this sensor.
        """
        if self._sensor is not None:
            return
        try:
            self._init()
            self._session_init_sequence = 0
            self.session_log.append({
                "session_id": self._session_id,
                "init_sequence": 0,
                "init_before_first_tick": True,
                "init_monotonic_ns": self._session_init_ns,
                "serial": self.serial,
            })
        except Exception as exc:
            self.state.fail(exc)
            self._schedule_retry(0)

    def read(self, sequence: int) -> dict:
        if self._sensor is None:
            if sequence < self.state._retry_at_seq:
                return observation(STATUS_ERROR, error=self.state.last_error,
                                   consecutive_errors=self.state.consecutive_errors)
            try:
                self._init()
                self._session_init_sequence = sequence
                self.session_log.append({
                    "session_id": self._session_id,
                    "init_sequence": sequence,
                    "init_monotonic_ns": self._session_init_ns,
                    "serial": self.serial,
                })
            except Exception as exc:
                self.state.fail(exc)
                self._schedule_retry(sequence)
                return observation(STATUS_ERROR, error=self.state.last_error,
                                   consecutive_errors=self.state.consecutive_errors)

        try:
            now = time.monotonic_ns()
            eco2, tvoc = self._sensor.iaq_measure()
            self._ticks_since_init += 1
            interval_ms = None
            if self._last_measure_ns is not None:
                interval_ms = (now - self._last_measure_ns) / 1e6
                self.intervals_ms.append(interval_ms)
            self._last_measure_ns = now
            self.measure_count += 1
            self.state.ok()

            # Status comes from the elapsed time of the initialisation session,
            # never from the measured values.
            session_elapsed_s = (now - self._session_init_ns) / 1e9
            warmup = session_elapsed_s < SGP30_WARMUP_S
            extra: dict = {
                "warmup": warmup,
                "measure_count": self.measure_count,
                "ticks_since_init": self._ticks_since_init,
                "session_id": self._session_id,
                "session_init_sequence": self._session_init_sequence,
                "session_elapsed_s": round(session_elapsed_s, 3),
            }
            if interval_ms is not None:
                extra["interval_ms"] = round(interval_ms, 3)
                if interval_ms > SGP30_INTERVAL_FAIL_MS:
                    extra["interval_violation"] = "hard"
                elif not (SGP30_INTERVAL_WARN_LOW_MS <= interval_ms <= SGP30_INTERVAL_WARN_HIGH_MS):
                    extra["interval_violation"] = "warn"
            return observation(
                STATUS_WARMING_UP if warmup else STATUS_OK,
                {"eco2_ppm": int(eco2), "tvoc_ppb": int(tvoc)},
                fresh=True, age_ms=0.0, extra=extra,
            )
        except Exception as exc:
            self.state.fail(exc)
            self._sensor = None
            self._schedule_retry(sequence)
            return observation(STATUS_ERROR, error=self.state.last_error,
                               consecutive_errors=self.state.consecutive_errors)


class Bme680Reader:
    """BME680 on /dev/i2c-7 @0x77, 1 Hz. Persistent driver instance.

    Temperature is a context value: the gas heater warms the die so it reads
    above ambient. The model's temperature channel is the NTC.
    """

    RETRY_BACKOFF_TICKS = (10, 20, 40, 80, 160, 300)

    def __init__(self, bus: int = 7, address: int = BME680_ADDRESS) -> None:
        self.bus = bus
        self.address = address
        self.state = SensorState("bme680")
        self._sensor = None
        self._retry_step = 0

    def _schedule_retry(self, sequence: int) -> None:
        step = min(self._retry_step, len(self.RETRY_BACKOFF_TICKS) - 1)
        self.state._retry_at_seq = sequence + self.RETRY_BACKOFF_TICKS[step]
        self._retry_step = min(self._retry_step + 1, len(self.RETRY_BACKOFF_TICKS) - 1)

    def _init(self) -> None:
        from adafruit_extended_bus import ExtendedI2C
        import adafruit_bme680
        i2c = ExtendedI2C(self.bus)
        self._sensor = adafruit_bme680.Adafruit_BME680_I2C(i2c, address=self.address)
        self._retry_step = 0

    def read_identity(self, i2c_port: str = BUS7) -> dict:
        """Read chip_id (0xD0) and variant_id (0xF0) with normal register reads.

        Two ordinary register reads, not an address scan. Called once at start
        so the run metadata records which physical part answered.
        """
        out: dict = {}
        try:
            fd = os.open(i2c_port, os.O_RDWR)
        except Exception:
            return out
        try:
            ioctl(fd, I2C_SLAVE, self.address)
            for key, reg in (("chip_id", 0xD0), ("variant_id", 0xF0)):
                try:
                    os.write(fd, bytes([reg]))
                    data = os.read(fd, 1)
                    if len(data) == 1:
                        out[key] = f"0x{data[0]:02x}"
                except Exception:
                    continue
        except Exception:
            pass
        finally:
            os.close(fd)
        return out

    def read(self, sequence: int) -> dict:
        if self._sensor is None:
            if sequence < self.state._retry_at_seq:
                return observation(STATUS_ERROR, error=self.state.last_error,
                                   consecutive_errors=self.state.consecutive_errors)
            try:
                self._init()
            except Exception as exc:
                self.state.fail(exc)
                self._schedule_retry(sequence)
                return observation(STATUS_ERROR, error=self.state.last_error,
                                   consecutive_errors=self.state.consecutive_errors)
        try:
            values = {
                "temperature_c": float(self._sensor.temperature),
                "humidity_pct": float(self._sensor.relative_humidity),
                "pressure_hpa": float(self._sensor.pressure),
                "gas_ohm": int(self._sensor.gas),
            }
            self.state.ok()
            return observation(STATUS_OK, values, fresh=True, age_ms=0.0)
        except Exception as exc:
            self.state.fail(exc)
            self._sensor = None
            self._schedule_retry(sequence)
            return observation(STATUS_ERROR, error=self.state.last_error,
                               consecutive_errors=self.state.consecutive_errors)


# --------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------
def _pct(values: list[float], p: float):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    idx = min(len(ordered) - 1, max(0, int(round((p / 100.0) * (len(ordered) - 1)))))
    return round(ordered[idx], 3)


def _mean(values: list[float]):
    return round(sum(values) / len(values), 3) if values else None


def _mx(values: list[float]):
    return round(max(values), 3) if values else None


def _mn(values: list[float]):
    return round(min(values), 3) if values else None


# --------------------------------------------------------------------------
# Collector
# --------------------------------------------------------------------------
class SensorCollector:
    """Master 1 Hz tick over all sensors, with per-bus serial access."""

    THERMAL_CHUNK_FRAMES = 30
    CT_RAW_CHUNK_TICKS = 60

    def __init__(self, run_dir, duration_s: float, *, save_ct_raw: bool = False,
                 save_thermal: bool = True, ct_burst_s: float = 0.5,
                 thermal_device: str = "/dev/video0") -> None:
        self.run_dir = run_dir
        self.duration_s = duration_s
        self.save_ct_raw = save_ct_raw
        self.save_thermal = save_thermal
        self.ct_burst_s = ct_burst_s
        self.thermal_device = thermal_device

        self.ads: Ads1115Owner | None = None
        self.writer = ChunkWriter()
        self.sgp30 = Sgp30Reader()
        self.bme680 = Bme680Reader()
        self.bus1 = Bus1Worker()
        self.thermal = ThermalWorker(thermal_device)

        self.ct_state = SensorState("ct1")
        self.ntc_state = SensorState("ntc")

        # timing accumulators
        self.jitters_ms: list[float] = []
        self.periods_ms: list[float] = []
        self.work_ms: list[float] = []
        self.ct_sample_counts: list[int] = []
        self.ct_sps: list[float] = []
        self.ct_clipping_ticks = 0
        self.flir_ages_ms: list[float] = []
        self.flir_ticks_with_frame = 0
        self.sps30_fresh_ticks = 0
        self.scd30_fresh_ticks = 0
        self.scd30_nonfresh_ticks = 0
        self.scd30_fresh_intervals_ms: list[float] = []
        self.scd30_ages_ms: list[float] = []
        self.sps30_ages_ms: list[float] = []
        self.snapshot_count = 0
        self.expected_ticks = 0
        self.stop_requested = False
        self.invalid_tick_count = 0
        self.invalid_reason_counts: dict[str, int] = {}
        self._window_buffer: list[dict] = []
        self.window_results: list[dict] = []

        self._last_sps30_seq = 0
        self._last_scd30_seq = 0
        self._last_scd30_fresh_ns: int | None = None
        self._thermal_buffer: list = []
        self._thermal_chunk_index = 0
        self._ct_raw_buffer: list = []
        self._ct_raw_chunk_index = 0
        self._last_thermal_seq = 0

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        """Startup-fatal work happens here: without the ADS1115 there is no run."""
        self.ads = Ads1115Owner(ct_burst_s=self.ct_burst_s)
        self.ads.probe()
        self.ads.enter_ct_mode()
        self.writer.start()
        self.bus1.start()
        self.thermal.start()
        self._await_workers()
        # SGP30 is initialised here, before the first tick, so that
        # sensor_manifest() can record its serial and the 15 s initialisation
        # phase is measured from the real iaq_init. Failure is non-fatal.
        self.sgp30.prime()

    def _await_workers(self, timeout_s: float = 8.0) -> None:
        """Wait for the background workers to produce their first reading.

        Without this the first ticks report status=error for sensors whose
        worker has simply not delivered anything yet - a startup artefact, not
        a fault. Bounded by timeout_s: a sensor that never appears must not
        block the run, it is reported as degraded instead.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            frame, _, _ = self.thermal.latest()
            sps, scd = self.bus1.latest()
            if frame is not None and sps is not None and scd is not None:
                return
            if self.thermal.startup_error and self.bus1.startup_error:
                return
            time.sleep(0.1)

    def shutdown(self) -> None:
        self.bus1.shutdown()
        self.thermal.shutdown()
        # Chunk writer last: flush_chunks() has already queued the partial
        # chunks, and shutdown() drains, flushes and joins.
        self.writer.shutdown()
        if self.ads is not None:
            self.ads.close()

    def request_stop(self) -> None:
        self.stop_requested = True

    # -- provenance --------------------------------------------------------
    def sensor_manifest(self) -> dict:
        """Physical inventory at the start of this run.

        Call after start(). Only facts actually read from the devices are
        recorded - a device that provides no unique serial simply has no
        `serial` key. Nothing here is synthesised or carried over from a
        previous run.

        This is the run-level physical inventory. Runtime initialisation
        history (per-session iaq_init records) stays in
        timing_report.sgp30.sessions and is a different thing.
        """
        manifest: dict = {
            "ads1115": {
                "bus": BUS7,
                "address": f"0x{ADS1115_ADDRESS:02x}",
                "note": "no unique serial available from this part",
            },
            "sgp30": {"bus": BUS7, "address": f"0x{SGP30_ADDRESS:02x}"},
            "bme680": {"bus": BUS7, "address": f"0x{BME680_ADDRESS:02x}"},
            "sps30": {"bus": BUS1, "address": f"0x{SPS30_ADDRESS:02x}"},
            "scd30": {"bus": BUS1, "address": f"0x{SCD30_ADDRESS:02x}"},
            "flir": {},
        }
        if self.sgp30.serial:
            manifest["sgp30"]["serial"] = self.sgp30.serial
        manifest["bme680"].update(self.bme680.read_identity())
        for name in ("sps30", "scd30"):
            manifest[name].update(self.bus1.identity.get(name, {}))
        manifest["flir"].update(self.thermal.read_identity())
        return manifest

    # -- per-tick sensor reads --------------------------------------------
    def _read_ct(self, sequence: int) -> dict:
        try:
            result = self.ads.ct_burst()
            codes = result.pop("_codes")
            if self.save_ct_raw:
                self._ct_raw_buffer.append((sequence, codes))
                if len(self._ct_raw_buffer) >= self.CT_RAW_CHUNK_TICKS:
                    self._flush_ct_raw()
            self.ct_state.ok()
            self.ct_sample_counts.append(result["sample_count"])
            if result["actual_sample_rate"]:
                self.ct_sps.append(result["actual_sample_rate"])
            if result["clipping"]:
                self.ct_clipping_ticks += 1
            return observation(STATUS_OK, result, fresh=True, age_ms=0.0)
        except Exception as exc:
            self.ct_state.fail(exc)
            return observation(STATUS_ERROR, error=self.ct_state.last_error,
                               consecutive_errors=self.ct_state.consecutive_errors)

    def _read_ntc(self) -> dict:
        try:
            values = self.ads.read_ntc()
            self.ntc_state.ok()
            return observation(STATUS_OK, values, fresh=True, age_ms=0.0)
        except Exception as exc:
            self.ntc_state.fail(exc)
            return observation(STATUS_ERROR, error=self.ntc_state.last_error,
                               consecutive_errors=self.ntc_state.consecutive_errors)

    def _observe_bus1(self, now_ns: int) -> tuple[dict, dict]:
        sps_reading, scd_reading = self.bus1.latest()

        if sps_reading is None:
            sps_obs = observation(
                STATUS_ERROR,
                error=self.bus1.startup_error or self.bus1.sps30_state.last_error,
                consecutive_errors=self.bus1.sps30_state.consecutive_errors)
        else:
            age_ms = (now_ns - sps_reading.monotonic_ns) / 1e6
            fresh = sps_reading.sequence != self._last_sps30_seq
            self._last_sps30_seq = sps_reading.sequence
            if fresh:
                self.sps30_fresh_ticks += 1
            self.sps30_ages_ms.append(age_ms)
            sps_obs = observation(
                STATUS_OK if age_ms <= 3000.0 else STATUS_STALE,
                sps_reading.values, fresh=fresh, age_ms=age_ms,
                extra={"measurement_sequence": sps_reading.sequence})

        if scd_reading is None:
            scd_obs = observation(
                STATUS_ERROR,
                error=self.bus1.startup_error or self.bus1.scd30_state.last_error,
                consecutive_errors=self.bus1.scd30_state.consecutive_errors)
        else:
            age_ms = (now_ns - scd_reading.monotonic_ns) / 1e6
            fresh = scd_reading.sequence != self._last_scd30_seq
            self._last_scd30_seq = scd_reading.sequence
            if fresh:
                self.scd30_fresh_ticks += 1
                if self._last_scd30_fresh_ns is not None:
                    self.scd30_fresh_intervals_ms.append(
                        (scd_reading.monotonic_ns - self._last_scd30_fresh_ns) / 1e6)
                self._last_scd30_fresh_ns = scd_reading.monotonic_ns
            else:
                self.scd30_nonfresh_ticks += 1
            self.scd30_ages_ms.append(age_ms)
            scd_obs = observation(
                STATUS_STALE if age_ms > SCD30_STALE_AGE_MS else STATUS_OK,
                scd_reading.values, fresh=fresh, age_ms=age_ms,
                extra={"measurement_sequence": scd_reading.sequence,
                       "measurement_monotonic_ns": scd_reading.monotonic_ns})
        return sps_obs, scd_obs

    def _observe_thermal(self, now_ns: int, sequence: int) -> dict:
        import numpy as np
        frame, frame_ns, frame_seq = self.thermal.latest()
        if frame is None:
            return observation(
                STATUS_ERROR,
                error=self.thermal.startup_error or self.thermal.state.last_error,
                consecutive_errors=self.thermal.state.consecutive_errors)

        age_ms = (now_ns - frame_ns) / 1e6
        self.flir_ages_ms.append(age_ms)
        self.flir_ticks_with_frame += 1
        fresh = frame_seq != self._last_thermal_seq
        self._last_thermal_seq = frame_seq

        celsius = frame.astype(np.float64) / 100.0 - 273.15
        values = {
            "min_c": float(celsius.min()),
            "max_c": float(celsius.max()),
            "mean_c": float(celsius.mean()),
            "shape": list(frame.shape),
        }
        extra: dict = {"frame_sequence": frame_seq, "frame_monotonic_ns": frame_ns}
        if self.save_thermal:
            extra["thermal_chunk"] = f"thermal_{self._thermal_chunk_index:06d}.npz"
            extra["thermal_index"] = len(self._thermal_buffer)
            self._thermal_buffer.append((sequence, frame))
            if len(self._thermal_buffer) >= self.THERMAL_CHUNK_FRAMES:
                self._flush_thermal()
        status = STATUS_OK if age_ms <= FLIR_AGE_WARN_MS else STATUS_STALE
        return observation(status, values, fresh=fresh, age_ms=age_ms, extra=extra)

    # -- chunk writers -----------------------------------------------------
    def _flush_thermal(self) -> None:
        """Hand the chunk to the background writer. No compression here."""
        if not self._thermal_buffer:
            return
        import numpy as np
        path = self.run_dir / f"thermal_{self._thermal_chunk_index:06d}.npz"
        self.writer.submit(path, {
            "frames": np.stack([f for _, f in self._thermal_buffer]),
            "sequences": np.asarray([s for s, _ in self._thermal_buffer], dtype=np.int64),
        })
        self._thermal_buffer = []
        self._thermal_chunk_index += 1

    def _flush_ct_raw(self) -> None:
        if not self._ct_raw_buffer:
            return
        import numpy as np
        path = self.run_dir / f"ct_raw_{self._ct_raw_chunk_index:06d}.npz"
        self.writer.submit(path, {
            "sequences": np.asarray([s for s, _ in self._ct_raw_buffer], dtype=np.int64),
            "codes": np.concatenate([c for _, c in self._ct_raw_buffer]),
            "lengths": np.asarray([len(c) for _, c in self._ct_raw_buffer], dtype=np.int64),
            "fsr_volts": np.float64(CT_PGA),
        })
        self._ct_raw_buffer = []
        self._ct_raw_chunk_index += 1

    def flush_chunks(self) -> None:
        self._flush_thermal()
        self._flush_ct_raw()

    # -- master tick -------------------------------------------------------
    def iter_snapshots(self):
        """Yield one snapshot per master tick, on absolute monotonic deadlines.

        Deadlines are ``start + n * period`` so lateness in one tick cannot
        accumulate into the next. If a tick's work overruns past the following
        deadline, the elapsed deadlines are counted as missed rather than
        silently shifting the whole schedule.
        """
        import datetime as _dt

        start_ns = time.monotonic_ns()
        self.start_monotonic_ns = start_ns
        expected = None if self.duration_s <= 0 else int(round(self.duration_s / MASTER_PERIOD_S))
        self.expected_ticks = expected or 0
        self.missed_ticks = 0
        n = 0
        prev_actual_ns: int | None = None

        while expected is None or n < expected:
            if self.stop_requested:
                break

            target_ns = start_ns + n * MASTER_PERIOD_NS
            now_ns = time.monotonic_ns()
            if now_ns >= target_ns + MASTER_PERIOD_NS:
                # The previous tick overran. Skip the deadlines already gone.
                skipped = int((now_ns - target_ns) // MASTER_PERIOD_NS)
                self.missed_ticks += skipped
                n += skipped
                if expected is not None and n >= expected:
                    break
                target_ns = start_ns + n * MASTER_PERIOD_NS
                now_ns = time.monotonic_ns()
            if now_ns < target_ns:
                time.sleep((target_ns - now_ns) / 1e9)

            actual_ns = time.monotonic_ns()
            jitter_ms = (actual_ns - target_ns) / 1e6
            self.jitters_ms.append(jitter_ms)
            if prev_actual_ns is not None:
                self.periods_ms.append((actual_ns - prev_actual_ns) / 1e6)
            prev_actual_ns = actual_ns

            # --- bus 7, serial, in timing-priority order --------------------
            sgp30_obs = self.sgp30.read(n)          # strict 1 Hz, goes first
            bme680_obs = self.bme680.read(n)
            ct_obs = self._read_ct(n)               # ~0.5 s burst
            ntc_obs = self._read_ntc()

            # --- buses/devices read from their own workers ------------------
            read_ns = time.monotonic_ns()
            sps30_obs, scd30_obs = self._observe_bus1(read_ns)
            flir_obs = self._observe_thermal(read_ns, n)

            work_ms = (time.monotonic_ns() - actual_ns) / 1e6
            self.work_ms.append(work_ms)

            sensors = {
                "ntc": ntc_obs,
                "ct1": ct_obs,
                # Only one CT front-end exists. CT2-4 are reported as
                # disabled, never zero-filled and never a copy of CT1.
                "ct2": observation(STATUS_DISABLED, extra={"reason": "no physical CT connected"}),
                "ct3": observation(STATUS_DISABLED, extra={"reason": "no physical CT connected"}),
                "ct4": observation(STATUS_DISABLED, extra={"reason": "no physical CT connected"}),
                "sps30": sps30_obs,
                "sgp30": sgp30_obs,
                "scd30": scd30_obs,
                "bme680": bme680_obs,
                "flir": flir_obs,
            }
            quality = tick_quality(sensors)
            if not quality["flir_frame_valid"]:
                self.invalid_tick_count += 1
                for reason in quality["invalid_reasons"]:
                    self.invalid_reason_counts[reason] = \
                        self.invalid_reason_counts.get(reason, 0) + 1

            snapshot = {
                "schema_version": SCHEMA_VERSION,
                "sequence": n,
                "timestamp_utc": _dt.datetime.now(_dt.timezone.utc)
                                    .strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "target_monotonic_ns": target_ns,
                "actual_monotonic_ns": actual_ns,
                "tick_jitter_ms": round(jitter_ms, 3),
                "tick_work_ms": round(work_ms, 3),
                "quality": quality,
                "sensors": sensors,
            }

            # Non-overlapping windows, evaluated as they complete. This is a
            # quality label only - it does not build model input.
            self._window_buffer.append(snapshot)
            if len(self._window_buffer) >= WINDOW_TICKS:
                self.window_results.append(window_quality(self._window_buffer))
                self._window_buffer = []

            self.snapshot_count += 1
            yield snapshot
            n += 1

    # -- timing report -----------------------------------------------------
    def timing_report(self) -> dict:
        abs_jitter = [abs(j) for j in self.jitters_ms]
        sgp_iv = self.sgp30.intervals_ms
        return {
            "master": {
                "snapshot_count": self.snapshot_count,
                "expected_ticks": self.expected_ticks,
                "missed_ticks": self.missed_ticks,
                "period_ms_mean": _mean(self.periods_ms),
                "period_ms_p50": _pct(self.periods_ms, 50),
                "period_ms_p95": _pct(self.periods_ms, 95),
                "period_ms_max": _mx(self.periods_ms),
                "jitter_ms_mean": _mean(self.jitters_ms),
                "abs_jitter_ms_p95": _pct(abs_jitter, 95),
                "abs_jitter_ms_max": _mx(abs_jitter),
                "tick_work_ms_mean": _mean(self.work_ms),
                "tick_work_ms_max": _mx(self.work_ms),
            },
            "sgp30": {
                "serial": self.sgp30.serial,
                "measurement_count": self.sgp30.measure_count,
                "interval_ms_mean": _mean(sgp_iv),
                "interval_ms_p95": _pct(sgp_iv, 95),
                "interval_ms_max": _mx(sgp_iv),
                "violations_gt_1100ms": sum(1 for v in sgp_iv if v > SGP30_INTERVAL_WARN_HIGH_MS),
                "violations_gt_1500ms": sum(1 for v in sgp_iv if v > SGP30_INTERVAL_FAIL_MS),
                "ok_count": self.sgp30.state.total_ok,
                "error_count": self.sgp30.state.total_errors,
                "consecutive_errors_at_end": self.sgp30.state.consecutive_errors,
                "initialisation_count": self.sgp30.init_count,
                "warmup_seconds": SGP30_WARMUP_S,
                "sessions": self.sgp30.session_log,
            },
            "sps30": {
                "fresh_snapshot_count": self.sps30_fresh_ticks,
                "age_ms_mean": _mean(self.sps30_ages_ms),
                "age_ms_max": _mx(self.sps30_ages_ms),
                "ok_count": self.bus1.sps30_state.total_ok,
                "error_count": self.bus1.sps30_state.total_errors,
            },
            "scd30": {
                "fresh_snapshot_count": self.scd30_fresh_ticks,
                "non_fresh_snapshot_count": self.scd30_nonfresh_ticks,
                "fresh_interval_ms_mean": _mean(self.scd30_fresh_intervals_ms),
                "fresh_interval_ms_max": _mx(self.scd30_fresh_intervals_ms),
                "age_ms_max": _mx(self.scd30_ages_ms),
                "ok_count": self.bus1.scd30_state.total_ok,
                "error_count": self.bus1.scd30_state.total_errors,
            },
            "bme680": {
                "ok_count": self.bme680.state.total_ok,
                "error_count": self.bme680.state.total_errors,
            },
            "ct1": {
                "burst_count": len(self.ct_sample_counts),
                "samples_per_burst_mean": _mean([float(v) for v in self.ct_sample_counts]),
                "samples_per_burst_min": _mn([float(v) for v in self.ct_sample_counts]),
                "samples_per_burst_max": _mx([float(v) for v in self.ct_sample_counts]),
                "actual_sps_mean": _mean(self.ct_sps),
                "actual_sps_min": _mn(self.ct_sps),
                "actual_sps_max": _mx(self.ct_sps),
                "clipping_tick_count": self.ct_clipping_ticks,
                "error_count": self.ct_state.total_errors,
            },
            "ntc": {
                "ok_count": self.ntc_state.total_ok,
                "error_count": self.ntc_state.total_errors,
            },
            "storage": self.writer.report(),
            "quality": {
                "policy": ("window is training-invalid when any tick has "
                           "flir status != ok or age_ms > 500 ms; raw data is kept, "
                           "stale frames are never duplicated or interpolated"),
                "window_ticks": WINDOW_TICKS,
                "invalid_tick_count": self.invalid_tick_count,
                "invalid_tick_reasons": self.invalid_reason_counts,
                "windows_evaluated": len(self.window_results),
                "windows_valid": sum(1 for w in self.window_results if w["valid"]),
                "windows_invalid": sum(1 for w in self.window_results if not w["valid"]),
                "invalid_window_details": [w for w in self.window_results if not w["valid"]],
            },
            "flir": {
                "ticks_with_frame": self.flir_ticks_with_frame,
                "age_ms_mean": _mean(self.flir_ages_ms),
                "age_ms_p95": _pct(self.flir_ages_ms, 95),
                "age_ms_max": _mx(self.flir_ages_ms),
                "frames_captured": self.thermal._sequence,
                "shape_failures": self.thermal.shape_failures,
                "error_count": self.thermal.state.total_errors,
            },
        }
