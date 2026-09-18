"""Hardware-free regression tests for the acquisition lifecycle."""

from __future__ import annotations

import errno
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "jetson_deploy"))

from sensors import collector  # noqa: E402
from sensors import runtime  # noqa: E402


class _FakeLock:
    instances = []

    def __init__(self):
        self.acquired = False
        self.released = False
        self.__class__.instances.append(self)

    def acquire(self):
        self.acquired = True
        return self

    def release(self):
        self.released = True


class CollectorErrorTests(unittest.TestCase):
    def test_nested_driver_errno_is_preserved(self):
        try:
            try:
                raise OSError(errno.EREMOTEIO, "kernel nack")
            except OSError as cause:
                raise RuntimeError("driver wrapper") from cause
        except RuntimeError as wrapped:
            record = collector._error_record("sensor.read", wrapped)
        self.assertEqual(record["errno"], errno.EREMOTEIO)
        self.assertEqual([e["exception_class"] for e in record["exception_chain"]],
                         ["RuntimeError", "OSError"])

    def test_sgp30_failure_is_not_retried(self):
        errors = []
        reader = collector.Sgp30Reader(
            on_error=lambda operation, exc, previous: errors.append(
                (operation, exc, previous)))
        failure = OSError(errno.EREMOTEIO, "synthetic I2C failure")
        with mock.patch.object(reader, "_init", side_effect=failure) as initialise:
            reader.prime()
            for sequence in (0, 10, 100, 1000):
                self.assertEqual(reader.read(sequence)["status"], collector.STATUS_ERROR)
        initialise.assert_called_once()
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0][0], "sgp30.prime")

    def test_bus1_failure_stops_worker_without_retry(self):
        errors = []
        worker = collector.Bus1Worker(
            on_error=lambda operation, exc, previous: errors.append(operation))

        class BrokenScd:
            calls = 0

            def get_data_ready(self):
                self.calls += 1
                raise OSError(errno.EREMOTEIO, "synthetic SCD30 failure")

        broken = BrokenScd()

        def setup():
            worker._scd = broken
            worker._sps = None

        with mock.patch.object(worker, "_setup", side_effect=setup):
            worker.run()
        self.assertEqual(broken.calls, 1)
        self.assertEqual(errors, ["scd30.poll_and_read"])
        self.assertTrue(worker._stop_event.is_set())

    def test_first_error_is_atomic_and_stops_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            instance = collector.SensorCollector(Path(tmp), 1, enable_sgp30=False)
            first = OSError(errno.EREMOTEIO, "first")
            instance._record_error("sensor.first", first, "sensor.previous")
            instance._record_error("sensor.second", RuntimeError("second"), None)
            self.assertEqual(instance.first_error["operation"], "sensor.first")
            self.assertEqual(instance.first_error["errno"], errno.EREMOTEIO)
            self.assertEqual(instance.first_error["last_successful_operation"],
                             "sensor.previous")
            self.assertTrue(instance.stopped_on_error)
            self.assertTrue(instance._stop_event.is_set())


class CollectorLifecycleTests(unittest.TestCase):
    def setUp(self):
        _FakeLock.instances.clear()

    def test_partial_start_failure_cleans_writer_ads_and_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            instance = collector.SensorCollector(Path(tmp), 1, enable_sgp30=False)

            class FakeAds:
                closed = False

                def __init__(self, **_):
                    pass

                def probe(self):
                    return 0

                def enter_ct_mode(self):
                    pass

                def close(self):
                    self.closed = True

            class BrokenBus:
                sps30_state = collector.SensorState("sps30")
                scd30_state = collector.SensorState("scd30")

                def start(self):
                    raise RuntimeError("synthetic thread start failure")

                def shutdown(self, **_):
                    return []

            instance.bus1 = BrokenBus()
            with mock.patch.object(runtime, "AcquisitionLock", _FakeLock), \
                 mock.patch.object(collector, "Ads1115Owner", FakeAds):
                with self.assertRaisesRegex(RuntimeError, "synthetic thread"):
                    instance.start()

            self.assertFalse(instance.writer.is_alive())
            self.assertTrue(instance.ads.closed)
            self.assertTrue(_FakeLock.instances[0].acquired)
            self.assertTrue(_FakeLock.instances[0].released)
            instance.shutdown()  # required to be idempotent
            report = instance.timing_report()
            self.assertEqual(report["first_error"]["operation"], "bus1.start")
            self.assertIn("shutdown_errors", report)

    def test_worker_readiness_timeout_is_a_startup_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            instance = collector.SensorCollector(Path(tmp), 1, enable_sgp30=False)

            class EmptyThermal:
                startup_error = None

                def latest(self):
                    return None, 0, 0

            class EmptyBus:
                startup_error = None

                def latest(self):
                    return None, None

            instance.thermal = EmptyThermal()
            instance.bus1 = EmptyBus()
            with self.assertRaisesRegex(
                    TimeoutError, "flir, sps30, scd30"):
                instance._await_workers(timeout_s=0)

    def test_worker_readiness_wait_honours_stop_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            instance = collector.SensorCollector(Path(tmp), 1, enable_sgp30=False)
            instance.request_stop()
            started = time.monotonic()
            with self.assertRaisesRegex(InterruptedError, "stop requested"):
                instance._await_workers(timeout_s=8)
            self.assertLess(time.monotonic() - started, 0.5)

    def test_partial_thermal_chunk_is_written_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instance = collector.SensorCollector(root, 1, enable_sgp30=False)
            instance.writer.start()
            first = np.full((120, 160), 1, dtype=np.uint16)
            second = np.full((120, 160), 2, dtype=np.uint16)
            instance._thermal_buffer = [(4, first), (5, second)]
            instance.flush_chunks()
            instance.writer.shutdown(timeout_s=5)

            target = root / "thermal_000000.npz"
            self.assertTrue(target.is_file())
            self.assertFalse((root / "thermal_000000.npz.tmp").exists())
            with np.load(target) as data:
                np.testing.assert_array_equal(data["sequences"], [4, 5])
                self.assertEqual(data["frames"].shape, (2, 120, 160))

    def test_live_worker_timeout_keeps_hardware_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            instance = collector.SensorCollector(Path(tmp), 1, enable_sgp30=False)
            lock = _FakeLock().acquire()
            instance._acquisition_lock = lock

            class TimedOutBus:
                def shutdown(self, **_):
                    return [collector._error_record(
                        "bus1.join", TimeoutError("still alive"))]

            instance.bus1 = TimedOutBus()
            instance.shutdown()
            self.assertFalse(lock.released)
            self.assertEqual(instance.shutdown_errors[0]["operation"], "bus1.join")

    def test_same_thermal_frame_keeps_tick_sequences_and_marks_nonfresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instance = collector.SensorCollector(root, 1, enable_sgp30=False)
            instance.writer.start()
            frame = np.full((120, 160), 30000, dtype=np.uint16)

            class Thermal:
                state = collector.SensorState("flir")
                startup_error = None

                def latest(self):
                    return frame, time.monotonic_ns(), 7

            instance.thermal = Thermal()
            first = instance._observe_thermal(0)
            second = instance._observe_thermal(1)
            self.assertTrue(first["fresh"])
            self.assertFalse(second["fresh"])
            self.assertEqual(len(instance._thermal_buffer), 2)
            self.assertEqual((first["thermal_chunk"], first["thermal_index"]),
                             ("thermal_000000.npz", 0))
            self.assertEqual((second["thermal_chunk"], second["thermal_index"]),
                             ("thermal_000000.npz", 1))

            instance.flush_chunks()
            instance.writer.shutdown(timeout_s=5)
            with np.load(root / "thermal_000000.npz") as data:
                np.testing.assert_array_equal(data["sequences"], [0, 1])
                np.testing.assert_array_equal(data["frames"][0], frame)
                np.testing.assert_array_equal(data["frames"][1], frame)


class CollectorTimingTests(unittest.TestCase):
    def test_ntc_failure_does_not_issue_restore_write(self):
        ads = collector.Ads1115Owner.__new__(collector.Ads1115Owner)
        ads._write_reg = mock.Mock(side_effect=OSError(errno.EREMOTEIO, "nack"))
        ads._read_reg = mock.Mock()
        ads.enter_ct_mode = mock.Mock()
        with self.assertRaises(OSError):
            ads.read_ntc()
        ads.enter_ct_mode.assert_not_called()

    def test_latest_age_is_never_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            instance = collector.SensorCollector(Path(tmp), 1, enable_sgp30=False)
            future = time.monotonic_ns() + 1_000_000_000
            reading = collector.Reading({"value": 1}, future, 1)

            class Bus:
                startup_error = None
                sps30_state = collector.SensorState("sps30")
                scd30_state = collector.SensorState("scd30")

                def latest(self):
                    return reading, reading

            instance.bus1 = Bus()
            sps, scd = instance._observe_bus1()
            self.assertEqual(sps["age_ms"], 0.0)
            self.assertEqual(scd["age_ms"], 0.0)

    def test_ct_and_ntc_expose_acquisition_intervals(self):
        with tempfile.TemporaryDirectory() as tmp:
            instance = collector.SensorCollector(Path(tmp), 1, enable_sgp30=False)
            now = time.monotonic_ns()

            class Ads:
                def ct_burst(self):
                    return {
                        "acquisition_start_monotonic_ns": now - 500_000_000,
                        "acquisition_end_monotonic_ns": now,
                        "acquisition_center_monotonic_ns": now - 250_000_000,
                        "sample_count": 2,
                        "capture_duration_ms": 500.0,
                        "actual_sample_rate": 4.0,
                        "vdiff_mean": 0.0,
                        "vdiff_min": 0.0,
                        "vdiff_max": 0.0,
                        "vrms": 0.0,
                        "current_a_nominal": 0.0,
                        "clipping": 0,
                        "_codes": np.array([0, 0]),
                    }

                def read_ntc(self):
                    return {
                        "acquisition_start_monotonic_ns": now,
                        "acquisition_end_monotonic_ns": now + 1,
                        "acquisition_center_monotonic_ns": now,
                        "temperature_c": 25.0,
                    }

            instance.ads = Ads()
            ct = instance._read_ct(0)
            ntc = instance._read_ntc()
            self.assertLess(ct["values"]["acquisition_start_monotonic_ns"],
                            ct["values"]["acquisition_end_monotonic_ns"])
            self.assertIn("acquisition_center_monotonic_ns", ntc["values"])
            self.assertGreaterEqual(ct["age_ms"], 0.0)
            self.assertGreaterEqual(ntc["age_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
