"""SPS30 soft-reset maintenance tool: fake sensor only, no I2C."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "jetson_deploy"))
spec = importlib.util.spec_from_file_location("reset_sps30_tool", ROOT / "jetson_deploy/scripts/13_reset_sps30.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class FakeSps30:
    def __init__(self, *, hung=False, never_ready=False):
        self.hung, self.never_ready, self.calls = hung, never_ready, []

    def _call(self, name, value=None):
        self.calls.append(name)
        if self.hung:
            raise TimeoutError("[Errno 110] Connection timed out")
        return value

    def read_product_type(self):
        return self._call("read_product_type", "00080000")

    def device_reset(self):
        return self._call("device_reset")

    def read_serial_number(self):
        return self._call("read_serial_number", "ABC")

    def read_firmware_version(self):
        return self._call("read_firmware_version", (2, 2))

    def read_device_status_register(self):
        return self._call("read_device_status_register", 0)

    def start_measurement(self, fmt):
        return self._call("start_measurement")

    def read_data_ready_flag(self):
        return self._call("read_data_ready_flag", 0 if self.never_ready else 1)

    def read_measurement_values_float(self):
        return self._call("read_measurement_values_float", (1.0, 2.5, 3.0, 4.0, 5, 6, 7, 8, 9, 0.5))

    def stop_measurement(self):
        return self._call("stop_measurement")


class Clock:
    def __init__(self):
        self.now = 0.0

    def sleep(self, s):
        self.now += s

    def monotonic(self):
        return self.now


def run(device, **kw):
    clock = Clock()
    with mock.patch.object(tool, "_float_format", return_value="FLOAT"):
        return tool.reset_sps30(device, sleep=clock.sleep, monotonic=clock.monotonic, **kw)


class ResetSequenceTests(unittest.TestCase):
    def test_healthy_sensor_is_reset_identified_and_measured_once(self):
        device = FakeSps30()
        record = run(device)
        self.assertTrue(record["ok"])
        self.assertEqual(record["measurement"]["pm2_5"], 2.5)
        self.assertEqual(device.calls, ["read_product_type", "device_reset", "read_product_type", "read_serial_number",
                                        "read_firmware_version", "read_device_status_register", "start_measurement",
                                        "read_data_ready_flag", "read_measurement_values_float", "stop_measurement"])
        json.dumps(record)

    def test_a_hung_sensor_is_reported_as_needing_a_power_cycle_without_retries(self):
        device = FakeSps30(hung=True)
        record = run(device)
        self.assertFalse(record["ok"])
        self.assertIn("power cycle", record["hint"])
        self.assertIn("Errno 110", record["steps"]["probe_before"]["error"])
        self.assertFalse(record["reset_command_acknowledged"])
        self.assertNotIn("start_measurement", device.calls, "no measurement on a sensor that did not answer")
        self.assertEqual(device.calls.count("device_reset"), 1, "never retried")

    def test_answering_sensor_without_data_is_not_ok_and_measurement_is_stopped(self):
        device = FakeSps30(never_ready=True)
        record = run(device)
        self.assertFalse(record["ok"])
        self.assertIn("did not produce a measurement", record["hint"])
        self.assertEqual(device.calls[-1], "stop_measurement")

    def test_no_measure_checks_identity_only(self):
        device = FakeSps30()
        record = run(device, measure=False)
        self.assertTrue(record["ok"])
        self.assertNotIn("start_measurement", device.calls)


class MainTests(unittest.TestCase):
    def test_busy_hardware_lock_refuses_before_touching_the_bus(self):
        opened = mock.Mock()
        busy = mock.Mock(side_effect=tool.AcquisitionBusyError("held by collector"))
        with mock.patch.object(tool.AcquisitionLock, "acquire", busy), \
                mock.patch.object(tool, "open_device", opened), \
                mock.patch("builtins.print"):
            self.assertEqual(tool.main([]), 3)
        opened.assert_not_called()

    def test_main_logs_outside_raw_data_and_closes_the_bus(self):
        transceiver = mock.Mock()
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(tool, "LOG_DIR", Path(tmp)), \
                mock.patch.object(tool, "AcquisitionLock") as lock, \
                mock.patch.object(tool, "open_device", return_value=(transceiver, FakeSps30())), \
                mock.patch.object(tool, "_float_format", return_value="FLOAT"), \
                mock.patch.object(tool.time, "sleep"), \
                mock.patch("builtins.print") as printed:
            self.assertEqual(tool.main([]), 0)
            logs = list(Path(tmp).glob("sps30-reset-*.json"))
            self.assertEqual(len(logs), 1)
            self.assertTrue(json.loads(logs[0].read_text())["ok"])
        transceiver.close.assert_called_once()
        lock.return_value.acquire.return_value.release.assert_called_once()
        self.assertTrue(json.loads(printed.call_args[0][0])["ok"])

    def test_hung_sensor_exit_code_is_1(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(tool, "LOG_DIR", Path(tmp)), \
                mock.patch.object(tool, "AcquisitionLock"), \
                mock.patch.object(tool, "open_device", return_value=(mock.Mock(), FakeSps30(hung=True))), \
                mock.patch.object(tool.time, "sleep"), \
                mock.patch("builtins.print"):
            self.assertEqual(tool.main([]), 1)


if __name__ == "__main__":
    unittest.main()
