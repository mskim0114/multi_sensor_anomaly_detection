"""Hardware-free process lifecycle and saved thermal reference regressions."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

JETSON = Path(__file__).resolve().parents[1] / "jetson_deploy"
sys.path.insert(0, str(JETSON))
import collect as control
from sensors import runtime
from sensors.runtime import AcquisitionBusyError, AcquisitionLock, atomic_json
from sensors.validation import validate_run


class ProcessControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.addCleanup(patch.stopall)
        patch.object(control, "runtime_dir", lambda: self.root).start()
        patch.object(runtime, "runtime_dir", lambda: self.root).start()

    def test_process_lock_prevents_second_owner_and_releases(self):
        with AcquisitionLock():
            code = (f"import sys; sys.path.insert(0, {str(JETSON)!r}); "
                    "from sensors.runtime import AcquisitionLock; "
                    f"from pathlib import Path; AcquisitionLock(Path({str(self.root / 'hardware.lock')!r})).acquire()")
            result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Acquisition already active", result.stderr)
        with AcquisitionLock():
            pass

    def test_pid_reuse_token_cannot_signal_current_process(self):
        identity = runtime.process_identity(os.getpid())
        identity["start_ticks"] = str(int(identity["start_ticks"]) - 1)
        with patch.object(signal, "pidfd_send_signal") as send:
            self.assertTrue(runtime.terminate_owned(identity, 0.1))
            send.assert_not_called()

    def test_background_start_duplicate_stop_and_status(self):
        # Use real Popen, pidfds and signals, substituting only the hardware run.
        fake = self.root / "fake_collector.py"
        fake.write_text(f'''
import sys, signal, time
from pathlib import Path
sys.path.insert(0, {str(JETSON)!r})
from sensors import runtime
runtime.runtime_dir = lambda: Path({str(self.root)!r})
import collect
def run(argv, *, run_dir, on_state):
    done = []
    signal.signal(signal.SIGTERM, lambda *_: done.append(True))
    with runtime.AcquisitionLock():
        on_state("starting")
        on_state("running")
        until = time.monotonic() + 15
        while not done and time.monotonic() < until:
            time.sleep(0.01)
        (run_dir / "flushed").write_text("yes")
        on_state("stopped", exit_code=143)
    return 143
collect.runner.main = run
raise SystemExit(collect.main())
''')
        original = subprocess.Popen
        children = []

        def spawn(args, **kwargs):
            args = list(args)
            args[2] = str(fake)
            child = original(args, **kwargs)
            children.append(child)
            return child

        try:
            with patch.object(control.subprocess, "Popen", side_effect=spawn):
                self.assertEqual(control.start(["--out-dir", str(self.root / "data")]), 0)
                first = control.read_json(self.root / "session.json")
                self.assertEqual(control.session_status(first)["status"], "running")
                self.assertEqual(control.start(["--out-dir", str(self.root / "other")]), 0)
                self.assertEqual(len(children), 1)
            self.assertEqual(control.stop(["--timeout", "5"]), 0)
            self.assertTrue((Path(first["run_dir"]) / "flushed").exists())
            self.assertFalse(control.session_status(first)["alive"])
            self.assertEqual(control.stop([]), 0)
        finally:
            for child in children:
                if child.poll() is None:
                    runtime.terminate_owned(runtime.process_identity(child.pid), 5)
                child.wait(timeout=5)

    def test_stale_session_is_failed_without_signal(self):
        session = {"identity": {"pid": 999999999, "start_ticks": "0", "boot_id": "old"},
                   "run_dir": str(self.root), "token": "t"}
        atomic_json(self.root / "control.json", {"status": "running", "token": "t"})
        self.assertEqual(control.session_status(session)["status"], "failed")

    def test_invalid_arguments_rejected_before_hardware(self):
        for argv in (["--duration", "nan"], ["--duration", "-1"],
                     ["--ct-burst", "1"], ["--ct-burst", "0"]):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                control.runner.parse_args(argv)
        self.assertTrue(control.runner.parse_args([]).disable_sgp30)


class SavedThermalTests(unittest.TestCase):
    def test_partial_chunk_missing_chunk_and_sequence_gap(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            np.savez_compressed(root / "thermal_000000.npz",
                                frames=np.zeros((2, 120, 160), dtype=np.uint16), sequences=[0, 1])
            def save(sequence=1):
                snapshots = [{"sequence": seq, "sensors": {"flir": {
                    "status": "ok", "fresh": True, "thermal_chunk": "thermal_000000.npz",
                    "thermal_index": i}}} for i, seq in enumerate((0, sequence))]
                (root / "snapshots.jsonl").write_text("\n".join(map(json.dumps, snapshots)))
                (root / "scalars.csv").write_text(f"sequence\n0\n{sequence}\n")
            save()
            self.assertTrue(validate_run(root, save_thermal=True)["passed"])
            self.assertFalse(validate_run(root, save_thermal=True, expected_snapshots=3)["passed"])
            (root / "scalars.csv").write_text("sequence\n0\n")
            self.assertFalse(validate_run(root, save_thermal=True)["passed"])
            save(2)
            self.assertFalse(validate_run(root, save_thermal=True)["passed"])
            save()
            (root / "thermal_000000.npz").unlink()
            self.assertFalse(validate_run(root, save_thermal=True)["passed"])


class RunnerFinalizationTests(unittest.TestCase):
    def test_sigterm_during_validation_finishes_report_and_restores_handler(self):
        collector = MagicMock()
        collector.sensor_profile.return_value = {}
        collector.sensor_manifest.return_value = {}
        collector.timing_report.return_value = {
            "master": {"snapshot_count": 1, "missed_ticks": 0},
            "storage": {"dropped_chunks": 0, "write_errors": 0, "degraded_events": []}}
        before = signal.getsignal(signal.SIGTERM)

        def validate(*args, **kwargs):
            os.kill(os.getpid(), signal.SIGTERM)
            return {"passed": True}

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(control.runner, "SensorCollector", return_value=collector), \
                patch.object(control.runner, "build_metadata", return_value={}), \
                patch.object(control.runner, "write_run", return_value={}), \
                patch.object(control.runner, "print_report"), \
                patch.object(control.runner, "validate_run", side_effect=validate):
            code = control.runner.main(["--duration", "1", "--out-dir", directory])
            report = json.loads(next(Path(directory).glob("*/timing_report.json")).read_text())
            self.assertEqual(code, 143)
            self.assertEqual(report["status"], "stopped")
            collector.shutdown.assert_called_once()
            collector.request_stop.assert_called_once()
        self.assertEqual(signal.getsignal(signal.SIGTERM), before)


if __name__ == "__main__":
    unittest.main()
