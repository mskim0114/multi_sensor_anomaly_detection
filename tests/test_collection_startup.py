"""Hardware-free regressions for failure and stop while a child is starting."""
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch


JETSON = Path(__file__).resolve().parents[1] / "jetson_deploy"
sys.path.insert(0, str(JETSON))
import collect as control
from sensors import runtime


class CollectionStartupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.children = []
        self.threads = []
        self.original_popen = subprocess.Popen
        for owner in (control, runtime):
            patcher = patch.object(owner, "runtime_dir", lambda: self.root)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self.cleanup_children)

    def cleanup_children(self):
        # Release test gates even if an assertion failed; fake runs have bounded
        # waits and never need a forced kill or access to real sensor devices.
        for name in ("finish_failure", "install_handler"):
            (self.root / name).touch()
        for child in self.children:
            child.wait(timeout=12)
        for thread in self.threads:
            thread.join(timeout=3)

    def write_fake(self, body):
        fake = self.root / "fake_collector.py"
        fake.write_text(
            "import signal, sys, time\n"
            "from pathlib import Path\n"
            f"sys.path.insert(0, {str(JETSON)!r})\n"
            "from sensors import runtime\n"
            f"ROOT = Path({str(self.root)!r})\n"
            "runtime.runtime_dir = lambda: ROOT\n"
            "import collect\n"
            "def wait_for(path):\n"
            "    deadline = time.monotonic() + 8\n"
            "    while not path.exists():\n"
            "        if time.monotonic() >= deadline:\n"
            "            raise RuntimeError('test gate timeout')\n"
            "        time.sleep(0.01)\n"
            + body
            + "\ncollect.runner.main = run\n"
            "raise SystemExit(collect.main())\n"
        )
        return fake

    def spawn(self, args, **kwargs):
        args = list(args)
        args[2] = str(self.root / "fake_collector.py")
        child = self.original_popen(args, **kwargs)
        self.children.append(child)
        return child

    def in_thread(self, action):
        result = {}
        done = threading.Event()

        def run():
            try:
                result["code"] = action()
            except BaseException as exc:
                result["error"] = exc
            finally:
                done.set()

        thread = threading.Thread(target=run, daemon=True)
        self.threads.append(thread)
        thread.start()
        return done, result

    def wait_for_file(self, name):
        deadline = time.monotonic() + 5
        while not (self.root / name).exists():
            if time.monotonic() >= deadline:
                self.fail(f"child did not create {name}")
            time.sleep(0.01)

    def assert_result(self, done, result, expected):
        self.assertTrue(done.wait(5), "controller did not return")
        if "error" in result:
            raise result["error"]
        self.assertEqual(result["code"], expected)

    def test_failed_start_waits_for_stopping_then_returns_failure(self):
        self.write_fake('''
def run(argv, *, run_dir, on_state):
    with runtime.AcquisitionLock():
        on_state("starting")
        on_state("stopping")
        (ROOT / "failure_stopping").touch()
        wait_for(ROOT / "finish_failure")
        (run_dir / "flushed").write_text("yes")
        on_state("failed", exit_code=1, failure="simulated device-open failure")
    return 1
''')
        with patch.object(control.subprocess, "Popen", side_effect=self.spawn):
            done, result = self.in_thread(
                lambda: control.start(["--out-dir", str(self.root / "data")]))
            self.wait_for_file("failure_stopping")
            self.assertFalse(done.wait(0.25), "start returned before failure was finalized")
            (self.root / "finish_failure").touch()
            self.assert_result(done, result, 1)
        session = control.read_json(self.root / "session.json")
        self.assertEqual(control.session_status(session)["status"], "failed")
        self.assertTrue((Path(session["run_dir"]) / "flushed").exists())
        self.assertEqual(self.children[0].wait(timeout=5), 1)
        with runtime.AcquisitionLock():
            pass

    def test_stop_during_startup_waits_for_handler_then_signals_and_flushes(self):
        self.write_fake('''
def run(argv, *, run_dir, on_state):
    stopped = []
    with runtime.AcquisitionLock():
        (ROOT / "before_handler").touch()
        wait_for(ROOT / "install_handler")
        signal.signal(signal.SIGTERM, lambda signum, _: stopped.append(signum))
        on_state("starting")
        deadline = time.monotonic() + 8
        while not stopped:
            if time.monotonic() >= deadline:
                raise RuntimeError("stop signal never arrived")
            time.sleep(0.01)
        on_state("stopping")
        (run_dir / "flushed").write_text(str(stopped[0]))
        on_state("stopped", exit_code=143)
    return 143
''')
        with patch.object(control.subprocess, "Popen", side_effect=self.spawn):
            start_done, start_result = self.in_thread(
                lambda: control.start(["--out-dir", str(self.root / "data")]))
            self.wait_for_file("before_handler")
            self.wait_for_file("session.json")
            session = control.read_json(self.root / "session.json")
            self.assertTrue(session)
            self.assertFalse((Path(session["run_dir"]) / "control.json").exists())
            with self.assertRaises(runtime.AcquisitionBusyError):
                with runtime.AcquisitionLock():
                    pass
            stop_done, stop_result = self.in_thread(
                lambda: control.stop(["--timeout", "5"]))
            self.assertFalse(stop_done.wait(0.25), "stop returned before its handler existed")
            (self.root / "install_handler").touch()
            self.assert_result(stop_done, stop_result, 0)
            self.assert_result(start_done, start_result, 0)
        self.assertEqual((Path(session["run_dir"]) / "flushed").read_text(), "15")
        self.assertEqual(self.children[0].wait(timeout=5), 143)
        state = control.session_status(session)
        self.assertEqual(state["status"], "stopped")
        self.assertFalse(state["alive"])
        with runtime.AcquisitionLock():
            pass


if __name__ == "__main__":
    unittest.main()
