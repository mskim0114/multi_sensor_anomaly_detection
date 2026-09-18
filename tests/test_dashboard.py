"""Hardware-free live preview, dashboard isolation, and control regressions."""
import base64
import csv
from email.message import Message
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

JETSON = Path(__file__).resolve().parents[1] / "jetson_deploy"
sys.path.insert(0, str(JETSON))
import collect
import dashboard
from sensors import live, runtime
from sensors.collector import write_run


def snapshot(sequence=0, *, stamp=None):
    return {
        "sequence": sequence,
        "timestamp_utc": "2026-09-17T00:00:00+00:00",
        "tick_jitter_ms": 0.0,
        "snapshot_created_monotonic_ns": time.monotonic_ns() if stamp is None else stamp,
        "sensors": {"flir": {"status": "ok", "fresh": True,
                             "frame_sequence": 100 + sequence}},
    }


class TemporaryDashboardTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.runtime = self.root / "runtime"
        self.runtime.mkdir()
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        for module, attribute, value in (
            (live, "preview_dir", self.cache),
            (dashboard, "preview_dir", self.cache),
            (runtime, "runtime_dir", self.runtime),
            (collect, "runtime_dir", self.runtime),
            (dashboard, "runtime_dir", self.runtime),
        ):
            patcher = patch.object(module, attribute, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.app = dashboard.Dashboard(out_dir=self.root / "output", cache_root=self.cache)

    def wait_until(self, predicate, message):
        deadline = time.monotonic() + 3
        while not predicate():
            if time.monotonic() >= deadline:
                self.fail(message)
            time.sleep(0.01)


class PreviewTests(TemporaryDashboardTest):
    def test_preview_contains_exact_selected_frame_and_snapshot_sequence(self):
        frame = np.arange(120 * 160, dtype=np.uint16).reshape(120, 160).astype(">u2")
        original = frame.copy()
        selected = snapshot(7)
        publisher = live.PreviewPublisher(self.run_dir, root=self.cache)
        publisher.start()
        try:
            publisher.offer(selected, (7, frame))
        finally:
            publisher.close()
        self.assertFalse(publisher.thread.is_alive())
        payload = dashboard.read_object(publisher.path)
        thermal = payload["thermal"]
        self.assertEqual(payload["snapshot"], selected)
        self.assertEqual(thermal["snapshot_sequence"], 7)
        self.assertEqual(thermal["frame_sequence"], 107)
        self.assertEqual(thermal["dtype"], "<u2")
        decoded = np.frombuffer(base64.b64decode(thermal["data"]), dtype="<u2").reshape(120, 160)
        np.testing.assert_array_equal(decoded, original)
        np.testing.assert_array_equal(frame, original)
        self.assertIsNone(live.thermal_payload(selected, (8, frame)))
        self.assertIsNone(live.thermal_payload(selected, None))

    def test_queue_is_bounded_and_keeps_latest_snapshot(self):
        publisher = live.PreviewPublisher(self.run_dir, root=self.cache)
        for sequence in range(1000):
            publisher.offer(snapshot(sequence))
        self.assertEqual(publisher.queue.qsize(), 1)
        self.assertEqual(publisher.stats()["dropped_updates"], 999)
        publisher.start()
        publisher.close()
        self.assertFalse(publisher.thread.is_alive())
        self.assertEqual(dashboard.read_object(publisher.path)["snapshot"]["sequence"], 999)

    def test_worker_cache_failure_does_not_interrupt_raw_write_run(self):
        failed = threading.Event()
        actual_write = live.volatile_json

        def fail_preview(path, value):
            if path.name.startswith("run-"):
                failed.set()
                raise OSError("simulated preview cache full")
            actual_write(path, value)

        def samples():
            yield snapshot(0)
            if not failed.wait(3):
                raise AssertionError("preview worker did not reach simulated failure")
            yield snapshot(1)
            yield snapshot(2)

        fake = SimpleNamespace(iter_snapshots=samples, preview_report={})
        with patch.object(live, "volatile_json", side_effect=fail_preview):
            result = write_run(fake, self.run_dir)
        self.assertEqual(result["snapshots_written"], 3)
        saved = [json.loads(line) for line in (self.run_dir / "snapshots.jsonl").read_text().splitlines()]
        self.assertEqual([row["sequence"] for row in saved], [0, 1, 2])
        with (self.run_dir / "scalars.csv").open() as stream:
            self.assertEqual([int(row["sequence"]) for row in csv.DictReader(stream)], [0, 1, 2])
        self.assertIn("simulated preview cache full", fake.preview_report["last_error"])
        self.assertFalse(fake.preview_report["worker_alive"])

    def test_preview_startup_failure_does_not_interrupt_raw_write_run(self):
        fake = SimpleNamespace(iter_snapshots=lambda: iter([snapshot(0), snapshot(1)]),
                               preview_report={})
        with patch.object(live, "PreviewPublisher", side_effect=PermissionError("cache unavailable")):
            result = write_run(fake, self.run_dir)
        self.assertEqual(result["snapshots_written"], 2)
        self.assertIn("cache unavailable", fake.preview_report["last_error"])

    def test_unexpected_offer_failure_does_not_interrupt_raw_write_run(self):
        fake = SimpleNamespace(iter_snapshots=lambda: iter([snapshot(0), snapshot(1)]),
                               preview_report={})
        with patch.object(live.PreviewPublisher, "offer", side_effect=MemoryError("display failure")) as offer:
            result = write_run(fake, self.run_dir)
        self.assertEqual(result["snapshots_written"], 2)
        self.assertEqual(offer.call_count, 1)
        self.assertIn("display failure", fake.preview_report["last_error"])
        self.assertFalse(fake.preview_report["worker_alive"])

    def test_cache_json_is_finite_and_bounded(self):
        path = self.cache / "finite.json"
        live.volatile_json(path, {"value": float("nan"), "nested": [float("inf"), 1]})
        self.assertEqual(json.loads(path.read_text()), {"value": None, "nested": [None, 1]})
        with self.assertRaises(ValueError):
            live.volatile_json(path, {"value": "x" * live.MAX_PREVIEW_BYTES})
        self.assertEqual(json.loads(path.read_text())["nested"], [None, 1])


class DashboardStateTests(TemporaryDashboardTest):
    def save_session_preview(self, *, identity=None, stamp=None, preview_identity=None):
        identity = runtime.process_identity(os.getpid()) if identity is None else identity
        runtime.atomic_json(self.runtime / "session.json", {
            "run_dir": str(self.run_dir), "identity": identity, "token": "session-token"})
        runtime.atomic_json(self.run_dir / "control.json", {
            "status": "running", "token": "session-token"})
        live.volatile_json(live.preview_path(self.run_dir, self.cache), {
            "run_dir": str(self.run_dir),
            "identity": identity if preview_identity is None else preview_identity,
            "snapshot": snapshot(4, stamp=stamp), "thermal": None, "publisher": {}})

    def test_current_preview_becomes_stale_without_faking_session_death(self):
        self.save_session_preview()
        state = self.app.state()
        self.assertTrue(state["session"]["alive"])
        self.assertTrue(state["preview_current"])
        self.assertEqual(state["session"]["snapshot_count"], 5)
        self.save_session_preview(stamp=time.monotonic_ns() - 4_000_000_000)
        state = self.app.state()
        self.assertTrue(state["session"]["alive"])
        self.assertFalse(state["preview_current"])
        self.assertGreaterEqual(state["preview_age_ms"], 4000)

    def test_reused_pid_token_marks_session_failed_and_preview_not_current(self):
        identity = runtime.process_identity(os.getpid())
        identity["start_ticks"] = str(int(identity["start_ticks"]) - 1)
        self.save_session_preview(identity=identity)
        state = self.app.state()
        self.assertFalse(state["session"]["alive"])
        self.assertEqual(state["session"]["status"], "failed")
        self.assertFalse(state["preview_current"])

    def test_preview_identity_mismatch_disables_controls(self):
        self.save_session_preview(preview_identity={"pid": 123})
        state = self.app.state()
        self.assertIsNone(state["preview"])
        self.assertFalse(state["controls"]["enabled"])
        self.assertIn("does not match", state["preview_error"])

    def test_missing_preview_reports_timeout_and_recovers_on_arrival(self):
        self.save_session_preview()
        live.preview_path(self.run_dir, self.cache).unlink()
        with patch.object(dashboard.time, "monotonic", return_value=10):
            first = self.app.state()
        self.assertTrue(first["session"]["alive"])
        self.assertIsNone(first["preview_error"])
        with patch.object(dashboard.time, "monotonic", return_value=13.1):
            delayed = self.app.state()
        self.assertIsNone(delayed["preview"])
        self.assertTrue(delayed["preview_error"])
        self.assertTrue(delayed["controls"]["enabled"])
        self.save_session_preview()
        recovered = self.app.state()
        self.assertTrue(recovered["preview_current"])
        self.assertIsNone(recovered["preview_error"])
        self.assertIsNone(self.app.missing_since)

    def test_ct_raw_is_opt_in_and_session_token_is_not_exposed(self):
        self.assertNotIn("--save-ct-raw", self.app.command("start"))
        opted_in = dashboard.Dashboard(out_dir=self.root / "output", save_ct_raw=True,
                                       cache_root=self.cache)
        self.assertIn("--save-ct-raw", opted_in.command("start"))
        self.save_session_preview()
        self.assertNotIn("token", self.app.state()["session"])

    def test_async_start_stop_uses_fixed_commands_and_rejects_duplicates(self):
        entered = threading.Event()
        release = threading.Event()
        self.addCleanup(release.set)
        commands = []

        def fake_run(args, **kwargs):
            commands.append((args, kwargs))
            self.assertTrue(stat.S_ISREG(os.fstat(kwargs["stdout"].fileno()).st_mode))
            kwargs["stdout"].write(b"fake control complete")
            if args[3] == "start":
                entered.set()
                if not release.wait(3):
                    raise AssertionError("test did not release start")
            return subprocess.CompletedProcess(args, 0)

        with patch.object(dashboard.subprocess, "run", side_effect=fake_run):
            try:
                self.app.submit("start")
                self.assertTrue(entered.wait(3))
                self.assertTrue(self.app.controls()["start_pending"])
                with self.assertRaises(RuntimeError):
                    self.app.submit("start")
                self.app.submit("stop")
                self.assertTrue(self.app.controls()["stop_pending"])
                with self.assertRaises(RuntimeError):
                    self.app.submit("stop")
                release.set()
                self.wait_until(lambda: not any(self.app.pending.values()), "async controls did not finish")
            finally:
                release.set()
                self.wait_until(lambda: not any(self.app.pending.values()), "async controls did not finish")
        self.assertEqual([args[3] for args, _ in commands], ["start", "stop"])
        start_args, options = commands[0]
        self.assertEqual(start_args[:3], [sys.executable, "-u", str(JETSON / "collect.py")])
        self.assertIn(str(self.root / "output"), start_args)
        self.assertNotIn("--save-ct-raw", start_args)
        self.assertFalse(options.get("shell", False))
        self.assertTrue(options["start_new_session"])
        self.assertTrue(options["close_fds"])
        self.assertEqual(options["stdin"], subprocess.DEVNULL)
        self.assertEqual(options["stderr"], subprocess.STDOUT)
        self.assertNotIn("capture_output", options)
        self.assertIsNone(self.app.controls()["last_result"]["error"])
        self.assertEqual(self.app.controls()["last_result"]["message"], "fake control complete")

    def test_async_control_failure_clears_pending_and_reports_exit(self):
        def fail(args, **kwargs):
            kwargs["stdout"].write(b"simulated failure")
            return subprocess.CompletedProcess(args, 2)

        with patch.object(dashboard.subprocess, "run", side_effect=fail):
            self.app.submit("start")
            self.wait_until(lambda: not self.app.controls()["start_pending"], "control did not finish")
        result = self.app.controls()["last_result"]
        self.assertIn("code 2", result["error"])
        self.assertEqual(result["message"], "simulated failure")


class DashboardApiTests(TemporaryDashboardTest):
    """Exercise HTTP handler routes without binding sockets or sensor actions."""
    host = "127.0.0.1:8765"

    def request(self, method="GET", path="/api/state", *, body=b"", headers=None):
        request_headers = {"Host": self.host, "X-Dashboard-Token": self.app.token}
        if method == "POST":
            request_headers.update({"Origin": "http://" + self.host,
                                    "Content-Type": "application/json",
                                    "Content-Length": str(len(body))})
        for key, value in (headers or {}).items():
            if value is None:
                request_headers.pop(key, None)
            else:
                request_headers[key] = value
        handler = object.__new__(dashboard.Handler)
        handler.server = SimpleNamespace(app=self.app, hosts={self.host, "localhost:8765"})
        handler.path = path
        handler.headers = Message()
        for key, value in request_headers.items():
            handler.headers[key] = value
        handler.rfile = io.BytesIO(body)
        response = []
        handler.respond = lambda status, value, *args: response.append((status, value))
        getattr(handler, "do_" + method)()
        self.assertEqual(len(response), 1)
        return response[0]

    def test_state_requires_token_and_rejects_foreign_host_and_origin(self):
        for headers, expected in (
            ({"X-Dashboard-Token": None}, 401),
            ({"X-Dashboard-Token": "wrong"}, 401),
            ({"Host": "attacker.example:8765"}, 403),
            ({"Host": None}, 403),
            ({"Origin": "http://attacker.example"}, 403),
        ):
            with self.subTest(headers=headers):
                self.assertEqual(self.request(headers=headers)[0], expected)
        self.assertEqual(self.request()[0], 200)

    def test_control_requires_matching_origin_and_token(self):
        with patch.object(self.app, "submit") as submit:
            for headers, expected in (
                ({"Origin": None}, 403),
                ({"Origin": "null"}, 403),
                ({"Origin": "http://localhost:8765"}, 403),
                ({"X-Dashboard-Token": None}, 401),
                ({"Host": "attacker.example:8765"}, 403),
            ):
                with self.subTest(headers=headers):
                    status, _ = self.request("POST", "/api/control", body=b'{"action":"start"}', headers=headers)
                    self.assertEqual(status, expected)
            submit.assert_not_called()

    def test_only_fixed_paths_are_available(self):
        for path in ("/../collect.py", "/%2e%2e/collect.py", "/api/state?file=/etc/passwd",
                     "/api/../../collect.py", "/api/control"):
            with self.subTest(path=path):
                self.assertEqual(self.request(path=path)[0], 404)
        with patch.object(self.app, "submit") as submit:
            self.assertEqual(self.request("POST", "/api/start", body=b'{"action":"start"}')[0], 404)
            submit.assert_not_called()

    def test_json_control_rejects_options_malformed_and_oversized_bodies(self):
        cases = [
            (b"", {}), (b"[]", {}), (b"null", {}), (b"{", {}), (b"\xff", {}),
            (b'{"action":"restart"}', {}), (b'{"action":["start"]}', {}),
            (b'{"action":"start","out_dir":"/tmp/other"}', {}),
            (b'{"action":"start"}', {"Content-Type": "text/plain"}),
            (b'{"action":"start"}', {"Content-Length": "-1"}),
            (b'{"action":"start"}', {"Content-Length": "garbage"}),
            (b'{"action":"start"}', {"Transfer-Encoding": "chunked"}),
            (b"x" * 257, {}),
        ]
        with patch.object(self.app, "submit") as submit:
            for body, headers in cases:
                with self.subTest(body=body, headers=headers):
                    self.assertEqual(self.request("POST", "/api/control", body=body, headers=headers)[0], 400)
            submit.assert_not_called()

    def test_valid_control_passes_only_action(self):
        with patch.object(self.app, "submit") as submit:
            status, body = self.request("POST", "/api/control", body=b'{"action":"start"}')
            self.assertEqual((status, body), (202, {"accepted": "start"}))
            submit.assert_called_once_with("start")

    def test_replay_rejects_controls_and_chunk_path_escape(self):
        recorded = snapshot(0)
        recorded["sensors"]["flir"].update(thermal_chunk="../thermal_escape.npz", thermal_index=0)
        (self.run_dir / "snapshots.jsonl").write_text(json.dumps(recorded) + "\n")
        self.app = dashboard.Dashboard(out_dir=self.root / "output", replay=self.run_dir,
                                       cache_root=self.cache)
        with patch.object(dashboard.subprocess, "run") as run, patch.object(np, "load") as load:
            status, result = self.request()
            self.assertEqual(status, 200)
            self.assertEqual(result["mode"], "replay")
            self.assertFalse(result["controls"]["enabled"])
            self.assertIn("Invalid thermal chunk path", result["preview_error"])
            self.assertIsNone(result["preview"]["thermal"])
            self.assertEqual(self.request("POST", "/api/control", body=b'{"action":"start"}')[0], 409)
            self.assertEqual(self.request("POST", "/api/control", body=b'{"action":"stop"}')[0], 409)
            run.assert_not_called()
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
