"""ZED side recorder: switch, active-run detection, tick alignment, segment files, upload gate.

No camera and no pyzed: a fake camera delivers frames on a 15 fps clock.
Run with ./jetson_deploy/run_python.sh -m unittest tests.test_zed_recorder
"""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "jetson_deploy"))
sys.path.insert(0, str(ROOT))

from uplink.units import discover, zed_still_writing  # noqa: E402
from zedcap import control  # noqa: E402
from zedcap.recorder import BEYOND_RANGE, NO_DATA, Session, depth_to_mm, read_tick_clock  # noqa: E402
from zedcap.service import Recorder  # noqa: E402

SECOND = 1_000_000_000
FAKE_JPEG = b"\xff\xd8\xff\xe0fake-jpeg"


def fake_encoder(quality, size=None):
    return lambda rgb: FAKE_JPEG + bytes([int(rgb[0, 0, 0])])


def read_index(run: Path) -> list[dict]:
    with open(run / "zed" / "frames.csv", newline="") as stream:
        return list(csv.DictReader(stream))


def me() -> dict:
    return control.process_identity(os.getpid())


def dead() -> dict:
    return dict(me(), boot_id="00000000-previous-boot")


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="zed_test_"))
        self.runtime = self.tmp / "runtime"
        self.runtime.mkdir(mode=0o700)
        self.roots = [str(self.tmp / "runs"), str(self.tmp / "dataset")]
        self.config_path = self.tmp / "zed.json"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_dir(self, name="20261013T000000Z", t0=None, kind="run") -> Path:
        run = self.tmp / ("runs" if kind == "run" else "dataset/dust") / name
        run.mkdir(parents=True)
        if kind == "run":
            (run / "control.json").write_text(json.dumps({"status": "running"}))
        else:
            (run / "experiment.json").write_text(json.dumps({"status": "running"}))
        if t0 is not None:
            (run / "snapshots.jsonl").write_text(json.dumps({"sequence": 0, "target_monotonic_ns": t0}) + "\n")
        return run

    def activate(self, run: Path, identity=None) -> None:
        (self.runtime / control.ACTIVE_NAME).write_text(json.dumps({"run_dir": str(run), "identity": identity or me()}))

    def session(self, run, t0=0, **config):
        cfg = dict(control.DEFAULT_CONFIG, **config)
        return Session(run, camera_info={"model": "fake"}, config=cfg, encode_jpeg=fake_encoder(85),
                       t0_ns=t0, tick_clock="test", recorder=me())


class SwitchTests(Base):
    def test_missing_or_broken_config_means_off(self):
        self.assertFalse(control.load_config(self.config_path)["enabled"])
        self.config_path.write_text("{not json")
        config = control.load_config(self.config_path)
        self.assertFalse(config["enabled"])
        self.assertIn("JSONDecodeError", config["config_error"])
        self.config_path.write_text(json.dumps({"enabled": True, "fps": 0}))
        config = control.load_config(self.config_path)
        self.assertFalse(config["enabled"], "an invalid value never switches recording on")
        self.assertIn("fps", config["config_error"])

    def test_set_enabled_flips_only_the_switch(self):
        self.config_path.write_text(json.dumps({"enabled": False, "jpeg_quality": 70, "roots": self.roots}))
        config = control.set_enabled(True, by="hub:boss", path=self.config_path)
        raw = json.loads(self.config_path.read_text())
        self.assertEqual((config["enabled"], raw["jpeg_quality"], raw["roots"], raw["changed_by"]),
                         (True, 70, self.roots, "hub:boss"))
        self.assertFalse(control.set_enabled(False, path=self.config_path)["enabled"])


class ActiveRunTests(Base):
    def test_only_a_live_collector_inside_the_roots_with_no_final_status_counts(self):
        self.assertIsNone(control.active_run(self.roots, self.runtime)[0])
        run = self.run_dir()
        self.activate(run, dead())
        self.assertEqual(control.active_run(self.roots, self.runtime), (None, "no collection running"))
        self.activate(run)
        active, reason = control.active_run(self.roots, self.runtime)
        self.assertEqual((active["run_id"], active["kind"], reason), (run.name, "run", "collecting"))
        self.assertIsNone(control.active_run([str(self.tmp / "elsewhere")], self.runtime)[0], "outside roots")
        (run / "control.json").write_text(json.dumps({"status": "completed"}))
        self.assertEqual(control.active_run(self.roots, self.runtime)[1], "run already finished")

    def test_trials_are_followed_until_their_experiment_is_final(self):
        trial = self.run_dir("trial_001", kind="trial")
        self.activate(trial)
        self.assertEqual(control.active_run(self.roots, self.runtime)[0]["kind"], "trial")
        (trial / "experiment.json").write_text(json.dumps({"status": "aborted"}))
        self.assertIsNone(control.active_run(self.roots, self.runtime)[0])

    def test_tick_clock_comes_from_the_first_complete_snapshot_line(self):
        run = self.run_dir()
        self.assertIsNone(read_tick_clock(run))
        (run / "snapshots.jsonl").write_text('{"sequence": 0, "target_monotonic_ns": 5')
        self.assertIsNone(read_tick_clock(run), "a half-written first line is not trusted")
        (run / "snapshots.jsonl").write_text(json.dumps({"sequence": 2, "target_monotonic_ns": 7 * SECOND}) + "\n")
        self.assertEqual(read_tick_clock(run), 5 * SECOND)


class SessionTests(Base):
    def feed(self, session, start_s, end_s, fps=15, phase_ms=7.0, skip=()):
        tick_frames = []
        n = int((end_s - start_s) * fps)
        for i in range(n):
            ns = int((start_s + i / fps) * SECOND + phase_ms * 1e6)
            if any(a <= ns / SECOND < b for a, b in skip):
                continue
            left = np.full((4, 6, 3), i % 251, dtype=np.uint8)
            depth = np.full((2, 3), 1.5, dtype=np.float32)
            if session.wants(ns):
                tick = session.offer(ns, ns + 10, left, depth, {"imu_az": -9.8})
                tick_frames.append(tick)
        return tick_frames

    def test_one_frame_per_tick_closest_to_the_tick_target(self):
        run = self.run_dir()
        session = self.session(run, segment_frames=4)
        ticks = self.feed(session, 0, 10.0, phase_ms=7.0)
        record = session.close("collection_ended")
        self.assertEqual(ticks, list(range(0, 10)))
        self.assertEqual((record["frames"], record["missed_ticks"], record["first_tick"], record["last_tick"]),
                         (10, 0, 0, 9))
        rows = read_index(run)
        self.assertEqual([int(r["tick"]) for r in rows], list(range(10)))
        self.assertTrue(all(abs(float(r["offset_ms"]) - 7.0) < 0.01 for r in rows))
        self.assertEqual(rows[0]["imu_az"], "-9.8")

    def test_gaps_are_counted_never_filled(self):
        run = self.run_dir()
        session = self.session(run)
        ticks = self.feed(session, 0, 8.0, skip=[(2.5, 5.5)])
        record = session.close("collection_ended")
        self.assertEqual(ticks, [0, 1, 2, 6, 7])
        self.assertEqual(record["missed_ticks"], 3)

    def test_segments_are_complete_files_that_the_index_points_into(self):
        run = self.run_dir()
        session = self.session(run, segment_frames=4, depth_mode="PERFORMANCE")
        self.feed(session, 0, 10.0)
        session.close("switched_off")
        files = sorted(p.name for p in (run / "zed").iterdir())
        self.assertEqual(files, ["frames.csv", "seg_01_00000_depth.npz", "seg_01_00000_left.npz",
                                 "seg_01_00001_depth.npz", "seg_01_00001_left.npz",
                                 "seg_01_00002_depth.npz", "seg_01_00002_left.npz", "zed.json"])
        rows = read_index(run)
        for row in rows:
            stem = f"seg_{int(row['session']):02d}_{int(row['segment']):05d}"
            left = np.load(run / "zed" / f"{stem}_left.npz")
            depth = np.load(run / "zed" / f"{stem}_depth.npz")["depth_mm"]
            offsets, data = left["left_offsets"], left["left_jpeg"]
            i = int(row["index"])
            self.assertTrue(data[offsets[i]:offsets[i + 1]].tobytes().startswith(FAKE_JPEG))
            self.assertEqual((depth.dtype, int(depth[i, 0, 0])), (np.uint16, 1500))
        with zipfile.ZipFile(run / "zed" / "seg_01_00000_left.npz") as archive:
            self.assertEqual(archive.getinfo("left_jpeg.npy").compress_type, zipfile.ZIP_STORED,
                             "JPEG members stay seekable for the hub")
        meta = json.loads((run / "zed" / "zed.json").read_text())
        self.assertEqual((meta["schema"], meta["status"], meta["sessions"][0]["stop_reason"]),
                         ("zed-v1", "complete", "switched_off"))
        self.assertNotIn("roots", meta["config"])

    def test_default_records_left_rgb_only(self):
        self.assertEqual(control.DEFAULT_CONFIG["depth_mode"], "NONE")
        run = self.run_dir()
        session = self.session(run, segment_frames=4)
        self.feed(session, 0, 10.0)
        session.close("collection_ended")
        files = sorted(p.name for p in (run / "zed").iterdir())
        self.assertEqual(files, ["frames.csv", "seg_01_00000_left.npz", "seg_01_00001_left.npz",
                                 "seg_01_00002_left.npz", "zed.json"])
        self.assertEqual(len(read_index(run)), 10)
        meta = json.loads((run / "zed" / "zed.json").read_text())
        self.assertIsNone(meta["depth_encoding"])
        self.assertIsNone(meta["files"]["depth"])

    def test_a_second_session_appends_and_marks_a_dead_one_interrupted(self):
        run = self.run_dir()
        first = self.session(run)
        self.feed(first, 0, 3.0)
        first.flush()
        first._queue.put(None)
        first._writer.join()                     # simulate a crash: never closed
        second = self.session(run)
        self.feed(second, 5, 7.0)
        second.close("collection_ended")
        meta = json.loads((run / "zed" / "zed.json").read_text())
        self.assertEqual([(s["session"], s["stop_reason"]) for s in meta["sessions"]],
                         [(1, "interrupted"), (2, "collection_ended")])
        sessions = {r["session"] for r in read_index(run)}
        self.assertEqual(sessions, {"1", "2"})

    def test_depth_encoding(self):
        values = np.array([[1.234, np.nan, np.inf, -np.inf, 0.0, 99.0]], dtype=np.float32)
        self.assertEqual(depth_to_mm(values).tolist(), [[1234, NO_DATA, BEYOND_RANGE, NO_DATA, NO_DATA, 65534]])

    def test_a_writer_failure_ends_the_session_with_that_error(self):
        run = self.run_dir()
        session = self.session(run, segment_frames=2)
        session._write_npz = lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
        with self.assertRaises(RuntimeError):
            for second in range(0, 50, 2):           # later frames see the writer's error
                self.feed(session, second, second + 2.0)
                time.sleep(0.05)
        record = session.close("recorder_error")
        self.assertIn("disk full", record["error"])


class FakeCamera:
    """15 fps frames on the real monotonic clock, delivered as fast as asked (no sleeping)."""

    def __init__(self, config, fail_open=False):
        self.fail_open = fail_open
        self.closed = False
        self.ns = time.monotonic_ns()

    def open(self):
        if self.fail_open:
            raise RuntimeError("CAMERA NOT DETECTED")
        return {"model": "fake", "serial_number": 1}

    def grab(self):
        time.sleep(0.002)
        self.ns = time.monotonic_ns()
        return self.ns, self.ns

    def retrieve(self):
        return np.zeros((4, 6, 3), np.uint8), np.ones((2, 3), np.float32), {}

    def close(self):
        self.closed = True


class ServiceTests(Base):
    def recorder(self, cameras):
        made = []

        def factory(config):
            camera = cameras.pop(0) if cameras else FakeCamera(config)
            made.append(camera)
            return camera
        rec = Recorder(camera_factory=factory, encoder_factory=fake_encoder, config_path=self.config_path,
                       status_root=self.runtime, poll_s=0.05, identity=me())
        return rec, made

    def write_config(self, enabled):
        self.config_path.write_text(json.dumps({"enabled": enabled, "roots": self.roots}))
        control.set_enabled(enabled, path=self.config_path)

    def test_records_while_on_and_collecting_and_stops_when_switched_off(self):
        run = self.run_dir(t0=time.monotonic_ns())
        self.activate(run)
        self.write_config(True)
        rec, made = self.recorder([])
        thread = threading.Thread(target=rec.run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and (control.read_status(self.runtime) or {}).get("frames", 0) < 2:
            time.sleep(0.05)
        self.assertEqual(control.read_status(self.runtime)["state"], "recording")
        control.set_enabled(False, path=self.config_path)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and (control.read_status(self.runtime) or {}).get("state") != "disabled":
            time.sleep(0.05)
        rec.stop.set()
        thread.join(5)
        meta = json.loads((run / "zed" / "zed.json").read_text())
        self.assertEqual((meta["status"], meta["sessions"][0]["stop_reason"]), ("complete", "switched_off"))
        self.assertGreaterEqual(meta["sessions"][0]["frames"], 2)
        self.assertTrue(made[0].closed, "camera is closed when not recording")

    def test_ends_when_the_collection_ends(self):
        run = self.run_dir(t0=time.monotonic_ns())
        self.activate(run)
        self.write_config(True)
        rec, _ = self.recorder([])
        thread = threading.Thread(target=rec.run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and (control.read_status(self.runtime) or {}).get("frames", 0) < 1:
            time.sleep(0.05)
        (run / "control.json").write_text(json.dumps({"status": "completed"}))
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and (control.read_status(self.runtime) or {}).get("state") != "idle":
            time.sleep(0.05)
        rec.stop.set()
        thread.join(5)
        meta = json.loads((run / "zed" / "zed.json").read_text())
        self.assertEqual(meta["sessions"][-1]["stop_reason"], "collection_ended")
        self.assertEqual(len(meta["sessions"]), 1, "no new session after the run finished")

    def test_an_open_failure_is_kept_and_not_retried_until_the_switch_changes(self):
        run = self.run_dir(t0=time.monotonic_ns())
        self.activate(run)
        self.write_config(True)
        rec, made = self.recorder([FakeCamera({}, fail_open=True)])
        rec.step()
        status = control.read_status(self.runtime)
        self.assertEqual(status["state"], "error")
        self.assertIn("CAMERA NOT DETECTED", status["error"])
        for _ in range(3):
            rec.step()
        self.assertEqual(len(made), 1, "no automatic retry for the same run")
        self.assertIn("switch ZED off and on", control.read_status(self.runtime)["reason"])
        self.assertFalse((run / "zed").exists(), "nothing written when the camera never opened")
        time.sleep(1.1)                          # changed_utc has one-second resolution
        control.set_enabled(False, path=self.config_path)
        control.set_enabled(True, path=self.config_path)
        rec.stop.set()                           # the retry records nothing: it stops at once
        rec.step()
        self.assertEqual(len(made), 2, "toggling the switch allows one new attempt")

    def test_disabled_never_opens_the_camera(self):
        run = self.run_dir(t0=time.monotonic_ns())
        self.activate(run)
        self.write_config(False)
        rec, made = self.recorder([])
        rec.step()
        self.assertEqual((made, control.read_status(self.runtime)["state"]), ([], "disabled"))


class UploadGateTests(Base):
    def test_unit_waits_while_its_zed_recorder_writes(self):
        run = self.tmp / "runs" / "20261013T010000Z"
        run.mkdir(parents=True)
        (run / "metadata.json").write_text(json.dumps({"run_id": run.name, "status": "completed"}))
        (run / "zed").mkdir()
        meta = run / "zed" / "zed.json"
        old = time.time() - 3600
        for status, recorder, expect in (("recording", me(), True), ("recording", dead(), False),
                                         ("complete", me(), False)):
            meta.write_text(json.dumps({"status": status, "recorder": recorder}))
            for path in (meta, run / "metadata.json"):
                os.utime(path, (old, old))
            with self.subTest(status=status, alive=recorder == me()):
                self.assertEqual(zed_still_writing(run), expect)
                units, skipped = discover({"runs": self.tmp / "runs"}, settle_s=60)
                self.assertEqual(bool(skipped), expect)
                self.assertEqual(len(units), 0 if expect else 1)


if __name__ == "__main__":
    unittest.main()
