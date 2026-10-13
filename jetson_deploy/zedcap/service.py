"""Recorder loop: wait for (switch on AND collection running), record one session, repeat.

* The camera is opened only while a collection runs and the switch is on, and closed otherwise.
* A failure (camera cannot open, frames stop arriving) is recorded once in zed.json and the live
  status. There is no automatic retry for that run: switch ZED off and on again, or start a new
  collection, to try again (repository rule: keep the first error, no hidden retry loops).
* It never writes outside ``<run>/zed/`` and never touches collector files.
"""
from __future__ import annotations

import base64
import os
import threading
import time
from pathlib import Path

from . import control
from .recorder import Session, read_tick_clock, utc_now

GRAB_FAILURE_S = 5.0          # no frame for this long ends the session with an error
CLOCK_WAIT_S = 30.0           # how long to wait for the collector's first snapshot


class Recorder:
    def __init__(self, *, camera_factory, encoder_factory, config_path: Path = control.CONFIG_PATH,
                 status_root: Path | None = None, stop: threading.Event | None = None,
                 poll_s: float = 1.0, identity: dict | None = None):
        self.camera_factory = camera_factory
        self.encoder_factory = encoder_factory
        self.config_path = Path(config_path)
        self.status_root = status_root
        self.stop = stop or threading.Event()
        self.poll_s = poll_s
        self.identity = identity or {"pid": os.getpid()}
        self.failed: dict[str, tuple] = {}        # run_dir -> (switch change stamp, error)
        self.status = {"state": "starting"}
        self.preview = None
        self._last_status = 0.0

    # ------------------------------------------------------------- status
    def publish(self, force: bool = False, **fields) -> None:
        self.status.update(fields)
        now = time.monotonic()
        if not force and now - self._last_status < 1.0:
            return
        self._last_status = now
        try:
            control.write_status(dict(self.status, pid=os.getpid(), preview=self.preview), self.status_root)
        except OSError:
            pass                                    # status is a display aid; recording continues

    # --------------------------------------------------------------- loop
    def run(self) -> None:
        while not self.stop.is_set():
            self.step()
            self.stop.wait(self.poll_s)
        self.publish(force=True, state="stopped", run_id=None)

    def step(self) -> None:
        config = control.load_config(self.config_path)
        base = {"enabled": config["enabled"], "config_error": config.get("config_error"),
                "switch_changed_utc": config.get("changed_utc"), "switch_changed_by": config.get("changed_by")}
        if not config["enabled"]:
            self.preview = None
            self.publish(force=True, state="disabled", run_id=None, reason="switched off", **base)
            return
        run, reason = control.active_run(config["roots"], self.status_root)
        if run is None:
            self.publish(force=True, state="idle", run_id=None, reason=reason, **base)
            return
        stamp = config.get("changed_utc")
        failed = self.failed.get(str(run["run_dir"]))
        if failed and failed[0] == stamp:
            self.publish(force=True, state="error", run_id=run["run_id"], reason="failed for this run; "
                         "switch ZED off and on to try again", error=failed[1], **base)
            return
        self.publish(force=True, state="opening", run_id=run["run_id"], reason="opening camera", error=None, **base)
        reason, error = self.record(run, config)
        if error:
            self.failed[str(run["run_dir"])] = (stamp, error)
        self.publish(force=True, state="error" if error else "idle", reason=reason, error=error)

    def _should_stop(self, run: dict, stamp) -> str | None:
        if self.stop.is_set():
            return "recorder_stopped"
        config = control.load_config(self.config_path)
        if not config["enabled"]:
            return "switched_off"
        if config.get("changed_utc") != stamp:
            return "switch_changed"
        current, _ = control.active_run(config["roots"], self.status_root)
        if current is None or current["run_dir"] != run["run_dir"]:
            return "collection_ended"
        return None

    def record(self, run: dict, config: dict) -> tuple[str, str | None]:
        """Record one session into run/zed until a stop condition. Returns (reason, error)."""
        stamp = config.get("changed_utc")
        camera = None
        session = None
        reason, error = "collection_ended", None
        try:
            try:
                camera = self.camera_factory(config)
                info = camera.open()
            except Exception as exc:
                return "camera_open_failed", f"{type(exc).__name__}: {exc}"[:300]
            encode = self.encoder_factory(config["jpeg_quality"], None)
            thumbnail = self.encoder_factory(70, (320, 180))
            # wait for the collector's tick clock while the camera warms up
            t0, source = None, "collector snapshots.jsonl (sequence 0 target_monotonic_ns)"
            deadline = time.monotonic() + CLOCK_WAIT_S
            first_frame = None
            while t0 is None:
                stop = self._should_stop(run, stamp)
                if stop:
                    return stop, None
                t0 = read_tick_clock(run["run_dir"])
                grabbed = camera.grab()
                if grabbed and first_frame is None:
                    first_frame = grabbed[0]
                if t0 is None and time.monotonic() > deadline:
                    if first_frame is None:
                        return "no_frames", "camera opened but delivered no frame"
                    t0, source = first_frame, "recorder (collector clock not found within 30 s)"
            session = Session(run["run_dir"], camera_info=info, config=config, encode_jpeg=encode, t0_ns=t0,
                              tick_clock=source, recorder=self.identity)
            self.publish(force=True, state="recording", run_id=run["run_id"], session=session.number,
                         reason="recording", camera={"model": info.get("model"),
                                                     "serial_number": info.get("serial_number")})
            last_frame = time.monotonic()
            last_check = 0.0
            last_preview = 0.0
            grabs = 0
            rate_start = time.monotonic()
            while True:
                now = time.monotonic()
                if now - last_check >= 0.5:
                    last_check = now
                    stop = self._should_stop(run, stamp)
                    if stop:
                        reason = stop
                        break
                grabbed = camera.grab()
                if grabbed is None:
                    if time.monotonic() - last_frame > GRAB_FAILURE_S:
                        reason, error = "grab_failed", f"no frame for {GRAB_FAILURE_S:.0f} s"
                        break
                    time.sleep(0.01)
                    continue
                last_frame = time.monotonic()
                grabs += 1
                frame_ns, image_ns = grabbed
                if session.wants(frame_ns):
                    left, depth, imu = camera.retrieve()
                    tick = session.offer(frame_ns, image_ns, left, depth, imu)
                    if tick is not None and now - last_preview >= float(config["preview_every_s"]):
                        last_preview = now
                        self.preview = {"tick": tick, "utc": utc_now(), "run_id": run["run_id"],
                                        "jpeg_b64": base64.b64encode(thumbnail(left)).decode("ascii")}
                elapsed = time.monotonic() - rate_start
                record = session.record
                self.publish(frames=record["frames"], missed_ticks=record["missed_ticks"],
                             last_tick=record["last_tick"], bytes_written=session.bytes_written,
                             grab_fps=round(grabs / elapsed, 1) if elapsed > 0 else None)
                if elapsed >= 10:
                    grabs, rate_start = 0, time.monotonic()
        except Exception as exc:
            reason, error = "recorder_error", f"{type(exc).__name__}: {exc}"[:300]
        finally:
            if camera is not None:
                try:
                    camera.close()
                except Exception:
                    pass
            if session is not None:
                try:
                    session.close(reason, error)
                except Exception as exc:
                    error = error or f"closing session failed: {type(exc).__name__}: {exc}"[:300]
        return reason, error
